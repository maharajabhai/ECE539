#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG Privacy + Blinder-style Anonymization Pipeline on PTB-XL
with time-domain overlay plots (original vs anonymized).

Run from project root:
    cd /Users/bluitel/Documents/ECE539/ECE539
    conda activate ecg_env   # whatever your env is
    python scripts/ecg_blinder_pipeline_with_overlay.py
"""

import os
import sys
import json
import pickle
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from scipy import signal
import matplotlib.pyplot as plt
from ecg_priv_common import compute_diag_metrics_thresholded
from ecg_priv_models import DiagnosisClassifier, compute_bpm_labels
from ptbxl_loader import load_ptbxl

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

PTBXL_BENCH_CODE = os.path.join(PROJECT_ROOT, "ecg_ptbxl_benchmarking", "code")
for p in [PROJECT_ROOT, PTBXL_BENCH_CODE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ecg_ptbxl_benchmarking imports (utils only)
from utils import utils


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@dataclass
class Config:
    datafolder: str = os.path.join(PROJECT_ROOT, "data")

    outputfolder: str = os.path.join(
        PROJECT_ROOT,
        "ecg_ptbxl_benchmarking",
        "output",
    )

    experiment: str = "exp0"
    modelname: str = "fastai_xresnet1d101"
    n_classes_pretrained: int = 71

    sampling_frequency: int = 500  # Hz

    # identity classifier
    identity_batch_size: int = 32
    identity_lr: float = 1e-3
    identity_epochs: int = 200
    min_samples_per_patient: int = 5
    identity_val_fraction: float = 0.2  # per-patient val fraction

    # VAE / Blinder-style autoencoder
    use_blinder: bool = True
    blinder_z_dim: int = 64
    blinder_epochs: int = 200
    blinder_batch_size: int = 2
    blinder_lr: float = 1e-3

    # utility surrogate evaluation
    utility_eval_batch_size: int = 32

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    identity_ckpt: str = os.path.join(PROJECT_ROOT, "scripts", "identity_best.pt")
    blinder_ckpt: str = os.path.join(PROJECT_ROOT, "scripts", "blinder_vae.pt")

    results_dir: str = os.path.join(PROJECT_ROOT, "privacy_baseline_outputs")

    seed: int = 1234


cfg = Config()
os.makedirs(cfg.results_dir, exist_ok=True)

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.device == "cuda":
    torch.cuda.manual_seed_all(cfg.seed)


# -------------------------------------------------------------------
# Utility label builder
# -------------------------------------------------------------------

def build_utility_labels_from_superclass(Y: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    all_super = []
    for lst in Y["diagnostic_superclass"].values:
        all_super.extend(lst)
    super_classes = sorted(list(set(all_super)))
    col2idx = {c: i for i, c in enumerate(super_classes)}

    N = len(Y)
    K = len(super_classes)
    y_util = np.zeros((N, K), dtype=np.float32)
    for i, lst in enumerate(Y["diagnostic_superclass"].values):
        for c in lst:
            y_util[i, col2idx[c]] = 1.0

    return y_util, super_classes


# -------------------------------------------------------------------
# Identity meta with per-patient train/val split
# -------------------------------------------------------------------

def bin_numeric(col: pd.Series, bins: List[int]) -> pd.Series:
    labels = list(range(len(bins) - 1))
    return pd.cut(col, bins=bins, labels=labels, include_lowest=True).astype(int)


def build_identity_meta(Y: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """
    Build metadata for identity classification:
    - filter to patients with >= min_samples_per_patient
    - create patient index + binned age/height/weight + sex_bin
    - create a per-patient train/val split 'id_fold' (0=train, 1=val)
    """
    meta = Y.copy().reset_index(drop=False)  # 'index' is ecg_id
    meta["row_idx"] = np.arange(len(meta))

    counts = meta["patient_id"].value_counts()
    if cfg.min_samples_per_patient and cfg.min_samples_per_patient > 1:
        valid_pids = counts[counts >= cfg.min_samples_per_patient].index
        meta = meta[meta["patient_id"].isin(valid_pids)].reset_index(drop=True)
    else:
        valid_pids = counts.index
        meta = meta.reset_index(drop=True)

    print(f"[Identity] After min_samples filter: {len(meta)} records "
          f"from {len(valid_pids)} patients")

    unique_pids = sorted(meta["patient_id"].unique())
    pid2idx = {pid: i for i, pid in enumerate(unique_pids)}
    meta["patient_idx"] = meta["patient_id"].map(pid2idx)

    meta["sex_bin"] = meta["sex"].fillna(0).astype(int)

    age_bins = [0, 30, 45, 60, 75, 200]
    meta["age_bin"] = bin_numeric(meta["age"].fillna(meta["age"].median()), age_bins)

    height_bins = [0, 150, 165, 180, 200, 300]
    meta["height_bin"] = bin_numeric(
        meta["height"].fillna(meta["height"].median()), height_bins
    )

    weight_bins = [0, 60, 80, 100, 120, 300]
    meta["weight_bin"] = bin_numeric(
        meta["weight"].fillna(meta["weight"].median()), weight_bins
    )

    folds = np.zeros(len(meta), dtype=int)
    for pid in unique_pids:
        idxs = np.where(meta["patient_id"].values == pid)[0]
        if len(idxs) == 1:
            folds[idxs] = 0
            continue
        n_val = max(1, int(np.round(cfg.identity_val_fraction * len(idxs))))
        val_idx = np.random.choice(idxs, size=n_val, replace=False)
        folds[val_idx] = 1
    meta["id_fold"] = folds

    print(
        f"[Identity] Split: "
        f"{(meta['id_fold']==0).sum()} train samples, "
        f"{(meta['id_fold']==1).sum()} val samples."
    )

    info = {
        "pid2idx": pid2idx,
        "idx2pid": {v: k for k, v in pid2idx.items()},
        "age_bins": age_bins,
        "height_bins": height_bins,
        "weight_bins": weight_bins,
        "n_patients": len(unique_pids),
        "n_age_bins": len(age_bins) - 1,
        "n_height_bins": len(height_bins) - 1,
        "n_weight_bins": len(weight_bins) - 1,
    }
    return meta, info


# -------------------------------------------------------------------
# Identity dataset + model
# -------------------------------------------------------------------

class IdentityDataset(Dataset):
    def __init__(self,
                 data_std: np.ndarray,  # (N, T, C)
                 meta: pd.DataFrame,
                 split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.meta = subset.reset_index(drop=True)

        row_idx = self.meta["row_idx"].values
        self.data = np.transpose(data_std[row_idx], (0, 2, 1)).astype(np.float32)

        self.patient = self.meta["patient_idx"].values.astype(int)
        self.sex = self.meta["sex_bin"].values.astype(int)
        self.age = self.meta["age_bin"].values.astype(int)
        self.height = self.meta["height_bin"].values.astype(int)
        self.weight = self.meta["weight_bin"].values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = {
            "patient": torch.tensor(self.patient[idx], dtype=torch.long),
            "sex": torch.tensor(self.sex[idx], dtype=torch.long),
            "age": torch.tensor(self.age[idx], dtype=torch.long),
            "height": torch.tensor(self.height[idx], dtype=torch.long),
            "weight": torch.tensor(self.weight[idx], dtype=torch.long),
        }
        return x, y


class IdentityNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_patients: int,
                 n_age_bins: int,
                 n_height_bins: int,
                 n_weight_bins: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, 2, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 7, 2, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head_patient = nn.Linear(128, n_patients)
        self.head_sex = nn.Linear(128, 2)
        self.head_age = nn.Linear(128, n_age_bins)
        self.head_height = nn.Linear(128, n_height_bins)
        self.head_weight = nn.Linear(128, n_weight_bins)

    def forward(self, x):
        h = self.features(x).squeeze(-1)
        return {
            "patient": self.head_patient(h),
            "sex": self.head_sex(h),
            "age": self.head_age(h),
            "height": self.head_height(h),
            "weight": self.head_weight(h),
            "feat": h,
        }


def identity_loss(outputs, targets):
    ce = nn.CrossEntropyLoss()
    loss = 0.0
    for key in ["patient", "sex", "age", "height", "weight"]:
        loss = loss + ce(outputs[key], targets[key])
    return loss


def train_identity(model: IdentityNet,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   cfg: Config):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.identity_lr)
    best_val_loss = float("inf")

    for epoch in range(cfg.identity_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = {k: v.to(cfg.device) for k, v in yb.items()}
            opt.zero_grad()
            out = model(xb)
            loss = identity_loss(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = {k: v.to(cfg.device) for k, v in yb.items()}
                out = model(xb)
                loss = identity_loss(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Identity] Epoch {epoch+1}/{cfg.identity_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), cfg.identity_ckpt.replace(".pt", "_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.identity_ckpt)

    print(f"[Identity] Best val loss: {best_val_loss:.4f}")


def eval_identity(model: IdentityNet, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    model.to(cfg.device)

    all_true = {k: [] for k in ["patient", "sex", "age", "height", "weight"]}
    all_pred = {k: [] for k in ["patient", "sex", "age", "height", "weight"]}

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            out = model(xb)
            for key in all_true.keys():
                logits = out[key].cpu().numpy()
                preds = np.argmax(logits, axis=1)
                all_pred[key].extend(preds.tolist())
                all_true[key].extend(yb[key].numpy().tolist())

    metrics = {}
    print("[Identity] Validation metrics:")
    for key in all_true.keys():
        acc = accuracy_score(all_true[key], all_pred[key])
        f1 = f1_score(all_true[key], all_pred[key], average="macro")
        metrics[f"{key}_acc"] = float(acc)
        metrics[f"{key}_f1"] = float(f1)
        print(f"  {key}_acc: {acc:.4f}")
        print(f"  {key}_f1:  {f1:.4f}")
    return metrics


# -------------------------------------------------------------------
# Utility model (fastai) + metrics
# -------------------------------------------------------------------

def train_or_load_utility_model_from_ptbxl(
    X: np.ndarray,
    Y_db: pd.DataFrame,
    y_util: np.ndarray,
    cfg: Config,
):
    print("[Utility] Preparing standardization...")
    util_dir = os.path.join(PROJECT_ROOT, "models", "ecg_privdiffuser")
    scaler_path = os.path.join(util_dir, "util_scaler.pkl")
    if os.path.exists(scaler_path):
        print(f"[Utility] Loading StandardScaler from {scaler_path}")
        standard_scaler = pickle.load(open(scaler_path, "rb"))
    else:
        print("[Utility] Fitting StandardScaler on full dataset...")
        X_flat = X.reshape(len(X), -1)
        standard_scaler = StandardScaler(copy=False)
        standard_scaler.fit(X_flat)
        os.makedirs(util_dir, exist_ok=True)
        pickle.dump(standard_scaler, open(scaler_path, "wb"))
        print(f"[Utility] Saved scaler to {scaler_path}")

    # Apply scaler directly (flatten then reshape)
    X_flat = X.reshape(len(X), -1)
    X_std = standard_scaler.transform(X_flat).reshape(X.shape).astype(np.float32, copy=False)
    del X_flat
    gc.collect()

    # Use full dataset for utility metrics (no strat_fold needed)
    num_classes = y_util.shape[1]
    C = X_std.shape[2]

    diag_ckpt = os.path.join(util_dir, "diag_surrogate.pt")
    model = DiagnosisClassifier(in_channels=C, n_classes=num_classes, embed_dim=96).to(cfg.device)
    if os.path.isfile(diag_ckpt):
        print(f"[Utility] Loading diagnosis surrogate from {diag_ckpt}")
        state = torch.load(diag_ckpt, map_location=cfg.device)
        model.load_state_dict(state)
        model.eval()
        # Batched inference to avoid GPU OOM
        eval_ds = TensorDataset(torch.tensor(np.transpose(X_std, (0, 2, 1))).float())
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.utility_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        all_probs = []
        with torch.no_grad():
            for (xb,) in eval_loader:
                xb = xb.to(cfg.device)
                logits, _ = model(xb)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
    else:
        print("[Utility] Missing diag_surrogate.pt; please run train_ecg_utils.py. Using zeros for metrics.")
        probs = np.zeros_like(y_util, dtype=np.float32)

    y_true = y_util.astype(np.float32)
    metrics_basic = compute_diag_metrics_thresholded(probs, y_true, thresh=0.5)
    metrics = {
        "utility_sample_acc": metrics_basic["sample_acc"],
        "utility_f1": metrics_basic["macro_f1"],
        "utility_mae": metrics_basic["mae"],
        "utility_macro_f1": metrics_basic["macro_f1"],
        "utility_macro_acc": metrics_basic["macro_acc"],
    }
    print("[Utility] Metrics on full set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics, X_std, standard_scaler


def eval_utility_on_data(model, X_val_std: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    if model is None:
        probs = np.zeros_like(y_val, dtype=np.float32)
    else:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            xb = torch.tensor(np.transpose(X_val_std, (0, 2, 1))).float().to(device)
            logits, _ = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

    metrics_basic = compute_diag_metrics_thresholded(probs, y_val.astype(np.float32), thresh=0.5)
    metrics = {
        "utility_sample_acc": metrics_basic["sample_acc"],
        "utility_f1": metrics_basic["macro_f1"],
        "utility_mae": metrics_basic["mae"],
        "utility_macro_f1": metrics_basic["macro_f1"],
        "utility_macro_acc": metrics_basic["macro_acc"],
    }
    print("[Utility|Anon] Validation metrics on anonymized data:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# -------------------------------------------------------------------
# Fidelity + interpretability helpers
# -------------------------------------------------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psd_correlation(x: np.ndarray, y: np.ndarray, fs: int) -> float:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape for PSD correlation.")

    if x.shape[0] < x.shape[1]:
        x_c = x.T
        y_c = y.T
    else:
        x_c = x
        y_c = y

    C, _ = x_c.shape
    psds_x, psds_y = [], []
    for c in range(C):
        _, p_x = signal.welch(x_c[c], fs=fs, nperseg=256)
        _, p_y = signal.welch(y_c[c], fs=fs, nperseg=256)
        psds_x.append(p_x)
        psds_y.append(p_y)

    psd_x = np.mean(np.stack(psds_x, axis=0), axis=0)
    psd_y = np.mean(np.stack(psds_y, axis=0), axis=0)

    x_mean = psd_x - psd_x.mean()
    y_mean = psd_y - psd_y.mean()
    denom = np.sqrt((x_mean ** 2).sum()) * np.sqrt((y_mean ** 2).sum())
    if denom == 0:
        return 0.0
    return float((x_mean * y_mean).sum() / denom)


def plot_fft(signal_np: np.ndarray, fs: int, title: str, filepath: str):
    if signal_np.shape[0] < signal_np.shape[1]:
        data = signal_np.T
    else:
        data = signal_np

    C, T = data.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    plt.figure(figsize=(8, 4))
    for c in range(C):
        fft_vals = np.fft.rfft(data[c])
        plt.plot(freqs, np.abs(fft_vals), alpha=0.4)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def tsne_plot(features: np.ndarray,
              labels: np.ndarray,
              label_name: str,
              filepath: str):
    features = np.nan_to_num(features)
    perplexity = min(30, max(5, features.shape[0] // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    Z = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=8, cmap="tab20")
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label(label_name)
    plt.title(f"t-SNE of identity features colored by {label_name}")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def plot_overlay_ecg(orig: np.ndarray,
                     anon: np.ndarray,
                     fs: int,
                     title: str,
                     filepath: str,
                     lead: int = 0,
                     max_seconds: float = 5.0):
    """
    Plot original vs anonymized ECG on the same axis for a single lead.
    orig, anon: (T, C) or (C, T)
    """
    # ensure shape (T, C)
    if orig.shape[0] < orig.shape[1]:
        orig = orig.T
    if anon.shape[0] < anon.shape[1]:
        anon = anon.T

    T = orig.shape[0]
    max_samples = min(T, int(max_seconds * fs))
    t = np.arange(max_samples) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, orig[:max_samples, lead], label="Original", alpha=0.8)
    plt.plot(t, anon[:max_samples, lead], label="Blinder-style anonymized", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title + f" (lead {lead})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


# -------------------------------------------------------------------
# Blinder-style VAE (Encoder + Decoder)
# -------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, z_dim, sample_size):
        super().__init__()
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.fc1 = nn.Linear(sample_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.z_mean = nn.Linear(128, z_dim)
        self.z_log_var = nn.Linear(128, z_dim)
        self.relu = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        z_m = self.z_mean(h4)
        z_l = self.z_log_var(h4)
        # clamp logvar to prevent explosions
        z_l = torch.clamp(z_l, min=-6.0, max=6.0)
        z = self.reparameterize(z_m, z_l)
        return z, z_m, z_l


class Decoder(nn.Module):
    def __init__(self, z_dim, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, sample_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        h5 = self.fc5(h4)
        return h5


class ECGBlinderVAE(nn.Module):
    def __init__(self, T: int, C: int, z_dim: int):
        super().__init__()
        self.T = T
        self.C = C
        self.sample_size = T * C
        self.encoder = Encoder(z_dim=z_dim, sample_size=self.sample_size)
        self.decoder = Decoder(z_dim=z_dim, sample_size=self.sample_size)

    def forward(self, x):
        """
        x: (N, T, C)
        """
        N, T, C = x.shape
        assert T == self.T and C == self.C
        x_flat = x.view(N, -1)
        z, z_m, z_l = self.encoder(x_flat)
        recon_flat = self.decoder(z)
        recon = recon_flat.view(N, T, C)
        return recon, z_m, z_l


def vae_loss(recon_x, x, mu, logvar):
    recon = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 1e-3 * kld


def train_blinder_vae(X_std: np.ndarray, cfg: Config) -> ECGBlinderVAE:
    N, T, C = X_std.shape
    model = ECGBlinderVAE(T=T, C=C, z_dim=cfg.blinder_z_dim).to(cfg.device)

    dataset = TensorDataset(torch.from_numpy(X_std.astype(np.float32)))
    loader = DataLoader(dataset,
                        batch_size=cfg.blinder_batch_size,
                        shuffle=True,
                        drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.blinder_lr)

    print(f"[BlinderVAE] Training for {cfg.blinder_epochs} epochs on {N} samples...")
    for epoch in range(cfg.blinder_epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(cfg.device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            # gradient clipping to avoid explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        print(f"[BlinderVAE] Epoch {epoch+1}/{cfg.blinder_epochs} loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), cfg.blinder_ckpt)
    print(f"[BlinderVAE] Saved to {cfg.blinder_ckpt}")
    return model


def load_or_train_blinder_vae(X_std: np.ndarray, cfg: Config) -> ECGBlinderVAE:
    N, T, C = X_std.shape
    model = ECGBlinderVAE(T=T, C=C, z_dim=cfg.blinder_z_dim).to(cfg.device)
    if os.path.exists(cfg.blinder_ckpt):
        print(f"[BlinderVAE] Loading existing checkpoint from {cfg.blinder_ckpt}")
        try:
            state = torch.load(cfg.blinder_ckpt, map_location=cfg.device)
            model.load_state_dict(state)
            return model
        except RuntimeError as e:
            print(f"[BlinderVAE] Checkpoint mismatch ({e}); retraining...")
            return train_blinder_vae(X_std, cfg)
    else:
        return train_blinder_vae(X_std, cfg)


def anonymize_ecg_with_blinder(model: ECGBlinderVAE,
                               X_std: np.ndarray,
                               cfg: Config) -> np.ndarray:
    model.eval()
    X = X_std.astype(np.float32)
    N, T, C = X.shape
    anon = []

    with torch.no_grad():
        for start in range(0, N, cfg.blinder_batch_size):
            end = min(start + cfg.blinder_batch_size, N)
            xb = torch.from_numpy(X[start:end]).to(cfg.device)
            recon, _, _ = model(xb)
            anon.append(recon.cpu().numpy())

    X_anon = np.concatenate(anon, axis=0)
    X_anon = np.nan_to_num(X_anon, nan=0.0, posinf=5.0, neginf=-5.0)
    assert X_anon.shape == X_std.shape
    return X_anon


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("=== ECG Privacy + Blinder-style Pipeline (with overlay plots) ===")

    # 1) Load PTB-XL
    X, Y = load_ptbxl(PROJECT_ROOT, cfg.datafolder, cfg.sampling_frequency)
    # Keep in float32 to cut RAM in half
    X = X.astype(np.float32, copy=False)

    # 2) Utility labels
    print("Building utility labels...")
    y_util, super_classes = build_utility_labels_from_superclass(Y)

    # 3) Utility model
    util_model, util_val_metrics, data_std, scaler = train_or_load_utility_model_from_ptbxl(
        X, Y, y_util, cfg
    )

    # 4) Identity metadata
    meta, id_info = build_identity_meta(Y, cfg)

    # 5) Identity datasets/loaders
    train_ds = IdentityDataset(data_std, meta, split="train")
    val_ds = IdentityDataset(data_std, meta, split="val")
    # Keep a small copy of raw validation subset, then drop full raw array to save RAM
    val_row_idx = val_ds.meta["row_idx"].values
    X_val_raw = X[val_row_idx].astype(np.float32, copy=False)
    del X
    gc.collect()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.identity_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.identity_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 6) Identity model
    model_id = IdentityNet(
        in_channels=data_std.shape[2],
        n_patients=id_info["n_patients"],
        n_age_bins=id_info["n_age_bins"],
        n_height_bins=id_info["n_height_bins"],
        n_weight_bins=id_info["n_weight_bins"],
    )

    if os.path.exists(cfg.identity_ckpt):
        print(f"[Identity] Loading existing model from {cfg.identity_ckpt}")
        state = torch.load(cfg.identity_ckpt, map_location=cfg.device)
        model_id.load_state_dict(state)
    else:
        print("[Identity] Training from scratch...")
        train_identity(model_id, train_loader, val_loader, cfg)
        print(f"[Identity] Model saved to {cfg.identity_ckpt}")

    print("[Identity] Evaluating on validation set (baseline)...")
    id_val_metrics_base = eval_identity(model_id, val_loader, cfg)

    # 7) Baseline fidelity (identity)
    print("[Fidelity] Baseline (identity) on validation subset...")
    X_val_anon_raw_base = X_val_raw

    rmses_base, psd_corrs_base = [], []
    for i in range(X_val_raw.shape[0]):
        rmses_base.append(rmse(X_val_raw[i], X_val_anon_raw_base[i]))
        psd_corrs_base.append(psd_correlation(X_val_raw[i],
                                              X_val_anon_raw_base[i],
                                              fs=cfg.sampling_frequency))
    # BPM utility on baseline
    bpm_raw = compute_bpm_labels(X_val_raw, fs=cfg.sampling_frequency)
    bpm_base_metrics = {
        "bpm_raw_mean": float(np.mean(bpm_raw)),
        "bpm_mae": 0.0,  # baseline vs itself
    }
    fidelity_baseline = {
        "rmse_mean": float(np.mean(rmses_base)),
        "rmse_std": float(np.std(rmses_base)),
        "psd_corr_mean": float(np.mean(psd_corrs_base)),
        "psd_corr_std": float(np.std(psd_corrs_base)),
    }

    # 8) FFT + t-SNE (baseline)
    print("[Plots] Baseline FFT and t-SNE...")
    example_raw = X_val_raw[0]
    fft_path_base = os.path.join(cfg.results_dir, "fft_val_raw_baseline.png")
    plot_fft(example_raw, cfg.sampling_frequency,
             "Validation ECG FFT (baseline raw)", fft_path_base)

    model_id.eval()
    all_feats_base, all_pid_base = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(cfg.device)
            out = model_id(xb)
            all_feats_base.append(out["feat"].cpu().numpy())
            all_pid_base.extend(yb["patient"].numpy().tolist())
    all_feats_base = np.concatenate(all_feats_base, axis=0)
    all_pid_base = np.array(all_pid_base)

    tsne_path_base = os.path.join(cfg.results_dir, "tsne_identity_val_baseline.png")
    tsne_plot(all_feats_base,
              all_pid_base,
              label_name="patient_id (baseline raw)",
              filepath=tsne_path_base)

    # -----------------------------------------------------------
    # 9) Blinder-style anonymization
    # -----------------------------------------------------------
    if cfg.use_blinder:
        print("[BlinderVAE] Loading / training anonymizer...")
        blinder_model = load_or_train_blinder_vae(data_std, cfg)

        print("[BlinderVAE] Anonymizing full standardized ECG...")
        data_std_anon = anonymize_ecg_with_blinder(blinder_model, data_std, cfg)

        # Identity dataset on anonymized ECG
        train_ds_anon = IdentityDataset(data_std_anon, meta, split="train")
        val_ds_anon = IdentityDataset(data_std_anon, meta, split="val")
        val_loader_anon = DataLoader(
            val_ds_anon,
            batch_size=cfg.identity_batch_size,
            shuffle=False,
            drop_last=False,
        )

        print("[Identity] Evaluating on anonymized validation set...")
        id_val_metrics_anon = eval_identity(model_id, val_loader_anon, cfg)

        # Utility on anonymized ECG (full set to match baseline)
        util_val_metrics_anon = eval_utility_on_data(util_model,
                                                     data_std_anon,
                                                     y_util)

        # Fidelity between raw and anonymized (val subset)
        print("[Fidelity] Blinder-style anonymization on validation subset...")
        X_val_std_anon_subset = data_std_anon[val_row_idx]
        N_val, T, C = X_val_std_anon_subset.shape
        X_val_anon_flat = X_val_std_anon_subset.reshape(N_val, -1)
        X_val_anon_raw = scaler.inverse_transform(X_val_anon_flat).reshape(N_val, T, C)

        rmses_blinder, psd_blinder = [], []
        for i in range(N_val):
            rmses_blinder.append(rmse(X_val_raw[i], X_val_anon_raw[i]))
            psd_blinder.append(
                psd_correlation(X_val_raw[i], X_val_anon_raw[i],
                                fs=cfg.sampling_frequency)
            )
        # BPM utility on anonymized
        bpm_anon = compute_bpm_labels(X_val_anon_raw, fs=cfg.sampling_frequency)
        bpm_blinder_metrics = {
            "bpm_raw_mean": float(np.mean(bpm_raw)),
            "bpm_anon_mean": float(np.mean(bpm_anon)),
            "bpm_mae": float(np.mean(np.abs(bpm_raw - bpm_anon))),
        }

        fidelity_blinder = {
            "rmse_mean": float(np.mean(rmses_blinder)),
            "rmse_std": float(np.std(rmses_blinder)),
            "psd_corr_mean": float(np.mean(psd_blinder)),
            "psd_corr_std": float(np.std(psd_blinder)),
        }

        # FFT after anonymization
        print("[Plots] FFT and t-SNE after Blinder-style anonymization...")
        example_anon_raw = X_val_anon_raw[0]
        fft_path_anon = os.path.join(cfg.results_dir, "fft_val_anon_blinder.png")
        plot_fft(example_anon_raw, cfg.sampling_frequency,
                 "Validation ECG FFT (Blinder-style anonymized)", fft_path_anon)

        model_id.eval()
        all_feats_anon, all_pid_anon = [], []
        with torch.no_grad():
            for xb, yb in val_loader_anon:
                xb = xb.to(cfg.device)
                out = model_id(xb)
                all_feats_anon.append(out["feat"].cpu().numpy())
                all_pid_anon.extend(yb["patient"].numpy().tolist())
        all_feats_anon = np.concatenate(all_feats_anon, axis=0)
        all_pid_anon = np.array(all_pid_anon)

        tsne_path_anon = os.path.join(cfg.results_dir,
                                      "tsne_identity_val_blinder.png")
        tsne_plot(all_feats_anon,
                  all_pid_anon,
                  label_name="patient_id (Blinder-style anon)",
                  filepath=tsne_path_anon)

        # ---------------------------------------------------
        # NEW: overlay ORIGINAL vs ANONYMIZED ECG plots
        # ---------------------------------------------------
        print("[Plots] Time-domain overlay plots (original vs anonymized)...")
        overlay_dir = os.path.join(cfg.results_dir, "overlay_examples")
        os.makedirs(overlay_dir, exist_ok=True)

        n_examples_to_plot = min(5, N_val)
        leads_to_plot = [0, 1]  # for example, first two leads

        for i in range(n_examples_to_plot):
            for lead in leads_to_plot:
                fname = f"overlay_val_idx{i}_lead{lead}.png"
                out_path = os.path.join(overlay_dir, fname)
                title = f"Original vs Blinder-style anonymized ECG (val idx {i})"
                plot_overlay_ecg(
                    orig=X_val_raw[i],        # (T, C)
                    anon=X_val_anon_raw[i],   # (T, C)
                    fs=cfg.sampling_frequency,
                    title=title,
                    filepath=out_path,
                    lead=lead,
                    max_seconds=5.0,
                )

        print(f"[Plots] Saved overlay plots to {overlay_dir}")

    else:
        id_val_metrics_anon = None
        util_val_metrics_anon = None
        fidelity_blinder = None
        fft_path_anon = None
        tsne_path_anon = None

    # 10) Save summary metrics
    results = {
        "identity_val_metrics_baseline": id_val_metrics_base,
        "identity_val_metrics_blinder": id_val_metrics_anon,
        "utility_val_metrics_baseline": util_val_metrics,
        "utility_val_metrics_blinder": util_val_metrics_anon,
        "bpm_baseline": bpm_base_metrics,
        "bpm_blinder": bpm_blinder_metrics if cfg.use_blinder else None,
        "fidelity_baseline": fidelity_baseline,
        "fidelity_blinder": fidelity_blinder,
        "plots": {
            "fft_val_raw_baseline": fft_path_base,
            "tsne_identity_val_baseline": tsne_path_base,
            "fft_val_anon_blinder": fft_path_anon,
            "tsne_identity_val_blinder": tsne_path_anon,
            "overlay_dir": os.path.join(cfg.results_dir, "overlay_examples"),
        },
    }
    out_json = os.path.join(cfg.results_dir, "privacy_blinder_metrics_with_overlay.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Done ===")
    print(f"Metrics + paths JSON: {out_json}")


if __name__ == "__main__":
    main()
