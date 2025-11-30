#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG + Multi-Attribute Blinder-style Anonymization (exact Blinder MLP architecture)
- One Blinder model per private attribute:
    patient, sex, age, height, weight
- Checkpoints:
    PROJECT_ROOT/models/blinder_<attr>.pt
- For each attribute:
    * Train Blinder (VAE + AUX) with that attribute as private label
    * Anonymize ECG by scrambling that attribute's label
    * Evaluate identity classifier
    * Compute fidelity metrics (RMSE, PSD correlation)
    * Save PSD, overlay plots, and saliency plots for that attribute
"""

import os
import sys
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy import signal
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Path setup (same as your old script)
# -------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

PTBXL_BENCH_CODE = os.path.join(PROJECT_ROOT, "ecg_ptbxl_benchmarking", "code")
for p in [PROJECT_ROOT, PTBXL_BENCH_CODE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ecg_ptbxl_benchmarking imports
from utils import utils
from models.fastai_model import fastai_model

# your dataset loader
from pipeline.datasets import ecg_ptbxl

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

@dataclass
class Config:
    datafolder: str = os.path.join(
        PROJECT_ROOT,
        "data",
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1",
    )
    outputfolder: str = os.path.join(
        PROJECT_ROOT,
        "ecg_ptbxl_benchmarking",
        "output",
    )
    experiment: str = "exp0"
    modelname: str = "fastai_xresnet1d101"
    n_classes_pretrained: int = 71
    sampling_frequency: int = 100  # Hz

    # identity classifier
    identity_batch_size: int = 64
    identity_lr: float = 1e-3
    identity_epochs: int = 50
    min_samples_per_patient: int = 5
    identity_val_fraction: float = 0.2

    # Blinder-style VAE + AUX (exact architecture)
    z_dim: int = 25
    alpha: float = 0.2            # weight for auxEncLoss (as in Blinder)
    blinder_epochs: int = 100
    blinder_batch_size: int = 256
    blinder_lr: float = 1e-3
    k_spt: int = 128              # support subset per batch (Blinder-style)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    identity_ckpt: str = os.path.join(PROJECT_ROOT, "scripts", "identity_best.pt")

    # directory for all Blinder models
    blinder_models_dir: str = os.path.join(PROJECT_ROOT, "models")

    results_dir: str = os.path.join(PROJECT_ROOT, "privacy_multiattr_blinder_outputs")
    seed: int = 1234

cfg = Config()
os.makedirs(cfg.results_dir, exist_ok=True)
os.makedirs(cfg.blinder_models_dir, exist_ok=True)

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.device == "cuda":
    torch.cuda.manual_seed_all(cfg.seed)

# -------------------------------------------------------------
# Utility label builder (public label = diagnostic superclass)
# -------------------------------------------------------------

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

# -------------------------------------------------------------
# Identity meta with per-patient split
# -------------------------------------------------------------

def bin_numeric(col: pd.Series, bins: List[int]) -> pd.Series:
    labels = list(range(len(bins) - 1))
    return pd.cut(col, bins=bins, labels=labels, include_lowest=True).astype(int)

def build_identity_meta(Y: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    meta = Y.copy().reset_index(drop=False)  # 'index' is ecg_id
    meta["row_idx"] = np.arange(len(meta))

    counts = meta["patient_id"].value_counts()
    valid_pids = counts[counts >= cfg.min_samples_per_patient].index
    meta = meta[meta["patient_id"].isin(valid_pids)].reset_index(drop=True)

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
        "n_sex": 2,
    }
    return meta, info

# -------------------------------------------------------------
# Identity dataset + model
# -------------------------------------------------------------

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

# -------------------------------------------------------------
# Utility model
# -------------------------------------------------------------

def train_or_load_utility_model_from_ptbxl(
    X: np.ndarray,
    Y_db: pd.DataFrame,
    y_util: np.ndarray,
    cfg: Config,
):
    print("[Utility] Preparing standardization...")
    scaler_dir = os.path.join(cfg.outputfolder, cfg.experiment, "data")
    os.makedirs(scaler_dir, exist_ok=True)
    standard_scaler_path = os.path.join(scaler_dir, "standard_scaler.pkl")

    if os.path.exists(standard_scaler_path):
        print(f"[Utility] Loading StandardScaler from {standard_scaler_path}")
        standard_scaler = pickle.load(open(standard_scaler_path, "rb"))
    else:
        print("[Utility] Fitting new StandardScaler on folds 1–9...")
        train_mask = Y_db["strat_fold"] < 10
        X_train_flat = X[train_mask].reshape(train_mask.sum(), -1)
        standard_scaler = StandardScaler()
        standard_scaler.fit(X_train_flat)
        utils.save_pickle(standard_scaler_path, standard_scaler)
        print(f"[Utility] Saved scaler to {standard_scaler_path}")

    X_std = utils.apply_standardizer(X, standard_scaler)

    train_mask = Y_db["strat_fold"] < 10
    val_mask = Y_db["strat_fold"] == 10

    X_train = X_std[train_mask]
    y_train = y_util[train_mask]
    X_val = X_std[val_mask]
    y_val = y_util[val_mask]

    num_classes = y_train.shape[1]
    T, C = X_train.shape[1], X_train.shape[2]
    input_shape = [T, C]

    pretrainedfolder = os.path.join(
        cfg.outputfolder,
        cfg.experiment,
        "models",
        cfg.modelname,
    )

    print("[Utility] Building fastai model...")
    model = fastai_model(
        cfg.modelname,
        num_classes,
        cfg.sampling_frequency,
        cfg.outputfolder,
        input_shape=input_shape,
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=cfg.n_classes_pretrained,
        pretrained=True,
        epochs_finetuning=0,
    )

    print("[Utility] Evaluating on validation set (fold 10)...")
    y_val_pred = model.predict(X_val)
    utils.evaluate_experiment(y_val, y_val_pred)

    probs = y_val_pred
    y_hat = (probs >= 0.5).astype(np.float32)
    y_true = y_val.astype(np.float32)

    sample_acc = np.mean(np.all(y_true == y_hat, axis=1))
    f1 = f1_score(y_true.flatten(), y_hat.flatten(), average="binary")
    mae = mean_absolute_error(y_true.flatten(), probs.flatten())

    metrics = {
        "utility_sample_acc": float(sample_acc),
        "utility_f1": float(f1),
        "utility_mae": float(mae),
    }
    print("[Utility] Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics, X_std, standard_scaler

def eval_utility_on_data(model, X_val_std: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    probs = model.predict(X_val_std)
    probs = np.nan_to_num(probs, nan=0.0, posinf=5.0, neginf=-5.0)
    y_hat = (probs >= 0.5).astype(np.float32)
    y_true = y_val.astype(np.float32)
    sample_acc = np.mean(np.all(y_true == y_hat, axis=1))
    f1 = f1_score(y_true.flatten(), y_hat.flatten(), average="binary")
    mae = mean_absolute_error(y_true.flatten(), probs.flatten())
    metrics = {
        "utility_sample_acc": float(sample_acc),
        "utility_f1": float(f1),
        "utility_mae": float(mae),
    }
    print("[Utility|Anon] Validation metrics on anonymized data:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics

# -------------------------------------------------------------
# Fidelity + plotting helpers
# -------------------------------------------------------------

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

def plot_psd(signal_np: np.ndarray,
             fs: int,
             title: str,
             filepath: str):
    if signal_np.shape[0] < signal_np.shape[1]:
        data = signal_np.T
    else:
        data = signal_np
    C, T = data.shape
    psds = []
    freqs = None
    for c in range(C):
        f, p = signal.welch(data[c], fs=fs, nperseg=256)
        psds.append(p)
        if freqs is None:
            freqs = f
    psds = np.stack(psds, axis=0)
    psd_mean = psds.mean(axis=0)
    psd_db = 10 * np.log10(psd_mean + 1e-12)
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power / Hz (dB)")
    plt.title(title)
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
    if orig.shape[0] < orig.shape[1]:
        orig = orig.T
    if anon.shape[0] < anon.shape[1]:
        anon = anon.T
    T = orig.shape[0]
    max_samples = min(T, int(max_seconds * fs))
    t = np.arange(max_samples) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, orig[:max_samples, lead], label="Original", alpha=0.8)
    plt.plot(t, anon[:max_samples, lead], label="Blinder anonymized", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title + f" (lead {lead})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()

# -------------------------------------------------------------
# Blinder models (exact architecture)
# -------------------------------------------------------------

class BlinderEncoder(nn.Module):
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
        z = self.reparameterize(z_m, z_l)
        return z, z_m, z_l

class BlinderDecoder(nn.Module):
    def __init__(self, z_dim, priv_dim, pub_dim, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.fc1 = nn.Linear(z_dim + priv_dim + pub_dim, 128)
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

class BlinderAUX(nn.Module):
    def __init__(self, nz, numLabels=1):
        super().__init__()
        self.nz = nz
        self.numLabels = numLabels
        self.aux1 = nn.Linear(nz, 128)
        self.aux2 = nn.Linear(128, 128)
        self.aux3 = nn.Linear(128, numLabels)

    def forward(self, z):
        z = torch.relu(self.aux1(z))
        z = torch.relu(self.aux2(z))
        logits = self.aux3(z)
        return logits

def blinder_kld(mu, logvar):
    return torch.mean(-2 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

# -------------------------------------------------------------
# Blinder dataset for a specific private attribute
# -------------------------------------------------------------

class BlinderDatasetAttr(Dataset):
    """
    X_std: (N, T, C)
    meta: DataFrame with row_idx and attribute columns
    y_pub: (N, K) diagnostic superclass one-hot
    attr_name: one of 'patient', 'sex', 'age', 'height', 'weight'
    """
    def __init__(self,
                 X_std: np.ndarray,
                 meta: pd.DataFrame,
                 y_pub: np.ndarray,
                 attr_name: str):
        self.attr_name = attr_name
        self.meta = meta.reset_index(drop=True)
        self.row_idx = self.meta["row_idx"].values
        self.X = X_std[self.row_idx]        # (N*, T, C)
        self.y_pub = y_pub[self.row_idx]    # (N*, K)

        if attr_name == "patient":
            self.priv_idx = self.meta["patient_idx"].values
        elif attr_name == "sex":
            self.priv_idx = self.meta["sex_bin"].values
        elif attr_name == "age":
            self.priv_idx = self.meta["age_bin"].values
        elif attr_name == "height":
            self.priv_idx = self.meta["height_bin"].values
        elif attr_name == "weight":
            self.priv_idx = self.meta["weight_bin"].values
        else:
            raise ValueError(f"Unknown attr_name {attr_name}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, C)
        x_flat = x.reshape(-1).astype(np.float32)
        priv = self.priv_idx[idx]
        return (
            torch.tensor(x_flat, dtype=torch.float32),
            torch.tensor(priv, dtype=torch.long),
            torch.tensor(self.y_pub[idx], dtype=torch.float32)
        )

# -------------------------------------------------------------
# Blinder training for a specific attribute
# -------------------------------------------------------------

def train_blinder_for_attr(
    X_std: np.ndarray,
    meta: pd.DataFrame,
    y_util: np.ndarray,
    attr_name: str,
    n_priv: int,
    cfg: Config,
    ckpt_path: str,
) -> Tuple[BlinderEncoder, BlinderDecoder, BlinderAUX]:

    sample_size = X_std.shape[1] * X_std.shape[2]
    n_pub = y_util.shape[1]

    ds = BlinderDatasetAttr(X_std, meta, y_util, attr_name)
    loader = DataLoader(
        ds,
        batch_size=cfg.blinder_batch_size,
        shuffle=True,
        drop_last=False,
    )

    encoder = BlinderEncoder(cfg.z_dim, sample_size).to(cfg.device)
    decoder = BlinderDecoder(cfg.z_dim, n_priv, n_pub, sample_size).to(cfg.device)
    aux = BlinderAUX(cfg.z_dim, numLabels=n_priv).to(cfg.device)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=cfg.blinder_lr)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=cfg.blinder_lr)
    opt_aux = torch.optim.Adam(aux.parameters(), lr=cfg.blinder_lr)

    bce_logits = nn.BCEWithLogitsLoss()

    print(f"[Blinder-{attr_name}] Training for {cfg.blinder_epochs} epochs "
          f"on {len(ds)} samples (sample_size={sample_size}, n_priv={n_priv})...")

    for epoch in range(cfg.blinder_epochs):
        encoder.train()
        decoder.train()
        aux.train()

        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0
        n_samples = 0

        for xb_flat, priv_idx, y_pub in loader:
            xb_flat = xb_flat.to(cfg.device)
            priv_idx = priv_idx.to(cfg.device)
            y_pub = y_pub.to(cfg.device)

            B = xb_flat.size(0)
            n_samples += B

            priv_onehot = torch.zeros(B, n_priv, device=cfg.device)
            priv_onehot[torch.arange(B), priv_idx] = 1.0

            # support subset
            k_spt = min(cfg.k_spt, B // 2) if B > 1 else B
            if k_spt == 0:
                continue

            train_x = xb_flat[:k_spt]
            train_priv = priv_onehot[:k_spt]
            train_pub = y_pub[:k_spt]

            # 1) Train AUX on mixed real + fake
            with torch.no_grad():
                z_fake, _, _ = encoder(train_x)
                # fake labels different from original
                priv_idx_train = torch.argmax(train_priv, dim=1).cpu().numpy()
                rest = np.arange(n_priv)
                fake_labels = []
                for p in priv_idx_train:
                    choices = np.delete(rest, p)
                    fake_labels.append(np.random.choice(choices))
                fake_labels = np.array(fake_labels)
                fake_onehot = torch.zeros_like(train_priv)
                fake_onehot[torch.arange(k_spt),
                            torch.from_numpy(fake_labels).to(cfg.device)] = 1.0
                fake_pub = train_pub

                cat_fake = torch.cat([z_fake, fake_onehot, fake_pub], dim=1)
                x_fake = decoder(cat_fake)

            train_x_fake = torch.cat([train_x, x_fake.detach()], dim=0)
            train_priv_fake = torch.cat([train_priv, fake_onehot], dim=0)
            shuffle_idx = torch.randperm(train_x_fake.size(0))
            train_x_fake = train_x_fake[shuffle_idx]
            train_priv_fake = train_priv_fake[shuffle_idx]

            with torch.no_grad():
                z_for_aux, _, _ = encoder(train_x_fake)
            priv_logits = aux(z_for_aux.detach())
            aux_loss = bce_logits(priv_logits, train_priv_fake)
            opt_aux.zero_grad()
            aux_loss.backward()
            opt_aux.step()

            # 2) Train encoder+decoder to reconstruct + foil AUX
            z, mu, logvar = encoder(train_x)
            cat_z = torch.cat([z, train_priv, train_pub], dim=1)
            xr = decoder(cat_z)

            # AUX prediction on z (we want encoder to FOOL this)
            priv_logits_enc = aux(z)
            auxEncLoss = bce_logits(priv_logits_enc, train_priv)

            # ---- NEW: rebalanced loss ----
            # plain MSE (no 512 / 150 scaling)
            mse = nn.MSELoss()(xr, train_x)

            # small KL weight: keep latents roughly Gaussian, but don't dominate
            kld_loss = blinder_kld(mu, logvar)
            kld_weight = 1e-4

            # softer adversarial term (was ~0.2)
            alpha_attr = 0.05

            recon_loss = mse                   # for logging
            loss = recon_loss + kld_weight * kld_loss - alpha_attr * auxEncLoss
        # --------------------------------

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss.backward()
            opt_enc.step()
            opt_dec.step()

            epoch_loss     += loss.item() * B
            epoch_aux_loss += aux_loss.item() * B
            epoch_recon    += recon_loss.item() * B
            epoch_kld      += kld_loss.item() * B


        if n_samples > 0:
            epoch_loss /= n_samples
            epoch_aux_loss /= n_samples
            epoch_recon /= n_samples
            epoch_kld /= n_samples

        print(f"[Blinder-{attr_name}] Epoch {epoch+1}/{cfg.blinder_epochs} "
              f"Loss={epoch_loss:.4f} Aux={epoch_aux_loss:.4f} "
              f"Recon={epoch_recon:.4f} KLD={epoch_kld:.4f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "aux": aux.state_dict(),
        "sample_size": sample_size,
        "z_dim": cfg.z_dim,
        "n_priv": n_priv,
        "n_pub": n_pub,
    }, ckpt_path)
    print(f"[Blinder-{attr_name}] Saved checkpoint to {ckpt_path}")
    return encoder, decoder, aux

# -------------------------------------------------------------
# Blinder anonymization for a specific attribute
# -------------------------------------------------------------

def anonymize_ecg_blinder_attr(
    encoder: BlinderEncoder,
    decoder: BlinderDecoder,
    X_std: np.ndarray,
    y_util: np.ndarray,
    meta: pd.DataFrame,
    attr_name: str,
    n_priv: int,
    cfg: Config,
) -> np.ndarray:
    encoder.eval()
    decoder.eval()
    encoder.to(cfg.device)
    decoder.to(cfg.device)

    sample_size = X_std.shape[1] * X_std.shape[2]

    meta_sorted = meta.sort_values("row_idx").reset_index(drop=True)
    row_idx = meta_sorted["row_idx"].values
    X_sub = X_std[row_idx]
    y_pub_sub = y_util[row_idx]

    if attr_name == "patient":
        priv_idx_attr = meta_sorted["patient_idx"].values
    elif attr_name == "sex":
        priv_idx_attr = meta_sorted["sex_bin"].values
    elif attr_name == "age":
        priv_idx_attr = meta_sorted["age_bin"].values
    elif attr_name == "height":
        priv_idx_attr = meta_sorted["height_bin"].values
    elif attr_name == "weight":
        priv_idx_attr = meta_sorted["weight_bin"].values
    else:
        raise ValueError(f"Unknown attr_name {attr_name}")

    X_flat = X_sub.reshape(len(X_sub), -1).astype(np.float32)
    X_anon_flat = np.zeros_like(X_flat)

    batch_size = cfg.blinder_batch_size
    all_priv = np.arange(n_priv)

    for start in range(0, len(X_flat), batch_size):
        end = min(start + batch_size, len(X_flat))
        xb = torch.tensor(X_flat[start:end], dtype=torch.float32, device=cfg.device)
        y_pub_batch = y_pub_sub[start:end]

        z, _, _ = encoder(xb)

        orig_priv = priv_idx_attr[start:end]
        new_priv = []
        for p in orig_priv:
            choices = np.delete(all_priv, p)
            if len(choices) == 0:
                choices = all_priv
            new_priv.append(np.random.choice(choices))
        new_priv = np.array(new_priv)

        priv_new_onehot = np.zeros((len(new_priv), n_priv), dtype=np.float32)
        priv_new_onehot[np.arange(len(new_priv)), new_priv] = 1.0

        priv_new_onehot_t = torch.tensor(priv_new_onehot, dtype=torch.float32, device=cfg.device)
        y_pub_t = torch.tensor(y_pub_batch, dtype=torch.float32, device=cfg.device)
        cat = torch.cat([z, priv_new_onehot_t, y_pub_t], dim=1)
        xr_flat = decoder(cat)
        X_anon_flat[start:end] = xr_flat.detach().cpu().numpy()

    X_anon = X_anon_flat.reshape(X_sub.shape[0], X_sub.shape[1], X_sub.shape[2])
    X_full_anon = np.copy(X_std)
    X_full_anon[row_idx] = X_anon
    return X_full_anon

# -------------------------------------------------------------
# Identity saliency for a specific attribute
# -------------------------------------------------------------

def compute_attr_saliency(
    model_id: IdentityNet,
    X_batch: torch.Tensor,       # (B, C, T)
    attr_targets: torch.Tensor,  # (B,) indices for that attr
    attr_name: str,
    cfg: Config,
) -> np.ndarray:
    """
    Input-gradient saliency for a specific identity attribute.
    - attr_name: 'patient', 'sex', 'age', 'height', 'weight'
    Returns saliency: (B, C, T)
    """
    model_id.eval()
    model_id.to(cfg.device)

    # IMPORTANT: work on a fresh tensor, don't mutate caller's X_batch
    X = X_batch.to(cfg.device).detach().clone()
    X.requires_grad_(True)

    out = model_id(X)
    logits = out[attr_name]  # (B, num_classes)
    chosen = logits[torch.arange(logits.size(0)), attr_targets.to(cfg.device)]
    loss = -chosen.mean()  # maximize that attribute logit

    model_id.zero_grad()
    loss.backward()

    saliency = X.grad.detach().abs().cpu().numpy()
    return saliency


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    print("=== ECG + Multi-Attribute Blinder Anonymization ===")

    # 1) Load PTB-XL
    data_dict = ecg_ptbxl.load_ptbxl_and_eda(
        ptbxl_root=cfg.datafolder,
        sampling_rate=cfg.sampling_frequency,
        output_dir=os.path.join(cfg.results_dir, "ptbxl_eda"),
        save_csv=False,
    )
    X = data_dict["X"]   # (N, T, C=12)
    Y = data_dict["Y"]

    # 2) Utility labels
    print("Building utility labels...")
    y_util, super_classes = build_utility_labels_from_superclass(Y)

    # 3) Utility model & standardization
    util_model, util_val_metrics, data_std, scaler = train_or_load_utility_model_from_ptbxl(
        X, Y, y_util, cfg
    )

    # 4) Identity metadata
    meta, id_info = build_identity_meta(Y, cfg)

    # 5) Identity datasets/loaders
    train_ds = IdentityDataset(data_std, meta, split="train")
    val_ds = IdentityDataset(data_std, meta, split="val")
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

    # 7) Baseline fidelity (no anonymization) on val subset
    val_row_idx = val_ds.meta["row_idx"].values
    X_val_raw = X[val_row_idx]

    rmses_base, psd_corrs_base = [], []
    for i in range(X_val_raw.shape[0]):
        rmses_base.append(rmse(X_val_raw[i], X_val_raw[i]))
        psd_corrs_base.append(
            psd_correlation(X_val_raw[i], X_val_raw[i], fs=cfg.sampling_frequency)
        )
    fidelity_baseline = {
        "rmse_mean": float(np.mean(rmses_base)),
        "rmse_std": float(np.std(rmses_base)),
        "psd_corr_mean": float(np.mean(psd_corrs_base)),
        "psd_corr_std": float(np.std(psd_corrs_base)),
    }

    example_raw = X_val_raw[0]
    psd_path_base = os.path.join(cfg.results_dir, "psd_val_raw_baseline.png")
    plot_psd(example_raw,
             cfg.sampling_frequency,
             "Validation ECG PSD (baseline raw)",
             psd_path_base)

    # ---------------------------------------------------------
    # Multi-attribute Blinder loop
    # ---------------------------------------------------------
    private_attrs = [
        ("patient", id_info["n_patients"]),
        ("sex", id_info["n_sex"]),
        ("age", id_info["n_age_bins"]),
        ("height", id_info["n_height_bins"]),
        ("weight", id_info["n_weight_bins"]),
    ]

    results_per_attr = {}

    # for saliency: grab one batch from val_loader once
    xb_raw_np, yb_batch = next(iter(val_loader))
    xb_raw = xb_raw_np.clone()  # (B, C, T)
    batch_size_for_sal = xb_raw.shape[0]
    # corresponding row_idx for that batch within val set
    row_idx_batch = val_ds.meta["row_idx"].values[:batch_size_for_sal]

    for attr_name, n_priv in private_attrs:
        print(f"\n=== Blinder for attribute: {attr_name} (n_priv={n_priv}) ===")

        ckpt_path = os.path.join(cfg.blinder_models_dir, f"blinder_{attr_name}.pt")

        encoder, decoder, aux = train_blinder_for_attr(
            data_std,
            meta,
            y_util,
            attr_name,
            n_priv,
            cfg,
            ckpt_path,
        )

        # Anonymize ECG w.r.t this attribute
        print(f"[Blinder-{attr_name}] Anonymizing standardized ECG...")
        data_std_anon_attr = anonymize_ecg_blinder_attr(
            encoder,
            decoder,
            data_std,
            y_util,
            meta,
            attr_name,
            n_priv,
            cfg,
        )

        # Identity evaluation on anonymized ECG
        train_ds_anon = IdentityDataset(data_std_anon_attr, meta, split="train")
        val_ds_anon = IdentityDataset(data_std_anon_attr, meta, split="val")
        val_loader_anon = DataLoader(
            val_ds_anon,
            batch_size=cfg.identity_batch_size,
            shuffle=False,
            drop_last=False,
        )

        print(f"[Identity] Evaluating on anonymized validation set (attr={attr_name})...")
        id_val_metrics_anon_attr = eval_identity(model_id, val_loader_anon, cfg)

        # Utility on anonymized ECG (fold 10)
        val_mask = Y["strat_fold"] == 10
        X_val_std_anon_attr = data_std_anon_attr[val_mask]
        y_val = y_util[val_mask]
        util_val_metrics_anon_attr = eval_utility_on_data(util_model, X_val_std_anon_attr, y_val)

        # Fidelity for this attribute (val subset)
        print(f"[Fidelity] Blinder anonymization for attr={attr_name} on validation subset...")
        X_val_std_anon_subset = data_std_anon_attr[val_row_idx]
        N_val, T, C = X_val_std_anon_subset.shape
        X_val_anon_flat = X_val_std_anon_subset.reshape(N_val, -1)
        X_val_anon_raw = scaler.inverse_transform(X_val_anon_flat).reshape(N_val, T, C)

        rmses_blinder, psd_blinder = [], []
        for i in range(N_val):
            rmses_blinder.append(rmse(X_val_raw[i], X_val_anon_raw[i]))
            psd_blinder.append(
                psd_correlation(X_val_raw[i], X_val_anon_raw[i], fs=cfg.sampling_frequency)
            )
        fidelity_attr = {
            "rmse_mean": float(np.mean(rmses_blinder)),
            "rmse_std": float(np.std(rmses_blinder)),
            "psd_corr_mean": float(np.mean(psd_blinder)),
            "psd_corr_std": float(np.std(psd_blinder)),
        }

        # PSD & overlay for example 0
        example_anon_raw = X_val_anon_raw[0]
        psd_path_anon = os.path.join(cfg.results_dir, f"psd_val_anon_blinder_{attr_name}.png")
        plot_psd(example_anon_raw,
                 cfg.sampling_frequency,
                 f"Validation ECG PSD (Blinder anonymized, {attr_name})",
                 psd_path_anon)

        overlay_dir_attr = os.path.join(cfg.results_dir, f"overlay_{attr_name}")
        os.makedirs(overlay_dir_attr, exist_ok=True)
        for lead in [0, 1]:
            fname = f"overlay_val_idx0_lead{lead}.png"
            out_path = os.path.join(overlay_dir_attr, fname)
            title = f"Original vs Blinder anonymized ECG (val idx 0, {attr_name})"
            plot_overlay_ecg(
                orig=example_raw,
                anon=example_anon_raw,
                fs=cfg.sampling_frequency,
                title=title,
                filepath=out_path,
                lead=lead,
                max_seconds=5.0,
            )

        # Saliency: feature importance for this attribute
        print(f"[Saliency-{attr_name}] Computing saliency before vs after anonymization...")
        # original
        attr_targets = yb_batch[attr_name]  # from IdentityDataset
        sal_raw_attr = compute_attr_saliency(model_id, xb_raw, attr_targets, attr_name, cfg)

        # anonymized input for same val rows
        X_anon_batch_std = data_std_anon_attr[row_idx_batch]
        xb_anon = torch.tensor(np.transpose(X_anon_batch_std, (0, 2, 1)), dtype=torch.float32)
        sal_anon_attr = compute_attr_saliency(model_id, xb_anon, attr_targets, attr_name, cfg)

        sal_dir_attr = os.path.join(cfg.results_dir, f"saliency_{attr_name}")
        os.makedirs(sal_dir_attr, exist_ok=True)
        t_axis = np.arange(xb_raw.shape[2]) / cfg.sampling_frequency

        for i in range(min(3, xb_raw.shape[0])):
            plt.figure(figsize=(10, 6))
            # top: ECG lead 0
            plt.subplot(2, 1, 1)
            plt.plot(t_axis, xb_raw_np[i, 0].numpy(), label="Original ECG (lead 0)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.title(f"ECG + saliency for {attr_name} (sample {i})")

            # bottom: saliency before vs after
            plt.subplot(2, 1, 2)
            s_raw = sal_raw_attr[i, 0]
            s_anon = sal_anon_attr[i, 0]
            s_raw_norm = s_raw / (s_raw.max() + 1e-8)
            s_anon_norm = s_anon / (s_anon.max() + 1e-8)
            plt.plot(t_axis, s_raw_norm, label="Saliency original", alpha=0.8)
            plt.plot(t_axis, s_anon_norm, label="Saliency anonymized", alpha=0.8)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized saliency")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(sal_dir_attr, f"saliency_{attr_name}_sample{i}_lead0.png"), dpi=200)
            plt.close()

        results_per_attr[attr_name] = {
            "identity_val_metrics_blinder": id_val_metrics_anon_attr,
            "utility_val_metrics_blinder": util_val_metrics_anon_attr,
            "fidelity_blinder": fidelity_attr,
            "psd_val_anon_blinder": psd_path_anon,
            "overlay_dir": overlay_dir_attr,
            "saliency_dir": sal_dir_attr,
            "blinder_ckpt": ckpt_path,
        }

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    results = {
        "identity_val_metrics_baseline": id_val_metrics_base,
        "utility_val_metrics_baseline": util_val_metrics,
        "fidelity_baseline": fidelity_baseline,
        "psd_val_raw_baseline": psd_path_base,
        "per_attribute": results_per_attr,
    }

    out_json = os.path.join(cfg.results_dir, "multiattr_blinder_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Done ===")
    print(f"Summary JSON: {out_json}")

if __name__ == "__main__":
    main()
