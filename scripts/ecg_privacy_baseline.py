#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG Privacy Baseline Pipeline (no anonymization yet)

Directory assumptions:
- Project root: /Users/bluitel/Documents/ECE539/ECE539
- PTB-XL DB:    <PROJECT_ROOT>/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1
- Benchmark repo: <PROJECT_ROOT>/ecg_ptbxl_benchmarking
- This script:  <PROJECT_ROOT>/scripts/ecg_privacy_baseline.py

What this script does:
1. Load PTB-XL directly from the database via pipeline.datasets.ecg_ptbxl.load_ptbxl_and_eda.
2. Load + finetune pretrained fastai utility model (xresnet1d101) as in Finetuning-Example.ipynb.
3. Build a multi-task identity classifier (PyTorch) over:
   - patient_id (multi-class)
   - age_bin
   - sex
   - height_bin
   - weight_bin
4. If identity checkpoint exists, load it; otherwise train it.
5. Evaluate:
   - Privacy: identity accuracy + macro F1 per attribute on validation set.
   - Utility: multi-label sample-wise accuracy, F1, MAE on validation set.
   - Fidelity (baseline upper bound): RMSE and PSD correlation using “anonymized = original”.
   - Interpretability: FFT plots for one validation ECG and t-SNE of identity features.

Run from project root:
    cd /Users/bluitel/Documents/ECE539/ECE539
    conda activate ecg_env
    python scripts/ecg_privacy_baseline.py
"""

import os
import sys

# -------------------------------------------------------------------
# Path setup so imports work from anywhere
# -------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ecg_ptbxl_benchmarking code directory (for utils, fastai_model, etc.)
PTBXL_BENCH_CODE = os.path.join(PROJECT_ROOT, "ecg_ptbxl_benchmarking", "code")

for p in [PROJECT_ROOT, PTBXL_BENCH_CODE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------------------------------------------------------
# Imports (now that paths are set)
# -------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy import signal
import matplotlib.pyplot as plt

# ecg_ptbxl_benchmarking repo imports
from utils import utils
from models.fastai_model import fastai_model

# your dataset loader
from pipeline.datasets import ecg_ptbxl


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

@dataclass
class Config:
    # PTB-XL *database* root (where ptbxl_database.csv lives)
    datafolder: str = os.path.join(
        PROJECT_ROOT,
        "data",
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1",
    )

    # ecg_ptbxl_benchmarking output root
    outputfolder: str = os.path.join(
        PROJECT_ROOT,
        "ecg_ptbxl_benchmarking",
        "output",
    )

    experiment: str = "exp0"
    modelname: str = "fastai_xresnet1d101"
    n_classes_pretrained: int = 71  # from exp0

    sampling_frequency: int = 100  # Hz

    # Identity classifier
    identity_batch_size: int = 64
    identity_lr: float = 1e-3
    identity_epochs: int = 300
    min_samples_per_patient: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Where to save your identity model
    identity_ckpt: str = os.path.join(PROJECT_ROOT, "scripts", "identity_best.pt")

    # Output folder for metrics & plots
    results_dir: str = os.path.join(PROJECT_ROOT, "privacy_baseline_outputs")


cfg = Config()
os.makedirs(cfg.results_dir, exist_ok=True)


# -------------------------------------------------------------------
# 1. UTILITY LABELS FROM PTB-XL DIAGNOSTIC SUPERCLASSES
# -------------------------------------------------------------------

def build_utility_labels_from_superclass(Y: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build a multi-hot label matrix from Y["diagnostic_superclass"].

    Returns:
        y_util: np.ndarray (N, num_superclasses) multi-hot labels
        super_classes: list[str] in same order as y_util columns
    """
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
# 2. IDENTITY META
# -------------------------------------------------------------------

def bin_numeric(col: pd.Series, bins: List[int]) -> pd.Series:
    labels = list(range(len(bins) - 1))
    return pd.cut(col, bins=bins, labels=labels, include_lowest=True).astype(int)


def build_identity_meta(Y: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """
    Build metadata for identity classification from ptbxl_database DataFrame Y.

    - Adds row_idx column to index into X / data_std.
    - Filters to patients with >= min_samples_per_patient.
    - Creates binned labels for age, height, weight; binary sex.
    """
    meta = Y.copy().reset_index(drop=False)  # ecg_id -> column
    meta["row_idx"] = np.arange(len(meta))

    counts = meta["patient_id"].value_counts()
    valid_pids = counts[counts >= cfg.min_samples_per_patient].index
    meta = meta[meta["patient_id"].isin(valid_pids)].reset_index(drop=True)

    print(f"Identity meta: {len(meta)} records from {len(valid_pids)} patients")

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
# 3. IDENTITY DATASET + MODEL
# -------------------------------------------------------------------

class IdentityDataset(Dataset):
    def __init__(self,
                 data_std: np.ndarray,  # (N, T, 12)
                 meta: pd.DataFrame,
                 cfg: Config,
                 split: str):
        """
        data_std: standardized ECG signals aligned with Y rows.
        meta: identity metadata from build_identity_meta, containing:
              - row_idx: index into data_std (0..N-1)
              - patient_idx, sex_bin, age_bin, height_bin, weight_bin
              - strat_fold
        split: 'train' -> folds 1–9; 'val' -> fold 10
        """
        self.cfg = cfg

        if split == "train":
            subset = meta[meta["strat_fold"] < 10]
        elif split == "val":
            subset = meta[meta["strat_fold"] == 10]
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
    for key in all_true.keys():
        acc = accuracy_score(all_true[key], all_pred[key])
        f1 = f1_score(all_true[key], all_pred[key], average="macro")
        metrics[f"{key}_acc"] = float(acc)
        metrics[f"{key}_f1"] = float(f1)

    print("[Identity] Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# -------------------------------------------------------------------
# 4. UTILITY MODEL + METRICS
# -------------------------------------------------------------------

def train_or_load_utility_model_from_ptbxl(
    X: np.ndarray,
    Y_db: pd.DataFrame,
    y_util: np.ndarray,
    cfg: Config,
):
    """
    Similar to Finetuning-Example, but using direct PTB-XL load.
    """
    import pickle
    from sklearn.preprocessing import StandardScaler

    print("Preparing utility data / standardization...")
    scaler_dir = os.path.join(cfg.outputfolder, cfg.experiment, "data")
    os.makedirs(scaler_dir, exist_ok=True)
    standard_scaler_path = os.path.join(scaler_dir, "standard_scaler.pkl")

    if os.path.exists(standard_scaler_path):
        print(f"Loading existing standard_scaler from {standard_scaler_path}")
        standard_scaler = pickle.load(open(standard_scaler_path, "rb"))
    else:
        print("Fitting new standard_scaler on training folds (1–9)...")
        train_mask = Y_db["strat_fold"] < 10
        X_train_flat = X[train_mask].reshape(train_mask.sum(), -1)
        standard_scaler = StandardScaler()
        standard_scaler.fit(X_train_flat)
        pickle.dump(standard_scaler, open(standard_scaler_path, "wb"))
        print(f"Saved new standard_scaler to {standard_scaler_path}")

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

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print("Building fastai utility model...")

    pretrainedfolder = os.path.join(
        cfg.outputfolder,
        cfg.experiment,
        "models",
        cfg.modelname,
    )

    model = fastai_model(
        cfg.modelname,
        num_classes,
        cfg.sampling_frequency,
        cfg.outputfolder,
        input_shape=input_shape,
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=cfg.n_classes_pretrained,
        pretrained=True,
        epochs_finetuning=2,
    )

    # print("Finetuning fastai utility model...")
    # model.fit(X_train, y_train, X_val, y_val)

    print("Evaluating fastai utility model on validation set...")
    y_val_pred = model.predict(X_val)
    print("Raw ecg_ptbxl_benchmarking metrics:")
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

    return model, metrics, X_std


# -------------------------------------------------------------------
# 5. FIDELITY + INTERPRETABILITY HELPERS
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


def tsne_plot(features: np.ndarray, labels: np.ndarray, title: str, filepath: str):
    perplexity = min(30, max(5, features.shape[0] // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    Z = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=8, cmap="tab20")
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# -------------------------------------------------------------------
# 6. MAIN
# -------------------------------------------------------------------

def main():
    print("Starting ECG Privacy Baseline Pipeline (direct PTB-XL load)...")

    # 1) Load PTB-XL via your loader
    data_dict = ecg_ptbxl.load_ptbxl_and_eda(
        ptbxl_root=cfg.datafolder,
        sampling_rate=cfg.sampling_frequency,
        output_dir=os.path.join(cfg.results_dir, "ptbxl_eda"),
        save_csv=False,
    )

    X = data_dict["X"]   # (N, T, 12)
    Y = data_dict["Y"]   # ptbxl_database (+ diag_superclass/subclass)

    # 2) Utility labels from diagnostic_superclass
    print("Building utility labels from diagnostic_superclass...")
    y_util, super_classes = build_utility_labels_from_superclass(Y)

    # 3) Train / load fastai utility model
    util_model, util_val_metrics, data_std = train_or_load_utility_model_from_ptbxl(
        X, Y, y_util, cfg
    )

    # 4) Identity metadata
    meta, id_info = build_identity_meta(Y, cfg)

    # 5) Identity datasets & loaders
    train_ds = IdentityDataset(data_std, meta, cfg, split="train")
    val_ds = IdentityDataset(data_std, meta, cfg, split="val")

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

    # 6) Identity model: load or train
    model_id = IdentityNet(
        in_channels=data_std.shape[2],
        n_patients=id_info["n_patients"],
        n_age_bins=id_info["n_age_bins"],
        n_height_bins=id_info["n_height_bins"],
        n_weight_bins=id_info["n_weight_bins"],
    )

    if os.path.exists(cfg.identity_ckpt):
        print(f"Loading existing identity model from {cfg.identity_ckpt}")
        state = torch.load(cfg.identity_ckpt, map_location=cfg.device)
        model_id.load_state_dict(state)
    else:
        print("Training identity model from scratch...")
        train_identity(model_id, train_loader, val_loader, cfg)
        print(f"Identity model saved to {cfg.identity_ckpt}")

    print("Evaluating identity model on validation set...")
    id_val_metrics = eval_identity(model_id, val_loader, cfg)

    # 7) Fidelity baseline (anonymized == original)
    print("Computing baseline fidelity metrics on validation subset...")
    val_row_idx = val_ds.meta["row_idx"].values
    X_val_raw = X[val_row_idx]
    X_val_anon = X_val_raw

    rmses = []
    psd_corrs = []
    for i in range(X_val_raw.shape[0]):
        orig = X_val_raw[i]
        anon = X_val_anon[i]
        rmses.append(rmse(orig, anon))
        psd_corrs.append(psd_correlation(orig, anon, fs=cfg.sampling_frequency))

    fidelity_metrics = {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "psd_corr_mean": float(np.mean(psd_corrs)),
        "psd_corr_std": float(np.std(psd_corrs)),
    }

    # 8) FFT interpretability plot
    print("Saving baseline FFT plot for one validation ECG...")
    example_raw = X_val_raw[0]
    fft_path = os.path.join(cfg.results_dir, "fft_val_raw_baseline.png")
    plot_fft(example_raw, cfg.sampling_frequency,
             "Validation ECG FFT (baseline raw)", fft_path)

    # 9) t-SNE interpretability plot
    print("Computing t-SNE over identity features (baseline)...")
    model_id.eval()
    all_feats = []
    all_pid = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(cfg.device)
            out = model_id(xb)
            all_feats.append(out["feat"].cpu().numpy())
            all_pid.extend(yb["patient"].numpy().tolist())
    all_feats = np.concatenate(all_feats, axis=0)
    all_pid = np.array(all_pid)

    tsne_path = os.path.join(cfg.results_dir, "tsne_identity_val_baseline.png")
    tsne_plot(all_feats, all_pid,
              "Identity features (baseline, raw ECG)", tsne_path)

    # 10) Save summary metrics
    results = {
        "identity_val_metrics": id_val_metrics,
        "utility_val_metrics": util_val_metrics,
        "fidelity_baseline": fidelity_metrics,
        "plots": {
            "fft_val_raw_baseline": fft_path,
            "tsne_identity_val_baseline": tsne_path,
        },
    }
    out_json = os.path.join(cfg.results_dir, "privacy_baseline_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Summary metrics written to {out_json}")
    print("Baseline FFT plot:", fft_path)
    print("Baseline t-SNE plot:", tsne_path)


if __name__ == "__main__":
    main()
