#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ninapro DB1 EMG + Blinder-style anonymization pipeline.

- Loads Ninapro_DB1.csv (Kaggle-converted Ninapro DB1).
- Builds windowed EMG segments (N, T, C=10).
- Trains a subject-identity CNN (private label = subject).
- Trains a simple gesture utility CNN (utility label = stimulus).
- Trains a Blinder-style VAE on standardized EMG.
- Anonymizes EMG, evaluates identity + utility again.
- Plots time-domain overlays (original vs anonymized) and FFTs.

Optional:
- Shows where to load an emg2qwerty Lightning checkpoint as the
  utility model (you will need to plug in the proper preprocessing).
"""


import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from scipy import signal
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    project_root: str = PROJECT_ROOT
    data_dir: str = None   # will be set in __post_init__
    ninapro_csv: str = None
    results_dir: str = None

    # EMG settings
    emg_channels: int = 10
    sampling_frequency: int = 100  # Ninapro DB1 EMG is 100 Hz
    window_size: int = 400         # samples per window (4 s at 100 Hz)
    step_size: int = 200           # stride between windows

    # Identity (subject) model
    identity_batch_size: int = 128
    identity_lr: float = 1e-3
    identity_epochs: int = 100
    min_windows_per_subject: int = 100
    identity_val_fraction: float = 0.2

    # Utility (gesture) model
    utility_batch_size: int = 128
    utility_lr: float = 1e-3
    utility_epochs: int = 100

    # Blinder-style VAE
    use_blinder: bool = True
    blinder_z_dim: int = 64
    blinder_epochs: int = 100
    blinder_batch_size: int = 256
    blinder_lr: float = 1e-3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoints (saved under project_root/models)
    models_dir: str = None
    identity_ckpt: str = None
    utility_ckpt: str = None
    blinder_ckpt: str = None

    # Optional emg2qwerty checkpoint (for you to plug in)
    emg2qwerty_ckpt: str = None

    # Random seed
    seed: int = 1234

    def __post_init__(self):
        self.data_dir = os.path.join(self.project_root, "data")
        self.ninapro_csv = os.path.join(self.data_dir, "Ninapro_DB1.csv")

        self.results_dir = os.path.join(self.project_root, "emg_privacy_outputs")
        os.makedirs(self.results_dir, exist_ok=True)

        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.identity_ckpt = os.path.join(self.models_dir, "emg_identity_subject.pt")
        self.utility_ckpt = os.path.join(self.models_dir, "emg_utility_gesture.pt")
        self.blinder_ckpt = os.path.join(self.models_dir, "emg_blinder_vae.pt")

        # optional: where you'd drop your emg2qwerty Lightning ckpt
        self.emg2qwerty_ckpt = os.path.join(self.models_dir, "emg2qwerty.ckpt")


cfg = Config()

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.device == "cuda":
    torch.cuda.manual_seed_all(cfg.seed)


# -------------------------------------------------------------------
# Data loading: Ninapro DB1 CSV -> windowed EMG
# -------------------------------------------------------------------

def load_ninapro_windows(cfg: Config) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load Ninapro_DB1.csv and create windowed EMG segments.

    Returns:
        X: (N, T, C) windows
        meta: DataFrame with columns
            - row_idx: 0..N-1
            - subject_id: subject integer
            - exercise: exercise ID
            - stimulus: majority stimulus in the window
    """
    print(f"[Data] Loading Ninapro CSV from {cfg.ninapro_csv}")
    df = pd.read_csv(cfg.ninapro_csv)

    # Expect EMG columns emg_0 .. emg_9 as per Kaggle script
    emg_cols = [f"emg_{i}" for i in range(cfg.emg_channels)]
    for col in emg_cols:
        if col not in df.columns:
            raise ValueError(f"Expected EMG column {col} not found in CSV.")

    required_meta_cols = ["subject", "exercise", "stimulus", "repetition"]
    for col in required_meta_cols:
        if col not in df.columns:
            raise ValueError(f"Expected metadata column {col} not found in CSV.")

    # Sort to keep time continuity: subject / exercise / repetition / index
    df = df.sort_values(by=["subject", "exercise", "repetition", "Unnamed: 0"])
    df = df.reset_index(drop=True)

    X_windows = []
    meta_rows = []

    window_size = cfg.window_size
    step = cfg.step_size

    # Group by subject/exercise/repetition and slide windows
    for (subj, ex, rep), group in df.groupby(["subject", "exercise", "repetition"]):
        emg_seq = group[emg_cols].values.astype(np.float32)  # (L, C)
        stim_seq = group["stimulus"].values.astype(int)      # (L,)

        L = emg_seq.shape[0]
        if L < window_size:
            continue

        for start in range(0, L - window_size + 1, step):
            end = start + window_size
            window_emg = emg_seq[start:end, :]  # (T, C)
            window_stim = stim_seq[start:end]

            # majority label in the window (excluding zeros if possible)
            uniq, counts = np.unique(window_stim, return_counts=True)
            if len(uniq) == 0:
                stim = 0
            else:
                # prefer non-zero if present
                non_zero = uniq[uniq > 0]
                if len(non_zero) > 0:
                    nz_counts = counts[uniq > 0]
                    stim = non_zero[np.argmax(nz_counts)]
                else:
                    stim = uniq[np.argmax(counts)]

            row_idx = len(X_windows)

            X_windows.append(window_emg)
            meta_rows.append(
                {
                    "row_idx": row_idx,
                    "subject_id": int(subj),
                    "exercise": int(ex),
                    "stimulus": int(stim),
                }
            )

    X = np.stack(X_windows, axis=0)  # (N, T, C)
    meta = pd.DataFrame(meta_rows)

    print(f"[Data] Created {X.shape[0]} windows of shape (T={X.shape[1]}, C={X.shape[2]})")
    n_subjects = meta["subject_id"].nunique()
    print(f"[Data] Unique subjects: {n_subjects}")
    return X, meta


# -------------------------------------------------------------------
# Identity metadata + datasets
# -------------------------------------------------------------------

def build_identity_meta_emg(meta: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter subjects with enough windows, assign subject_idx and per-subject train/val split.
    """
    counts = meta["subject_id"].value_counts()
    valid_subj = counts[counts >= cfg.min_windows_per_subject].index
    meta_f = meta[meta["subject_id"].isin(valid_subj)].copy().reset_index(drop=True)

    print(f"[Identity] After min_windows filter: {len(meta_f)} windows "
          f"from {len(valid_subj)} subjects.")

    unique_subj = sorted(meta_f["subject_id"].unique())
    subj2idx = {s: i for i, s in enumerate(unique_subj)}
    meta_f["subject_idx"] = meta_f["subject_id"].map(subj2idx)

    # per-subject split into train / val
    folds = np.zeros(len(meta_f), dtype=int)
    for s in unique_subj:
        idxs = np.where(meta_f["subject_id"].values == s)[0]
        if len(idxs) == 1:
            folds[idxs] = 0
            continue
        n_val = max(1, int(np.round(cfg.identity_val_fraction * len(idxs))))
        val_idx = np.random.choice(idxs, size=n_val, replace=False)
        folds[val_idx] = 1
    meta_f["id_fold"] = folds

    print(f"[Identity] Split: "
          f"{(meta_f['id_fold'] == 0).sum()} train windows, "
          f"{(meta_f['id_fold'] == 1).sum()} val windows.")

    info = {
        "subj2idx": subj2idx,
        "idx2subj": {v: k for k, v in subj2idx.items()},
        "n_subjects": len(unique_subj),
    }
    return meta_f, info


class IdentityDatasetEMG(Dataset):
    def __init__(self, X_std: np.ndarray, meta: pd.DataFrame, split: str):
        """
        X_std: (N, T, C)
        meta: must have columns 'row_idx', 'subject_idx', 'id_fold'
        """
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]  # (n, T, C)

        # PyTorch expects (N, C, T)
        self.X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
        self.y = self.meta["subject_idx"].values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class SubjectIdentityNet(nn.Module):
    def __init__(self, in_channels: int, n_subjects: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_subjects)

    def forward(self, x):
        h = self.features(x).squeeze(-1)
        logits = self.head(h)
        return {"logits": logits, "feat": h}


def train_identity(
    model: SubjectIdentityNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
):
    model.to(cfg.device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.identity_lr)
    best_val_loss = float("inf")

    for epoch in range(cfg.identity_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            opt.zero_grad()
            out = model(xb)
            loss = ce(out["logits"], yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                out = model(xb)
                loss = ce(out["logits"], yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"[Identity] Epoch {epoch+1}/{cfg.identity_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        torch.save(model.state_dict(), cfg.identity_ckpt.replace(".pt", "_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.identity_ckpt)

    print(f"[Identity] Best val loss: {best_val_loss:.4f}")


def eval_identity(model: SubjectIdentityNet, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    model.to(cfg.device)

    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            out = model(xb)
            logits = out["logits"].cpu().numpy()
            preds = np.argmax(logits, axis=1)
            all_pred.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    print("[Identity] Validation metrics:")
    print(f"  subject_acc: {acc:.4f}")
    print(f"  subject_f1:  {f1:.4f}")
    return {"subject_acc": float(acc), "subject_f1": float(f1)}


# -------------------------------------------------------------------
# Utility model (gesture classifier) on Ninapro stimulus
# -------------------------------------------------------------------

class UtilityDatasetEMG(Dataset):
    def __init__(self, X_std: np.ndarray, meta: pd.DataFrame, split: str):
        """
        X_std: (N, T, C)
        meta: must have columns 'row_idx', 'stimulus', 'id_fold'
        """
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]
        self.X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)

        # Map stimulus labels to 0..K-1
        uniq_stim = sorted(self.meta["stimulus"].unique())
        self.stim2idx = {s: i for i, s in enumerate(uniq_stim)}
        self.idx2stim = {i: s for s, i in self.stim2idx.items()}
        self.y = self.meta["stimulus"].map(self.stim2idx).values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class GestureUtilityNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.features(x).squeeze(-1)
        logits = self.head(h)
        return {"logits": logits, "feat": h}


def train_utility(
    model: GestureUtilityNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
):
    model.to(cfg.device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.utility_lr)
    best_val_loss = float("inf")

    for epoch in range(cfg.utility_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad()
            out = model(xb)
            loss = ce(out["logits"], yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                out = model(xb)
                loss = ce(out["logits"], yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"[Utility] Epoch {epoch+1}/{cfg.utility_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        torch.save(model.state_dict(), cfg.utility_ckpt.replace(".pt", "_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.utility_ckpt)

    print(f"[Utility] Best val loss: {best_val_loss:.4f}")


def eval_utility(model: GestureUtilityNet, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    model.to(cfg.device)

    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            out = model(xb)
            logits = out["logits"].cpu().numpy()
            preds = np.argmax(logits, axis=1)
            all_pred.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    print("[Utility] Validation metrics:")
    print(f"  gesture_acc: {acc:.4f}")
    print(f"  gesture_f1:  {f1:.4f}")
    return {"gesture_acc": float(acc), "gesture_f1": float(f1)}


# -------------------------------------------------------------------
# Optional: emg2qwerty checkpoint hook (for you to customize)
# -------------------------------------------------------------------

def load_emg2qwerty_from_ckpt(ckpt_path: str, device: str = "cpu"):
    """
    Example hook for loading emg2qwerty Lightning checkpoint.

    NOTE:
    - The actual emg2qwerty TDSConvCTCModule expects 5D spectrogram inputs
      shaped like (T, N, bands, electrode_channels, freq_bins).
    - You will need to apply the same preprocessing and transforms as in the
      original repo (see emg2qwerty/data.py and transforms.py).
    - Here we just show how to load the module; we DO NOT integrate it into
      the running pipeline, since Ninapro -> emg2qwerty preprocessing is nontrivial.
    """
    if not os.path.exists(ckpt_path):
        print(f"[emg2qwerty] Checkpoint not found at {ckpt_path}, skipping.")
        return None

    try:
        from lightning import TDSConvCTCModule  # your local emg2qwerty.lightning
    except ImportError:
        print("[emg2qwerty] Could not import TDSConvCTCModule from lightning.py.")
        return None

    print(f"[emg2qwerty] Loading Lightning module from {ckpt_path}...")
    model = TDSConvCTCModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)
    return model


# -------------------------------------------------------------------
# Fidelity + plotting helpers
# -------------------------------------------------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psd_correlation(x: np.ndarray, y: np.ndarray, fs: int) -> float:
    """
    Compute correlation between average PSDs across channels.
    x, y: (T, C) or (C, T)
    """
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
    """
    signal_np: (T, C) or (C, T)
    """
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


def plot_overlay_emg(
    orig: np.ndarray,
    anon: np.ndarray,
    fs: int,
    title: str,
    filepath: str,
    channel: int = 0,
    max_seconds: float = 5.0,
):
    """
    Plot original vs anonymized EMG on the same axis for one channel.
    orig, anon: (T, C) or (C, T)
    """
    if orig.shape[0] < orig.shape[1]:
        orig = orig.T
    if anon.shape[0] < anon.shape[1]:
        anon = anon.T

    T = orig.shape[0]
    max_samples = min(T, int(max_seconds * fs))
    t = np.arange(max_samples) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, orig[:max_samples, channel], label="Original", alpha=0.8)
    plt.plot(t, anon[:max_samples, channel], label="Blinder anonymized", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title + f" (ch {channel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# -------------------------------------------------------------------
# Blinder-style VAE (same structure as ECG script, but for EMG)
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
        # clamp logvar
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
        out = self.fc5(h4)
        return out


class EMGBlinderVAE(nn.Module):
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
        z, mu, logvar = self.encoder(x_flat)
        recon_flat = self.decoder(z)
        recon = recon_flat.view(N, T, C)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 1e-3 * kld


def train_blinder_vae(X_std: np.ndarray, cfg: Config) -> EMGBlinderVAE:
    N, T, C = X_std.shape
    model = EMGBlinderVAE(T=T, C=C, z_dim=cfg.blinder_z_dim).to(cfg.device)

    dataset = TensorDataset(torch.from_numpy(X_std.astype(np.float32)))
    loader = DataLoader(
        dataset,
        batch_size=cfg.blinder_batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.blinder_lr)

    print(f"[BlinderVAE] Training for {cfg.blinder_epochs} epochs on {N} windows...")
    for epoch in range(cfg.blinder_epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(cfg.device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        print(f"[BlinderVAE] Epoch {epoch+1}/{cfg.blinder_epochs} loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), cfg.blinder_ckpt)
    print(f"[BlinderVAE] Saved to {cfg.blinder_ckpt}")
    return model


def load_or_train_blinder_vae(X_std: np.ndarray, cfg: Config) -> EMGBlinderVAE:
    N, T, C = X_std.shape
    model = EMGBlinderVAE(T=T, C=C, z_dim=cfg.blinder_z_dim).to(cfg.device)
    if os.path.exists(cfg.blinder_ckpt):
        print(f"[BlinderVAE] Loading existing checkpoint from {cfg.blinder_ckpt}")
        state = torch.load(cfg.blinder_ckpt, map_location=cfg.device)
        model.load_state_dict(state)
        return model
    else:
        return train_blinder_vae(X_std, cfg)


def anonymize_emg_with_blinder(model: EMGBlinderVAE, X_std: np.ndarray, cfg: Config) -> np.ndarray:
    model.eval()
    X = X_std.astype(np.float32)
    N, T, C = X.shape
    anon_chunks = []

    with torch.no_grad():
        for start in range(0, N, cfg.blinder_batch_size):
            end = min(start + cfg.blinder_batch_size, N)
            xb = torch.from_numpy(X[start:end]).to(cfg.device)
            recon, _, _ = model(xb)
            anon_chunks.append(recon.cpu().numpy())

    X_anon = np.concatenate(anon_chunks, axis=0)
    X_anon = np.nan_to_num(X_anon, nan=0.0, posinf=5.0, neginf=-5.0)
    assert X_anon.shape == X_std.shape
    return X_anon


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main():
    print("=== Ninapro EMG + Blinder-style anonymization pipeline ===")

    # 1) Load windowed EMG + metadata
    X, meta_raw = load_ninapro_windows(cfg)  # X: (N, T, C)

    # 2) Identity meta
    meta, id_info = build_identity_meta_emg(meta_raw, cfg)

    # 3) Standardize across all windows
    print("[Standardization] Fitting StandardScaler on all windows...")
    N, T, C = X.shape
    X_flat = X.reshape(N, -1)
    scaler = StandardScaler()
    X_flat_std = scaler.fit_transform(X_flat)
    X_std = X_flat_std.reshape(N, T, C)
    print("[Standardization] Done.")

    # 4) Identity datasets/loaders (baseline)
    train_ds_id = IdentityDatasetEMG(X_std, meta, split="train")
    val_ds_id = IdentityDatasetEMG(X_std, meta, split="val")
    train_loader_id = DataLoader(
        train_ds_id,
        batch_size=cfg.identity_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader_id = DataLoader(
        val_ds_id,
        batch_size=cfg.identity_batch_size,
        shuffle=False,
        drop_last=False,
    )

    id_model = SubjectIdentityNet(
        in_channels=cfg.emg_channels,
        n_subjects=id_info["n_subjects"],
    )

    if os.path.exists(cfg.identity_ckpt):
        print(f"[Identity] Loading existing model from {cfg.identity_ckpt}")
        state = torch.load(cfg.identity_ckpt, map_location=cfg.device)
        id_model.load_state_dict(state)
    else:
        print("[Identity] Training subject-identity model from scratch...")
        train_identity(id_model, train_loader_id, val_loader_id, cfg)
        print(f"[Identity] Model saved to {cfg.identity_ckpt}")

    print("[Identity] Evaluating baseline (original, standardized)...")
    id_metrics_base = eval_identity(id_model, val_loader_id, cfg)

    # 5) Utility datasets/loaders (gesture)
    train_ds_ut = UtilityDatasetEMG(X_std, meta, split="train")
    val_ds_ut = UtilityDatasetEMG(X_std, meta, split="val")
    train_loader_ut = DataLoader(train_ds_ut, batch_size=cfg.utility_batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader_ut = DataLoader(val_ds_ut, batch_size=cfg.utility_batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Infer number of gesture classes directly from metadata
    n_gestures = len(train_ds_ut.stim2idx)
    ut_model = GestureUtilityNet(in_channels=C, n_classes=n_gestures)



    if os.path.exists(cfg.utility_ckpt):
        print(f"[Utility] Loading existing model from {cfg.utility_ckpt}")
        state = torch.load(cfg.utility_ckpt, map_location=cfg.device)
        ut_model.load_state_dict(state)
    else:
        print("[Utility] Training gesture utility model from scratch...")
        train_utility(ut_model, train_loader_ut, val_loader_ut, cfg)
        print(f"[Utility] Model saved to {cfg.utility_ckpt}")

    print("[Utility] Evaluating baseline (original, standardized)...")
    ut_metrics_base = eval_utility(ut_model, val_loader_ut, cfg)

    # 6) Baseline fidelity in raw space (original vs itself: sanity)
    print("[Fidelity] Baseline sanity check...")
    val_row_idx = val_ds_id.meta["row_idx"].values
    X_val_raw = X[val_row_idx]  # original raw, subset

    rmses_base, psd_corrs_base = [], []
    for i in range(X_val_raw.shape[0]):
        rmses_base.append(rmse(X_val_raw[i], X_val_raw[i]))
        psd_corrs_base.append(
            psd_correlation(
                X_val_raw[i],
                X_val_raw[i],
                fs=cfg.sampling_frequency,
            )
        )

    fidelity_baseline = {
        "rmse_mean": float(np.mean(rmses_base)),
        "rmse_std": float(np.std(rmses_base)),
        "psd_corr_mean": float(np.mean(psd_corrs_base)),
        "psd_corr_std": float(np.std(psd_corrs_base)),
    }

    # 7) Train / load Blinder VAE + anonymize standardized EMG
    if cfg.use_blinder:
        print("[BlinderVAE] Loading / training anonymizer...")
        blinder_model = load_or_train_blinder_vae(X_std, cfg)

        print("[BlinderVAE] Anonymizing all standardized EMG windows...")
        X_std_anon = anonymize_emg_with_blinder(blinder_model, X_std, cfg)

        # Anonymized identity datasets
        train_ds_id_anon = IdentityDatasetEMG(X_std_anon, meta, split="train")
        val_ds_id_anon = IdentityDatasetEMG(X_std_anon, meta, split="val")
        val_loader_id_anon = DataLoader(
            val_ds_id_anon,
            batch_size=cfg.identity_batch_size,
            shuffle=False,
            drop_last=False,
        )

        print("[Identity] Evaluating on anonymized validation set...")
        id_metrics_anon = eval_identity(id_model, val_loader_id_anon, cfg)

        # Anonymized utility datasets
        train_ds_ut_anon = UtilityDatasetEMG(X_std_anon, meta, split="train")
        val_ds_ut_anon = UtilityDatasetEMG(X_std_anon, meta, split="val")
        val_loader_ut_anon = DataLoader(
            val_ds_ut_anon,
            batch_size=cfg.utility_batch_size,
            shuffle=False,
            drop_last=False,
        )

        print("[Utility] Evaluating on anonymized validation set...")
        ut_metrics_anon = eval_utility(ut_model, val_loader_ut_anon, cfg)

        # 8) Fidelity between original raw and anonymized raw (val subset)
        print("[Fidelity] Computing fidelity metrics (original vs anonymized)...")
        X_val_std_anon = X_std_anon[val_row_idx]
        N_val, T_val, C_val = X_val_std_anon.shape
        X_val_anon_flat = X_val_std_anon.reshape(N_val, -1)
        X_val_anon_raw = scaler.inverse_transform(X_val_anon_flat).reshape(
            N_val, T_val, C_val
        )

        rmses_blinder, psd_blinder = [], []
        for i in range(N_val):
            rmses_blinder.append(rmse(X_val_raw[i], X_val_anon_raw[i]))
            psd_blinder.append(
                psd_correlation(
                    X_val_raw[i],
                    X_val_anon_raw[i],
                    fs=cfg.sampling_frequency,
                )
            )

        fidelity_blinder = {
            "rmse_mean": float(np.mean(rmses_blinder)),
            "rmse_std": float(np.std(rmses_blinder)),
            "psd_corr_mean": float(np.mean(psd_blinder)),
            "psd_corr_std": float(np.std(psd_blinder)),
        }

        # 9) FFT + overlays
        print("[Plots] FFT + overlay examples...")
        example_raw = X_val_raw[0]
        example_anon_raw = X_val_anon_raw[0]

        fft_raw_path = os.path.join(cfg.results_dir, "emg_fft_val_raw.png")
        fft_anon_path = os.path.join(cfg.results_dir, "emg_fft_val_anon.png")
        plot_fft(
            example_raw,
            cfg.sampling_frequency,
            "EMG FFT (original)",
            fft_raw_path,
        )
        plot_fft(
            example_anon_raw,
            cfg.sampling_frequency,
            "EMG FFT (Blinder anonymized)",
            fft_anon_path,
        )

        overlay_dir = os.path.join(cfg.results_dir, "overlay_examples")
        os.makedirs(overlay_dir, exist_ok=True)

        n_examples_to_plot = min(5, N_val)
        channels_to_plot = [0, 1]  # first two EMG channels

        for i in range(n_examples_to_plot):
            for ch in channels_to_plot:
                fname = f"overlay_val_idx{i}_ch{ch}.png"
                out_path = os.path.join(overlay_dir, fname)
                title = f"Original vs Blinder anonymized EMG (val idx {i})"
                plot_overlay_emg(
                    orig=X_val_raw[i],
                    anon=X_val_anon_raw[i],
                    fs=cfg.sampling_frequency,
                    title=title,
                    filepath=out_path,
                    channel=ch,
                    max_seconds=4.0,
                )

        print(f"[Plots] Overlays saved to {overlay_dir}")

    else:
        id_metrics_anon = None
        ut_metrics_anon = None
        fidelity_blinder = None
        fft_raw_path = None
        fft_anon_path = None
        overlay_dir = None

    # 10) Save summary JSON
    results = {
        "identity_baseline": id_metrics_base,
        "identity_blinder": id_metrics_anon,
        "utility_baseline": ut_metrics_base,
        "utility_blinder": ut_metrics_anon,
        "fidelity_baseline": fidelity_baseline,
        "fidelity_blinder": fidelity_blinder,
        "plots": {
            "fft_raw": fft_raw_path,
            "fft_anon": fft_anon_path,
            "overlay_dir": overlay_dir,
        },
    }
    out_json = os.path.join(cfg.results_dir, "emg_blinder_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Done ===")
    print(f"[Results] Saved metrics + plot paths to {out_json}")


if __name__ == "__main__":
    main()
