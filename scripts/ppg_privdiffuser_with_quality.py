#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPG + PrivDiffuser-style anonymization + quality assessment.

Data expectation:
    - A NumPy archive at cfg.data_npz containing:
        X:    (N, T) raw PPG waveforms
        meta: pandas DataFrame pickled at cfg.meta_pkl with columns:
              row_idx, subject_id, id_fold (0=train, 1=val)
    You can produce X/meta from the seeing-red pipeline after beat filtering,
    or any other PPG source, as long as the shapes/columns match.
"""

import os
import sys
import json
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import subprocess

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TOP_LEVEL = os.path.dirname(PROJECT_ROOT)
PRIVDIFFUSER_DIR = os.path.join(PROJECT_ROOT, "PrivDiffuser")
PPGRAW_DIR = os.path.join(PROJECT_ROOT, "PPGraw", "src")
SEEINGRED_EXTRACTED = os.path.join(PROJECT_ROOT, "seeing-red", "data", "extracted")

for p in [THIS_DIR, PROJECT_ROOT, TOP_LEVEL, PRIVDIFFUSER_DIR, PPGRAW_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# PrivDiffuser imports
from unet import Unet
from diffusion import GaussianDiffusion
from embedding import ConditionalEmbedding
from scheduler import GradualWarmupScheduler
from utils import get_named_beta_schedule

# PPG quality scoring
from PPGraw import PPGraw

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

@dataclass
class PPGConfig:
    data_npz: str = os.path.join(PROJECT_ROOT, "data", "ppg_windows.npz")
    meta_pkl: str = os.path.join(PROJECT_ROOT, "data", "ppg_meta.pkl")
    anon_npz: str = os.path.join(PROJECT_ROOT, "data", "ppg_anon.npz")
    sampling_frequency: int = 60  # Hz
    identity_batch_size: int = 64
    identity_lr: float = 5e-4
    identity_epochs: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    identity_ckpt: str = os.path.join(PROJECT_ROOT, "models", "ppg_privdiffuser", "identity.pt")
    results_dir: str = os.path.join(PROJECT_ROOT, "privacy_ppg_outputs")
    run_seeingred_identity: bool = True


@dataclass
class PPGDiffusionConfig:
    inch: int = 1
    modch: int = 64
    outch: int = 1
    T: int = 1000
    chmul: Tuple[int, ...] = (1, 2)
    numres: int = 2
    cdim: int = 64
    useconv: bool = True
    droprate: float = 0.1
    dtype: torch.dtype = torch.float32
    lr: float = 2e-4
    epoch: int = 40
    multiplier: int = 2
    threshold: float = 0.3
    num_steps: int = 50
    eta: float = 0.0
    select: str = "linear"
    w1: float = 2.0
    w2: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    model_dir: str = os.path.join(PROJECT_ROOT, "models", "ppg_privdiffuser")
    hr_epochs: int = 30
    hr_batch_size: int = 128
    hr_lr: float = 1e-3


# -------------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------------

def load_ppg_data(cfg: PPGConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    # Auto-build from seeing-red extracted signals if npz/pkl are missing
    if not (os.path.isfile(cfg.data_npz) and os.path.isfile(cfg.meta_pkl)):
        X, meta = build_ppg_from_seeingred(cfg)
    else:
        data_npz = np.load(cfg.data_npz)
        if "X" not in data_npz:
            raise KeyError("npz must contain 'X'")
        X = data_npz["X"]  # (N, T)
        with open(cfg.meta_pkl, "rb") as f:
            meta = pickle.load(f)
    # ensure contiguous subject labels
    subj_ids = sorted(meta["subject_id"].unique())
    subj_map = {sid: i for i, sid in enumerate(subj_ids)}
    meta["subject_idx"] = meta["subject_id"].map(subj_map)
    if "row_idx" not in meta.columns or "id_fold" not in meta.columns:
        raise ValueError("meta must contain row_idx and id_fold")
    return X, meta


def build_ppg_from_seeingred(cfg: PPGConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    if not os.path.isdir(SEEINGRED_EXTRACTED):
        raise FileNotFoundError(f"Missing {cfg.data_npz} and no seeing-red extracted dir at {SEEINGRED_EXTRACTED}")

    records = []
    for subj in sorted(os.listdir(SEEINGRED_EXTRACTED)):
        subj_dir = os.path.join(SEEINGRED_EXTRACTED, subj)
        if not os.path.isdir(subj_dir):
            continue
        files = [f for f in os.listdir(subj_dir) if f.endswith(".csv")]
        for fname in sorted(files):
            records.append((int(subj), os.path.join(subj_dir, fname), fname))

    if len(records) == 0:
        raise FileNotFoundError("No extracted CSVs found under seeing-red/data/extracted")

    # First pass to determine min length after resample
    resampled = []
    min_len = None
    src_fs = 240  # from seeing-red params.yaml
    for subj_id, path, fname in records:
        df = pd.read_csv(path)
        if "luma_mean" not in df.columns:
            continue
        luma = df["luma_mean"].values.astype(np.float32)
        red = df["r_ch_mean"].values.astype(np.float32) if "r_ch_mean" in df.columns else np.zeros_like(luma)
        sig = np.stack([luma, red], axis=1)  # (T, C=2)
        target_len = int(sig.shape[0] * cfg.sampling_frequency / src_fs)
        target_len = max(target_len, cfg.sampling_frequency)  # ensure at least 1s worth
        sig_rs = np.stack([resample(sig[:, 0], target_len), resample(sig[:, 1], target_len)], axis=1)
        resampled.append((subj_id, fname, sig_rs))
        min_len = target_len if min_len is None else min(min_len, target_len)

    if min_len is None:
        raise ValueError("No valid luma_mean signals found in extracted CSVs")

    X_list, meta_rows = [], []
    subj_ids = sorted({r[0] for r in resampled})
    subj_map = {sid: i for i, sid in enumerate(subj_ids)}
    for idx, (subj_id, fname, sig_rs) in enumerate(resampled):
        sig_trim = sig_rs[:min_len]
        X_list.append(sig_trim)
        meta_rows.append({"row_idx": idx, "subject_id": subj_id, "subject_idx": subj_map[subj_id], "filename": fname})

    X = np.stack(X_list, axis=0)  # (N, T, C=2)
    meta = pd.DataFrame(meta_rows)

    # simple per-subject split: 80% train, 20% val (shuffled)
    folds = []
    rng = np.random.default_rng(42)
    for subj_id in meta["subject_id"].unique():
        idxs = meta.index[meta["subject_id"] == subj_id].tolist()
        rng.shuffle(idxs)
        n_val = max(1, int(0.2 * len(idxs)))
        val_idx = set(idxs[:n_val])
        for i in idxs:
            folds.append((i, 1 if i in val_idx else 0))
    fold_map = dict(folds)
    meta["id_fold"] = meta.index.map(fold_map)

    os.makedirs(os.path.dirname(cfg.data_npz), exist_ok=True)
    np.savez(cfg.data_npz, X=X)
    with open(cfg.meta_pkl, "wb") as f:
        pickle.dump(meta, f)
    print(f"[PPG] Built {cfg.data_npz} and {cfg.meta_pkl} from seeing-red extracted signals (len={min_len}, N={len(X)})")
    return X, meta


# -------------------------------------------------------------------------
# Identity dataset / model
# -------------------------------------------------------------------------

class PPGIdentityDataset(Dataset):
    def __init__(self, X_std: np.ndarray, meta: pd.DataFrame, split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")
        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]
        # (n,C,T)
        self.X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
        # use contiguous subject_idx labels
        y_col = "subject_idx" if "subject_idx" in self.meta.columns else "subject_id"
        self.y = self.meta[y_col].values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = self.X[idx]
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mean) / std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=s) if in_ch != out_ch or s != 1 else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, s, p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, 1, p),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = out + self.proj(x)
        return self.act(out)


class PPGIdentityNet(nn.Module):
    def __init__(self, n_subjects: int, in_channels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, k=5, s=2, p=2, dropout=dropout),
            ResidualBlock1D(64, 128, k=5, s=2, p=2, dropout=dropout),
            ResidualBlock1D(128, 256, k=3, s=2, p=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, n_subjects)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        logits = self.head(h)
        return {"logits": logits, "feat": h}


def train_identity(model: PPGIdentityNet, train_loader: DataLoader, val_loader: DataLoader, cfg: PPGConfig):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.identity_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.identity_epochs, eta_min=1e-5)
    # class weights to handle imbalance
    meta_df = getattr(train_loader.dataset, "meta", None)
    if meta_df is not None and "subject_idx" in meta_df.columns:
        counts = meta_df["subject_idx"].value_counts().sort_index()
        weights = (1.0 / (counts + 1e-6))
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights.values, device=cfg.device, dtype=torch.float32)
        ce = nn.CrossEntropyLoss(weight=class_weights)
    else:
        ce = nn.CrossEntropyLoss()
    best_val = float("inf")
    for ep in range(cfg.identity_epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            out = model(xb)
            loss = ce(out["logits"], yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                out = model(xb)
                loss = ce(out["logits"], yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[PPG Identity] Epoch {ep+1}/{cfg.identity_epochs} train={tr_loss:.4f} val={val_loss:.4f}")
        scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(cfg.identity_ckpt), exist_ok=True)
            torch.save(model.state_dict(), cfg.identity_ckpt)
    print(f"[PPG Identity] Best val loss={best_val:.4f}")


def eval_identity(model: PPGIdentityNet, loader: DataLoader, cfg: PPGConfig) -> Dict[str, Any]:
    model.to(cfg.device)
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            out = model(xb)
            loss = ce(out["logits"], yb)
            loss_sum += loss.item() * xb.size(0)
            preds = out["logits"].argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_true.extend(yb.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
    f1 = f1_score(all_true, all_pred, average="macro") if all_true else 0.0
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total), "f1_macro": float(f1)}


# -------------------------------------------------------------------------
# Pulse-rate surrogate (utility)
# -------------------------------------------------------------------------

def estimate_bpm(sig_raw: np.ndarray, fs: int) -> float:
    # if multichannel, use first channel
    if sig_raw.ndim == 2:
        sig_raw = sig_raw[:, 0]
    sig = sig_raw - np.nanmean(sig_raw)
    sig = np.nan_to_num(sig, nan=0.0)
    b, a = signal.butter(2, [0.5 / (fs / 2), 5.0 / (fs / 2)], btype="band")
    sig_filt = signal.filtfilt(b, a, sig)
    peaks, _ = signal.find_peaks(sig_filt, distance=int(0.3 * fs), prominence=np.std(sig_filt) * 0.5)
    bpm = None
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        rr = rr[rr > 1e-3]
        if len(rr) > 0:
            bpm = 60.0 / np.median(rr)
    # fallback: autocorrelation
    if bpm is None:
        corr = np.correlate(sig_filt, sig_filt, mode="full")
        corr = corr[corr.size // 2:]
        # ignore zero lag
        corr[0] = 0
        peak_idx = np.argmax(corr[: int(fs * 2)])  # up to ~2s lag (30 bpm)
        if peak_idx > 0:
            bpm = 60.0 * fs / peak_idx
    if bpm is None:
        return 0.0
    return float(np.clip(bpm, 30.0, 220.0))


def detect_peaks(sig_raw: np.ndarray, fs: int):
    if sig_raw.ndim == 2:
        sig_raw = sig_raw[:, 0]
    sig = sig_raw - np.nanmean(sig_raw)
    sig = np.nan_to_num(sig, nan=0.0)
    b, a = signal.butter(2, [0.5 / (fs / 2), 5.0 / (fs / 2)], btype="band")
    sig_filt = signal.filtfilt(b, a, sig)
    peaks, _ = signal.find_peaks(sig_filt, distance=int(0.3 * fs), prominence=np.std(sig_filt) * 0.5)
    return peaks, sig_filt


def compute_hrv_metrics(signals: np.ndarray, fs: int) -> Dict[str, float]:
    rmssd_list, sdnn_list, pnn50_list = [], [], []
    for sig in signals:
        sig_use = sig[:, 0] if sig.ndim == 2 else sig
        peaks, _ = detect_peaks(sig_use, fs)
        if len(peaks) < 3:
            continue
        rr = np.diff(peaks) / fs
        rr = rr[rr > 1e-3]
        if len(rr) < 2:
            continue
        diff_rr = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        sdnn = np.std(rr)
        nn50 = np.sum(np.abs(diff_rr) > 0.05)
        pnn50 = nn50 / max(1, len(diff_rr))
        rmssd_list.append(rmssd)
        sdnn_list.append(sdnn)
        pnn50_list.append(pnn50)
    return {
        "rmssd_mean": float(np.nanmean(rmssd_list)) if rmssd_list else 0.0,
        "sdnn_mean": float(np.nanmean(sdnn_list)) if sdnn_list else 0.0,
        "pnn50_mean": float(np.nanmean(pnn50_list)) if pnn50_list else 0.0,
    }


def estimate_resp_rate(signals: np.ndarray, fs: int) -> Dict[str, float]:
    rates = []
    for sig in signals:
        sig = sig[:, 0] if sig.ndim == 2 else sig
        sig = sig - np.nanmean(sig)
        sig = np.nan_to_num(sig, nan=0.0)
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 1024))
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not np.any(mask):
            continue
        freq_band = freqs[mask]
        psd_band = psd[mask]
        if psd_band.size == 0:
            continue
        peak_freq = freq_band[np.argmax(psd_band)]
        rates.append(peak_freq * 60.0)
    return {
        "resp_rate_mean": float(np.nanmean(rates)) if rates else 0.0,
        "resp_rate_std": float(np.nanstd(rates)) if rates else 0.0,
    }


class PulseRateDataset(Dataset):
    def __init__(self, X_std: np.ndarray, bpm: np.ndarray, meta: pd.DataFrame, split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")
        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]
        # (n,C,T)
        self.X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
        self.bpm = bpm[idx]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.bpm[idx], dtype=torch.float32)


class PulseRateRegressor(nn.Module):
    def __init__(self, cdim: int, in_channels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, k=5, s=2, p=2, dropout=dropout),
            ResidualBlock1D(64, 128, k=5, s=2, p=2, dropout=dropout),
            ResidualBlock1D(128, 256, k=3, s=2, p=1, dropout=dropout),
            ResidualBlock1D(256, 256, k=3, s=2, p=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, 1)
        self.proj = nn.Linear(256, cdim)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        bpm = self.head(h).squeeze(-1)
        z = torch.nn.functional.normalize(self.proj(h), dim=-1)
        return bpm, z


def train_pulse_regressor(model: PulseRateRegressor,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          diff_cfg: PPGDiffusionConfig) -> Dict[str, float]:
    device = diff_cfg.device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=diff_cfg.hr_lr)
    mae = nn.L1Loss()
    best_val = float("inf")
    for ep in range(diff_cfg.hr_epochs):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = mae(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * xb.size(0)
        tr /= len(train_loader.dataset)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                loss = mae(pred, yb)
                vl += loss.item() * xb.size(0)
        vl /= len(val_loader.dataset)
        print(f"[PulseRate] Epoch {ep+1}/{diff_cfg.hr_epochs} train_mae={tr:.3f} val_mae={vl:.3f}")
        if vl < best_val:
            best_val = vl
    return {"train_mae": tr, "val_mae": best_val}


class PPGSurrogate(nn.Module):
    def __init__(self, base_model: PulseRateRegressor, z_dim: int):
        super().__init__()
        self.base = base_model
        self.z_proj = nn.Identity()

    def forward(self, x_img):
        # x_img is (B,1,C,T); drop the dummy spatial dim -> (B,C,T)
        if x_img.dim() == 4:
            x = x_img.squeeze(1)
        else:
            x = x_img
        B, C, T = x.shape
        bpm, z = self.base(x)
        return bpm, self.z_proj(z)


def build_frozen_surrogate(base_model: PulseRateRegressor, z_dim: int, device: str) -> PPGSurrogate:
    surrogate = PPGSurrogate(base_model, z_dim=z_dim).to(device)
    for p in surrogate.base.parameters():
        p.requires_grad = False
    surrogate.eval()
    return surrogate


# -------------------------------------------------------------------------
# Diffusion dataset
# -------------------------------------------------------------------------

class PPGDiffuserDataset(Dataset):
    def __init__(self, X_std: np.ndarray, meta: pd.DataFrame, split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")
        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]                # (n, T, C)
        X_ct = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)  # (n,C,T)
        self.X = X_ct[:, None, :, :]      # (n,1,C,T)
        y_col = "subject_idx" if "subject_idx" in subset.columns else "subject_id"
        self.y_priv = subset[y_col].values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_priv[idx], dtype=torch.long),
        )


# -------------------------------------------------------------------------
# Quality metrics (PPGraw)
# -------------------------------------------------------------------------

def compute_ppg_quality_batch(signals: np.ndarray, fs: int) -> Dict[str, float]:
    metrics = {
        "timebase_mean": [],
        "ampl_span": [],
        "granularity": [],
        "norm_min": [],
        "norm_max": [],
        "freq_if_max": [],
    }
    for sig in signals:
        sig_use = sig[:, 0] if sig.ndim == 2 else sig
        try:
            pr = PPGraw(signal=sig_use.tolist(), fs=fs)
            tb = pr.review_timebase() if hasattr(pr, "review_timebase") else {}
            amp = pr.review_amplitude()
            gran = pr.review_granularity()
            norm = pr.review_normalization()
            freq = pr.review_frequency()
            metrics["timebase_mean"].append(tb.get("mean", 0.0) if isinstance(tb, dict) else 0.0)
            metrics["ampl_span"].append(amp.get("span", 0.0))
            metrics["granularity"].append(gran.get("granularity", 0.0))
            metrics["norm_min"].append(norm.get("min", 0.0))
            metrics["norm_max"].append(norm.get("max", 0.0))
            metrics["freq_if_max"].append(freq.get("IF_max", 0.0))
        except Exception:
            # If PPGraw fails on a sample, record zeros to avoid crashing.
            metrics["timebase_mean"].append(0.0)
            metrics["ampl_span"].append(0.0)
            metrics["granularity"].append(0.0)
            metrics["norm_min"].append(0.0)
            metrics["norm_max"].append(0.0)
            metrics["freq_if_max"].append(0.0)
    return {k: float(np.nanmean(v)) for k, v in metrics.items()}


# -------------------------------------------------------------------------
# Train-or-load diffusion and anonymize
# -------------------------------------------------------------------------

def train_and_apply_privdiffuser_ppg(X_std: np.ndarray,
                                     meta: pd.DataFrame,
                                     diff_cfg: PPGDiffusionConfig,
                                     surrogate_base: PulseRateRegressor,
                                     id_model: PPGIdentityNet) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(diff_cfg.model_dir, exist_ok=True)
    device = diff_cfg.device

    ckpt_unet = os.path.join(diff_cfg.model_dir, "unet.pt")
    ckpt_cemb = os.path.join(diff_cfg.model_dir, "cond_emb.pt")

    train_ds = PPGDiffuserDataset(X_std, meta, split="train")
    val_ds = PPGDiffuserDataset(X_std, meta, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=diff_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=diff_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
    )

    surrogate = build_frozen_surrogate(surrogate_base, z_dim=diff_cfg.cdim, device=device)
    # identity classifier for negative guidance
    id_model = id_model.to(device)
    id_model.eval()
    class IdentityWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x, emb=None):
            out = self.base(x)
            return out["logits"], out["feat"]
    id_wrap = IdentityWrapper(id_model)

    net = Unet(
        in_ch=diff_cfg.inch,
        mod_ch=diff_cfg.modch,
        out_ch=diff_cfg.outch,
        ch_mul=diff_cfg.chmul,
        num_res_blocks=diff_cfg.numres,
        cdim=diff_cfg.cdim,
        use_conv=diff_cfg.useconv,
        droprate=diff_cfg.droprate,
        dtype=diff_cfg.dtype,
    ).to(device)

    betas = get_named_beta_schedule(num_diffusion_timesteps=diff_cfg.T)
    diffusion = GaussianDiffusion(
        dtype=diff_cfg.dtype,
        model=net,
        betas=betas,
        w=diff_cfg.w1,
        v=0.1,
        device=device,
    )

    cemblayer = ConditionalEmbedding(diff_cfg.cdim, diff_cfg.cdim, diff_cfg.cdim).to(device)

    if os.path.isfile(ckpt_unet) and os.path.isfile(ckpt_cemb):
        print("[PPG Diffusion] Loading existing checkpoints...")
        net.load_state_dict(torch.load(ckpt_unet, map_location=device))
        cemblayer.load_state_dict(torch.load(ckpt_cemb, map_location=device))
    else:
        print(f"[PPG Diffusion] Training for {diff_cfg.epoch} epochs...")
        optimizer = torch.optim.AdamW(
            list(diffusion.model.parameters()) + list(cemblayer.parameters()),
            lr=diff_cfg.lr,
            weight_decay=1e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=diff_cfg.epoch,
            eta_min=0.0,
            last_epoch=-1,
        )
        warmup_scheduler = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=diff_cfg.multiplier,
            warm_epoch=diff_cfg.epoch // 10,
            after_scheduler=cosine_scheduler,
        )

        for ep in range(diff_cfg.epoch):
            diffusion.model.train()
            cemblayer.train()
            surrogate.eval()
            ep_loss, nb = 0.0, 0
            for x_img, y_priv in train_loader:
                x_img = x_img.to(device)
                y_priv = y_priv.to(device)
                b = x_img.size(0)
                optimizer.zero_grad()
                with torch.no_grad():
                    _, emb = surrogate(x_img)
                cemb = cemblayer(emb)
                drop_mask = (torch.rand(b, device=device) < diff_cfg.threshold).float().view(b, 1)
                cemb = cemb * (1.0 - drop_mask)
                loss = diffusion.trainloss(x_img, cemb=cemb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                nb += 1
            warmup_scheduler.step()
            print(f"[PPG Diffusion] Epoch {ep+1}/{diff_cfg.epoch} loss={ep_loss/max(1,nb):.4f}")
        torch.save(net.state_dict(), ckpt_unet)
        torch.save(cemblayer.state_dict(), ckpt_cemb)
        print(f"[PPG Diffusion] Saved to {ckpt_unet} / {ckpt_cemb}")

    print("[PPG Diffusion] Sampling anonymized validation split...")
    diffusion.model.eval()
    cemblayer.eval()
    surrogate.eval()

    val_row_idx = val_ds.meta["row_idx"].values
    recon_imgs = np.empty((len(val_ds), 1, X_std.shape[2], X_std.shape[1]), dtype=np.float32)
    with torch.no_grad():
        offset = 0
        for x_img, y_priv in val_loader:
            x_img = x_img.to(device)
            y_priv = y_priv.to(device)
            B = x_img.size(0)
            _, emb = surrogate(x_img)
            cemb = cemblayer(emb)
            genshape = x_img.shape
            generated = diffusion.ddim_sample(
                genshape,
                diff_cfg.num_steps,
                diff_cfg.eta,
                diff_cfg.select,
                priv_classifier=id_wrap,
                priv_y=y_priv,
                emb=emb,
                w1=diff_cfg.w1,
                w2=diff_cfg.w2,
                cemb=cemb,
            )
            recon_imgs[offset:offset+B] = generated.detach().cpu().numpy()
            offset += B

    recon = recon_imgs[:, 0, :, :]  # (N_val, C, T_padded)
    X_std_anon = np.copy(X_std)
    # -> (N_val,T,C)
    X_val_anon_std = np.transpose(recon, (0, 2, 1))
    for row_idx, anon_sig in zip(val_row_idx, X_val_anon_std):
        X_std_anon[row_idx] = anon_sig
    return X_std_anon, val_row_idx


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    cfg = PPGConfig()
    diff_cfg = PPGDiffusionConfig()
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(diff_cfg.model_dir, exist_ok=True)

    X_raw, meta = load_ppg_data(cfg)
    orig_len = X_raw.shape[1]
    print(f"[Data] PPG shape: {X_raw.shape}")
    pad_len = (4 - (X_raw.shape[1] % 4)) % 4  # ensure length divisible by 4 for Unet downsampling
    if pad_len > 0:
        X_raw = np.pad(X_raw, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
        print(f"[Data] Padded signals by {pad_len} to length {X_raw.shape[1]}")

    # Standardization: fit on train only
    train_idx = meta[meta["id_fold"] == 0]["row_idx"].values
    scaler = StandardScaler()
    train_flat = X_raw[train_idx].reshape(len(train_idx), -1)
    scaler.fit(train_flat)
    X_std = scaler.transform(X_raw.reshape(len(X_raw), -1)).reshape(X_raw.shape)

    # Identity classifier for negative guidance
    n_subjects = meta["subject_id"].nunique()
    id_model = PPGIdentityNet(n_subjects=n_subjects, in_channels=X_std.shape[2])
    train_ds_id = PPGIdentityDataset(X_std, meta, split="train")
    val_ds_id = PPGIdentityDataset(X_std, meta, split="val")
    train_loader_id = DataLoader(train_ds_id, batch_size=cfg.identity_batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
    val_loader_id = DataLoader(val_ds_id, batch_size=cfg.identity_batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    if os.path.isfile(cfg.identity_ckpt):
        print(f"[PPG Identity] Loading checkpoint {cfg.identity_ckpt}")
        id_model.load_state_dict(torch.load(cfg.identity_ckpt, map_location=cfg.device))
    else:
        print("[PPG Identity] Training identity model...")
        train_identity(id_model, train_loader_id, val_loader_id, cfg)
    id_metrics_base = eval_identity(id_model, val_loader_id, cfg)
    print(f"[PPG Identity] Baseline acc={id_metrics_base['acc']:.4f}")
    id_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pulse-rate surrogate
    print("[PulseRate] Estimating BPM labels...")
    bpm_labels = np.array([estimate_bpm(sig, cfg.sampling_frequency) for sig in X_raw], dtype=np.float32)
    train_ds_hr = PulseRateDataset(X_std, bpm_labels, meta, split="train")
    val_ds_hr = PulseRateDataset(X_std, bpm_labels, meta, split="val")
    train_loader_hr = DataLoader(train_ds_hr, batch_size=diff_cfg.hr_batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
    val_loader_hr = DataLoader(val_ds_hr, batch_size=diff_cfg.hr_batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    hr_model = PulseRateRegressor(cdim=diff_cfg.cdim, in_channels=X_std.shape[2])
    hr_metrics = train_pulse_regressor(hr_model, train_loader_hr, val_loader_hr, diff_cfg)

    # Diffusion
    X_std_anon, val_row_idx = train_and_apply_privdiffuser_ppg(X_std, meta, diff_cfg, hr_model, id_model=id_model)

    # Inverse-standardize anonymized signals for downstream physiology metrics
    X_anon_raw = scaler.inverse_transform(X_std_anon.reshape(len(X_std_anon), -1)).reshape(X_std_anon.shape)
    # Baseline/anon slices for validation split
    X_raw_val = X_raw[val_row_idx, :orig_len]
    X_anon_val = X_anon_raw[val_row_idx, :orig_len]

    # Identity on anonymized
    id_model.to(cfg.device)
    val_ds_id_anon = PPGIdentityDataset(X_std_anon, meta, split="val")
    val_loader_id_anon = DataLoader(val_ds_id_anon, batch_size=cfg.identity_batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)
    id_metrics_anon = eval_identity(id_model, val_loader_id_anon, cfg)

    # Physiology metrics
    hrv_base = compute_hrv_metrics(X_raw_val, cfg.sampling_frequency)
    hrv_anon = compute_hrv_metrics(X_anon_val, cfg.sampling_frequency)
    resp_base = estimate_resp_rate(X_raw_val, cfg.sampling_frequency)
    resp_anon = estimate_resp_rate(X_anon_val, cfg.sampling_frequency)

    results = {
        "identity_baseline": id_metrics_base,
        "identity_privdiffuser": id_metrics_anon,
        "pulse_rate": hr_metrics,
        "hrv_baseline": hrv_base,
        "hrv_privdiffuser": hrv_anon,
        "resp_baseline": resp_base,
        "resp_privdiffuser": resp_anon,
        "val_indices": val_row_idx.tolist(),
    }
    out_json = os.path.join(cfg.results_dir, "ppg_privdiffuser_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {out_json}")

    # save anonymized signals for downstream identity attack
    np.savez(cfg.anon_npz, X_anon=X_std_anon, meta_row_idx=meta["row_idx"].values)
    print(f"[Export] Saved anonymized signals to {cfg.anon_npz}")

    if cfg.run_seeingred_identity:
        if not os.path.isfile(cfg.anon_npz):
            print(f"[Seeingred] Skipping attack: missing {cfg.anon_npz}")
        else:
            try:
                subprocess.run(
                    ["bash", "scripts/run_ppg_identity_attack_seeingred.sh"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"[Seeingred] Identity attack script failed: {e}")


if __name__ == "__main__":
    main()
