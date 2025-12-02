#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared ECG utility models and datasets (diagnosis, heart rate) without external dependencies.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from scipy import signal


class DiagnosisDataset(Dataset):
    """Multi-label diagnosis targets from PTB-XL diagnostic_superclass."""
    def __init__(self, X_std: np.ndarray, y_diag: np.ndarray, meta, split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")
        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]
        self.X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)  # (n,C,T)
        self.y = y_diag[idx].astype(np.float32)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class ECGHeartRateDataset(Dataset):
    def __init__(self, X_std: np.ndarray, bpm: np.ndarray, meta, split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]                # (n, T, C)
        self.X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)  # (n, C, T)
        self.bpm = bpm[idx].astype(np.float32)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.bpm[idx], dtype=torch.float32)
        return x, y


class DiagnosisClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, embed_dim: int = 128):
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
        self.proj = nn.Linear(128, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        h = self.features(x).squeeze(-1)
        z = torch.nn.functional.normalize(self.proj(h), dim=-1)
        logits = self.head(z)
        return logits, z


class HeartRateRegressor(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, 1)
        self.embed_proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        h = self.features(x).squeeze(-1)   # (B, 128)
        h = self.dropout(h)
        bpm_pred = self.head(h).squeeze(-1)
        z = self.embed_proj(h)
        z = torch.nn.functional.normalize(z, dim=-1)
        return bpm_pred, z


def estimate_bpm_from_window(x_raw: np.ndarray, fs: int) -> float:
    if x_raw.shape[0] < x_raw.shape[1]:
        x = x_raw.T
    else:
        x = x_raw
    lead0 = x[:, 0]
    b, a = signal.butter(2, [5, 25], btype="bandpass", fs=fs)
    filt = signal.filtfilt(b, a, lead0)
    distance = int(0.3 * fs)
    peaks, _ = signal.find_peaks(filt, distance=distance, prominence=np.std(filt) * 0.5)
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / fs
    rr = rr[rr > 1e-3]
    if len(rr) == 0:
        return 0.0
    bpm = 60.0 / np.median(rr)
    return float(np.clip(bpm, 20.0, 220.0))


def compute_bpm_labels(X_raw: np.ndarray, fs: int) -> np.ndarray:
    # prefer wfdb gqrs if available for better peak detection
    try:
        import wfdb
        labels = []
        for win in X_raw:
            sig = win if win.shape[0] > win.shape[1] else win.T
            lead0 = sig[:, 0]
            qrs_locs = wfdb.processing.gqrs_detect(sig=lead0, fs=fs)
            # optional refinement
            try:
                qrs_locs = wfdb.processing.correct_peaks(lead0, qrs_locs, search_radius=int(0.1 * fs), fs=fs, tol=0.1)
            except Exception:
                pass
            if len(qrs_locs) < 2:
                labels.append(0.0)
                continue
            rr = np.diff(qrs_locs) / fs
            rr = rr[(rr > 0.25) & (rr < 2.0)]  # 30–240 bpm
            if len(rr) == 0:
                labels.append(0.0)
                continue
            bpm = 60.0 / np.median(rr)
            labels.append(float(np.clip(bpm, 30.0, 220.0)))
        return np.array(labels, dtype=np.float32)
    except Exception:
        return np.array([estimate_bpm_from_window(win, fs) for win in X_raw], dtype=np.float32)
