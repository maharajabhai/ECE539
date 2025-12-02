#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PTB-XL ECG + PrivDiffuser-style anonymization + overlays.

This mirrors the EMG PrivDiffuser pipeline, but uses PTB-XL ECG data and
replaces the fastai utility model with a lightweight heart-rate surrogate:
    - Utility task: regress heart rate (BPM) from each ECG window.
    - The surrogate's penultimate embedding conditions the diffusion model.
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ecg_priv_common import compute_diag_metrics_thresholded
from ecg_priv_models import (
    DiagnosisDataset,
    DiagnosisClassifier,
    ECGHeartRateDataset,
    HeartRateRegressor,
    compute_bpm_labels,
)
from ptbxl_loader import load_ptbxl

# -------------------------------------------------------------------------
# Resolve paths
# -------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TOP_LEVEL = os.path.dirname(PROJECT_ROOT)
PRIVDIFFUSER_DIR = os.path.join(PROJECT_ROOT, "PrivDiffuser")

for p in [THIS_DIR, PROJECT_ROOT, TOP_LEVEL, PRIVDIFFUSER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------------------------------------------------------------
# Import ECG Blinder pipeline pieces
# -------------------------------------------------------------------------
from ecg_blinder_pipeline_with_overlay import (
    Config as ECGConfig,
    build_identity_meta,
    build_utility_labels_from_superclass,
    IdentityDataset,
    IdentityNet,
    train_identity,
    eval_identity,
    rmse,
    psd_correlation,
    plot_fft,
    plot_overlay_ecg,
)

# -------------------------------------------------------------------------
# Import PrivDiffuser core modules (root of PrivDiffuser-main)
# -------------------------------------------------------------------------
from unet import Unet
from diffusion import GaussianDiffusion
from embedding import ConditionalEmbedding
from scheduler import GradualWarmupScheduler
from PrivDiffuser.utils import get_named_beta_schedule

# -------------------------------------------------------------------------
# Diffusion configuration
# -------------------------------------------------------------------------

@dataclass
class ECGDiffusionConfig:
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    model_dir: str = os.path.join(PROJECT_ROOT, "models", "ecg_privdiffuser")
    identity_ckpt: str = os.path.join(PROJECT_ROOT, "models", "ecg_privdiffuser", "identity_best.pt")
    hr_ckpt: str = os.path.join(PROJECT_ROOT, "models", "ecg_privdiffuser", "hr_surrogate.pt")

    # Heart-rate surrogate
    hr_epochs: int = 40
    hr_batch_size: int = 128
    hr_lr: float = 1e-3
    diag_epochs: int = 40
    diag_lr: float = 1e-3
    diag_pos_weight: float = 1.0
    # utility_target: "diagnosis" or "heart_rate"
    utility_target: str = "diagnosis"


# -------------------------------------------------------------------------
# Utility: Heart rate (BPM) from ECG
# -------------------------------------------------------------------------

def estimate_bpm_from_window(x_raw: np.ndarray, fs: int) -> float:
    """
    Estimate heart rate from a single ECG window using R-peak detection.
    x_raw: (T, C) or (C, T)
    """
    if x_raw.shape[0] < x_raw.shape[1]:
        x = x_raw.T  # -> (T, C)
    else:
        x = x_raw

    lead0 = x[:, 0]
    # Simple bandpass to emphasize QRS
    b, a = signal.butter(2, [5, 25], btype="bandpass", fs=fs)
    filt = signal.filtfilt(b, a, lead0)
    distance = int(0.3 * fs)  # min 200 bpm upper bound
    peaks, _ = signal.find_peaks(filt, distance=distance, prominence=np.std(filt) * 0.5)

    if len(peaks) < 2:
        return 0.0

    rr = np.diff(peaks) / fs  # seconds
    rr = rr[rr > 1e-3]
    if len(rr) == 0:
        return 0.0
    bpm = 60.0 / np.median(rr)
    return float(np.clip(bpm, 20.0, 220.0))


def compute_bpm_labels(X_raw: np.ndarray, fs: int) -> np.ndarray:
    return np.array([estimate_bpm_from_window(win, fs) for win in X_raw], dtype=np.float32)


# -------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------

class ECGPrivDiffuserDataset(Dataset):
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
        X_ct = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
        self.X = X_ct[:, None, :, :]      # (n, 1, C, T)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_priv = torch.tensor(self.meta["patient_idx"].values[idx], dtype=torch.long)
        return x, y_priv


class ECGHeartRateDataset(Dataset):
    def __init__(self, X_std: np.ndarray, bpm: np.ndarray, meta: pd.DataFrame, split: str):
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


class DiagnosisDataset(Dataset):
    """Multi-label diagnosis targets from PTB-XL diagnostic_superclass."""
    def __init__(self, X_std: np.ndarray, y_diag: np.ndarray, meta: pd.DataFrame, split: str):
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


# -------------------------------------------------------------------------
# Heart-rate surrogate model
# -------------------------------------------------------------------------

class HeartRateRegressor(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128):
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
        self.head = nn.Linear(128, 1)
        self.embed_proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        h = self.features(x).squeeze(-1)   # (B, 128)
        bpm_pred = self.head(h).squeeze(-1)
        z = self.embed_proj(h)
        z = torch.nn.functional.normalize(z, dim=-1)
        return bpm_pred, z


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


def train_heart_rate_regressor(model: HeartRateRegressor,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               diff_cfg: ECGDiffusionConfig) -> Dict[str, float]:
    device = diff_cfg.device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=diff_cfg.hr_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=diff_cfg.hr_epochs, eta_min=diff_cfg.hr_lr * 0.1)
    best_val = float("inf")
    mae_loss = nn.L1Loss()

    for ep in range(diff_cfg.hr_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = mae_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred, _ = model(xb)
                loss = mae_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[HeartRate] Epoch {ep+1}/{diff_cfg.hr_epochs} "
              f"train_mae={train_loss:.3f} val_mae={val_loss:.3f}")
        if val_loss < best_val:
            best_val = val_loss
        scheduler.step()

    return {"train_mae": float(train_loss), "val_mae": float(best_val)}


# -------------------------------------------------------------------------
# Surrogate wrapper for diffusion
# -------------------------------------------------------------------------

class ECGSurrogate(nn.Module):
    def __init__(self, base_model: nn.Module, z_dim: int):
        super().__init__()
        self.base = base_model
        # base_model produces normalized embeddings of size z_dim already
        self.z_proj = nn.Identity()

    def forward(self, x_img):
        B, _, C, T = x_img.shape
        x = x_img.view(B, C, T)
        logits, z_base = self.base(x)
        z = self.z_proj(z_base)
        return logits, z


class IdentityWrapper(nn.Module):
    """Squeeze dummy dim and return logits for cond_fn compatibility."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x, emb=None):
        if x.dim() == 4:
            x = x.squeeze(1)  # (B,C,T)
        out = self.base(x)
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out


def build_frozen_surrogate(base_model: HeartRateRegressor,
                           z_dim: int,
                           device: str) -> ECGSurrogate:
    surrogate = ECGSurrogate(base_model, z_dim=z_dim).to(device)
    for p in surrogate.base.parameters():
        p.requires_grad = False
    surrogate.eval()
    return surrogate


def train_diagnosis_classifier(
    model: DiagnosisClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ECGDiffusionConfig,
) -> Dict:
    model.to(cfg.device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.diag_lr)
    best_val = float("inf")

    for ep in range(cfg.diag_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits, _ = model(xb)
                loss = bce(logits, yb)
                val_loss += loss.item() * xb.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())
        val_loss /= len(val_loader.dataset)

    try:
        from sklearn.metrics import roc_auc_score
        logits_cat = torch.cat(all_logits, dim=0).numpy()
        labels_cat = torch.cat(all_labels, dim=0).numpy()
        auroc_macro = roc_auc_score(labels_cat, logits_cat, average="macro")
    except Exception:
        auroc_macro = None

        if val_loss < best_val:
            best_val = val_loss
        # optional thresholded metrics for logging
        diag_metrics = {}
        try:
            probs = torch.sigmoid(torch.cat(all_logits, dim=0)).cpu().numpy()
            labels = torch.cat(all_labels, dim=0).cpu().numpy()
            diag_metrics = compute_diag_metrics_thresholded(probs, labels, thresh=0.5)
        except Exception:
            diag_metrics = {}

        msg = f"[Diagnosis] Epoch {ep+1}/{cfg.diag_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        if diag_metrics:
            msg += f" sample_acc={diag_metrics['sample_acc']:.4f} macro_f1={diag_metrics['macro_f1']:.4f}"
        print(msg)

    return {"train_bce": train_loss, "val_bce": val_loss, "val_auroc_macro": auroc_macro}


def eval_diagnosis(model: DiagnosisClassifier, loader: DataLoader, device: str) -> Dict:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    losses = []
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = bce(logits, yb)
            losses.append(loss.item() * xb.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
    total = sum(len(xb) for xb, _ in loader)
    val_loss = sum(losses) / max(1, total)
    try:
        from sklearn.metrics import roc_auc_score
        logits_cat = torch.cat(all_logits, dim=0).numpy()
        labels_cat = torch.cat(all_labels, dim=0).numpy()
        auroc_macro = roc_auc_score(labels_cat, logits_cat, average="macro")
    except Exception:
        auroc_macro = None
    return {"val_bce": val_loss, "val_auroc_macro": auroc_macro}


# -------------------------------------------------------------------------
# Train-or-load PrivDiffuser and anonymize validation set
# -------------------------------------------------------------------------

def train_and_apply_privdiffuser_ecg(X_std: np.ndarray,
                                     meta: pd.DataFrame,
                                     diff_cfg: ECGDiffusionConfig,
                                     utility_surrogate: nn.Module,
                                     id_model: IdentityNet,
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(diff_cfg.model_dir, exist_ok=True)
    device = diff_cfg.device

    ckpt_unet = os.path.join(diff_cfg.model_dir, "unet.pt")
    ckpt_cemb = os.path.join(diff_cfg.model_dir, "cond_emb.pt")

    train_ds = ECGPrivDiffuserDataset(X_std, meta, split="train")
    val_ds = ECGPrivDiffuserDataset(X_std, meta, split="val")

    train_loader_diff = DataLoader(
        train_ds,
        batch_size=diff_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader_diff = DataLoader(
        val_ds,
        batch_size=diff_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
    )

    surrogate = build_frozen_surrogate(utility_surrogate, z_dim=diff_cfg.cdim, device=device)

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

    # Use ConditionalEmbedding as an MLP over continuous embeddings
    cemblayer = ConditionalEmbedding(diff_cfg.cdim, diff_cfg.cdim, diff_cfg.cdim).to(device)

    if os.path.isfile(ckpt_unet) and os.path.isfile(ckpt_cemb):
        print("[Diffusion-ECG] Loading existing PrivDiffuser weights...")
        net_state = torch.load(ckpt_unet, map_location=device)
        cemb_state = torch.load(ckpt_cemb, map_location=device)
        net.load_state_dict(net_state)
        cemblayer.load_state_dict(cemb_state)
    else:
        print(f"[Diffusion-ECG] Training for {diff_cfg.epoch} epochs...")
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
            epoch_loss = 0.0
            num_batches = 0
            for batch in train_loader_diff:
                x_img, _ = batch
                x_img = x_img.to(device)
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
                epoch_loss += loss.item()
                num_batches += 1

            warmup_scheduler.step()
            avg_loss = epoch_loss / max(1, num_batches)
            lr_cur = optimizer.state_dict()["param_groups"][0]["lr"]
            print(f"[Diffusion-ECG] Epoch {ep+1}/{diff_cfg.epoch} "
                  f"loss={avg_loss:.4f} lr={lr_cur:.6f}")

        torch.save(net.state_dict(), ckpt_unet)
        torch.save(cemblayer.state_dict(), ckpt_cemb)
        print(f"[Diffusion-ECG] Saved U-Net to {ckpt_unet}")
        print(f"[Diffusion-ECG] Saved ConditionalEmbedding to {ckpt_cemb}")

    print("[Diffusion-ECG] Sampling anonymized ECG for validation split...")
    diffusion.model.eval()
    cemblayer.eval()
    surrogate.eval()

    val_row_idx = val_ds.meta["row_idx"].values
    recon_imgs = np.empty((len(val_ds), 1, val_ds.X.shape[2], val_ds.X.shape[3]), dtype=np.float32)

    id_model = id_model.to(device)
    id_wrapper = IdentityWrapper(id_model)
    id_wrapper.eval()

    with torch.no_grad():
        offset = 0
        for batch in val_loader_diff:
            x_img, y_priv = batch
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
                priv_classifier=id_wrapper,
                priv_y=y_priv,
                emb=emb,
                w1=diff_cfg.w1,
                w2=diff_cfg.w1,  # reuse w1 for negative guidance strength
                cemb=cemb,
            )
            recon_np = generated.detach().cpu().numpy()
            recon_imgs[offset:offset + B] = recon_np
            offset += B

    recon_ct = np.squeeze(recon_imgs, axis=1)              # (N_val, C, T)
    X_val_anon_std = np.transpose(recon_ct, (0, 2, 1))     # (N_val, T, C)

    X_std_anon = np.copy(X_std)
    for row_idx, anon_win in zip(val_row_idx, X_val_anon_std):
        X_std_anon[row_idx] = anon_win

    return X_std_anon, val_row_idx


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=== PTB-XL ECG + PrivDiffuser anonymization ===")
    cfg = ECGConfig()
    diff_cfg = ECGDiffusionConfig()
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(diff_cfg.model_dir, exist_ok=True)

    # 1) Load PTB-XL
    X_raw, Y = load_ptbxl(PROJECT_ROOT, cfg.datafolder, cfg.sampling_frequency)
    print(f"[Data] ECG shape: {X_raw.shape}")

    # 2) Identity metadata
    meta, id_info = build_identity_meta(Y, cfg)

    # 3) Standardization (fit on train split only to avoid leakage)
    N, T, C = X_raw.shape
    scaler = StandardScaler()
    train_idx = meta[meta["id_fold"] == 0]["row_idx"].values
    X_train_flat = X_raw[train_idx].reshape(len(train_idx), -1)
    scaler.fit(X_train_flat)
    X_flat = X_raw.reshape(N, -1)
    X_std_flat = scaler.transform(X_flat)
    X_std = X_std_flat.reshape(N, T, C)

    # 4) Identity model
    train_ds_id = IdentityDataset(X_std, meta, split="train")
    val_ds_id = IdentityDataset(X_std, meta, split="val")
    train_loader_id = DataLoader(
        train_ds_id,
        batch_size=cfg.identity_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader_id = DataLoader(
        val_ds_id,
        batch_size=cfg.identity_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    id_model = IdentityNet(
        in_channels=C,
        n_patients=id_info["n_patients"],
        n_age_bins=id_info["n_age_bins"],
        n_height_bins=id_info["n_height_bins"],
        n_weight_bins=id_info["n_weight_bins"],
    )

    if os.path.isfile(diff_cfg.identity_ckpt):
        print(f"[Identity] Loading checkpoint from {diff_cfg.identity_ckpt}")
        state = torch.load(diff_cfg.identity_ckpt, map_location=cfg.device)
        id_model.load_state_dict(state)
    elif os.path.isfile(cfg.identity_ckpt):
        print(f"[Identity] Loading checkpoint from {cfg.identity_ckpt}")
        state = torch.load(cfg.identity_ckpt, map_location=cfg.device)
        id_model.load_state_dict(state)
    else:
        print("[Identity] No checkpoint found; training from scratch...")
        train_identity(id_model, train_loader_id, val_loader_id, cfg)
        os.makedirs(os.path.dirname(diff_cfg.identity_ckpt), exist_ok=True)
        torch.save(id_model.state_dict(), diff_cfg.identity_ckpt)
        print(f"[Identity] Saved to {diff_cfg.identity_ckpt}")

    print("[Identity] Evaluating baseline on standardized ECG...")
    id_metrics_base = eval_identity(id_model, val_loader_id, cfg)

    # Free identity GPU memory during diffusion training
    id_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) Utility surrogate
    utility_kind = diff_cfg.utility_target.lower()
    diag_metrics = diag_metrics_raw = diag_metrics_anon = None
    diag_thresh_raw = diag_thresh_anon = None
    hr_metrics = None

    if utility_kind == "diagnosis":
        print("[Diagnosis] Building multi-label diagnosis targets...")
        y_diag, diag_classes = build_utility_labels_from_superclass(Y)
        train_ds_diag = DiagnosisDataset(X_std, y_diag, meta, split="train")
        val_ds_diag = DiagnosisDataset(X_std, y_diag, meta, split="val")
        train_loader_diag = DataLoader(
            train_ds_diag,
            batch_size=diff_cfg.hr_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader_diag = DataLoader(
            val_ds_diag,
            batch_size=diff_cfg.hr_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        diag_model = DiagnosisClassifier(in_channels=C, n_classes=y_diag.shape[1], embed_dim=diff_cfg.cdim)
        diag_ckpt = os.path.join(diff_cfg.model_dir, "diag_surrogate.pt")
        if os.path.isfile(diag_ckpt):
            print(f"[Diagnosis] Loading surrogate from {diag_ckpt}")
            state = torch.load(diag_ckpt, map_location=diff_cfg.device)
            diag_model.load_state_dict(state)
            diag_metrics = {"train_bce": None, "val_bce": None, "val_auroc_macro": None}
        else:
            print("[Diagnosis] Training diagnosis surrogate (multi-label)...")
            diag_metrics = train_diagnosis_classifier(diag_model, train_loader_diag, val_loader_diag, diff_cfg)
            torch.save(diag_model.state_dict(), diag_ckpt)
            print(f"[Diagnosis] Saved surrogate to {diag_ckpt}")
        diag_model.to(diff_cfg.device)
        diag_metrics_raw = eval_diagnosis(diag_model, val_loader_diag, diff_cfg.device)
        utility_model = diag_model
    else:
        print("[HeartRate] Computing BPM labels...")
        bpm_labels = compute_bpm_labels(X_raw, fs=cfg.sampling_frequency)
        train_ds_hr = ECGHeartRateDataset(X_std, bpm_labels, meta, split="train")
        val_ds_hr = ECGHeartRateDataset(X_std, bpm_labels, meta, split="val")
        train_loader_hr = DataLoader(
            train_ds_hr,
            batch_size=diff_cfg.hr_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader_hr = DataLoader(
            val_ds_hr,
            batch_size=diff_cfg.hr_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        hr_model = HeartRateRegressor(in_channels=C, embed_dim=diff_cfg.cdim)
        if os.path.isfile(diff_cfg.hr_ckpt):
            print(f"[HeartRate] Loading surrogate from {diff_cfg.hr_ckpt}")
            state = torch.load(diff_cfg.hr_ckpt, map_location=diff_cfg.device)
            hr_model.load_state_dict(state)
            hr_metrics = {"train_mae": None, "val_mae": None}
        else:
            print("[HeartRate] Training surrogate regressor (BPM)...")
            hr_metrics = train_heart_rate_regressor(hr_model, train_loader_hr, val_loader_hr, diff_cfg)
            torch.save(hr_model.state_dict(), diff_cfg.hr_ckpt)
            print(f"[HeartRate] Saved surrogate to {diff_cfg.hr_ckpt}")
        utility_model = hr_model

    # 6) PrivDiffuser training + anonymization
    X_std_anon, val_row_idx = train_and_apply_privdiffuser_ecg(
        X_std=X_std,
        meta=meta,
        diff_cfg=diff_cfg,
        utility_surrogate=utility_model,
        id_model=id_model,
    )

    # 7) Identity evaluation on anonymized
    id_model.to(cfg.device)
    val_ds_id_anon = IdentityDataset(X_std_anon, meta, split="val")
    val_loader_id_anon = DataLoader(
        val_ds_id_anon,
        batch_size=cfg.identity_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print("[Identity] Evaluating on PrivDiffuser-anonymized ECG...")
    id_metrics_anon = eval_identity(id_model, val_loader_id_anon, cfg)

    # Diagnosis evaluation on anonymized
    if utility_kind == "diagnosis":
        val_ds_diag_anon = DiagnosisDataset(X_std_anon, y_diag, meta, split="val")
        val_loader_diag_anon = DataLoader(
            val_ds_diag_anon,
            batch_size=diff_cfg.hr_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        diag_metrics_anon = eval_diagnosis(utility_model, val_loader_diag_anon, diff_cfg.device)

        # Thresholded diagnosis metrics (aligned with Blinder)
        def collect_probs(loader):
            all_probs, all_labels = [], []
            utility_model.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(diff_cfg.device)
                    yb = yb.to(diff_cfg.device)
                    logits, _ = utility_model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(yb.cpu().numpy())
            return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

        probs_raw, labels_raw = collect_probs(val_loader_diag)
        probs_anon, labels_anon = collect_probs(val_loader_diag_anon)
        diag_thresh_raw = compute_diag_metrics_thresholded(probs_raw, labels_raw, thresh=0.5)
        diag_thresh_anon = compute_diag_metrics_thresholded(probs_anon, labels_anon, thresh=0.5)

    # 8) Fidelity metrics (raw vs anonymized on val subset)
    print("[Fidelity] Computing RMSE and PSD correlation (raw vs anonymized)...")
    X_val_raw = X_raw[val_row_idx]
    X_val_std_anon = X_std_anon[val_row_idx]
    N_val, T_val, C_val = X_val_std_anon.shape
    X_val_anon_flat = X_val_std_anon.reshape(N_val, -1)
    X_val_anon_raw = scaler.inverse_transform(X_val_anon_flat).reshape(N_val, T_val, C_val)

    rmses_anon, psd_corrs_anon = [], []
    for i in range(N_val):
        rmses_anon.append(rmse(X_val_raw[i], X_val_anon_raw[i]))
        psd_corrs_anon.append(
            psd_correlation(X_val_raw[i], X_val_anon_raw[i], fs=cfg.sampling_frequency)
        )
    fidelity_anon = {
        "rmse_mean": float(np.mean(rmses_anon)),
        "rmse_std": float(np.std(rmses_anon)),
        "psd_corr_mean": float(np.mean(psd_corrs_anon)),
        "psd_corr_std": float(np.std(psd_corrs_anon)),
    }

    # 9) Plots
    print("[Plots] Creating FFT + overlay plots...")
    example_idx = 0
    example_raw = X_val_raw[example_idx]
    example_anon = X_val_anon_raw[example_idx]

    fft_raw_path = os.path.join(cfg.results_dir, "ecg_fft_val_raw.png")
    fft_anon_path = os.path.join(cfg.results_dir, "ecg_fft_val_privdiffuser.png")
    plot_fft(example_raw, cfg.sampling_frequency, "ECG FFT (original val)", fft_raw_path)
    plot_fft(example_anon, cfg.sampling_frequency, "ECG FFT (PrivDiffuser val)", fft_anon_path)

    overlay_dir = os.path.join(cfg.results_dir, "ecg_overlay_privdiffuser")
    os.makedirs(overlay_dir, exist_ok=True)
    n_examples_plot = min(5, N_val)
    leads_to_plot = [0, 1]
    for i in range(n_examples_plot):
        for lead in leads_to_plot:
            out_path = os.path.join(overlay_dir, f"ecg_overlay_example_{i}_lead{lead}.png")
            plot_overlay_ecg(
                X_val_raw[i],
                X_val_anon_raw[i],
                fs=cfg.sampling_frequency,
                title=f"PrivDiffuser ECG overlay (example {i}, lead {lead})",
                filepath=out_path,
                lead=lead,
                max_seconds=5.0,
            )

    # 10) Save metrics
    results = {
        "utility_target": utility_kind,
        "diagnosis_surrogate_train": diag_metrics,
        "diagnosis_baseline": diag_metrics_raw,
        "diagnosis_privdiffuser": diag_metrics_anon,
        "diagnosis_baseline_threshold": diag_thresh_raw,
        "diagnosis_privdiffuser_threshold": diag_thresh_anon,
        "heart_rate_surrogate": hr_metrics,
        "identity_baseline": id_metrics_base,
        "identity_privdiffuser": id_metrics_anon,
        "fidelity_privdiffuser": fidelity_anon,
        "plots": {
            "fft_raw": fft_raw_path,
            "fft_anon": fft_anon_path,
            "overlay_dir": overlay_dir,
        },
    }
    out_json = os.path.join(cfg.results_dir, "ecg_privdiffuser_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Done (ECG + PrivDiffuser) ===")
    print(f"[Results] Saved to {out_json}")


if __name__ == "__main__":
    main()
