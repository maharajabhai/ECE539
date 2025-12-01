#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ninapro EMG + PrivDiffuser-style anonymization + overlays.

Pipeline:
1) Load Ninapro EMG windows + metadata (from your Blinder EMG pipeline).
2) Train / load:
   - Identity model (SubjectIdentityNet) for subject classification.
   - Utility model (GestureUtilityNet) for gesture classification.
3) Train / load a PrivDiffuser-style diffusion model:
   - Surrogate utility encoder = frozen GestureUtilityNet backbone.
   - U-Net + GaussianDiffusion + ConditionalEmbedding from PrivDiffuser repo.
   - Classifier-free guidance using the surrogate embedding.
4) Use diffusion to anonymize the validation set.
5) Re-evaluate identity + utility on anonymized EMG.
6) Compute fidelity metrics and generate overlays + FFTs.

Run:
    python emg_privdiffuser_with_overlay.py
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# Resolve paths
# -------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../ECE539/scripts
PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # .../ECE539
TOP_LEVEL = os.path.dirname(PROJECT_ROOT)                      # .../
PRIVDIFFUSER_DIR = os.path.join(PROJECT_ROOT, "PrivDiffuser")

for p in [THIS_DIR, PROJECT_ROOT, TOP_LEVEL, PRIVDIFFUSER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------------------------------------------------------------
# Import your EMG Blinder pipeline pieces (lives in scripts/)
# -------------------------------------------------------------------------
from emg_blinder_pipeline_with_overlay import (
    Config as EMGConfig,
    load_ninapro_windows,
    build_identity_meta_emg,
    IdentityDatasetEMG,
    UtilityDatasetEMG,
    SubjectIdentityNet,
    GestureUtilityNet,
    train_identity,
    eval_identity,
    train_utility,
    eval_utility,
    rmse,
    psd_correlation,
    plot_fft,
    plot_overlay_emg,
)

# -------------------------------------------------------------------------
# Import PrivDiffuser core modules (root of PrivDiffuser-main)
# -------------------------------------------------------------------------
from unet import Unet
from diffusion import GaussianDiffusion
from embedding import ConditionalEmbedding
from scheduler import GradualWarmupScheduler
from utils import get_named_beta_schedule


# -------------------------------------------------------------------------
# Diffusion configuration
# -------------------------------------------------------------------------

@dataclass
class EMGDiffusionConfig:
    inch: int = 1               # Unet input channels (we use (B,1,C,T))
    modch: int = 64             # base channels for Unet (smaller than original)
    outch: int = 1              # Unet output channels
    T: int = 1000               # diffusion steps
    chmul: Tuple[int, ...] = (1, 2)  # channel multipliers
    numres: int = 2             # resblocks per stage
    cdim: int = 60              # conditional embedding dimension
    useconv: bool = True
    droprate: float = 0.1
    dtype: torch.dtype = torch.float32
    lr: float = 2e-4
    epoch: int = 40             # diffusion training epochs
    multiplier: int = 2         # warmup scheduler multiplier
    threshold: float = 0.3      # classifier-free guidance drop prob
    num_steps: int = 50         # DDIM steps
    eta: float = 0.0            # DDIM noise eta
    select: str = "linear"      # DDIM schedule strategy
    w1: float = 2.0             # classifier-free guidance strength
    w2: float = 1.0             # negative identity guidance strength
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64         # smaller to reduce GPU footprint
    # store under project_root/models/emg_privdiffuser
    model_dir: str = os.path.join(PROJECT_ROOT, "models", "emg_privdiffuser")


# -------------------------------------------------------------------------
# Dataset for PrivDiffuser: EMG -> image-like (B,1,C,T)
# -------------------------------------------------------------------------

class EMGPrivDiffuserDataset(Dataset):
    """
    Wrap standardized EMG windows into image-like tensors for PrivDiffuser.

    X_std: (N, T, C)
    meta: DataFrame with columns including:
        - row_idx
        - subject_idx
        - stimulus (gesture label)

    We split using meta["id_fold"]:
        id_fold == 0 -> train
        id_fold == 1 -> val
    """
    def __init__(self, X_std: np.ndarray, meta, split: str):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]                # (n, T, C)

        # (n, C, T)
        X_ct = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
        # (n, 1, C, T) for Conv2d-based Unet
        self.X = X_ct[:, None, :, :]

        # public label = stimulus (gesture)
        self.y_pub = self.meta["stimulus"].values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_pub = torch.tensor(self.y_pub[idx], dtype=torch.long)
        y_priv = torch.tensor(self.meta["subject_idx"].values[idx], dtype=torch.long)
        return x, y_pub, y_priv


class IdentityWrapper(nn.Module):
    """Allow PrivDiffuser cond_fn to call identity classifier with optional emb argument."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x, emb=None):
        # cond_fn passes (B,1,C,T); identity expects (B,C,T)
        if x.dim() == 4:
            x = x.squeeze(1)
        out = self.base(x)
        # base returns dict with 'logits'; cond_fn expects logits tensor
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out


# -------------------------------------------------------------------------
# Surrogate Utility Encoder: reuse GestureUtilityNet as backbone
# -------------------------------------------------------------------------

class EMGSurrogate(nn.Module):
    """
    Surrogate utility classifier for PrivDiffuser.

    - Backbone: GestureUtilityNet from your EMG pipeline (pretrained).
    - Input: (B, 1, C, T)
    - Output:
        logits: (B, n_gestures)
        z:      (B, cdim) normalized embedding used for conditioning.
    """
    def __init__(self, base_model: GestureUtilityNet, z_dim: int):
        super().__init__()
        self.base = base_model                   # uses Conv1d over (C,T)
        self.z_proj = nn.Linear(128, z_dim)      # base feat dim is 128

    def forward(self, x_img):
        B, _, C, T = x_img.shape
        x = x_img.view(B, C, T)                  # -> (B,C,T) for GestureUtilityNet
        out = self.base(x)
        h = out["feat"]                          # (B,128)
        logits = out["logits"]
        z = self.z_proj(h)                       # (B,z_dim)
        z = torch.nn.functional.normalize(z, dim=-1)
        return logits, z


def build_frozen_surrogate_from_utility(
    utility_model: GestureUtilityNet,
    z_dim: int,
    device: str,
) -> EMGSurrogate:
    """
    Wrap the *baseline* utility model into an EMGSurrogate and freeze its weights.

    - utility_model: already trained / loaded GestureUtilityNet.
    - We DO NOT update its weights or re-save its checkpoint.
    """
    surrogate = EMGSurrogate(utility_model, z_dim=z_dim)
    surrogate.to(device)

    # freeze backbone so its checkpoint never changes
    for p in surrogate.base.parameters():
        p.requires_grad = False

    surrogate.eval()
    return surrogate


# -------------------------------------------------------------------------
# Train-or-load PrivDiffuser and anonymize validation set
# -------------------------------------------------------------------------

def train_and_apply_privdiffuser_emg(
    X_std: np.ndarray,
    meta,
    diff_cfg: EMGDiffusionConfig,
    utility_model: GestureUtilityNet,
    identity_model: SubjectIdentityNet,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train OR load PrivDiffuser-style diffusion model on EMG and anonymize val windows.

    - Uses a frozen baseline utility model as surrogate.
    - Does NOT update the baseline utility checkpoint.

    Returns:
        X_std_anon: full (N, T, C) standardized EMG with val windows anonymized.
        val_row_idx: indices of validation rows (for metrics / raw mapping).
    """
    os.makedirs(diff_cfg.model_dir, exist_ok=True)
    device = diff_cfg.device

    ckpt_unet = os.path.join(diff_cfg.model_dir, "unet.pt")
    ckpt_cemb = os.path.join(diff_cfg.model_dir, "cond_emb.pt")

    # ----------------------- Datasets & loaders ------------------------
    train_ds = EMGPrivDiffuserDataset(X_std, meta, split="train")
    val_ds   = EMGPrivDiffuserDataset(X_std, meta, split="val")

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

    # ----------------------- Surrogate utility (frozen) ----------------
    surrogate = build_frozen_surrogate_from_utility(
        utility_model=utility_model,
        z_dim=diff_cfg.cdim,
        device=device,
    )
    identity_model = identity_model.to(device)
    identity_model.eval()

    # ----------------------- U-Net + Diffusion ------------------------
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

    # Conditional embedding layer
    cemblayer = ConditionalEmbedding(10, diff_cfg.cdim, diff_cfg.cdim).to(device)

    # ----------------------- Load or train ----------------------------
    if os.path.isfile(ckpt_unet) and os.path.isfile(ckpt_cemb):
        print("[Diffusion-EMG] Loading existing PrivDiffuser weights...")
        net_state = torch.load(ckpt_unet, map_location=device)
        cemb_state = torch.load(ckpt_cemb, map_location=device)
        net.load_state_dict(net_state)
        cemblayer.load_state_dict(cemb_state)
    else:
        print(f"[Diffusion-EMG] No checkpoints found. Training for {diff_cfg.epoch} epochs...")

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

        # ----------------------- Train diffusion -------------------------
        for ep in range(diff_cfg.epoch):
            diffusion.model.train()
            cemblayer.train()
            surrogate.eval()  # keep it frozen

            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader_diff:
                if len(batch) == 3:
                    x_img, _, _ = batch
                else:
                    x_img, _ = batch
                x_img = x_img.to(device)
                b = x_img.size(0)
                optimizer.zero_grad()

                # Surrogate embedding (NO grad, NO weight updates)
                with torch.no_grad():
                    _, emb = surrogate(x_img)           # (B,cdim)

                cemb = cemblayer(emb)                   # (B,cdim)

                # Classifier-free guidance: randomly drop some cemb rows
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
            print(f"[Diffusion-EMG] Epoch {ep+1}/{diff_cfg.epoch} loss={avg_loss:.4f}  lr={lr_cur:.6f}")

        # Save trained weights so we don't have to retrain next time
        torch.save(net.state_dict(), ckpt_unet)
        torch.save(cemblayer.state_dict(), ckpt_cemb)
        print(f"[Diffusion-EMG] Saved U-Net to {ckpt_unet}")
        print(f"[Diffusion-EMG] Saved ConditionalEmbedding to {ckpt_cemb}")

    # ----------------------- Sampling (anonymization) ----------------
    print("[Diffusion-EMG] Sampling anonymized EMG for validation split...")
    diffusion.model.eval()
    cemblayer.eval()
    surrogate.eval()

    val_row_idx = val_ds.meta["row_idx"].values  # index mapping back to full X_std
    recon_imgs = np.empty((len(val_ds), 1, val_ds.X.shape[2], val_ds.X.shape[3]), dtype=np.float32)

    with torch.no_grad():
        offset = 0
        for batch in val_loader_diff:
            # Dataset always returns (x_img, y_pub, y_priv); keep priv labels for guidance
            x_img, _, y_priv = batch
            x_img = x_img.to(device)
            y_priv = y_priv.to(device)
            B = x_img.size(0)

            _, emb = surrogate(x_img)
            cemb = cemblayer(emb)

            genshape = x_img.shape  # (B,1,C,T)

            priv_classifier = IdentityWrapper(identity_model.to(device))
            priv_classifier.eval()
            generated = diffusion.ddim_sample(
                genshape,
                diff_cfg.num_steps,
                diff_cfg.eta,
                diff_cfg.select,
                priv_classifier=priv_classifier,
                priv_y=y_priv,
                emb=emb,
                w1=diff_cfg.w1,
                w2=diff_cfg.w2,
                cemb=cemb,
            )

            recon_np = generated.detach().cpu().numpy()
            recon_imgs[offset:offset + B] = recon_np
            offset += B

    # (N_val,1,C,T)
    # -> (N_val,C,T) -> (N_val,T,C)
    recon_ct = np.squeeze(recon_imgs, axis=1)              # (N_val,C,T)
    X_val_anon_std = np.transpose(recon_ct, (0, 2, 1))     # (N_val,T,C)

    # ----------------------- Merge back into full array --------------
    X_std_anon = np.copy(X_std)
    for row_idx, anon_win in zip(val_row_idx, X_val_anon_std):
        X_std_anon[row_idx] = anon_win

    return X_std_anon, val_row_idx


# -------------------------------------------------------------------------
# Top-level main for EMG
# -------------------------------------------------------------------------

def main():
    print("=== Ninapro EMG + PrivDiffuser anonymization ===")

    cfg = EMGConfig()
    os.makedirs(cfg.results_dir, exist_ok=True)

    # 1) Load Ninapro EMG windows
    X, meta_raw = load_ninapro_windows(cfg)   # X: (N,T,C)
    print(f"[Data] Windows shape: {X.shape}")

    # 2) Identity meta (subject_idx + id_fold)
    meta, id_info = build_identity_meta_emg(meta_raw, cfg)

    # 3) Standardization
    N, T, C = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(N, -1)
    X_std_flat = scaler.fit_transform(X_flat)
    X_std = X_std_flat.reshape(N, T, C)

    # 4) Identity model: train / load + baseline on original std
    train_ds_id = IdentityDatasetEMG(X_std, meta, split="train")
    val_ds_id = IdentityDatasetEMG(X_std, meta, split="val")
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

    n_subjects = id_info["n_subjects"]
    id_model = SubjectIdentityNet(in_channels=C, n_subjects=n_subjects)

    if os.path.isfile(cfg.identity_ckpt):
        print(f"[Identity] Loading checkpoint from {cfg.identity_ckpt}")
        state = torch.load(cfg.identity_ckpt, map_location=cfg.device)
        id_model.load_state_dict(state)
    else:
        print("[Identity] Training identity model from scratch...")
        train_identity(id_model, train_loader_id, val_loader_id, cfg)

    print("[Identity] Evaluating baseline on original standardized EMG...")
    id_metrics_base = eval_identity(id_model, val_loader_id, cfg)
    # Move identity model off GPU to free memory before diffusion training
    id_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) Utility (gesture) model: train / load + baseline
    train_ds_ut = UtilityDatasetEMG(X_std, meta, split="train")
    val_ds_ut = UtilityDatasetEMG(X_std, meta, split="val")
    train_loader_ut = DataLoader(
        train_ds_ut,
        batch_size=cfg.utility_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader_ut = DataLoader(
        val_ds_ut,
        batch_size=cfg.utility_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    n_gestures = len(train_ds_ut.stim2idx)
    ut_model = GestureUtilityNet(in_channels=C, n_classes=n_gestures)

    if os.path.isfile(cfg.utility_ckpt):
        print(f"[Utility] Loading checkpoint from {cfg.utility_ckpt}")
        state = torch.load(cfg.utility_ckpt, map_location=cfg.device)
        ut_model.load_state_dict(state)
    else:
        print("[Utility] Training gesture model from scratch...")
        train_utility(ut_model, train_loader_ut, val_loader_ut, cfg)

    print("[Utility] Evaluating baseline on original standardized EMG...")
    ut_metrics_base = eval_utility(ut_model, val_loader_ut, cfg)

    # IMPORTANT: do not train ut_model any further; treat as frozen surrogate backbone.
    ut_model.eval()
    ut_model.to(cfg.device)

    # 6) Train-or-load PrivDiffuser + anonymize validation set
    diff_cfg = EMGDiffusionConfig()
    X_std_anon, val_row_idx = train_and_apply_privdiffuser_emg(
        X_std=X_std,
        meta=meta,
        diff_cfg=diff_cfg,
        utility_model=ut_model,
        identity_model=id_model,
    )

    # 7) Build anonymized datasets & loaders for evaluation
    val_ds_id_anon = IdentityDatasetEMG(X_std_anon, meta, split="val")
    val_loader_id_anon = DataLoader(
        val_ds_id_anon,
        batch_size=cfg.identity_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    val_ds_ut_anon = UtilityDatasetEMG(X_std_anon, meta, split="val")
    val_loader_ut_anon = DataLoader(
        val_ds_ut_anon,
        batch_size=cfg.utility_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Bring identity model back for evaluation on anonymized data
    id_model.to(cfg.device)

    print("[Identity] Evaluating on PrivDiffuser-anonymized EMG...")
    id_metrics_anon = eval_identity(id_model, val_loader_id_anon, cfg)

    print("[Utility] Evaluating on PrivDiffuser-anonymized EMG...")
    ut_metrics_anon = eval_utility(ut_model, val_loader_ut_anon, cfg)

    # 8) Fidelity metrics in raw space (original vs anonymized)
    print("[Fidelity] Computing RMSE and PSD correlation (raw vs anonymized)...")
    X_val_raw = X[val_row_idx]  # original raw val windows

    # anonymized val windows: take val rows from X_std_anon and inverse-transform
    X_val_std_anon = X_std_anon[val_row_idx]
    N_val, T_val, C_val = X_val_std_anon.shape
    X_val_anon_flat = X_val_std_anon.reshape(N_val, -1)
    X_val_anon_raw = scaler.inverse_transform(X_val_anon_flat)
    X_val_anon_raw = X_val_anon_raw.reshape(N_val, T_val, C_val)

    rmses_anon, psd_corrs_anon = [], []
    for i in range(N_val):
        rmses_anon.append(rmse(X_val_raw[i], X_val_anon_raw[i]))
        psd_corrs_anon.append(psd_correlation(
            X_val_raw[i], X_val_anon_raw[i], fs=cfg.sampling_frequency
        ))

    fidelity_anon = {
        "rmse_mean": float(np.mean(rmses_anon)),
        "rmse_std": float(np.std(rmses_anon)),
        "psd_corr_mean": float(np.mean(psd_corrs_anon)),
        "psd_corr_std": float(np.std(psd_corrs_anon)),
    }

    # 9) Example FFT + overlays
    print("[Plots] Creating example FFT + overlay plots...")
    example_idx = 0
    example_raw = X_val_raw[example_idx]
    example_anon = X_val_anon_raw[example_idx]

    fft_raw_path = os.path.join(cfg.results_dir, "emg_fft_val_raw.png")
    fft_anon_path = os.path.join(cfg.results_dir, "emg_fft_val_privdiffuser.png")
    plot_fft(example_raw, cfg.sampling_frequency, "EMG FFT (original val)", fft_raw_path)
    plot_fft(example_anon, cfg.sampling_frequency, "EMG FFT (PrivDiffuser val)", fft_anon_path)

    overlay_dir = os.path.join(cfg.results_dir, "emg_overlay_privdiffuser")
    os.makedirs(overlay_dir, exist_ok=True)

    n_examples_plot = min(5, N_val)
    channels_to_plot = [0, 1]
    for i in range(n_examples_plot):
        for ch in channels_to_plot:
            out_path = os.path.join(overlay_dir, f"emg_overlay_example_{i}_ch{ch}.png")
            plot_overlay_emg(
                X_val_raw[i],
                X_val_anon_raw[i],
                fs=cfg.sampling_frequency,
                title=f"PrivDiffuser EMG overlay (example {i}, ch {ch})",
                filepath=out_path,
                channel=ch,
            )

    # 10) Save metrics
    results = {
        "identity_baseline": id_metrics_base,
        "identity_privdiffuser": id_metrics_anon,
        "utility_baseline": ut_metrics_base,
        "utility_privdiffuser": ut_metrics_anon,
        "fidelity_privdiffuser": fidelity_anon,
        "plots": {
            "fft_raw": fft_raw_path,
            "fft_anon": fft_anon_path,
            "overlay_dir": overlay_dir,
        },
    }
    out_json = os.path.join(cfg.results_dir, "emg_privdiffuser_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Done (EMG + PrivDiffuser) ===")
    print(f"[Results] Saved to {out_json}")


if __name__ == "__main__":
    main()
