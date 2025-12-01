#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ECG utility surrogates (diagnosis, optional BPM) on PTB-XL and save checkpoints/metrics.
Designed to run once on a small GPU (V100/A100) and be reused by Blinder/PrivDiffuser.
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TOP_LEVEL = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TOP_LEVEL not in sys.path:
    sys.path.insert(0, TOP_LEVEL)

from pipeline.datasets import ecg_ptbxl
from ecg_blinder_pipeline_with_overlay import (
    Config as ECGConfig,
    build_identity_meta,
    build_utility_labels_from_superclass,
)
from ecg_privdiffuser_with_overlay import (
    DiagnosisDataset,
    DiagnosisClassifier,
    train_diagnosis_classifier,
    eval_diagnosis,
    compute_bpm_labels,
    ECGHeartRateDataset,
    HeartRateRegressor,
    train_heart_rate_regressor,
)
from ecg_priv_common import compute_diag_metrics_thresholded


@dataclass
class UtilTrainConfig:
    model_dir: str = os.path.join(PROJECT_ROOT, "models", "ecg_privdiffuser")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hr_batch_size: int = 128
    hr_lr: float = 1e-3
    hr_epochs: int = 40
    diag_batch_size: int = 128
    diag_lr: float = 1e-3
    diag_epochs: int = 40
    cdim: int = 64


def main():
    cfg = ECGConfig()
    util_cfg = UtilTrainConfig()
    os.makedirs(util_cfg.model_dir, exist_ok=True)

    # 1) Load PTB-XL
    data_dict = ecg_ptbxl.load_ptbxl_and_eda(
        ptbxl_root=cfg.datafolder,
        sampling_rate=cfg.sampling_frequency,
        output_dir=os.path.join(cfg.results_dir, "ptbxl_eda"),
        save_csv=False,
    )
    X_raw = data_dict["X"]   # (N, T, C)
    Y = data_dict["Y"]
    print(f"[Data] ECG shape: {X_raw.shape}")

    # 2) Identity split (for consistent folds)
    meta, _ = build_identity_meta(Y, cfg)

    # 3) Standardize (train split only)
    N, T, C = X_raw.shape
    scaler = StandardScaler()
    train_idx = meta[meta["id_fold"] == 0]["row_idx"].values
    X_train_flat = X_raw[train_idx].reshape(len(train_idx), -1)
    scaler.fit(X_train_flat)
    X_std = scaler.transform(X_raw.reshape(N, -1)).reshape(N, T, C)

    results = {}

    # ---------------- Diagnosis surrogate ----------------
    print("[Diagnosis] Building labels...")
    y_diag, diag_classes = build_utility_labels_from_superclass(Y)
    train_ds_diag = DiagnosisDataset(X_std, y_diag, meta, split="train")
    val_ds_diag = DiagnosisDataset(X_std, y_diag, meta, split="val")
    train_loader_diag = DataLoader(
        train_ds_diag,
        batch_size=util_cfg.diag_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader_diag = DataLoader(
        val_ds_diag,
        batch_size=util_cfg.diag_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    diag_model = DiagnosisClassifier(in_channels=C, n_classes=y_diag.shape[1], embed_dim=util_cfg.cdim)
    diag_ckpt = os.path.join(util_cfg.model_dir, "diag_surrogate.pt")
    if os.path.isfile(diag_ckpt):
        print(f"[Diagnosis] Found checkpoint at {diag_ckpt}, skipping training.")
        state = torch.load(diag_ckpt, map_location=util_cfg.device)
        diag_model.load_state_dict(state)
        diag_metrics_train = {"train_bce": None, "val_bce": None, "val_auroc_macro": None}
    else:
        print("[Diagnosis] Training surrogate...")
        diag_metrics_train = train_diagnosis_classifier(diag_model, train_loader_diag, val_loader_diag, util_cfg)
        torch.save(diag_model.state_dict(), diag_ckpt)
        print(f"[Diagnosis] Saved to {diag_ckpt}")
    diag_model.to(util_cfg.device)
    diag_metrics_val = eval_diagnosis(diag_model, val_loader_diag, util_cfg.device)
    # Thresholded metrics
    def collect_probs(loader):
        all_probs, all_labels = [], []
        diag_model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(util_cfg.device)
                yb = yb.to(util_cfg.device)
                logits, _ = diag_model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(yb.cpu().numpy())
        return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)
    probs_raw, labels_raw = collect_probs(val_loader_diag)
    diag_thresh = compute_diag_metrics_thresholded(probs_raw, labels_raw, thresh=0.5)

    results["diagnosis_train"] = diag_metrics_train
    results["diagnosis_val"] = diag_metrics_val
    results["diagnosis_val_threshold"] = diag_thresh

    # ---------------- Optional: Heart-rate surrogate ----------------
    hr_ckpt = os.path.join(util_cfg.model_dir, "hr_surrogate.pt")
    hr_metrics_train = hr_metrics_val = None
    if not os.path.isfile(hr_ckpt):
        print("[HeartRate] Computing BPM labels...")
        bpm_labels = compute_bpm_labels(X_raw, fs=cfg.sampling_frequency)
        train_ds_hr = ECGHeartRateDataset(X_std, bpm_labels, meta, split="train")
        val_ds_hr = ECGHeartRateDataset(X_std, bpm_labels, meta, split="val")
        train_loader_hr = DataLoader(
            train_ds_hr,
            batch_size=util_cfg.hr_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader_hr = DataLoader(
            val_ds_hr,
            batch_size=util_cfg.hr_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        hr_model = HeartRateRegressor(in_channels=C, embed_dim=util_cfg.cdim)
        hr_metrics_train = train_heart_rate_regressor(hr_model, train_loader_hr, val_loader_hr, util_cfg)
        torch.save(hr_model.state_dict(), hr_ckpt)
        print(f"[HeartRate] Saved to {hr_ckpt}")
        # Evaluate val MAE
        hr_model.to(util_cfg.device)
        hr_model.eval()
        maes = []
        with torch.no_grad():
            for xb, yb in val_loader_hr:
                xb = xb.to(util_cfg.device)
                yb = yb.to(util_cfg.device)
                preds, _ = hr_model(xb)
                maes.append(torch.abs(preds - yb).cpu().numpy())
        hr_metrics_val = {"val_mae": float(np.mean(np.concatenate(maes)))}
        results["heart_rate_train"] = hr_metrics_train
        results["heart_rate_val"] = hr_metrics_val
    else:
        print(f"[HeartRate] Found checkpoint at {hr_ckpt}, skipping HR training.")

    # ------------- Save metrics summary -----------------
    out_json = os.path.join(util_cfg.model_dir, "ecg_util_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Saved utility metrics to {out_json}")


if __name__ == "__main__":
    main()
