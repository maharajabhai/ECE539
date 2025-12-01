#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for ECG privacy pipelines (Blinder/PrivDiffuser):
- Diagnosis metrics (thresholded) to keep reporting consistent.
"""

from typing import Dict, Optional
import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error


def compute_diag_metrics_thresholded(probs: np.ndarray, y_true: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    """
    Compute simple, thresholded multi-label metrics:
    - sample-level exact match accuracy
    - macro F1 over labels
    - macro per-class accuracy
    - MAE on probabilities
    """
    probs = np.nan_to_num(probs, nan=0.0, posinf=5.0, neginf=-5.0)
    y_hat = (probs >= thresh).astype(np.float32)
    y_true = y_true.astype(np.float32)

    sample_acc = float(np.mean(np.all(y_true == y_hat, axis=1)))
    macro_f1 = float(f1_score(y_true, y_hat, average="macro", zero_division=0))
    per_class_acc = (y_true == y_hat).mean(axis=0)
    macro_acc = float(np.mean(per_class_acc))
    mae = float(mean_absolute_error(y_true.flatten(), probs.flatten()))

    return {
        "sample_acc": sample_acc,
        "macro_f1": macro_f1,
        "macro_acc": macro_acc,
        "mae": mae,
    }


def eval_diag_thresholded(model, loader, device: str, thresh: float = 0.5) -> Dict[str, float]:
    """
    Run model over a loader and return thresholded diagnosis metrics.
    Assumes model returns (logits, embedding) and loader yields (xb, yb).
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.cpu().numpy())
    probs_cat = np.concatenate(all_probs, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    return compute_diag_metrics_thresholded(probs_cat, labels_cat, thresh=thresh)
