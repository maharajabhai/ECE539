#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPG Blinder-style autoencoder anonymization + quality assessment.

Data expectation:
    - A NumPy archive at cfg.data_npz containing:
        X:    (N, T) raw PPG waveforms
    - A meta pickle at cfg.meta_pkl with columns:
        row_idx, subject_id, id_fold (0=train, 1=val)

If you only have videos (seeing-red dataset in data/videos), first run the
seeing-red preprocessing pipeline (signal_extractor -> ... -> feature_extractor)
or add a small conversion step to dump windowed signals into ppg_windows.npz.
"""

import os
import sys
import json
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scipy import signal
from scipy.signal import resample
import os
import sys
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TOP_LEVEL = os.path.dirname(PROJECT_ROOT)
PPGRAW_DIR = os.path.join(PROJECT_ROOT, "PPGraw", "src")
SEEINGRED_EXTRACTED = os.path.join(PROJECT_ROOT, "seeing-red", "data", "extracted")

for p in [THIS_DIR, PROJECT_ROOT, TOP_LEVEL, PPGRAW_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from PPGraw import PPGraw


@dataclass
class PPGConfig:
    data_npz: str = os.path.join(PROJECT_ROOT, "data", "ppg_windows.npz")
    meta_pkl: str = os.path.join(PROJECT_ROOT, "data", "ppg_meta.pkl")
    anon_npz: str = os.path.join(PROJECT_ROOT, "data", "ppg_anon_blinder.npz")
    sampling_frequency: int = 60
    identity_batch_size: int = 64
    identity_lr: float = 5e-4
    identity_epochs: int = 120
    blinder_batch_size: int = 128
    blinder_lr: float = 1e-3
    blinder_epochs: int = 200
    z_dim: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    identity_ckpt: str = os.path.join(PROJECT_ROOT, "models", "ppg_blinder", "identity.pt")
    blinder_ckpt: str = os.path.join(PROJECT_ROOT, "models", "ppg_blinder", "ppg_blinder.pt")
    results_dir: str = os.path.join(PROJECT_ROOT, "privacy_ppg_outputs")
    run_seeingred_identity: bool = True


def load_ppg(cfg: PPGConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    if not (os.path.isfile(cfg.data_npz) and os.path.isfile(cfg.meta_pkl)):
        X, meta = build_ppg_from_seeingred(cfg)
    else:
        X = np.load(cfg.data_npz)["X"]
        with open(cfg.meta_pkl, "rb") as f:
            meta = pickle.load(f)
    subj_ids = sorted(meta["subject_id"].unique())
    subj_map = {sid: i for i, sid in enumerate(subj_ids)}
    meta["subject_idx"] = meta["subject_id"].map(subj_map)
    return X, meta


def build_ppg_from_seeingred(cfg: PPGConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    if not os.path.isdir(SEEINGRED_EXTRACTED):
        raise FileNotFoundError(f"Missing {cfg.data_npz} and no seeing-red extracted dir at {SEEINGRED_EXTRACTED}")

    records = []
    for subj in sorted(os.listdir(SEEINGRED_EXTRACTED)):
        subj_dir = os.path.join(SEEINGRED_EXTRACTED, subj)
        if not os.path.isdir(subj_dir):
            continue
        for fname in sorted([f for f in os.listdir(subj_dir) if f.endswith('.csv')]):
            records.append((int(subj), os.path.join(subj_dir, fname), fname))

    if not records:
        raise FileNotFoundError("No extracted CSVs found under seeing-red/data/extracted")

    src_fs = 240  # from seeing-red params.yaml
    resampled, min_len = [], None
    for subj_id, path, fname in records:
        df = pd.read_csv(path)
        if "luma_mean" not in df.columns:
            continue
        sig = df["luma_mean"].values.astype(np.float32)
        target_len = int(len(sig) * cfg.sampling_frequency / src_fs)
        target_len = max(target_len, cfg.sampling_frequency)
        sig_rs = resample(sig, target_len)
        resampled.append((subj_id, fname, sig_rs))
        min_len = target_len if min_len is None else min(min_len, target_len)

    if min_len is None:
        raise ValueError("No valid luma_mean signals found in extracted CSVs")

    X_list, meta_rows = [], []
    subj_ids = sorted({r[0] for r in resampled})
    subj_map = {sid: i for i, sid in enumerate(subj_ids)}
    for idx, (subj_id, fname, sig_rs) in enumerate(resampled):
        X_list.append(sig_rs[:min_len])
        meta_rows.append({"row_idx": idx, "subject_id": subj_id, "subject_idx": subj_map[subj_id], "filename": fname})

    X = np.stack(X_list, axis=0)
    meta = pd.DataFrame(meta_rows)

    # simple 80/20 split per subject (shuffled)
    rng = np.random.default_rng(42)
    fold_map = {}
    for subj_id in meta["subject_id"].unique():
        idxs = meta.index[meta["subject_id"] == subj_id].tolist()
        rng.shuffle(idxs)
        n_val = max(1, int(0.2 * len(idxs)))
        val_idx = set(idxs[:n_val])
        for i in idxs:
            fold_map[i] = 1 if i in val_idx else 0
    meta["id_fold"] = meta.index.map(fold_map)

    os.makedirs(os.path.dirname(cfg.data_npz), exist_ok=True)
    np.savez(cfg.data_npz, X=X)
    with open(cfg.meta_pkl, "wb") as f:
        pickle.dump(meta, f)
    print(f"[PPG] Built {cfg.data_npz} and {cfg.meta_pkl} from seeing-red extracted signals (len={min_len}, N={len(X)})")
    return X, meta


class IdentityDS(Dataset):
    def __init__(self, X_std, meta, split):
        if split == "train":
            subset = meta[meta["id_fold"] == 0]
        elif split == "val":
            subset = meta[meta["id_fold"] == 1]
        else:
            raise ValueError
        self.meta = subset.reset_index(drop=True)
        idx = self.meta["row_idx"].values
        X_sel = X_std[idx]
        self.X = X_sel[:, None, :]
        y_col = "subject_idx" if "subject_idx" in self.meta.columns else "subject_id"
        self.y = self.meta[y_col].values.astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        x = self.X[i]
        mean = x.mean()
        std = x.std() + 1e-6
        x = (x - mean) / std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.long)


class IdentityNet(nn.Module):
    def __init__(self, n_subjects: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, 2, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, n_subjects)

    def forward(self, x):
        h = self.feat(x).squeeze(-1)
        return {"logits": self.head(h), "feat": h}


def train_identity(model, train_loader, val_loader, cfg):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.identity_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.identity_epochs, eta_min=1e-5)
    ce = nn.CrossEntropyLoss()
    best = float("inf")
    for ep in range(cfg.identity_epochs):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            out = model(xb)
            loss = ce(out["logits"], yb)
            loss.backward()
            opt.step()
            tr += loss.item() * xb.size(0)
        tr /= len(train_loader.dataset)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                out = model(xb)
                vl += ce(out["logits"], yb).item() * xb.size(0)
        vl /= len(val_loader.dataset)
        print(f"[PPG ID] {ep+1}/{cfg.identity_epochs} train={tr:.4f} val={vl:.4f}")
        scheduler.step()
        if vl < best:
            best = vl
            os.makedirs(os.path.dirname(cfg.identity_ckpt), exist_ok=True)
            torch.save(model.state_dict(), cfg.identity_ckpt)
    print(f"[PPG ID] best val loss {best:.4f}")


def eval_identity(model, loader, cfg):
    model.to(cfg.device).eval()
    ce = nn.CrossEntropyLoss()
    loss, corr, tot = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            out = model(xb)
            loss += ce(out["logits"], yb).item() * xb.size(0)
            corr += (out["logits"].argmax(1) == yb).sum().item()
            tot += xb.size(0)
            all_true.extend(yb.cpu().numpy().tolist())
            all_pred.extend(out["logits"].argmax(1).cpu().numpy().tolist())
    f1 = f1_score(all_true, all_pred, average="macro") if all_true else 0.0
    return {"loss": loss / max(1, tot), "acc": corr / max(1, tot), "f1_macro": float(f1)}


class VAE(nn.Module):
    def __init__(self, T, z_dim):
        super().__init__()
        self.T = T
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, T),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x).squeeze(-1)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    rec = nn.MSELoss()(recon, x.squeeze(1))
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + 1e-3 * kld


def train_vae(model, X_std, cfg):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_std[:, None, :], dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.blinder_batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.blinder_lr)
    print(f"[PPG Blinder] training {cfg.blinder_epochs} epochs on {len(dataset)} samples")
    for ep in range(cfg.blinder_epochs):
        model.train()
        loss_sum = 0.0
        for (xb,) in loader:
            xb = xb.to(cfg.device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * xb.size(0)
        loss_sum /= len(dataset)
        print(f"[PPG Blinder] {ep+1}/{cfg.blinder_epochs} loss={loss_sum:.4f}")
    torch.save(model.state_dict(), cfg.blinder_ckpt)
    print(f"[PPG Blinder] saved to {cfg.blinder_ckpt}")


def anonymize_with_vae(model, X_std, cfg):
    model.to(cfg.device).eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(X_std), cfg.blinder_batch_size):
            end = min(start + cfg.blinder_batch_size, len(X_std))
            xb = torch.tensor(X_std[start:end, None, :], dtype=torch.float32, device=cfg.device)
            recon, _, _ = model(xb)
            outs.append(recon.cpu().numpy())
    anon = np.concatenate(outs, axis=0)
    return anon


def compute_ppg_quality(signals, fs):
    metrics = {"ampl_span": [], "granularity": [], "norm_min": [], "norm_max": []}
    for sig in signals:
        try:
            pr = PPGraw(signal=sig.tolist(), fs=fs)
            amp = pr.review_amplitude()
            gran = pr.review_granularity()
            norm = pr.review_normalization()
            metrics["ampl_span"].append(amp.get("span", 0.0))
            metrics["granularity"].append(gran.get("granularity", 0.0))
            metrics["norm_min"].append(norm.get("min", 0.0))
            metrics["norm_max"].append(norm.get("max", 0.0))
        except Exception:
            metrics["ampl_span"].append(0.0)
            metrics["granularity"].append(0.0)
            metrics["norm_min"].append(0.0)
            metrics["norm_max"].append(0.0)
    return {k: float(np.nanmean(v)) for k, v in metrics.items()}


def estimate_bpm(sig, fs):
    sig = sig - np.nanmean(sig)
    sig = np.nan_to_num(sig, nan=0.0)
    b, a = signal.butter(2, [0.5 / (fs / 2), 5.0 / (fs / 2)], btype="band")
    filt = signal.filtfilt(b, a, sig)
    peaks, _ = signal.find_peaks(filt, distance=int(0.3 * fs), prominence=np.std(filt) * 0.5)
    bpm = None
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        rr = rr[rr > 1e-3]
        if len(rr):
            bpm = 60.0 / np.median(rr)
    if bpm is None:
        corr = np.correlate(filt, filt, mode="full")
        corr = corr[corr.size // 2:]
        corr[0] = 0
        peak_idx = np.argmax(corr[: int(fs * 2)])  # search up to ~2s lag
        if peak_idx > 0:
            bpm = 60.0 * fs / peak_idx
    return float(np.clip(bpm, 30.0, 220.0)) if bpm is not None else 0.0


def detect_peaks(sig, fs):
    sig = sig - np.nanmean(sig)
    sig = np.nan_to_num(sig, nan=0.0)
    b, a = signal.butter(2, [0.5 / (fs / 2), 5.0 / (fs / 2)], btype="band")
    filt = signal.filtfilt(b, a, sig)
    peaks, _ = signal.find_peaks(filt, distance=int(0.3 * fs), prominence=np.std(filt) * 0.5)
    return peaks


def compute_hrv(signals, fs):
    rmssd, sdnn, pnn50 = [], [], []
    for sig in signals:
        peaks = detect_peaks(sig, fs)
        if len(peaks) < 3:
            continue
        rr = np.diff(peaks) / fs
        rr = rr[rr > 1e-3]
        if len(rr) < 2:
            continue
        diff_rr = np.diff(rr)
        rmssd.append(np.sqrt(np.mean(diff_rr ** 2)))
        sdnn.append(np.std(rr))
        nn50 = np.sum(np.abs(diff_rr) > 0.05)
        pnn50.append(nn50 / max(1, len(diff_rr)))
    return {
        "rmssd_mean": float(np.nanmean(rmssd)) if rmssd else 0.0,
        "sdnn_mean": float(np.nanmean(sdnn)) if sdnn else 0.0,
        "pnn50_mean": float(np.nanmean(pnn50)) if pnn50 else 0.0,
    }


def estimate_resp(signals, fs):
    rates = []
    for sig in signals:
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
        rates.append(freq_band[np.argmax(psd_band)] * 60.0)
    return {
        "resp_rate_mean": float(np.nanmean(rates)) if rates else 0.0,
        "resp_rate_std": float(np.nanstd(rates)) if rates else 0.0,
    }


def main():
    cfg = PPGConfig()
    os.makedirs(cfg.results_dir, exist_ok=True)
    X_raw, meta = load_ppg(cfg)
    print(f"[Data] PPG shape {X_raw.shape}, meta {len(meta)}")

    # standardize on train split only
    train_idx = meta[meta["id_fold"] == 0]["row_idx"].values
    scaler = StandardScaler()
    scaler.fit(X_raw[train_idx])
    X_std = scaler.transform(X_raw)

    # identity (skipped)
    id_model = None
    id_base = {"loss": None, "acc": None, "f1_macro": None}

    # blinder VAE
    vae = VAE(T=X_std.shape[1], z_dim=cfg.z_dim).to(cfg.device)
    if os.path.isfile(cfg.blinder_ckpt):
        print(f"[PPG Blinder] loading {cfg.blinder_ckpt}")
        vae.load_state_dict(torch.load(cfg.blinder_ckpt, map_location=cfg.device))
    else:
        train_vae(vae, X_std, cfg)

    X_std_anon = anonymize_with_vae(vae, X_std, cfg)

    # identity on anonymized
    # identity on anonymized (skipped)
    id_anon = {"loss": None, "acc": None, "f1_macro": None}

    # quality metrics
    val_rows = val_ds.meta["row_idx"].values
    X_val_raw = X_raw[val_rows]
    X_val_anon_raw = scaler.inverse_transform(X_std_anon[val_rows])
    q_raw = compute_ppg_quality(X_val_raw, cfg.sampling_frequency)
    q_anon = compute_ppg_quality(X_val_anon_raw, cfg.sampling_frequency)

    # pulse rate as simple utility readout
    bpm_raw = [estimate_bpm(sig, cfg.sampling_frequency) for sig in X_val_raw]
    bpm_anon = [estimate_bpm(sig, cfg.sampling_frequency) for sig in X_val_anon_raw]
    util_metrics = {
        "bpm_raw_mean": float(np.nanmean(bpm_raw)),
        "bpm_anon_mean": float(np.nanmean(bpm_anon)),
        "bpm_abs_diff_mean": float(np.nanmean(np.abs(np.array(bpm_raw) - np.array(bpm_anon)))),
    }
    hrv_raw = compute_hrv(X_val_raw, cfg.sampling_frequency)
    hrv_anon = compute_hrv(X_val_anon_raw, cfg.sampling_frequency)
    resp_raw = estimate_resp(X_val_raw, cfg.sampling_frequency)
    resp_anon = estimate_resp(X_val_anon_raw, cfg.sampling_frequency)

    results = {
        "identity_baseline": id_base,
        "identity_blinder": id_anon,
        "quality_raw": q_raw,
        "quality_blinder": q_anon,
        "pulse_rate": util_metrics,
        "hrv_raw": hrv_raw,
        "hrv_blinder": hrv_anon,
        "resp_rate_raw": resp_raw,
        "resp_rate_blinder": resp_anon,
    }
    out_json = os.path.join(cfg.results_dir, "ppg_blinder_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] saved to {out_json}")

    # Save anonymized signals for downstream identity attack
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
