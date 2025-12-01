#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export anonymized PPG signals to CSVs mirroring seeing-red's extracted tree.

Inputs:
    - Original extracted CSVs under seeing-red/data/extracted/<user>/<file>.csv
    - Anonymized waveforms npz at cfg.anon_npz (from Blinder or PrivDiffuser)
      whose order aligns with cfg.meta_pkl (row_idx).

Outputs:
    - An anonymized CSV tree at cfg.output_root/extracted/<user>/<file>.csv
      with the same filenames as original, single column 'luma_mean'.
"""

import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportConfig:
    meta_pkl: str = os.path.join(Path(__file__).resolve().parents[1], "data", "ppg_meta.pkl")
    anon_npz: str = os.path.join(Path(__file__).resolve().parents[1], "data", "ppg_anon.npz")
    extracted_root: str = os.path.join(Path(__file__).resolve().parents[1], "seeing-red", "data", "extracted")
    output_root: str = os.path.join(Path(__file__).resolve().parents[1], "seeing-red", "data_anon")


def main():
    cfg = ExportConfig()
    anon_path = cfg.anon_npz
    if not os.path.isfile(anon_path):
        raise FileNotFoundError(f"Missing anonymized npz at {cfg.anon_npz}")
    if not os.path.isfile(cfg.meta_pkl):
        raise FileNotFoundError(f"Missing meta at {cfg.meta_pkl}")

    with open(cfg.meta_pkl, "rb") as f:
        meta = pickle.load(f)
    anon_npz = np.load(anon_path)
    if "X_anon" in anon_npz:
        X_anon = anon_npz["X_anon"]
    elif "X_std_anon" in anon_npz:
        X_anon = anon_npz["X_std_anon"]
    else:
        # fallback: assume single array
        X_anon = list(anon_npz.values())[0]

    # ensure lengths align
    if len(X_anon) != len(meta):
        raise ValueError(f"Length mismatch: anon {len(X_anon)} vs meta {len(meta)}")

    out_extracted = Path(cfg.output_root) / "extracted"
    out_preprocessed = Path(cfg.output_root) / "preprocessed"
    out_beats = Path(cfg.output_root) / "beats"
    out_fid = Path(cfg.output_root) / "fiducial_points"
    out_postfta = Path(cfg.output_root) / "beats-post-FTA"
    out_feat = Path(cfg.output_root) / "features"
    out_feat_sel1 = Path(cfg.output_root) / "features-selected1"
    out_feat_sel2 = Path(cfg.output_root) / "features-selected2"
    # clear existing anonymized tree to rebuild fresh
    for d in [out_extracted.parent, out_preprocessed, out_beats, out_fid, out_postfta, out_feat, out_feat_sel1, out_feat_sel2]:
        if d.exists() and d.is_dir():
            for root, dirs, files in os.walk(d, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
    out_extracted.mkdir(parents=True, exist_ok=True)

    for _, row in meta.iterrows():
        row_idx = int(row["row_idx"])
        subj = str(int(row["subject_id"])).zfill(5)
        fname = row.get("filename", f"{row_idx}.csv")
        sig = X_anon[row_idx]
        # write luma_mean (and r_ch_mean if available)
        if sig.ndim == 2 and sig.shape[1] >= 2:
            df = pd.DataFrame({
                "luma_mean": sig[:, 0],
                "r_ch_mean": sig[:, 1],
            })
        else:
            sig_1d = sig[:, 0] if sig.ndim == 2 else sig
            df = pd.DataFrame({"luma_mean": sig_1d})
        subj_dir = out_extracted / subj
        subj_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(subj_dir / fname, index=False, float_format="%.6f")

    print(f"[Export] Wrote anonymized CSVs to {out_extracted}")


if __name__ == "__main__":
    main()
