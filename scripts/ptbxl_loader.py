#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight PTB-XL loader without external package layout.
Reads ptbxl_database.csv + scp_statements.csv and loads WFDB records.
Requires wfdb installed and PTB-XL stored under data_root.
"""
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import wfdb


def _load_metadata(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_path = os.path.join(data_root, "ptbxl_database.csv")
    scp_path = os.path.join(data_root, "scp_statements.csv")
    if not os.path.isfile(db_path) or not os.path.isfile(scp_path):
        raise FileNotFoundError("Missing ptbxl_database.csv or scp_statements.csv in PTB-XL root.")
    df = pd.read_csv(db_path, index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)
    return df, scp


def _extract_superclass(scp_codes: dict, scp_df: pd.DataFrame) -> List[str]:
    classes = []
    for code in scp_codes.keys():
        row = scp_df.loc[code]
        if row["diagnostic_class"] == "NORM":
            continue
        if pd.isna(row["diagnostic_class"]):
            continue
        classes.append(row["diagnostic_class"])
    return list(set(classes))


def load_ptbxl(project_root: str, data_root: str, sampling_rate: int) -> Tuple[np.ndarray, pd.DataFrame]:
    df, scp_df = _load_metadata(data_root)
    # choose filename column based on sampling rate
    if sampling_rate == 100:
        fname_col = "filename_lr"
    elif sampling_rate == 500:
        fname_col = "filename_hr"
    else:
        fname_col = "filename_lr"
    X_list = []
    meta_rows = []
    for ecg_id, row in df.iterrows():
        record_path = os.path.join(data_root, row[fname_col])
        sig, _ = wfdb.rdsamp(record_path)
        X_list.append(sig.astype(np.float32))
        scp_codes = eval(row["scp_codes"])
        diag_super = _extract_superclass(scp_codes, scp_df)
        meta_rows.append(
            {
                "row_idx": len(meta_rows),
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "age": row["age"],
                "sex": row["sex"],
                "height": row["height"],
                "weight": row["weight"],
                "filename": row[fname_col],
                "diagnostic_superclass": diag_super,
            }
        )
    X = np.stack(X_list, axis=0)  # (N, T, C)
    Y = pd.DataFrame(meta_rows)
    return X, Y
