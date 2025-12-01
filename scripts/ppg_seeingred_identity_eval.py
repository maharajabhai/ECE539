#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate PPG identity using the seeing-red hand-crafted feature/classifier setup
with 20-beat aggregation.

Assumes seeing-red feature outputs already exist under:
    seeing-red/data/features-selected2/features.csv (and optionally features-FTA.csv)

This mirrors exp1 from seeing-red/classify.py, but only reports EER for the
aggregation window size of 20 beats to serve as a stronger privacy attack check.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

try:
    import xgboost as xgb
except Exception:
    xgb = None

# Roots for raw and anonymized seeing-red data
DEFAULT_RAW_ROOT = Path(__file__).resolve().parent.parent / "seeing-red" / "data"
RAW_ROOT = Path(os.environ.get("SEEINGRED_RAW_ROOT", DEFAULT_RAW_ROOT))
ANON_ROOT = Path(os.environ.get("SEEINGRED_ANON_ROOT", RAW_ROOT.parent / "data_anon"))


def load_feature_file(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=False)


def aggregation(values, size, reps=100, fn=np.mean):
    out = []
    for _ in range(reps):
        perm = np.random.permutation(values)
        out.append(fn(perm[:size]))
    return out


def eer_from_scores(pos_scores, neg_scores):
    y_true = np.concatenate((np.ones(len(pos_scores)), np.zeros(len(neg_scores))))
    y_pred = np.concatenate((pos_scores, neg_scores))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer)


def evaluate_exp1_window20(df: pd.DataFrame, clf_name: str):
    classifiers = {
        "SVM": svm.SVC(kernel="rbf", C=1.0, gamma="auto", probability=True),
        "GBT": xgb.XGBClassifier() if xgb is not None else None,
        "RF": RandomForestClassifier(n_estimators=50),
    }
    clf = classifiers.get(clf_name)
    if clf is None:
        return None

    X = df.drop(["filename", "meta_counter", "folder_name"], axis=1).values
    y_raw = df["folder_name"].values
    # map labels to contiguous ints
    classes = sorted(np.unique(y_raw))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in y_raw], dtype=int)

    skfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    aggr_size = 20
    eers = []

    for train_i, test_i in skfold.split(X, y):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        clf.fit(X_train, y_train)

        # one-vs-rest style per-user scores
        for user in np.unique(y):
            X_user = X_test[y_test == user]
            X_others = X_test[y_test != user]
            y_others = y_test[y_test != user]

            user_scores = clf.predict_proba(X_user)[:, clf.classes_ == user].flatten()
            other_scores = clf.predict_proba(X_others)[:, clf.classes_ == user].flatten()

            user_aggr = aggregation(user_scores, aggr_size, reps=100, fn=np.mean)
            other_aggr = []
            for other_u in np.unique(y_others):
                _s = other_scores[y_others == other_u]
                other_aggr.extend(aggregation(_s, aggr_size, reps=10, fn=np.mean))

            eers.append(eer_from_scores(user_aggr, other_aggr))

    return {"eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers))}


def main():
    results = {}

    def eval_root(root: Path, label: str):
        feats_dir = root / "features-selected2"
        found = False
        for fname in ["features.csv", "features-FTA.csv"]:
            path = feats_dir / fname
            df = load_feature_file(path)
            if df is None or df.empty:
                continue
            found = True
            suffix = "FTA" if "FTA" in fname else "NOFTA"
            for clf_name in ["SVM", "GBT", "RF"]:
                res = evaluate_exp1_window20(df, clf_name)
                if res is not None:
                    results[f"{label}_{suffix}_{clf_name}_win20"] = res
        return found

    eval_root(RAW_ROOT, "raw")
    eval_root(ANON_ROOT, "anon")

    out_path = Path(__file__).resolve().parent.parent / "privacy_ppg_outputs" / "seeingred_identity_win20.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Saved seeing-red identity (20-beat) metrics to {out_path}")


if __name__ == "__main__":
    main()
