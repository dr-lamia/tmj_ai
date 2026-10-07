"""Reproduce the patient-grouped VAS improvement analysis used in TMJ AI Studio.

Study coding supplied for this analysis:
0 = no pain; 100 = worst pain.
Improvement = last-visit VAS < preoperative VAS (any reduction).

The purpose of this script is transparent internal validation. The resulting
probability is exploratory and requires external validation before clinical use.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).parent
DATA = BASE / "tmj_clean_master_deidentified.csv"

NUMERIC = [
    "age_years", "pre_mio_mm", "pre_mahan_dir_present",
    "pre_joint_noise_present", "pre_muscle_pain_present",
    "pre_joint_pain_present", "pre_vas",
]
CATEGORICAL = ["gender_clean", "site_clean", "diet_preop_clean", "meds_preop_clean"]


def build_model():
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocessor", prep), ("model", rf)])


def patient_cluster_auc_ci(y, p, groups, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    unique = np.array(pd.unique(groups))
    by_patient = {u: np.where(groups == u)[0] for u in unique}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_patient[u] for u in sampled])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        values.append(roc_auc_score(yy, pp))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    df = pd.read_csv(DATA)
    pre = pd.to_numeric(df["pre_comparison_analog"], errors="coerce")
    post = pd.to_numeric(df["lv_comparison_analog"], errors="coerce")
    keep = pre.notna() & post.notna() & df["patient_id"].notna()
    a = df.loc[keep].copy()
    a["pre_vas"] = pre.loc[keep].astype(float)
    y = (post.loc[keep].astype(float) < pre.loc[keep].astype(float)).astype(int).to_numpy()
    groups = a["patient_id"].astype(str).to_numpy()
    X = a[NUMERIC + CATEGORICAL]

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    p = cross_val_predict(build_model(), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    out = {
        "definition": "postoperative VAS < preoperative VAS (0=no pain, 100=worst pain; any reduction)",
        "validation_scheme": "StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) by patient_id",
        "n_encounters": int(len(y)),
        "n_unique_patients": int(pd.Series(groups).nunique()),
        "improved_n": int(y.sum()),
        "improved_pct": float(y.mean() * 100),
        "roc_auc": float(roc_auc_score(y, p)),
        "auc_95ci_patient_cluster_bootstrap": patient_cluster_auc_ci(y, p, groups),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "decision_threshold": 0.5,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    (BASE / "vas_improvement_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
