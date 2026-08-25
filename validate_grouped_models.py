"""Reproduce the patient-grouped internal validation used for the TMJ AI manuscript.

Primary validation:
- StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
- patient_id is the grouping variable
- pooled out-of-fold probabilities
- 0.50 decision threshold

The script also performs:
- patient-cluster bootstrap 95% CI for AUC
- Brier score and simple calibration intercept/slope
- sensitivity analysis excluding preoperative MIO from the functional-response model
- exploratory continuous postoperative-MIO regression
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from regression_utils import grouped_cross_validation


BASE = Path(__file__).parent
DATA_PATH = BASE / "tmj_clean_master_deidentified.csv"

NUMERIC_FEATURES = [
    "age_years",
    "pre_mio_mm",
    "pre_mahan_dir_present",
    "pre_joint_noise_present",
    "pre_muscle_pain_present",
    "pre_joint_pain_present",
]
CATEGORICAL_FEATURES = [
    "gender_clean",
    "site_clean",
    "diet_preop_clean",
    "meds_preop_clean",
]


def build_classifier(numeric_features: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        min_samples_leaf=4,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    # Effectively unregularized logistic recalibration.
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(logit_p, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def patient_cluster_auc_ci(
    analysis_df: pd.DataFrame,
    probabilities: np.ndarray,
    target: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    patient_values = analysis_df["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patient_values)
    patient_to_indices = {
        patient: np.flatnonzero(patient_values == patient)
        for patient in unique_patients
    }
    y_all = analysis_df[target].astype(int).to_numpy()
    aucs = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_to_indices[p] for p in sampled])
        y = y_all[idx]
        p = probabilities[idx]
        if np.unique(y).size < 2:
            continue
        aucs.append(roc_auc_score(y, p))

    return (
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def evaluate_classifier(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    numeric_features = numeric_features or NUMERIC_FEATURES
    features = numeric_features + CATEGORICAL_FEATURES

    analysis_df = df.loc[
        df[target].notna() & df["patient_id"].notna()
    ].copy()
    X = analysis_df[features]
    y = analysis_df[target].astype(int).to_numpy()
    groups = analysis_df["patient_id"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    probabilities = cross_val_predict(
        build_classifier(numeric_features),
        X,
        y,
        cv=cv,
        groups=groups,
        method="predict_proba",
    )[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
    calibration_intercept, calibration_slope = calibration_stats(y, probabilities)
    auc_ci = patient_cluster_auc_ci(analysis_df, probabilities, target)

    metrics = {
        "validation_scheme": "StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)",
        "n": int(len(y)),
        "unique_patients": int(pd.Series(groups).nunique()),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "auc_95ci_patient_cluster_bootstrap": list(auc_ci),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted)),
        "recall": float(recall_score(y, predicted)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, predicted)),
        "brier": float(brier_score_loss(y, probabilities)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "decision_threshold": 0.5,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    oof = analysis_df[["encounter_id", "patient_id"]].copy()
    oof["observed"] = y
    oof["predicted_probability"] = probabilities
    oof["predicted_class"] = predicted
    return metrics, oof


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    outcome_metrics, outcome_oof = evaluate_classifier(
        df,
        target="mio_improvement_ge_10mm",
    )
    severity_metrics, severity_oof = evaluate_classifier(
        df,
        target="advanced_stage_iv_v",
    )

    sensitivity_numeric = [
        feature for feature in NUMERIC_FEATURES if feature != "pre_mio_mm"
    ]
    sensitivity_metrics, sensitivity_oof = evaluate_classifier(
        df,
        target="mio_improvement_ge_10mm",
        numeric_features=sensitivity_numeric,
    )

    regression_metrics, regression_oof = grouped_cross_validation(df, n_splits=3)

    all_metrics = {
        "functional_response": outcome_metrics,
        "severity_stratification": severity_metrics,
        "functional_response_without_preop_mio": sensitivity_metrics,
        "exploratory_postop_mio_regression": regression_metrics,
    }

    (BASE / "grouped_validation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )
    outcome_oof.to_csv(BASE / "oof_functional_response_grouped.csv", index=False)
    severity_oof.to_csv(BASE / "oof_severity_grouped.csv", index=False)
    sensitivity_oof.to_csv(BASE / "oof_functional_response_without_preop_mio.csv", index=False)
    regression_oof.to_csv(BASE / "oof_postop_mio_regression.csv", index=False)

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
