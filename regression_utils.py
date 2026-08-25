"""Utilities for postoperative MIO regression in the TMJ AI project.

The regression target is the observed last-visit maximal interincisal opening
(lv_mio_mm), not the change score. Expected improvement is calculated only
at inference time as predicted postoperative MIO minus preoperative MIO.

This avoids defining the regression target directly from preoperative MIO
while also using preoperative MIO as a predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "lv_mio_mm"
GROUP = "patient_id"


def build_regression_pipeline() -> Pipeline:
    """Build the preprocessing + Random Forest regression pipeline."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_regression_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return X, y, patient groups, and the filtered analysis dataframe."""
    required = FEATURES + [TARGET, GROUP]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    analysis_df = df.loc[df[TARGET].notna() & df[GROUP].notna()].copy()
    X = analysis_df[FEATURES].copy()
    y = pd.to_numeric(analysis_df[TARGET], errors="coerce")
    groups = analysis_df[GROUP].astype(str)

    keep = y.notna()
    analysis_df = analysis_df.loc[keep].copy()
    X = X.loc[keep].copy()
    y = y.loc[keep].astype(float)
    groups = groups.loc[keep]

    return X, y, groups, analysis_df


def grouped_cross_validation(
    df: pd.DataFrame,
    n_splits: int = 3,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate postoperative MIO prediction with patient-grouped CV."""
    X, y, groups, analysis_df = prepare_regression_data(df)

    cv = GroupKFold(n_splits=n_splits)
    pipeline = build_regression_pipeline()

    oof_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        groups=groups,
        method="predict",
        n_jobs=None,
    )

    mae = mean_absolute_error(y, oof_pred)
    rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
    r2 = r2_score(y, oof_pred)

    metrics = {
        "n_encounters": int(len(y)),
        "n_unique_patients": int(groups.nunique()),
        "n_splits": int(n_splits),
        "mae_mm": float(mae),
        "rmse_mm": float(rmse),
        "r2": float(r2),
        "mean_observed_postop_mio_mm": float(y.mean()),
        "sd_observed_postop_mio_mm": float(y.std(ddof=1)),
    }

    oof = pd.DataFrame(
        {
            "encounter_id": analysis_df.get("encounter_id", pd.Series(analysis_df.index, index=analysis_df.index)).astype(str).values,
            "patient_id": groups.values,
            "pre_mio_mm": pd.to_numeric(analysis_df["pre_mio_mm"], errors="coerce").values,
            "observed_postop_mio_mm": y.values,
            "predicted_postop_mio_mm": oof_pred,
        }
    )
    oof["observed_mio_change_mm"] = oof["observed_postop_mio_mm"] - oof["pre_mio_mm"]
    oof["predicted_mio_change_mm"] = oof["predicted_postop_mio_mm"] - oof["pre_mio_mm"]
    oof["absolute_error_mm"] = np.abs(oof["observed_postop_mio_mm"] - oof["predicted_postop_mio_mm"])

    return metrics, oof


def fit_final_regression_model(df: pd.DataFrame) -> Pipeline:
    """Fit the final regression model on all eligible encounters."""
    X, y, _groups, _analysis_df = prepare_regression_data(df)
    pipeline = build_regression_pipeline()
    pipeline.fit(X, y)
    return pipeline


def predict_postop_mio(model: Pipeline, patient_rows: pd.DataFrame) -> pd.DataFrame:
    """Predict postoperative MIO and derive expected change in millimetres."""
    X = patient_rows[FEATURES].copy()
    predicted_postop = model.predict(X)
    pre_mio = pd.to_numeric(patient_rows["pre_mio_mm"], errors="coerce").to_numpy(dtype=float)

    result = patient_rows.copy()
    result["predicted_postop_mio_mm"] = predicted_postop
    result["predicted_mio_change_mm"] = predicted_postop - pre_mio
    return result


def load_or_fit_regression_model(data_path: str | Path) -> Pipeline:
    """Convenience function for Streamlit: fit from the de-identified master data."""
    df = pd.read_csv(data_path)
    return fit_final_regression_model(df)


def save_regression_model(model: Pipeline, output_path: str | Path) -> None:
    joblib.dump(model, output_path)


# The grouped validation workflow runs automatically when this module changes.
