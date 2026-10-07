from __future__ import annotations

from pathlib import Path
import collections

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "age_years", "pre_mio_mm", "pre_mahan_dir_present",
    "pre_joint_noise_present", "pre_muscle_pain_present", "pre_joint_pain_present",
]
CATEGORICAL_FEATURES = ["gender_clean", "site_clean", "diet_preop_clean", "meds_preop_clean"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
VAS_NUMERIC_FEATURES = NUMERIC_FEATURES + ["pre_vas"]
VAS_FEATURES = VAS_NUMERIC_FEATURES + CATEGORICAL_FEATURES

PRETTY = {
    "age_years": "Age", "pre_mio_mm": "Pre-op MIO",
    "pre_mahan_dir_present": "Directional limitation", "pre_joint_noise_present": "Joint noise",
    "pre_muscle_pain_present": "Muscle pain", "pre_joint_pain_present": "Joint pain",
    "pre_vas": "Pre-op VAS",
    "gender_clean": "Sex", "site_clean": "Site", "diet_preop_clean": "Pre-op diet",
    "meds_preop_clean": "Pre-op medications",
}

OUTCOMES = {
    "joint_pain_absent": {"label": "No joint pain", "positive": "joint pain absent at last visit", "summary": "Joint pain absent"},
    "joint_noise_absent": {"label": "No joint noise", "positive": "joint noise absent at last visit", "summary": "Joint noise absent"},
    "no_medication": {"label": "No medication", "positive": "no medication recorded at last visit", "summary": "No medication"},
    "regular_diet": {"label": "Regular diet", "positive": "regular-consistency diet at last visit", "summary": "Regular-consistency diet"},
}

METRICS = {
    "Functional response": {"n":466,"patients":451,"positive":162,"auc":0.869,"ci":[0.836,0.899],"sensitivity":0.802,"specificity":0.770,"brier":0.146},
    "Joint pain absent": {"n":473,"patients":458,"positive":239,"auc":0.5801416157,"ci":[0.5257176768,0.6308931412],"sensitivity":0.5104602510,"specificity":0.5726495726,"brier":0.2479598175},
    "Joint noise absent": {"n":474,"patients":459,"positive":263,"auc":0.6247815040,"ci":[0.5699041121,0.6729795703],"sensitivity":0.6768060837,"specificity":0.5023696682,"brier":0.2354849212},
    "No medication": {"n":261,"patients":252,"positive":72,"auc":0.8182686655,"ci":[0.7550467497,0.8783236565],"sensitivity":0.5833333333,"specificity":0.8835978836,"brier":0.1518574480},
    "Regular-consistency diet": {"n":469,"patients":454,"positive":291,"auc":0.5835553496,"ci":[0.5304355222,0.6377464531],"sensitivity":0.6838487973,"specificity":0.4719101124,"brier":0.2394816631},
}

VAS_IMPROVEMENT_METRICS = {
    "definition": "postoperative VAS < preoperative VAS; 0=no pain, 100=worst pain; any reduction",
    "n": 253,
    "patients": 251,
    "positive": 27,
    "prevalence": 0.1067193676,
    "auc": 0.6196329072,
    "ci": [0.4901017830, 0.7381854018],
    "average_precision": 0.2683876283,
    "accuracy": 0.8814229249,
    "sensitivity": 0.2222222222,
    "specificity": 0.9601769912,
    "precision": 0.4,
    "f1": 0.2857142857,
    "brier": 0.1243185659,
    "threshold": 0.5,
}

GLOBAL_SHAP = {
    "Functional response": [["Pre-op MIO",0.2151101510],["Pre-op diet",0.0485786515],["Age",0.0402969397],["Pre-op medications",0.0331328642],["Joint noise",0.0319955025],["Site",0.0156473211]],
    "Joint pain absent": [["Pre-op diet",0.0458285961],["Age",0.0327031401],["Site",0.0304504593],["Sex",0.0304212957],["Joint pain",0.0263933936],["Muscle pain",0.0259282437]],
    "Joint noise absent": [["Joint noise",0.0731137321],["Pre-op medications",0.0418529387],["Age",0.0369170147],["Site",0.0324760838],["Pre-op MIO",0.0257044507],["Pre-op diet",0.0227288340]],
    "No medication": [["Pre-op medications",0.1512891141],["Pre-op diet",0.1422894255],["Directional limitation",0.0505805691],["Age",0.0348368991],["Site",0.0247465731],["Pre-op MIO",0.0179442107]],
    "Regular-consistency diet": [["Pre-op diet",0.0638611695],["Pre-op medications",0.0560429233],["Age",0.0313225239],["Joint noise",0.0239312640],["Pre-op MIO",0.0199778838],["Site",0.0185844076]],
}

VAS_GLOBAL_SHAP = [
    ["Pre-op MIO",0.0885108822],["Age",0.0696150486],["Site",0.0427119473],
    ["Pre-op VAS",0.0403634417],["Pre-op diet",0.0270019459],
    ["Directional limitation",0.0213925738],["Joint noise",0.0207974636],
    ["Sex",0.0195951127],["Joint pain",0.0188717869],["Muscle pain",0.0185219266],
]

VAS = {
    "n":253,"patients":251,"pre_mean":46.9169960474,"post_mean":73.5375494071,
    "change_mean":26.6205533597,"mae":17.7421659789,"rmse":22.4552068097,"r2":-0.0413267457
}


def _build_pipeline(numeric_features: list[str], n_estimators: int = 500) -> Pipeline:
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, numeric_features), ("cat", cat, CATEGORICAL_FEATURES)])
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def build_pipeline(n_estimators: int = 500) -> Pipeline:
    return _build_pipeline(NUMERIC_FEATURES, n_estimators)


def target(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "joint_pain_absent":
        x = pd.to_numeric(df["lv_joint_pain_present"], errors="coerce")
        return pd.Series(np.where(x.notna(), (x == 0).astype(float), np.nan), index=df.index)
    if name == "joint_noise_absent":
        x = pd.to_numeric(df["lv_joint_noise_present"], errors="coerce")
        return pd.Series(np.where(x.notna(), (x == 0).astype(float), np.nan), index=df.index)
    if name == "no_medication":
        x = df["meds_last_visit_clean"].astype(str).str.lower()
        return pd.Series(np.where(x.isin(["yes","no"]), (x == "no").astype(float), np.nan), index=df.index)
    if name == "regular_diet":
        x = df["diet_last_visit_clean"].astype(str).str.lower()
        regular = ["regular","rc","regular_comp","rgular"]
        return pd.Series(np.where(x.isin(regular + ["soft"]), x.isin(regular).astype(float), np.nan), index=df.index)
    raise ValueError(name)


def fit_secondary_models(data_path: str | Path) -> dict[str, Pipeline]:
    df = pd.read_csv(data_path)
    models = {}
    for name in OUTCOMES:
        y = target(df, name)
        keep = y.notna()
        model = build_pipeline(500)
        model.fit(df.loc[keep, FEATURES], y.loc[keep].astype(int))
        models[name] = model
    return models


def fit_vas_improvement_model(data_path: str | Path) -> Pipeline:
    """Fit VAS improvement classifier on complete paired VAS records.

    User-confirmed orientation: 0=no pain, 100=worst pain.
    Improvement is therefore any reduction at last visit.
    """
    df = pd.read_csv(data_path)
    pre = pd.to_numeric(df["pre_comparison_analog"], errors="coerce")
    post = pd.to_numeric(df["lv_comparison_analog"], errors="coerce")
    keep = pre.notna() & post.notna()
    analysis = df.loc[keep].copy()
    analysis["pre_vas"] = pre.loc[keep].astype(float)
    y = (post.loc[keep].astype(float) < pre.loc[keep].astype(float)).astype(int)
    model = _build_pipeline(VAS_NUMERIC_FEATURES, 500)
    model.fit(analysis[VAS_FEATURES], y)
    return model


def transformed_names(pipe: Pipeline, numeric_features: list[str] | None = None) -> list[str]:
    numeric_features = numeric_features or NUMERIC_FEATURES
    pre = pipe.named_steps["preprocessor"]
    oh = pre.named_transformers_["cat"].named_steps["onehot"]
    return numeric_features + list(oh.get_feature_names_out(CATEGORICAL_FEATURES))


def _local_shap(pipe: Pipeline, row: pd.DataFrame, feature_names: list[str], numeric_features: list[str]) -> pd.DataFrame:
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    xt = pre.transform(row[feature_names])
    names = transformed_names(pipe, numeric_features)
    sv = shap.TreeExplainer(model).shap_values(xt)
    if isinstance(sv, list):
        arr = np.asarray(sv[1] if len(sv) > 1 else sv[0])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
    vals = np.ravel(arr[0])
    collapsed = collections.defaultdict(float)
    for name, value in zip(names, vals):
        base = name
        for c in CATEGORICAL_FEATURES:
            if name.startswith(c + "_"):
                base = c
                break
        collapsed[PRETTY.get(base, base)] += float(value)
    return pd.DataFrame([{"feature":k,"shap_value":v} for k,v in collapsed.items()]).sort_values(
        "shap_value", key=lambda s:s.abs(), ascending=False
    )


def local_shap(pipe: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    return _local_shap(pipe, row, FEATURES, NUMERIC_FEATURES)


def vas_local_shap(pipe: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    return _local_shap(pipe, row, VAS_FEATURES, VAS_NUMERIC_FEATURES)
