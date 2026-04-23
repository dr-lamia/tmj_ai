
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).parent

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

PRETTY_NAME_MAP = {
    "age_years": "Age",
    "pre_mio_mm": "Pre-op MIO",
    "pre_mahan_dir_present": "Directional limitation",
    "pre_joint_noise_present": "Joint noise",
    "pre_muscle_pain_present": "Muscle pain",
    "pre_joint_pain_present": "Joint pain",
    "gender_clean": "Gender",
    "site_clean": "Site",
    "diet_preop_clean": "Pre-op diet",
    "meds_preop_clean": "Pre-op meds",
}


def find_file(name: str) -> Path:
    candidates = [BASE / name, BASE / "shap_outputs" / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required file: {name}")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(find_file(name))


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads(find_file(name).read_text(encoding="utf-8"))


@st.cache_resource
def load_model(name: str):
    return joblib.load(find_file(name))


def fmt_metric(x) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"


def display_value(label: str) -> str:
    return "Unknown / missing" if str(label) == "nan" else str(label)


def get_options(df1: pd.DataFrame, df2: pd.DataFrame, col: str, fallback: list[str]) -> list[str]:
    vals = []
    for df in (df1, df2):
        if col in df.columns:
            vals.extend(df[col].astype(str).fillna("nan").tolist())
    cleaned = sorted({v if v not in ("", "None", "none") else "nan" for v in vals})
    return cleaned or fallback


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def build_manual_input(prefix: str, gender_opts, site_opts, diet_opts, meds_opts) -> pd.DataFrame:
    c1, c2 = st.columns(2)

    with c1:
        age_txt = st.text_input("Age (years)", value="35", key=f"{prefix}_age")
        gender = st.selectbox("Gender", gender_opts, format_func=display_value, key=f"{prefix}_gender")
        site = st.selectbox("Site", site_opts, format_func=display_value, key=f"{prefix}_site")
        diet = st.selectbox("Pre-op diet", diet_opts, format_func=display_value, key=f"{prefix}_diet")
        meds = st.selectbox("Pre-op meds", meds_opts, format_func=display_value, key=f"{prefix}_meds")

    with c2:
        pre_mio_txt = st.text_input("Pre-op MIO (mm)", value="30", key=f"{prefix}_mio")
        pre_mahan_txt = st.selectbox(
            "Directional limitation present",
            ["0", "1"],
            format_func=lambda x: "No" if x == "0" else "Yes",
            key=f"{prefix}_mahan",
        )
        joint_noise_txt = st.selectbox(
            "Joint noise present",
            ["0", "1"],
            format_func=lambda x: "No" if x == "0" else "Yes",
            key=f"{prefix}_joint_noise",
        )
        muscle_pain_txt = st.selectbox(
            "Muscle pain present",
            ["0", "1"],
            format_func=lambda x: "No" if x == "0" else "Yes",
            key=f"{prefix}_muscle_pain",
        )
        joint_pain_txt = st.selectbox(
            "Joint pain present",
            ["0", "1"],
            format_func=lambda x: "No" if x == "0" else "Yes",
            key=f"{prefix}_joint_pain",
        )

    return pd.DataFrame([{
        "age_years": to_float(age_txt, 35.0),
        "gender_clean": gender,
        "site_clean": site,
        "diet_preop_clean": diet,
        "meds_preop_clean": meds,
        "pre_mio_mm": to_float(pre_mio_txt, 30.0),
        "pre_mahan_dir_present": float(pre_mahan_txt),
        "pre_joint_noise_present": float(joint_noise_txt),
        "pre_muscle_pain_present": float(muscle_pain_txt),
        "pre_joint_pain_present": float(joint_pain_txt),
    }])


def load_uploaded_patient(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing required columns: " + ", ".join(missing))
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def get_transformed_feature_map(fitted_pipe):
    preprocessor = fitted_pipe.named_steps["preprocessor"]
    num_names = NUMERIC_FEATURES.copy()
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return num_names + cat_names


def collapse_shap_by_base_feature(shap_row, transformed_feature_names):
    feature_contrib = {}
    for name, value in zip(transformed_feature_names, shap_row):
        matched = False
        for base in CATEGORICAL_FEATURES:
            if name.startswith(base + "_"):
                feature_contrib[base] = feature_contrib.get(base, 0.0) + float(value)
                matched = True
                break
        if not matched:
            feature_contrib[name] = feature_contrib.get(name, 0.0) + float(value)
    return feature_contrib


def explain_single_prediction(fitted_pipe, row: pd.DataFrame):
    preprocessor = fitted_pipe.named_steps["preprocessor"]
    model = fitted_pipe.named_steps["model"]

    transformed = preprocessor.transform(row)
    transformed_names = get_transformed_feature_map(fitted_pipe)

    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[1])[0]
    else:
        shap_row = np.array(shap_values)[0]

    collapsed = collapse_shap_by_base_feature(shap_row, transformed_names)
    local_df = pd.DataFrame(
        [{"feature": k, "shap_value": v, "feature_pretty": PRETTY_NAME_MAP.get(k, k)} for k, v in collapsed.items()]
    ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)

    return local_df


def plain_language_explanation(local_df: pd.DataFrame, positive_label: str) -> str:
    top = local_df.head(5)
    positive = top[top["shap_value"] > 0]["feature_pretty"].tolist()
    negative = top[top["shap_value"] < 0]["feature_pretty"].tolist()

    parts = []
    if positive:
        parts.append(
            "The prediction was pushed more toward **{}** mainly by: {}.".format(
                positive_label, ", ".join(positive)
            )
        )
    if negative:
        parts.append(
            "The prediction was pushed away from **{}** mainly by: {}.".format(
                positive_label, ", ".join(negative)
            )
        )
    if not parts:
        parts.append("No strong feature contributions were found for this case.")

    parts.append(
        "This explanation reflects how the saved model used the entered baseline clinical features for this individual case."
    )
    return " ".join(parts)


def render_local_shap(local_df: pd.DataFrame, title: str):
    st.subheader(title)
    plot_df = local_df.head(10).copy()
    st.bar_chart(plot_df.set_index("feature_pretty")["shap_value"])
    st.dataframe(
        plot_df[["feature_pretty", "shap_value"]],
        use_container_width=True,
        hide_index=True
    )


st.set_page_config(page_title="TMJ AI Studio", page_icon="🦷", layout="wide")

study1_df = load_csv("study1_mio_model_dataset.csv")
study2_df = load_csv("study2_stage_model_dataset.csv")
summary = load_json("model_summary.json")

model1 = load_model("model_study1_mio_improvement.joblib")
model2 = load_model("model_study2_advanced_stage.joblib")

gender_options = get_options(study1_df, study2_df, "gender_clean", ["F", "M"])
site_options = get_options(study1_df, study2_df, "site_clean", ["LT", "RT", "BL", "nan"])
diet_options = get_options(study1_df, study2_df, "diet_preop_clean", ["regular", "soft", "regular_comp", "rc", "unknown", "nan"])
meds_options = get_options(study1_df, study2_df, "meds_preop_clean", ["yes", "no", "nan"])

st.title("TMJ AI Studio")
st.caption("Upload a patient CSV or enter one patient manually, then get prediction and SHAP-based explanation.")

with st.sidebar:
    st.header("Model snapshot")
    st.write("**Study 1**")
    st.write(f"AUC: {fmt_metric(summary.get('study1', {}).get('roc_auc'))}")
    st.write(f"Accuracy: {fmt_metric(summary.get('study1', {}).get('accuracy'))}")
    st.write("**Study 2**")
    st.write(f"AUC: {fmt_metric(summary.get('study2', {}).get('roc_auc'))}")
    st.write(f"Accuracy: {fmt_metric(summary.get('study2', {}).get('accuracy'))}")

tabs = st.tabs(["Study 1", "Study 2", "CSV template help"])

with tabs[0]:
    st.subheader("Study 1: Predict meaningful postoperative MIO improvement")
    mode1 = st.selectbox("Input method", ["Manual entry", "Upload patient CSV"], key="study1_mode")

    row1 = None
    if mode1 == "Manual entry":
        row1 = build_manual_input("s1", gender_options, site_options, diet_options, meds_options)
    else:
        uploaded1 = st.file_uploader("Upload patient CSV for Study 1", type=["csv"], key="u1")
        if uploaded1 is not None:
            try:
                row1 = load_uploaded_patient(uploaded1)
                st.success("CSV loaded successfully.")
                st.dataframe(row1, use_container_width=True)
            except Exception as e:
                st.error(str(e))

    if row1 is not None and st.button("Run Study 1 prediction", use_container_width=True):
        probs = model1.predict_proba(row1)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = row1.copy()
        results["probability_mio_improvement_ge_10mm"] = probs
        results["predicted_class"] = ["Responder" if p == 1 else "Non-responder" for p in preds]

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)

        local_df1 = explain_single_prediction(model1, row1.iloc[[0]])

        st.subheader("Explanation")
        render_local_shap(local_df1, "Local SHAP explanation for first patient")
        st.write(
            plain_language_explanation(
                local_df1,
                positive_label="meaningful MIO improvement"
            )
        )

with tabs[1]:
    st.subheader("Study 2: Predict advanced Wilkes stage")
    mode2 = st.selectbox("Input method", ["Manual entry", "Upload patient CSV"], key="study2_mode")

    row2 = None
    if mode2 == "Manual entry":
        row2 = build_manual_input("s2", gender_options, site_options, diet_options, meds_options)
    else:
        uploaded2 = st.file_uploader("Upload patient CSV for Study 2", type=["csv"], key="u2")
        if uploaded2 is not None:
            try:
                row2 = load_uploaded_patient(uploaded2)
                st.success("CSV loaded successfully.")
                st.dataframe(row2, use_container_width=True)
            except Exception as e:
                st.error(str(e))

    if row2 is not None and st.button("Run Study 2 prediction", use_container_width=True):
        probs = model2.predict_proba(row2)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = row2.copy()
        results["probability_advanced_stage_iv_v"] = probs
        results["predicted_class"] = ["Advanced stage" if p == 1 else "Lower stage" for p in preds]

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)

        local_df2 = explain_single_prediction(model2, row2.iloc[[0]])

        st.subheader("Explanation")
        render_local_shap(local_df2, "Local SHAP explanation for first patient")
        st.write(
            plain_language_explanation(
                local_df2,
                positive_label="advanced stage disease"
            )
        )

with tabs[2]:
    st.subheader("Required CSV columns")
    template = pd.DataFrame([{
        "age_years": 35,
        "gender_clean": "F",
        "site_clean": "RT",
        "diet_preop_clean": "soft",
        "meds_preop_clean": "yes",
        "pre_mio_mm": 28,
        "pre_mahan_dir_present": 1,
        "pre_joint_noise_present": 1,
        "pre_muscle_pain_present": 1,
        "pre_joint_pain_present": 1,
    }])

    st.write("Your uploaded CSV must include these columns exactly:")
    st.dataframe(template, use_container_width=True)

    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="tmj_patient_template.csv",
        mime="text/csv"
    )
