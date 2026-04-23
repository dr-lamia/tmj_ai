import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE = Path(__file__).parent


def find_file(name: str) -> Path:
    """Look first in repo root, then in shap_outputs/."""
    candidates = [
        BASE / name,
        BASE / "shap_outputs" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required file: {name}")


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads(find_file(name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(find_file(name))


@st.cache_resource
def load_model(name: str):
    return joblib.load(find_file(name))


def safe_load_csv(name: str) -> pd.DataFrame:
    try:
        return load_csv(name)
    except Exception:
        return pd.DataFrame()


def safe_load_json(name: str) -> dict:
    try:
        return load_json(name)
    except Exception:
        return {}


def fmt_metric(x) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"


def get_options(df1: pd.DataFrame, df2: pd.DataFrame, col: str, fallback: list[str]) -> list[str]:
    vals = []
    for df in (df1, df2):
        if col in df.columns:
            vals.extend(df[col].astype(str).fillna("nan").tolist())
    cleaned = sorted({v if v not in ("", "None", "none") else "nan" for v in vals})
    return cleaned or fallback


def display_value(label: str) -> str:
    return "Unknown / missing" if label == "nan" else label


def build_input(prefix: str, gender_opts: list[str], site_opts: list[str], diet_opts: list[str], meds_opts: list[str]) -> pd.DataFrame:
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age (years)",
            min_value=5.0,
            max_value=100.0,
            value=35.0,
            step=1.0,
            key=f"{prefix}_age",
        )
        gender = st.selectbox(
            "Gender",
            gender_opts,
            format_func=display_value,
            key=f"{prefix}_gender",
        )
        site = st.selectbox(
            "Site",
            site_opts,
            format_func=display_value,
            key=f"{prefix}_site",
        )
        diet = st.selectbox(
            "Pre-op diet",
            diet_opts,
            format_func=display_value,
            key=f"{prefix}_diet",
        )
        meds = st.selectbox(
            "Meds pre-op",
            meds_opts,
            format_func=display_value,
            key=f"{prefix}_meds",
        )

    with c2:
        pre_mio = st.number_input(
            "Pre-op MIO (mm)",
            min_value=0.0,
            max_value=80.0,
            value=30.0,
            step=1.0,
            key=f"{prefix}_mio",
        )
        pre_mahan = st.selectbox(
            "Directional limitation present",
            [0.0, 1.0],
            format_func=lambda x: "No" if x == 0.0 else "Yes",
            key=f"{prefix}_mahan",
        )
        joint_noise = st.selectbox(
            "Joint noise present",
            [0.0, 1.0],
            format_func=lambda x: "No" if x == 0.0 else "Yes",
            key=f"{prefix}_joint_noise",
        )
        muscle_pain = st.selectbox(
            "Muscle pain present",
            [0.0, 1.0],
            format_func=lambda x: "No" if x == 0.0 else "Yes",
            key=f"{prefix}_muscle_pain",
        )
        joint_pain = st.selectbox(
            "Joint pain present",
            [0.0, 1.0],
            format_func=lambda x: "No" if x == 0.0 else "Yes",
            key=f"{prefix}_joint_pain",
        )

    return pd.DataFrame(
        [
            {
                "age_years": age,
                "gender_clean": gender,
                "site_clean": site,
                "diet_preop_clean": diet,
                "meds_preop_clean": meds,
                "pre_mio_mm": pre_mio,
                "pre_mahan_dir_present": pre_mahan,
                "pre_joint_noise_present": joint_noise,
                "pre_muscle_pain_present": muscle_pain,
                "pre_joint_pain_present": joint_pain,
            }
        ]
    )


def render_global_shap(study: str, title: str):
    st.subheader(title)

    png_name = f"{study}_global_shap_bar.png"
    csv_name = f"{study}_global_importance.csv"

    try:
        st.image(str(find_file(png_name)), use_container_width=True)
    except Exception:
        st.info("Global SHAP image not found.")

    df = safe_load_csv(csv_name)
    if not df.empty:
        cols = [c for c in ["feature_pretty", "mean_abs_shap"] if c in df.columns]
        if cols:
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Global SHAP table not found.")


def render_representative_local_shap(study: str, title: str):
    st.subheader(title)

    png_name = f"{study}_local_shap_bar.png"
    csv_name = f"{study}_representative_local_shap.csv"

    try:
        st.image(str(find_file(png_name)), use_container_width=True)
    except Exception:
        st.info("Representative local SHAP image not found.")

    df = safe_load_csv(csv_name)
    if not df.empty:
        cols = [c for c in ["feature_pretty", "shap_value"] if c in df.columns]
        if cols:
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Representative local SHAP table not found.")


st.set_page_config(page_title="TMJ AI Studio", page_icon="🦷", layout="wide")

summary = safe_load_json("model_summary.json")
shap_summary = safe_load_json("shap_summary.json")

data = load_csv("tmj_clean_master_deidentified.csv")
study1_df = safe_load_csv("study1_mio_model_dataset.csv")
study2_df = safe_load_csv("study2_stage_model_dataset.csv")

model1 = load_model("model_study1_mio_improvement.joblib")
model2 = load_model("model_study2_advanced_stage.joblib")

gender_options = get_options(study1_df, study2_df, "gender_clean", ["F", "M"])
site_options = get_options(study1_df, study2_df, "site_clean", ["LT", "RT", "BL", "nan"])
diet_options = get_options(study1_df, study2_df, "diet_preop_clean", ["regular", "soft", "regular_comp", "rc", "unknown", "nan"])
meds_options = get_options(study1_df, study2_df, "meds_preop_clean", ["yes", "no", "nan"])

st.title("TMJ AI Studio")
st.caption("De-identified cleaned dataset with 2 baseline predictive models built from the TMJ arthroscopy spreadsheet.")

with st.sidebar:
    st.header("Model snapshot")

    s1 = summary.get("study1", {})
    s2 = summary.get("study2", {})

    st.write("**Study 1**")
    st.write(f"AUC: {fmt_metric(s1.get('roc_auc'))}")
    st.write(f"Accuracy: {fmt_metric(s1.get('accuracy'))}")
    st.write(f"F1: {fmt_metric(s1.get('f1'))}")

    st.write("**Study 2**")
    st.write(f"AUC: {fmt_metric(s2.get('roc_auc'))}")
    st.write(f"Accuracy: {fmt_metric(s2.get('accuracy'))}")
    st.write(f"F1: {fmt_metric(s2.get('f1'))}")

    st.divider()

    st.download_button(
        "Download de-identified clean CSV",
        data=find_file("tmj_clean_master_deidentified.csv").read_bytes(),
        file_name="tmj_clean_master_deidentified.csv",
        mime="text/csv",
    )

tabs = st.tabs(
    [
        "Overview",
        "Study 1 Predictor",
        "Study 2 Predictor",
        "SHAP",
        "Data Explorer",
    ]
)

with tabs[0]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Study 1")
        st.write("Predict clinically meaningful postoperative MIO improvement (10 mm or more).")
        if s1:
            st.json(s1)
        else:
            st.info("Study 1 summary not found.")

        if shap_summary.get("study1", {}).get("top_global"):
            st.write("**Top global drivers**")
            st.dataframe(
                pd.DataFrame(shap_summary["study1"]["top_global"]),
                use_container_width=True,
                hide_index=True,
            )

    with c2:
        st.subheader("Study 2")
        st.write("Predict advanced Wilkes stage (IV to V) versus lower stage (II to III).")
        if s2:
            st.json(s2)
        else:
            st.info("Study 2 summary not found.")

        if shap_summary.get("study2", {}).get("top_global"):
            st.write("**Top global drivers**")
            st.dataframe(
                pd.DataFrame(shap_summary["study2"]["top_global"]),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.write(
        "This app uses the de-identified cleaned dataset for exploration and the saved joblib models for prediction. "
        "The SHAP section shows precomputed explanation outputs."
    )

with tabs[1]:
    st.subheader("Study 1 Predictor")
    st.write("Baseline clinical inputs only.")
    row1 = build_input("s1", gender_options, site_options, diet_options, meds_options)

    if st.button("Predict MIO improvement", use_container_width=True):
        prob = float(model1.predict_proba(row1)[0, 1])
        pred = int(prob >= 0.5)

        a, b = st.columns(2)
        with a:
            st.metric("Probability of clinically meaningful improvement", f"{prob:.1%}")
        with b:
            st.metric("Predicted class", "Responder" if pred else "Non-responder")

        if pred:
            st.success("Predicted responder: likely to achieve 10 mm or more MIO gain.")
        else:
            st.warning("Predicted non-responder: less likely to achieve 10 mm or more MIO gain.")

        st.write("**Input record used for prediction**")
        st.dataframe(row1, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Study 2 Predictor")
    st.write("Baseline clinical inputs only.")
    row2 = build_input("s2", gender_options, site_options, diet_options, meds_options)

    if st.button("Predict advanced stage", use_container_width=True):
        prob = float(model2.predict_proba(row2)[0, 1])
        pred = int(prob >= 0.5)

        a, b = st.columns(2)
        with a:
            st.metric("Probability of advanced stage (IV to V)", f"{prob:.1%}")
        with b:
            st.metric("Predicted class", "Advanced stage" if pred else "Lower stage")

        if pred:
            st.success("Predicted advanced stage.")
        else:
            st.info("Predicted lower stage.")

        st.write("**Input record used for prediction**")
        st.dataframe(row2, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("SHAP Explanations")

    c1, c2 = st.columns(2)
    with c1:
        render_global_shap("study1", "Study 1 Global SHAP")
        render_representative_local_shap("study1", "Study 1 Representative Local SHAP")

    with c2:
        render_global_shap("study2", "Study 2 Global SHAP")
        render_representative_local_shap("study2", "Study 2 Representative Local SHAP")

with tabs[4]:
    st.subheader("Data Explorer")

    default_cols = [
        c
        for c in [
            "encounter_id",
            "age_years",
            "gender_clean",
            "site_clean",
            "pre_mio_mm",
            "lv_mio_mm",
            "mio_change_mm",
            "wilkes_stage_clean",
        ]
        if c in data.columns
    ]

    selected_cols = st.multiselect(
        "Columns",
        data.columns.tolist(),
        default=default_cols if default_cols else data.columns.tolist()[:8],
    )

    rows = st.slider(
        "Rows to preview",
        min_value=10,
        max_value=min(200, len(data)),
        value=min(25, len(data)),
        step=5,
    )

    st.dataframe(data[selected_cols].head(rows), use_container_width=True)
