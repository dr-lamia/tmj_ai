import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
summary = json.loads((BASE / "model_summary.json").read_text())
data = pd.read_csv(BASE / "tmj_clean_master_deidentified.csv")
model1 = joblib.load(BASE / "model_study1_mio_improvement.joblib")
model2 = joblib.load(BASE / "model_study2_advanced_stage.joblib")

PRETTY = {
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

@st.cache_data
def load_global_importance(study: str):
    return pd.read_csv(BASE / "shap_outputs" / f"{study}_global_importance.csv")

def base_feature(name: str) -> str:
    if name.startswith("num__"):
        return name.split("num__", 1)[1]
    if name.startswith("cat__"):
        rest = name.split("cat__", 1)[1]
        for p in ["gender_clean_", "site_clean_", "diet_preop_clean_", "meds_preop_clean_"]:
            if rest.startswith(p):
                return p[:-1]
        return rest
    return name

def aggregate_local_shap(pipe, row: pd.DataFrame) -> pd.DataFrame:
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    Xt = pre.transform(row)
    feature_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xt)
    if isinstance(shap_values, list):
        s = shap_values[1][0]
    elif getattr(shap_values, "ndim", None) == 3:
        s = shap_values[0, :, 1]
    else:
        s = shap_values[0]
    out = pd.DataFrame({
        "feature_raw": feature_names,
        "feature": [base_feature(x) for x in feature_names],
        "shap_value": s,
    })
    agg = out.groupby("feature", as_index=False)["shap_value"].sum()
    agg["abs_val"] = agg["shap_value"].abs()
    agg["feature_pretty"] = agg["feature"].map(PRETTY).fillna(agg["feature"])
    return agg.sort_values("abs_val", ascending=False)

def plot_bar(df: pd.DataFrame, value_col: str, title: str, top_n: int = 10):
    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(plot_df["feature_pretty"], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP value|" if value_col == "mean_abs_shap" else "SHAP contribution")
    plt.tight_layout()
    return fig

st.set_page_config(page_title="TMJ AI Studio", page_icon="🦷", layout="wide")
st.title("TMJ AI Studio")
st.caption("Cleaned dataset + baseline models + SHAP explanations built from the uploaded TMJ arthroscopy spreadsheet")

with st.sidebar:
    st.header("Quick metrics")
    st.write(f"Study 1 AUC: {summary['study1']['roc_auc']:.3f}")
    st.write(f"Study 2 AUC: {summary['study2']['roc_auc']:.3f}")

overview_tab, study1_tab, study2_tab, data_tab = st.tabs(["Overview", "Study 1 Predictor", "Study 2 Predictor", "Data Explorer"])

with overview_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Study 1")
        st.write("Predict clinically meaningful postoperative MIO improvement (>=10 mm).")
        st.json(summary['study1'])
        global1 = load_global_importance("study1")
        st.pyplot(plot_bar(global1.assign(feature_pretty=global1['feature_pretty']), "mean_abs_shap", "Study 1 global SHAP importance"))
        st.dataframe(global1[["feature_pretty", "mean_abs_shap"]].head(10), use_container_width=True)
    with c2:
        st.subheader("Study 2")
        st.write("Predict advanced Wilkes stage (IV-V) vs II-III.")
        st.json(summary['study2'])
        global2 = load_global_importance("study2")
        st.pyplot(plot_bar(global2.assign(feature_pretty=global2['feature_pretty']), "mean_abs_shap", "Study 2 global SHAP importance"))
        st.dataframe(global2[["feature_pretty", "mean_abs_shap"]].head(10), use_container_width=True)

def build_input(prefix: str) -> pd.DataFrame:
    age = st.number_input("Age (years)", min_value=5.0, max_value=100.0, value=35.0, step=1.0, key=f"{prefix}_age")
    gender = st.selectbox("Gender", ["F", "M"], key=f"{prefix}_gender")
    site = st.selectbox("Site", ["LT", "RT", "BL"], key=f"{prefix}_site")
    diet = st.selectbox("Pre-op diet", ["regular", "soft", "regular_comp", "rc", "unknown"], key=f"{prefix}_diet")
    meds = st.selectbox("Meds pre-op", ["yes", "no", "unknown", "mixed"], key=f"{prefix}_meds")
    pre_mio = st.number_input("Pre-op MIO (mm)", min_value=0.0, max_value=80.0, value=30.0, step=1.0, key=f"{prefix}_mio")
    pre_mahan = st.selectbox("Directional limitation present", [0.0, 1.0], format_func=lambda x: "No" if x == 0.0 else "Yes", key=f"{prefix}_mahan")
    joint_noise = st.selectbox("Joint noise present", [0.0, 1.0], format_func=lambda x: "No" if x == 0.0 else "Yes", key=f"{prefix}_joint_noise")
    muscle_pain = st.selectbox("Muscle pain present", [0.0, 1.0], format_func=lambda x: "No" if x == 0.0 else "Yes", key=f"{prefix}_muscle_pain")
    joint_pain = st.selectbox("Joint pain present", [0.0, 1.0], format_func=lambda x: "No" if x == 0.0 else "Yes", key=f"{prefix}_joint_pain")
    return pd.DataFrame([{
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
    }])

with study1_tab:
    st.subheader("Study 1 predictor")
    row = build_input("s1")
    if st.button("Predict MIO improvement"):
        prob = float(model1.predict_proba(row)[0, 1])
        pred = int(prob >= 0.5)
        st.metric("Probability of >=10 mm MIO gain", f"{prob:.1%}")
        st.success("Predicted responder") if pred else st.warning("Predicted non-responder")
        local1 = aggregate_local_shap(model1, row)
        st.pyplot(plot_bar(local1, "shap_value", "Study 1 local SHAP explanation"))
        st.dataframe(local1[["feature_pretty", "shap_value"]].head(10), use_container_width=True)
        st.dataframe(row, use_container_width=True)

with study2_tab:
    st.subheader("Study 2 predictor")
    row = build_input("s2")
    if st.button("Predict advanced stage"):
        prob = float(model2.predict_proba(row)[0, 1])
        pred = int(prob >= 0.5)
        st.metric("Probability of advanced stage (IV-V)", f"{prob:.1%}")
        st.success("Predicted advanced stage") if pred else st.info("Predicted lower stage")
        local2 = aggregate_local_shap(model2, row)
        st.pyplot(plot_bar(local2, "shap_value", "Study 2 local SHAP explanation"))
        st.dataframe(local2[["feature_pretty", "shap_value"]].head(10), use_container_width=True)
        st.dataframe(row, use_container_width=True)

with data_tab:
    st.subheader("Data explorer")
    default_cols = ['encounter_id','age_years','gender_clean','site_clean','pre_mio_mm','lv_mio_mm','mio_change_mm','wilkes_stage_clean']
    choices = st.multiselect("Columns", data.columns.tolist(), default=default_cols)
    rows = st.slider("Rows to preview", min_value=10, max_value=min(200, len(data)), value=25)
    st.dataframe(data[choices].head(rows), use_container_width=True)
