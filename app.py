from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from regression_utils import load_or_fit_regression_model, predict_postop_mio
from secondary_outcomes import (
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    OUTCOMES, METRICS, GLOBAL_SHAP, VAS,
    fit_secondary_models, local_shap,
)

BASE = Path(__file__).parent


def find_file(name):
    p = BASE / name
    if p.exists(): return p
    p = BASE / "shap_outputs" / name
    if p.exists(): return p
    raise FileNotFoundError(name)


def fmt(x):
    try: return f"{float(x):.3f}"
    except Exception: return "—"


def build_primary_pipeline():
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", min_samples_leaf=4, random_state=42, n_jobs=-1)
    return Pipeline([("preprocessor", pre), ("model", rf)])


@st.cache_resource
def load_models():
    p = pd.read_csv(find_file("study1_mio_model_dataset.csv"))
    y = pd.to_numeric(p["mio_improvement_ge_10mm"], errors="coerce"); keep = y.notna()
    primary = build_primary_pipeline(); primary.fit(p.loc[keep, FEATURES], y.loc[keep].astype(int))
    secondary = fit_secondary_models(find_file("tmj_clean_master_deidentified.csv"))
    regression = load_or_fit_regression_model(find_file("tmj_clean_master_deidentified.csv"))
    return primary, secondary, regression


@st.cache_data
def load_clean(): return pd.read_csv(find_file("tmj_clean_master_deidentified.csv"))


def options(df, col, fallback):
    if col not in df: return fallback
    vals = sorted({str(v) for v in df[col].dropna().tolist() if str(v) not in ("", "None")})
    return vals or fallback


def patient_form(prefix, clean):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 5.0, 100.0, 35.0, 1.0, key=prefix+"age")
        sex = st.selectbox("Sex", options(clean,"gender_clean",["F","M"]), key=prefix+"sex")
        site = st.selectbox("TMJ site", options(clean,"site_clean",["LT","RT","BL"]), key=prefix+"site")
        diet = st.selectbox("Preoperative diet", options(clean,"diet_preop_clean",["regular","soft","rc","regular_comp","unknown"]), key=prefix+"diet")
        meds = st.selectbox("Preoperative medication status", options(clean,"meds_preop_clean",["yes","no","unknown"]), key=prefix+"meds")
    with c2:
        mio = st.number_input("Preoperative MIO (mm)", 0.0, 80.0, 30.0, 1.0, key=prefix+"mio")
        lim = st.selectbox("Directional limitation", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"lim")
        noise = st.selectbox("Joint noise", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"noise")
        mpain = st.selectbox("Muscle pain", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"mpain")
        jpain = st.selectbox("Joint pain", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"jpain")
    with st.expander("Optional research field: preoperative pain VAS"):
        vas = st.number_input("Pain VAS (0=no pain, 100=worst pain)",0.0,100.0,50.0,1.0,key=prefix+"vas")
        st.caption("VAS is recorded for context only and is not used in the deployed models because the historical postoperative VAS field is internally discordant with binary pain status.")
    row = pd.DataFrame([{
        "age_years":age,"gender_clean":sex,"site_clean":site,"diet_preop_clean":diet,"meds_preop_clean":meds,
        "pre_mio_mm":mio,"pre_mahan_dir_present":lim,"pre_joint_noise_present":noise,
        "pre_muscle_pain_present":mpain,"pre_joint_pain_present":jpain,
    }])
    return row, vas


def upload_form(upload):
    df = pd.read_csv(upload); missing=[c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    return df[FEATURES].copy()


def evidence(auc, primary=False):
    if primary: return "Primary model — good internal discrimination"
    if auc >= .80: return "Promising exploratory discrimination"
    if auc >= .60: return "Limited internal discrimination"
    return "Weak internal discrimination"


def show_local(model, row, label, regression=False):
    try:
        d = local_shap(model,row).head(8)
        st.bar_chart(d.set_index("feature")["shap_value"])
        pos=d[d.shap_value>0].feature.tolist(); neg=d[d.shap_value<0].feature.tolist()
        if regression:
            if pos: st.write("Pushes estimated MIO higher: "+", ".join(pos[:4]))
            if neg: st.write("Pushes estimated MIO lower: "+", ".join(neg[:4]))
        else:
            if pos: st.write("Pushes toward **"+label+"**: "+", ".join(pos[:4]))
            if neg: st.write("Pushes away from **"+label+"**: "+", ".join(neg[:4]))
        st.caption("SHAP explains model attribution, not causation or treatment effect.")
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")


st.set_page_config(page_title="TMJ AI Studio", page_icon="🦷", layout="wide")
clean = load_clean()
try:
    primary_model, secondary_models, reg_model = load_models()
except Exception as e:
    st.error("Models could not be initialized."); st.exception(e); st.stop()

st.title("TMJ AI Studio")
st.subheader("Explainable multidimensional outcome forecasting after TMJ arthroscopy")
st.warning("Research decision-support prototype only. Internal validation does not establish clinical readiness; external validation, recalibration and prospective evaluation are required.")

with st.sidebar:
    st.header("Model snapshot")
    st.write(f"Primary MIO-response AUC: **{METRICS['Functional response']['auc']:.3f}**")
    for cfg in OUTCOMES.values():
        m=METRICS[cfg['summary']]; st.caption(f"{cfg['label']}: AUC {m['auc']:.3f}")
    st.divider(); st.caption("Exact MIO regression: MAE 4.77 mm · R² 0.024")
    st.caption("Wilkes-stage prediction has been removed from the active app.")

assessment, explain, evidence_tab, template_tab = st.tabs(["Patient Assessment","Explainability","Model Evidence","CSV Template"])

with assessment:
    st.write("Primary output: probability of ≥10-mm MIO improvement. Secondary outputs: favorable joint-pain, joint-noise, medication and diet status at last visit.")
    mode=st.radio("Input method",["Manual entry","Upload CSV"],horizontal=True)
    row=None; pre_vas=None
    if mode=="Manual entry": row,pre_vas=patient_form("a_",clean)
    else:
        up=st.file_uploader("Patient CSV",type=["csv"])
        if up:
            try: row=upload_form(up); st.dataframe(row,width="stretch",hide_index=True)
            except Exception as e: st.error(str(e))
    if row is not None and st.button("Run multidimensional assessment",type="primary",width="stretch"):
        primary_p=primary_model.predict_proba(row)[:,1]
        reg=predict_postop_mio(reg_model,row)
        result=row.copy(); result["probability_mio_improvement_ge_10mm"]=primary_p
        for key in OUTCOMES: result["probability_"+key]=secondary_models[key].predict_proba(row)[:,1]
        result["predicted_postop_mio_mm_exploratory"]=reg["predicted_postop_mio_mm"].values
        result["predicted_mio_change_mm_exploratory"]=reg["predicted_mio_change_mm"].values
        st.session_state["last_row"]=row; st.session_state["last_result"]=result
        first=result.iloc[0]
        st.markdown("### Primary functional prognosis")
        c1,c2,c3=st.columns(3)
        c1.metric("Probability of ≥10-mm MIO improvement",f"{first['probability_mio_improvement_ge_10mm']*100:.1f}%")
        c2.metric("Exploratory postoperative MIO",f"{first['predicted_postop_mio_mm_exploratory']:.1f} mm")
        c3.metric("Exploratory expected MIO change",f"{first['predicted_mio_change_mm_exploratory']:+.1f} mm")
        st.caption("The exact-MIO estimate is exploratory (MAE ≈4.77 mm; R² ≈0.024).")
        st.markdown("### Secondary postoperative outcome profile")
        cols=st.columns(4)
        for col,(key,cfg) in zip(cols,OUTCOMES.items()):
            p=float(first["probability_"+key]); m=METRICS[cfg['summary']]
            col.metric(cfg['label'],f"{p*100:.1f}%"); col.caption(f"AUC {m['auc']:.3f} · {evidence(m['auc'])}")
        st.dataframe(result,width="stretch",hide_index=True)
        st.info("No-medication status has the strongest secondary discrimination (AUC ≈0.818) but remains exploratory and is based on fewer complete follow-up records. Joint-pain and diet models are weak; joint-noise prediction is limited.")
        st.warning("Postoperative VAS prediction is intentionally suppressed. On the confirmed scale (0=no pain, 100=worst pain), historical VAS values conflict with the binary pain field and final-VAS regression was poor.")

with explain:
    st.write("Global and patient-level SHAP explanations are provided for transparency. They describe model behavior and do not establish causation.")
    name=st.selectbox("Global model",list(GLOBAL_SHAP.keys()))
    g=pd.DataFrame(GLOBAL_SHAP[name],columns=["Feature","Mean |SHAP|"])
    st.bar_chart(g.set_index("Feature")["Mean |SHAP|"]); st.dataframe(g,width="stretch",hide_index=True)
    if "last_row" in st.session_state:
        r=st.session_state["last_row"].iloc[[0]]
        with st.expander("Primary functional-response explanation",expanded=True): show_local(primary_model,r,"≥10-mm MIO improvement")
        for key,cfg in OUTCOMES.items():
            with st.expander(cfg['label']): show_local(secondary_models[key],r,cfg['positive'])
        st.caption("Exact-MIO SHAP is not shown here because it is an exploratory regression with very low R².")
    else: st.info("Run a patient assessment to view patient-level explanations.")

with evidence_tab:
    table=[]
    for label in ["Functional response","Joint pain absent","Joint noise absent","No medication","Regular-consistency diet"]:
        m=METRICS[label]
        table.append({"Outcome":label,"Encounters":m['n'],"Patients":m['patients'],"AUC":m['auc'],"95% CI":f"{m['ci'][0]:.3f}–{m['ci'][1]:.3f}","Sensitivity":m['sensitivity'],"Specificity":m['specificity'],"Brier":m['brier'],"Interpretation":evidence(m['auc'],label=="Functional response")})
    st.dataframe(pd.DataFrame(table),width="stretch",hide_index=True)
    st.markdown("### Pain VAS integrity check")
    st.write(f"Paired direct VAS records: {VAS['n']}. Mean VAS increased from {VAS['pre_mean']:.1f} preoperatively to {VAS['post_mean']:.1f} at last visit on the confirmed 0=no pain to 100=worst pain scale. This conflicts with the binary joint-pain field and requires source-record reconciliation.")
    st.caption(f"Exploratory final-VAS regression: MAE {VAS['mae']:.1f} points · RMSE {VAS['rmse']:.1f} · R² {VAS['r2']:.3f}.")
    st.markdown("### Scope")
    st.info("Wilkes-stage prediction is not part of the current app. Legacy Wilkes research files may remain in the repository for provenance only.")

with template_tab:
    t=pd.DataFrame([{"age_years":35,"gender_clean":"F","site_clean":"RT","diet_preop_clean":"soft","meds_preop_clean":"yes","pre_mio_mm":28,"pre_mahan_dir_present":1,"pre_joint_noise_present":1,"pre_muscle_pain_present":1,"pre_joint_pain_present":1}])
    st.dataframe(t,width="stretch",hide_index=True)
    st.download_button("Download CSV template",t.to_csv(index=False).encode(),"tmj_patient_template.csv","text/csv")
