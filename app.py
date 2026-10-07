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

# Streamlit can rerun app.py in the same Python process after a Git pull.
# If secondary_outcomes.py changed, Python may still hold the previous module
# in memory. Reload it only when the new VAS API is not yet visible.
import importlib
import secondary_outcomes as _secondary_outcomes

_REQUIRED_SECONDARY_API = (
    "VAS_IMPROVEMENT_METRICS",
    "VAS_GLOBAL_SHAP",
    "fit_vas_improvement_model",
    "vas_local_shap",
)
if not all(hasattr(_secondary_outcomes, name) for name in _REQUIRED_SECONDARY_API):
    _secondary_outcomes = importlib.reload(_secondary_outcomes)

FEATURES = _secondary_outcomes.FEATURES
NUMERIC_FEATURES = _secondary_outcomes.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _secondary_outcomes.CATEGORICAL_FEATURES
OUTCOMES = _secondary_outcomes.OUTCOMES
METRICS = _secondary_outcomes.METRICS
GLOBAL_SHAP = _secondary_outcomes.GLOBAL_SHAP
VAS = _secondary_outcomes.VAS
VAS_IMPROVEMENT_METRICS = _secondary_outcomes.VAS_IMPROVEMENT_METRICS
VAS_GLOBAL_SHAP = _secondary_outcomes.VAS_GLOBAL_SHAP
fit_secondary_models = _secondary_outcomes.fit_secondary_models
fit_vas_improvement_model = _secondary_outcomes.fit_vas_improvement_model
local_shap = _secondary_outcomes.local_shap
vas_local_shap = _secondary_outcomes.vas_local_shap

BASE = Path(__file__).parent


def find_file(name):
    p = BASE / name
    if p.exists():
        return p
    p = BASE / "shap_outputs" / name
    if p.exists():
        return p
    raise FileNotFoundError(name)


def build_primary_pipeline():
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES),
    ])
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocessor", pre), ("model", rf)])


@st.cache_resource
def load_models():
    primary_df = pd.read_csv(find_file("study1_mio_model_dataset.csv"))
    y = pd.to_numeric(primary_df["mio_improvement_ge_10mm"], errors="coerce")
    keep = y.notna()
    primary = build_primary_pipeline()
    primary.fit(primary_df.loc[keep, FEATURES], y.loc[keep].astype(int))

    clean_path = find_file("tmj_clean_master_deidentified.csv")
    secondary = fit_secondary_models(clean_path)
    vas_model = fit_vas_improvement_model(clean_path)
    regression = load_or_fit_regression_model(clean_path)
    return primary, secondary, vas_model, regression


@st.cache_data
def load_clean():
    return pd.read_csv(find_file("tmj_clean_master_deidentified.csv"))


def options(df, col, fallback):
    if col not in df:
        return fallback
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
        vas = st.number_input(
            "Preoperative pain VAS",
            0.0, 100.0, 50.0, 1.0,
            help="Study coding used here: 0 = no pain and 100 = worst pain.",
            key=prefix+"vas",
        )
    with c2:
        mio = st.number_input("Preoperative MIO (mm)", 0.0, 80.0, 30.0, 1.0, key=prefix+"mio")
        lim = st.selectbox("Directional limitation", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"lim")
        noise = st.selectbox("Joint noise", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"noise")
        mpain = st.selectbox("Muscle pain", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"mpain")
        jpain = st.selectbox("Joint pain", [0.0,1.0], format_func=lambda x:"No" if x==0 else "Yes", key=prefix+"jpain")

    row = pd.DataFrame([{
        "age_years":age, "gender_clean":sex, "site_clean":site,
        "diet_preop_clean":diet, "meds_preop_clean":meds,
        "pre_mio_mm":mio, "pre_mahan_dir_present":lim,
        "pre_joint_noise_present":noise, "pre_muscle_pain_present":mpain,
        "pre_joint_pain_present":jpain,
    }])
    vas_row = row.copy()
    vas_row["pre_vas"] = float(vas)
    return row, vas_row


def upload_form(upload):
    df = pd.read_csv(upload)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if "pre_vas" not in df.columns:
        raise ValueError("Missing column: pre_vas (0=no pain, 100=worst pain)")
    return df[FEATURES].copy(), df[FEATURES + ["pre_vas"]].copy()


def evidence(auc, primary=False):
    if primary:
        return "Primary model — good internal discrimination"
    if auc >= .80:
        return "Promising exploratory discrimination"
    if auc >= .60:
        return "Limited internal discrimination"
    return "Weak internal discrimination"


def show_local(model, row, label, vas=False):
    try:
        d = (vas_local_shap(model,row) if vas else local_shap(model,row)).head(8)
        st.bar_chart(d.set_index("feature")["shap_value"])
        pos = d[d.shap_value > 0].feature.tolist()
        neg = d[d.shap_value < 0].feature.tolist()
        if pos:
            st.write("Pushes toward **" + label + "**: " + ", ".join(pos[:4]))
        if neg:
            st.write("Pushes away from **" + label + "**: " + ", ".join(neg[:4]))
        st.caption("SHAP explains model attribution, not causation or treatment effect.")
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")


def _probability_band(p):
    if p >= 0.70:
        return "relatively favorable"
    if p >= 0.50:
        return "intermediate"
    return "cautious"


def _patient_shap_drivers(model, row, vas=False, n=3):
    """Return the strongest patient-level model attributions for counselling text."""
    try:
        d = (vas_local_shap(model, row) if vas else local_shap(model, row)).copy()
        toward = d[d["shap_value"] > 0].head(n)["feature"].tolist()
        away = d[d["shap_value"] < 0].head(n)["feature"].tolist()
        return toward, away
    except Exception:
        return [], []


def personalized_counselling(first, row, vas_row, primary_model, vas_model):
    """Deterministic, evidence-weighted counselling summary.

    This is intentionally not a treatment-selection rule. It translates the
    already-displayed predictions into reproducible counselling and follow-up
    points while preserving uncertainty from the internally validated models.
    """
    p_mio = float(first["probability_mio_improvement_ge_10mm"])
    p_vas = float(first["probability_vas_improvement"])
    p_pain = float(first["probability_joint_pain_absent"])
    p_noise = float(first["probability_joint_noise_absent"])
    p_med = float(first["probability_no_medication"])
    p_diet = float(first["probability_regular_diet"])

    band = _probability_band(p_mio)
    if band == "relatively favorable":
        summary = (
            f"The functional forecast is relatively favorable: the model estimates a "
            f"{p_mio*100:.1f}% probability of at least 10 mm MIO improvement. "
            "This is the most clinically reliable prediction in the current system."
        )
    elif band == "intermediate":
        summary = (
            f"The functional forecast is intermediate: the model estimates a "
            f"{p_mio*100:.1f}% probability of at least 10 mm MIO improvement. "
            "Functional benefit is possible, but the prediction should be discussed together with clinical findings."
        )
    else:
        summary = (
            f"The functional forecast is cautious: the model estimates a "
            f"{p_mio*100:.1f}% probability of at least 10 mm MIO improvement. "
            "This should prompt careful expectation-setting rather than a treatment decision."
        )

    strengths = []
    cautions = []
    follow_up = []

    if p_mio >= 0.65:
        strengths.append(f"Substantial MIO improvement: {p_mio*100:.1f}% predicted probability.")
    elif p_mio < 0.50:
        cautions.append(f"Functional improvement is less certain ({p_mio*100:.1f}%).")

    # Medication is the strongest of the secondary models (AUC ~0.818).
    if p_med >= 0.60:
        strengths.append(
            f"No medication requirement: {p_med*100:.1f}% predicted probability "
            "(strongest-performing secondary model)."
        )
    elif p_med < 0.45:
        cautions.append(
            f"Continued medication requirement may remain relevant "
            f"(only {p_med*100:.1f}% probability of being medication-free)."
        )

    # VAS remains patient-important but statistically uncertain.
    if p_vas >= 0.60:
        strengths.append(
            f"Pain-VAS improvement signal: {p_vas*100:.1f}%, but this secondary model remains exploratory."
        )
    elif p_vas < 0.50:
        cautions.append(
            f"Pain relief is uncertain ({p_vas*100:.1f}% probability of VAS improvement); "
            "functional improvement and pain relief should be counselled as separate outcomes."
        )
    else:
        cautions.append(
            f"Pain-VAS prediction is intermediate ({p_vas*100:.1f}%) and remains exploratory."
        )

    # Weaker secondary models are used only to identify counselling/follow-up domains.
    if p_pain < 0.50:
        cautions.append("Persistent joint pain should remain an explicit follow-up concern.")
        follow_up.append("Joint pain presence and pain intensity (VAS)")
    if p_noise < 0.50:
        cautions.append("Persistence of joint noise remains possible.")
        follow_up.append("Joint noise")
    if p_diet < 0.50:
        cautions.append("Return to a regular-consistency diet is uncertain.")
        follow_up.append("Diet progression")
    if p_med < 0.50:
        follow_up.append("Medication requirement")
    if p_mio < 0.65:
        follow_up.append("MIO and functional gain")
    if p_vas < 0.60:
        follow_up.append("Pain VAS trajectory")

    if not strengths:
        strengths.append(
            "No outcome crosses the app's predefined favorable counselling threshold; "
            "the profile should be presented as uncertain rather than unfavorable."
        )
    if not cautions:
        cautions.append(
            "No major low-probability domain was identified, but all estimates remain internally validated research outputs."
        )

    # Preserve order while removing duplicates.
    follow_up = list(dict.fromkeys(follow_up))
    if not follow_up:
        follow_up = ["MIO", "Pain VAS", "Joint symptoms", "Medication use", "Diet progression"]

    func_toward, func_away = _patient_shap_drivers(primary_model, row, vas=False, n=3)
    pain_toward, pain_away = _patient_shap_drivers(vas_model, vas_row, vas=True, n=3)

    return {
        "band": band,
        "summary": summary,
        "strengths": strengths,
        "cautions": cautions,
        "follow_up": follow_up,
        "functional_toward": func_toward,
        "functional_away": func_away,
        "pain_toward": pain_toward,
        "pain_away": pain_away,
    }


def render_personalized_counselling(rec):
    st.markdown("### Personalized counselling recommendation")
    if rec["band"] == "relatively favorable":
        st.success(rec["summary"])
    elif rec["band"] == "intermediate":
        st.info(rec["summary"])
    else:
        st.warning(rec["summary"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Potential strengths**")
        for item in rec["strengths"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Counselling cautions**")
        for item in rec["cautions"]:
            st.markdown(f"- {item}")

    st.markdown("**Suggested follow-up focus**")
    st.write(" · ".join(rec["follow_up"]))

    st.markdown("**Patient-specific model-attributed factors**")
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Functional-response model")
        if rec["functional_toward"]:
            st.write("Toward improvement: " + ", ".join(rec["functional_toward"]))
        if rec["functional_away"]:
            st.write("Away from improvement: " + ", ".join(rec["functional_away"]))
        if not rec["functional_toward"] and not rec["functional_away"]:
            st.write("Patient-level SHAP attribution unavailable.")
    with d2:
        st.caption("VAS-improvement model")
        if rec["pain_toward"]:
            st.write("Toward pain improvement: " + ", ".join(rec["pain_toward"]))
        if rec["pain_away"]:
            st.write("Away from pain improvement: " + ", ".join(rec["pain_away"]))
        if not rec["pain_toward"] and not rec["pain_away"]:
            st.write("Patient-level SHAP attribution unavailable.")

    st.caption(
        "SHAP factors explain this model's prediction; they are not causal risk factors and are not treatment targets."
    )
    st.warning(
        "Research counselling aid only — this section does not recommend performing, withholding, "
        "or changing arthroscopy. Treatment decisions require full clinical assessment and clinician judgment."
    )


st.set_page_config(page_title="TMJ AI Studio", page_icon="🦷", layout="wide")
clean = load_clean()
try:
    primary_model, secondary_models, vas_model, reg_model = load_models()
except Exception as e:
    st.error("Models could not be initialized.")
    st.exception(e)
    st.stop()

st.title("TMJ AI Studio")
st.subheader("Explainable postoperative outcome recommendation system after TMJ arthroscopy")
st.warning(
    "Research decision-support prototype only. Internal validation does not establish clinical readiness; "
    "external validation, recalibration and prospective evaluation are required."
)

with st.sidebar:
    st.header("Model snapshot")
    st.write(f"Functional-response AUC: **{METRICS['Functional response']['auc']:.3f}**")
    st.write(f"VAS-improvement AUC: **{VAS_IMPROVEMENT_METRICS['auc']:.3f}**")
    for cfg in OUTCOMES.values():
        m = METRICS[cfg['summary']]
        st.caption(f"{cfg['label']}: AUC {m['auc']:.3f}")
    st.divider()
    st.caption("Exact MIO regression: MAE 4.77 mm · R² 0.024")
    st.caption("Wilkes-stage prediction is not part of the active app.")

assessment, explain, evidence_tab, template_tab = st.tabs(
    ["Patient Assessment","Explainability","Model Evidence","CSV Template"]
)

with assessment:
    st.write(
        "Primary output: probability of ≥10-mm MIO improvement. Secondary exploratory outcomes include "
        "pain-VAS improvement, joint-pain status, joint-noise status, medication requirement and diet consistency."
    )
    mode = st.radio("Input method",["Manual entry","Upload CSV"],horizontal=True)
    row = None
    vas_row = None
    if mode == "Manual entry":
        row, vas_row = patient_form("a_", clean)
    else:
        up = st.file_uploader("Patient CSV", type=["csv"])
        if up:
            try:
                row, vas_row = upload_form(up)
                st.dataframe(vas_row, width="stretch", hide_index=True)
            except Exception as e:
                st.error(str(e))

    if row is not None and st.button("Run multidimensional assessment", type="primary", width="stretch"):
        primary_p = primary_model.predict_proba(row)[:,1]
        vas_p = vas_model.predict_proba(vas_row)[:,1]
        reg = predict_postop_mio(reg_model,row)

        result = vas_row.copy()
        result["probability_mio_improvement_ge_10mm"] = primary_p
        result["probability_vas_improvement"] = vas_p
        result["vas_improvement_class_0_5"] = np.where(vas_p >= 0.5, "Likely improved", "Not predicted to improve")
        for key in OUTCOMES:
            result["probability_"+key] = secondary_models[key].predict_proba(row)[:,1]
        result["predicted_postop_mio_mm_exploratory"] = reg["predicted_postop_mio_mm"].values
        result["predicted_mio_change_mm_exploratory"] = reg["predicted_mio_change_mm"].values

        st.session_state["last_row"] = row
        st.session_state["last_vas_row"] = vas_row
        st.session_state["last_result"] = result
        first = result.iloc[0]

        st.markdown("### Primary and key secondary patient-centered forecasts")
        c1,c2 = st.columns(2)
        c1.metric("Probability of ≥10-mm MIO improvement", f"{first['probability_mio_improvement_ge_10mm']*100:.1f}%")
        c1.caption("Primary functional-response model · AUC 0.869")

        c2.metric("Probability of VAS pain improvement", f"{first['probability_vas_improvement']*100:.1f}%")
        c2.caption(
            f"{first['vas_improvement_class_0_5']} at threshold 0.50 · exploratory AUC "
            f"{VAS_IMPROVEMENT_METRICS['auc']:.3f} "
            f"(95% CI {VAS_IMPROVEMENT_METRICS['ci'][0]:.3f}–{VAS_IMPROVEMENT_METRICS['ci'][1]:.3f})"
        )
        st.info(
            "VAS improvement is operationally defined as any reduction in postoperative VAS relative to baseline "
            "(0=no pain, 100=worst pain). This model is a major patient-centered research output, but its internal "
            "discrimination is limited and the confidence interval includes 0.50."
        )

        st.markdown("### Secondary postoperative outcome profile")
        cols = st.columns(4)
        for col,(key,cfg) in zip(cols,OUTCOMES.items()):
            p = float(first["probability_"+key])
            m = METRICS[cfg['summary']]
            col.metric(cfg['label'], f"{p*100:.1f}%")
            col.caption(f"AUC {m['auc']:.3f} · {evidence(m['auc'])}")

        rec = personalized_counselling(
            first, row.iloc[[0]], vas_row.iloc[[0]], primary_model, vas_model
        )
        render_personalized_counselling(rec)

        st.markdown("### Exploratory exact MIO")
        c3,c4 = st.columns(2)
        c3.metric("Predicted postoperative MIO", f"{first['predicted_postop_mio_mm_exploratory']:.1f} mm")
        c4.metric("Predicted MIO change", f"{first['predicted_mio_change_mm_exploratory']:+.1f} mm")
        st.caption("Exact-MIO regression remains exploratory (MAE ≈4.77 mm; R² ≈0.024).")
        st.dataframe(result, width="stretch", hide_index=True)

with explain:
    st.write(
        "Global and patient-level SHAP explanations are provided for transparency. "
        "They describe model behavior and do not establish causation."
    )
    global_options = list(GLOBAL_SHAP.keys()) + ["VAS improvement"]
    name = st.selectbox("Global model", global_options)
    vals = VAS_GLOBAL_SHAP if name == "VAS improvement" else GLOBAL_SHAP[name]
    g = pd.DataFrame(vals, columns=["Feature","Mean |SHAP|"])
    st.bar_chart(g.set_index("Feature")["Mean |SHAP|"])
    st.dataframe(g, width="stretch", hide_index=True)

    if "last_row" in st.session_state:
        r = st.session_state["last_row"].iloc[[0]]
        vr = st.session_state["last_vas_row"].iloc[[0]]
        with st.expander("Functional-response explanation", expanded=True):
            show_local(primary_model, r, "≥10-mm MIO improvement")
        with st.expander("VAS-improvement explanation", expanded=True):
            show_local(vas_model, vr, "postoperative VAS improvement", vas=True)
        for key,cfg in OUTCOMES.items():
            with st.expander(cfg['label']):
                show_local(secondary_models[key], r, cfg['positive'])
    else:
        st.info("Run a patient assessment to view patient-level explanations.")

with evidence_tab:
    table = []
    primary_order = ["Functional response"]
    for label in primary_order:
        m = METRICS[label]
        table.append({
            "Outcome":label, "Encounters":m['n'], "Patients":m['patients'],
            "AUC":m['auc'], "95% CI":f"{m['ci'][0]:.3f}–{m['ci'][1]:.3f}",
            "Sensitivity":m['sensitivity'], "Specificity":m['specificity'],
            "Brier":m['brier'], "Interpretation":evidence(m['auc'], True)
        })

    vm = VAS_IMPROVEMENT_METRICS
    table.append({
        "Outcome":"VAS pain improvement", "Encounters":vm['n'], "Patients":vm['patients'],
        "AUC":vm['auc'], "95% CI":f"{vm['ci'][0]:.3f}–{vm['ci'][1]:.3f}",
        "Sensitivity":vm['sensitivity'], "Specificity":vm['specificity'],
        "Brier":vm['brier'], "Interpretation":"Limited and statistically uncertain internal discrimination"
    })

    for label in ["Joint pain absent","Joint noise absent","No medication","Regular-consistency diet"]:
        m = METRICS[label]
        table.append({
            "Outcome":label, "Encounters":m['n'], "Patients":m['patients'],
            "AUC":m['auc'], "95% CI":f"{m['ci'][0]:.3f}–{m['ci'][1]:.3f}",
            "Sensitivity":m['sensitivity'], "Specificity":m['specificity'],
            "Brier":m['brier'], "Interpretation":evidence(m['auc'])
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    st.markdown("### VAS improvement endpoint")
    st.write(
        f"Paired VAS data were available for {vm['n']} encounters from {vm['patients']} patients. "
        f"Using 0=no pain and 100=worst pain, improvement was defined as any postoperative decrease. "
        f"{vm['positive']} encounters ({vm['prevalence']*100:.1f}%) met this endpoint. "
        f"Patient-grouped AUC was {vm['auc']:.3f} "
        f"(95% CI {vm['ci'][0]:.3f}–{vm['ci'][1]:.3f})."
    )
    st.caption(
        "The probability is displayed because pain improvement is a clinically important study concern; "
        "however, the model must remain exploratory because discrimination is limited and the confidence interval includes chance."
    )

    st.markdown("### Scope")
    st.info("Wilkes-stage prediction is not part of the current app. Legacy Wilkes research files may remain in the repository for provenance only.")

with template_tab:
    t = pd.DataFrame([{
        "age_years":35, "gender_clean":"F", "site_clean":"RT",
        "diet_preop_clean":"soft", "meds_preop_clean":"yes",
        "pre_mio_mm":28, "pre_mahan_dir_present":1,
        "pre_joint_noise_present":1, "pre_muscle_pain_present":1,
        "pre_joint_pain_present":1, "pre_vas":70,
    }])
    st.dataframe(t, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV template",
        t.to_csv(index=False).encode(),
        "tmj_patient_template.csv",
        "text/csv"
    )
