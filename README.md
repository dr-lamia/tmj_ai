# TMJ AI Studio

This repository contains a cleaned TMJ arthroscopy dataset, explainable machine-learning models, and a clinician-facing Streamlit research prototype.

## Integrated prediction framework

The project is presented as one integrated study with two principal prediction tasks and one exploratory continuous module:

1. **Primary prognostic classifier** — predicts the probability of clinically meaningful postoperative maximal interincisal opening (MIO) improvement (`>=10 mm`).
2. **Severity stratification** — predicts advanced Wilkes stage (`IV-V`) versus lower stage (`II-III`).
3. **Exploratory postoperative MIO regression** — estimates last-visit postoperative MIO in millimetres. Expected improvement is then calculated as predicted postoperative MIO minus preoperative MIO.

The clinician-facing interface also provides SHAP-based case-level explanations. The system is a research prototype and is not a substitute for clinical judgment.

## Main files

- `tmj_clean_master_deidentified.csv`
- `study1_mio_model_dataset.csv`
- `study2_stage_model_dataset.csv`
- `model_study1_mio_improvement.joblib`
- `model_study2_advanced_stage.joblib`
- `regression_utils.py`
- `regression_summary.json`
- `app.py`
- `model_summary.json`
- `data_cleaning_report.md`
- SHAP output files

## Classification validation snapshot

The displayed classification metrics are aligned with the patient-grouped internal-validation values used in the manuscript.

### Functional response classifier

- Encounters: 466
- Positive cases: 162
- AUC: 0.864
- Accuracy: 0.796
- Sensitivity: 0.809
- Specificity: 0.789
- F1: 0.734

### Severity stratification

- Encounters: 457
- Advanced-stage cases: 260
- AUC: 0.716
- Accuracy: 0.659
- Sensitivity: 0.673
- Specificity: 0.640
- F1: 0.692

## Exploratory exact-MIO regression

The continuous model predicts **last-visit postoperative MIO (`lv_mio_mm`) directly** rather than using `mio_change_mm` as the regression target. Preoperative MIO remains one of the baseline predictors. The app derives expected MIO change only after prediction:

`expected MIO change = predicted postoperative MIO - preoperative MIO`

The regression pipeline uses the same ten baseline clinical variables and a Random Forest regressor with within-pipeline imputation and categorical one-hot encoding.

Patient-grouped 3-fold internal validation produced:

- Encounters: 466
- Unique patients: 451
- MAE: **4.77 mm**
- RMSE: **6.60 mm**
- R²: **0.024**

Because the R² was very low, the exact-MIO estimate is intentionally labelled **exploratory** in the interface and should not be treated as a primary validated clinical output. The probability of ≥10 mm improvement remains the principal functional-prognosis output.

A change-score regression was also explored during development, but its apparent performance was strongly dependent on preoperative MIO because the change score itself contains the baseline MIO value. For scientific reporting, the project therefore avoids presenting that result as evidence of an independently predictive continuous-outcome model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required patient inputs

- age
- sex
- affected site
- preoperative diet
- preoperative medication status
- preoperative MIO
- directional limitation
- joint noise
- muscle pain
- joint pain

## Important research note

The models and interface require external validation, calibration assessment where applicable, and prospective usability/clinical-impact testing before any clinical deployment.
