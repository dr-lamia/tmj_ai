# TMJ AI Studio

This repository contains a cleaned TMJ arthroscopy dataset, explainable machine-learning models, and a clinician-facing Streamlit research prototype.

## Integrated prediction framework

The project is presented as one integrated study with two principal prediction tasks and one exploratory continuous module:

1. **Primary prognostic classifier** — predicts the probability of clinically meaningful postoperative maximal interincisal opening (MIO) improvement (`>=10 mm`).
2. **Severity stratification** — predicts advanced Wilkes stage (`IV-V`) versus lower stage (`II-III`).
3. **Exploratory postoperative MIO regression** — estimates last-visit postoperative MIO in millimetres. Expected improvement is then calculated as predicted postoperative MIO minus preoperative MIO.

The clinician-facing interface also provides SHAP-based case-level explanations. The system is a research prototype and is not a substitute for clinical judgment.

## Reproducible patient-grouped validation

The manuscript metrics are generated with:

`StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`

using `patient_id` as the grouping variable, pooled out-of-fold probabilities, and a decision threshold of 0.50. This prevents encounters from the same patient appearing in both training and validation folds.

Run:

```bash
python validate_grouped_models.py
```

The script writes the grouped validation metrics and out-of-fold prediction files.

### Functional response classifier

- Encounters: 466
- Unique patients: 451
- Positive cases: 162
- AUC: **0.869** (patient-cluster bootstrap 95% CI 0.836-0.899)
- Accuracy: 0.781
- Sensitivity: 0.802
- Specificity: 0.770
- Precision: 0.650
- F1: 0.718
- Brier score: 0.146
- Calibration intercept: -0.399
- Calibration slope: 1.424

### Severity stratification

- Encounters: 457
- Unique patients: 443
- Advanced-stage cases: 260
- AUC: **0.710** (patient-cluster bootstrap 95% CI 0.660-0.756)
- Accuracy: 0.672
- Sensitivity: 0.700
- Specificity: 0.635
- Precision: 0.717
- F1: 0.708
- Brier score: 0.217
- Calibration intercept: 0.172
- Calibration slope: 1.123

## Sensitivity analysis: role of preoperative MIO

Because the functional-response outcome is defined using change from baseline MIO, preoperative MIO is mathematically related to the endpoint. To assess how much of the predictive signal depended on this variable, the grouped validation was repeated after removing preoperative MIO from the predictor set.

- AUC without preoperative MIO: **0.634**
- Accuracy: 0.620
- F1: 0.475

This substantial performance decrease is important for interpretation: the primary model predicts the probability of achieving the defined change threshold, but the dominance of baseline MIO should not be interpreted as a causal treatment-response mechanism.

## Exploratory exact-MIO regression

The continuous model predicts **last-visit postoperative MIO (`lv_mio_mm`) directly** rather than using `mio_change_mm` as the regression target. Preoperative MIO remains a baseline predictor and expected MIO change is calculated only after prediction:

`expected MIO change = predicted postoperative MIO - preoperative MIO`

Patient-grouped 3-fold validation produced:

- Encounters: 466
- Unique patients: 451
- MAE: **4.77 mm**
- RMSE: **6.60 mm**
- R²: **0.024**

Because the R² was very low, the exact-MIO estimate is intentionally labelled **exploratory** in the interface and should not be treated as a primary validated clinical output. The probability of ≥10 mm improvement remains the principal functional-prognosis output.

A direct change-score regression was explored during development, but its apparent performance depended strongly on preoperative MIO because the change score itself contains the baseline value. It is therefore not presented as an independently validated continuous prognostic model.

## XGBoost benchmark

XGBoost was tested before replacing the Random Forest models using nested patient-grouped validation.

- Functional-response AUC: Random Forest **0.8687** vs XGBoost **0.8751**. The AUC increase was only 0.0064, with a paired patient-cluster bootstrap 95% CI of **-0.0077 to 0.0197**.
- Severity-stratification AUC: Random Forest **0.7097** vs XGBoost **0.6926**.
- Exact postoperative-MIO regression: Random Forest MAE **4.7745 mm**, R² **0.0242** vs XGBoost MAE **4.7683 mm**, R² **0.0231**.

The deployed models were therefore **not replaced**. XGBoost showed only a small, statistically uncertain numerical gain for the primary classifier, performed worse for severity stratification, and did not meaningfully improve exact-MIO regression. Full details are in `xgboost_benchmark.md` and `xgboost_benchmark_results.json`.

## Main files

- `tmj_clean_master_deidentified.csv`
- `study1_mio_model_dataset.csv`
- `study2_stage_model_dataset.csv`
- `model_study1_mio_improvement.joblib`
- `model_study2_advanced_stage.joblib`
- `validate_grouped_models.py`
- `grouped_validation_metrics.json`
- `regression_utils.py`
- `regression_summary.json`
- `xgboost_benchmark.md`
- `xgboost_benchmark_results.json`
- `app.py`
- `model_summary.json`
- `data_cleaning_report.md`
- SHAP output files

## Run the app locally

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

The models and interface require external validation, prospective usability assessment, and evaluation of clinical impact before any clinical deployment.
