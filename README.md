# TMJ AI Studio

This repository contains a cleaned TMJ arthroscopy dataset, explainable machine-learning models, and a clinician-facing Streamlit research prototype.

## Integrated prediction framework

The project is presented as one integrated study with three linked prediction outputs:

1. **Functional response classifier** — predicts the probability of clinically meaningful postoperative maximal interincisal opening (MIO) improvement (`>=10 mm`).
2. **Postoperative MIO regression** — predicts the expected postoperative MIO in millimetres. Expected improvement is then calculated as predicted postoperative MIO minus preoperative MIO.
3. **Severity stratification** — predicts advanced Wilkes stage (`IV-V`) versus lower stage (`II-III`).

The clinician-facing interface also provides SHAP-based case-level explanations. The system is a research prototype and is not a substitute for clinical judgment.

## Main files

- `tmj_clean_master_deidentified.csv`
- `study1_mio_model_dataset.csv`
- `study2_stage_model_dataset.csv`
- `model_study1_mio_improvement.joblib`
- `model_study2_advanced_stage.joblib`
- `regression_utils.py`
- `app.py`
- `model_summary.json`
- `data_cleaning_report.md`
- SHAP output files

## Regression design

The continuous model predicts **last-visit postoperative MIO (`lv_mio_mm`) directly** rather than using `mio_change_mm` as the regression target. Preoperative MIO remains one of the baseline predictors. The app derives expected MIO change only after prediction:

`expected MIO change = predicted postoperative MIO - preoperative MIO`

The regression pipeline uses the same ten baseline clinical variables used by the classification models and a Random Forest regressor with within-pipeline imputation and categorical one-hot encoding.

Internal regression validation is implemented with **3-fold GroupKFold using patient ID as the grouping variable**, so encounters belonging to the same patient are not divided between training and validation folds. The app calculates and displays MAE, RMSE, and R² for this module.

## Classification validation snapshot

The displayed classification metrics are aligned with patient-grouped 3-fold internal validation used for the manuscript.

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

The regression model and interface should undergo external validation, calibration assessment, and prospective usability/clinical-impact testing before any clinical deployment.
