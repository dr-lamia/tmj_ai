# TMJ AI UI

This folder contains a cleaned TMJ arthroscopy dataset, two trained baseline machine-learning models, and a Streamlit interface.

## Included studies
1. **Study 1**: Predict clinically meaningful postoperative MIO improvement (`>=10 mm`).
2. **Study 2**: Predict advanced Wilkes stage (`IV-V`) versus `II-III`.

## Main files
- `tmj_clean_master.csv`
- `tmj_clean_master_deidentified.csv`
- `study1_mio_model_dataset.csv`
- `study2_stage_model_dataset.csv`
- `model_study1_mio_improvement.joblib`
- `model_study2_advanced_stage.joblib`
- `app.py`
- `data_cleaning_report.md`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Baseline metrics
### Study 1
- Rows: 466
- Positive cases: 162
- AUC: 0.870
- Accuracy: 0.796
- F1: 0.734

### Study 2
- Rows: 457
- Positive cases: 260
- AUC: 0.669
- Accuracy: 0.626
- F1: 0.676

## Notes
- Rows are treated as **encounters**, not perfectly deduplicated patient-level records.
- The app uses only cleaner preoperative variables so the workflow stays clinically interpretable.
- You can later swap the classifier with XGBoost, CatBoost, or a calibrated ensemble without changing the UI layout.
