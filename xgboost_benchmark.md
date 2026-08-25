# XGBoost benchmark against the Random Forest models

This benchmark was performed before changing the deployed TMJ AI Studio models. The purpose was to determine whether replacing Random Forest with XGBoost produced a meaningful improvement under the same patient-grouped validation framework.

## Validation design

- Classification outer validation: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` grouped by `patient_id`.
- XGBoost tuning: nested 3-fold patient-grouped validation inside each outer training fold, using five prespecified conservative XGBoost parameter sets and selecting by inner-fold AUC.
- Binary decision threshold: 0.50.
- Difference in AUC: paired patient-cluster bootstrap with 3,000 resamples.
- Exact-MIO regression: 3-fold `GroupKFold` by patient, with nested grouped tuning for XGBoost.

## Functional-response classifier

| Metric | Random Forest | Nested XGBoost |
|---|---:|---:|
| AUC | 0.8687 | 0.8751 |
| Accuracy | 0.7811 | 0.8090 |
| Sensitivity | 0.8025 | 0.8210 |
| Specificity | 0.7697 | 0.8026 |
| Precision | 0.6500 | 0.6891 |
| F1 | 0.7182 | 0.7493 |
| Brier score | 0.1462 | 0.1408 |

The observed AUC increase was only **0.0064**. Patient-cluster bootstrap 95% CI for the AUC difference was **-0.0077 to 0.0197**, which includes zero. XGBoost therefore showed a small numerical improvement, but the evidence was not strong enough to justify replacing the Random Forest solely on this internal dataset.

## Severity stratification

| Metric | Random Forest | Nested XGBoost |
|---|---:|---:|
| AUC | 0.7097 | 0.6926 |
| Accuracy | 0.6718 | 0.6521 |
| Sensitivity | 0.7000 | 0.6692 |
| Specificity | 0.6345 | 0.6294 |
| Precision | 0.7165 | 0.7045 |
| F1 | 0.7082 | 0.6864 |
| Brier score | 0.2174 | 0.2230 |

XGBoost performed worse than Random Forest for severity stratification. The observed AUC difference was **-0.0172** (XGBoost minus Random Forest), with a patient-cluster bootstrap 95% CI of **-0.0412 to 0.0051**.

## Exploratory postoperative-MIO regression

| Metric | Random Forest | Nested XGBoost |
|---|---:|---:|
| MAE | 4.7745 mm | 4.7683 mm |
| RMSE | 6.6017 mm | 6.6054 mm |
| R² | 0.0242 | 0.0231 |

The difference was negligible. XGBoost did not solve the limited predictive signal for exact postoperative MIO.

## Decision

**The deployed models were not replaced.** Random Forest remains the production/research-prototype model because XGBoost did not show a clear, robust advantage across the linked prediction tasks. The XGBoost results are retained as a formal algorithm-comparison sensitivity analysis and can be reported in the manuscript or supplementary material.

Detailed numerical results are stored in `xgboost_benchmark_results.json`.
