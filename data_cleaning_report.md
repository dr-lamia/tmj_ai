# TMJ AI dataset cleaning report

## Cleaning actions applied
- Imported the main encounter sheet from the uploaded `.xlsb` file.
- Standardized mixed date formats into clean date columns.
- Normalized demographics: gender and TMJ site.
- Standardized diet and medication fields.
- Converted symptom text into binary presence variables.
- Extracted numeric Wilkes stage from noisy diagnosis text.
- Created two model targets:
  - `mio_improvement_ge_10mm`
  - `advanced_stage_iv_v`
- Preserved raw source text in parallel `*_raw` columns for auditability.
- Saved de-identified and study-specific modeling tables.

## Why encounter-level modeling was kept
The source spreadsheet contains repeated patient IDs that appear to represent repeat surgery events, second-side events, or later encounters. To avoid mixing timelines incorrectly, each spreadsheet row was preserved as an encounter.

## Final baseline model snapshot
### Study 1
- AUC: 0.870
- Accuracy: 0.796
- Recall: 0.809

### Study 2
- AUC: 0.669
- Accuracy: 0.626
- Recall: 0.689
