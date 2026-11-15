# CarePredict experiment decision log

## Audit and protocol

- The original saved artifact is HistGradientBoosting (`max_iter=120`, `max_depth=6`, `learning_rate=0.08`). Its estimator has no class-weight setting; the legacy notebook trained with positive-class sample weights, but the serialized estimator cannot verify those fit-time weights.
- The legacy notebook calibrated and selected a threshold on the same full validation data, used test labels for permutation importance, and exported to the original baseline path. It is not the selection workflow.
- The original baseline and locked test parquet remain unchanged. The test data has prior evaluation in earlier workflow versions and is not an untouched holdout.
- The new chronological protocol reserves May 30 for calibration, May 31 for model selection, June 1 for threshold selection, and June 2 for a fixed-policy validation audit. Preprocessing is fit only on training data.
- Model families/configurations: 37 candidate rows spanning Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM, CatBoost, the baseline/booking-context/same-day-context feature sets, and the saved original model.
- Candidate selection rule: precision at a 90% recall operating point where attainable; otherwise recall first, then PR-AUC. Calibration is selected by validation Brier score, then log loss and ECE. The final threshold is chosen on June 1 and is not tuned on the June 2 audit or locked test.

## Results

- Selected: **XGBoost / unweighted_depth3 / baseline**
- Calibration: **sigmoid**
- Frozen threshold: **0.145**
- May 31 selection precision/recall/F1: **25.8% / 90.1% / 0.402**
- June 1 threshold-selection precision/recall/F1: **26.5% / 90.5% / 0.410**
- June 2 audit precision/recall/F1: **27.1% / 86.4% / 0.413**
- Locked test: **not decoded or scored in this stage**. Run `python finalize_test.py` once only after reviewing the frozen artifact.

See `model_selection_report.md` and the generated CSV/JSON files for the complete candidate table, calibration, thresholds, audit subgroups, and global validation permutation importance.


## Final locked-test evaluation

- Frozen model hash: `acf33b58c0a103100a8b0ee30832cead8ad5c28c6730d90004e6dccec110ee0c`; locked test hash: `983fed6ff1adc7541e680173169740da35a481c09f8e721ebde90e7958fc8465`.
- Test use: one finalized evaluation in this workflow; earlier versions had prior historical evaluation.
- Selected policy threshold: **0.145**.
- Precision/recall/F1: **27.1% / 89.9% / 0.4161**.
- ROC-AUC / PR-AUC / Brier: **0.7330 / 0.3388 / 0.1354**.
- Confusion counts TN/FP/FN/TP: **6,486/7,921/329/2,939**.
- Same-day n/precision/recall: **6,111/28.7%/26.1%**.
- No model or threshold was changed in response to test results.
