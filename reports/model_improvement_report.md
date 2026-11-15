# CarePredict model improvement report

## Frozen model and validation-selected policy

- Model: **XGBoost** (unweighted_depth3)
- Feature set: **baseline**
- Calibration: **sigmoid**
- Decision threshold: **0.145**, selected on June 1 validation data.
- Calibration fit: May 30; model/configuration selection: May 31; threshold selection: June 1; validation audit: June 2.
- The locked June 3–8, 2016 test set was accessed only after the candidate, calibrator, and policy were frozen in this run. It was evaluated in prior workflow versions, so it is not an untouched research holdout.
- The test set did not influence model, feature, calibration, or threshold selection.

## Final locked-test metrics

| Metric | Value |
|---|---:|
| Appointments | 17,675 |
| Observed no-shows | 3,268 (18.5%) |
| Precision | 27.1% |
| Recall | 89.9% |
| F1-score | 0.4161 |
| ROC-AUC | 0.7330 |
| PR-AUC | 0.3388 |
| Brier score | 0.1354 |
| Log loss | 0.4207 |
| Confusion matrix (TN, FP, FN, TP) | 6,486, 7,921, 329, 2,939 |

## Operating strategy trade-offs

| Strategy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation-selected high-recall threshold | 27.1% | 89.9% | 0.4161 | 6,486 | 7,921 | 329 | 2,939 |
| Validation-selected balanced (max F1) | 29.7% | 75.0% | 0.4258 | 8,613 | 5,794 | 817 | 2,451 |
| Resource-efficient (top 30/day) | 46.7% | 1.7% | 0.0331 | 14,343 | 64 | 3,212 | 56 |

High-recall favors detecting more no-shows and increases false-positive alerts. Balanced maximizes validation F1. Resource-efficient prioritizes up to 30 appointments per day; that capacity is illustrative, and its dynamic ranking is not a fixed threshold.

## Same-day and subgroup performance

Same-day: 6,111 appointments; 253 no-shows (4.1%); precision 28.7%; recall 26.1%; TN/FP/FN/TP 5,694/164/187/66.

Detailed age, lead-time, and neighbourhood metrics are in `subgroup_test.csv`. They are descriptive and should not be interpreted as causal or as evidence of equal performance across groups.

## Calibration and feature interpretation

The selected calibration method was chosen on the May 31 validation window using Brier score, log loss, and ECE. The calibration comparison and genuine validation reliability curve are in `calibration_comparison.csv` and `validation_curves.json`. The locked-test reliability curve is included in model metadata for final evaluation only.

Global permutation importance is measured on the May 31 model-selection validation window using average-precision drop. It is not a per-patient explanation. The system does not automatically contact patients, cancel appointments, or modify schedules.

## Limitations

Historical 2016 data from Vitória, Brazil may not represent current operations or another hospital. The binary outcome does not distinguish cancellations from no-shows. The locked test had historical prior use in earlier iterations. Same-day, age, lead-time, and neighbourhood performance require ongoing human review. SMS, clinical, scholarship, and other sensitive attributes were not added.
