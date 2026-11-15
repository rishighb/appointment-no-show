# CarePredict model-selection report

## Selection protocol

The existing binary `no_show` target and chronological splits are unchanged. Preprocessing is fit on the training split only. May 30 is reserved for calibrator fitting, May 31 for model/configuration/feature-set selection, June 1 for operating-threshold selection, and June 2 for a one-time validation audit. The locked June 3–8 test records and labels are not decoded or scored during this run; only the file hash is checked. The test was evaluated in prior workflow versions, so it is not described as an untouched holdout.

All six requested model families were compared with the same chronological data and the same one-hot feature encoding. Each family has two prespecified configurations; weighting and capacity settings are reported rather than tuned on the test data. `same_day_context` tests log lead time and same-day-by-age/neighbourhood interactions. These are booking-time features; no separate same-day classifier is used.

The selected calibration method minimizes May 31 Brier score, then log loss and ECE. Model selection prioritizes precision at a 90% recall operating point when that target is reached, followed by PR-AUC, same-day recall, Brier score, and inference time. The deployed threshold is selected separately on June 1. June 2 is an audit, not a selection set.

## Selected candidate

- Model: **XGBoost** (unweighted_depth3)
- Feature set: **baseline**
- Calibration: **sigmoid**
- June 1 high-recall threshold: **0.145**
- Recall target met during May 31 model selection: **True**
- June 1 precision / recall / F1: **26.5% / 90.5% / 0.410**
- June 2 audit precision / recall / F1: **27.1% / 86.4% / 0.413**

## Candidate comparison (May 31 selection window)

| model | feature_set | configuration | calibration_method | roc_auc | pr_auc | brier_score | high_recall_precision | high_recall_recall | high_recall_f1 | same_day_recall | fit_seconds | inference_ms_per_1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | baseline | balanced_C0.1 | sigmoid | 0.7010 | 0.3071 | 0.1373 | 0.2443 | 0.9035 | 0.3846 | 0.0000 | 1.2731 | 5.7784 |
| Logistic Regression | baseline | unweighted_C1 | sigmoid | 0.7009 | 0.3039 | 0.1373 | 0.2457 | 0.9022 | 0.3863 | 0.0000 | 1.2219 | 8.0358 |
| Random Forest | baseline | balanced_depth12 | sigmoid | 0.7060 | 0.2931 | 0.1368 | 0.2487 | 0.9022 | 0.3899 | 0.1029 | 9.7921 | 31.4514 |
| Random Forest | baseline | unweighted_depth20 | sigmoid | 0.7071 | 0.2924 | 0.1365 | 0.2551 | 0.9048 | 0.3979 | 0.1618 | 10.5404 | 66.1175 |
| HistGradientBoosting | baseline | weighted_baseline_params | sigmoid | 0.7057 | 0.2960 | 0.1363 | 0.2531 | 0.9009 | 0.3952 | 0.2059 | 5.0615 | 16.3665 |
| HistGradientBoosting | baseline | unweighted_regularized | sigmoid | 0.7068 | 0.2976 | 0.1363 | 0.2491 | 0.9035 | 0.3905 | 0.1765 | 7.3661 | 31.7170 |
| XGBoost | baseline | unweighted_depth3 | sigmoid | 0.7071 | 0.3050 | 0.1361 | 0.2583 | 0.9009 | 0.4015 | 0.2059 | 5.7349 | 11.9614 |
| XGBoost | baseline | balanced_depth5 | sigmoid | 0.7076 | 0.2940 | 0.1362 | 0.2548 | 0.9022 | 0.3974 | 0.1765 | 9.0458 | 15.6609 |
| LightGBM | baseline | unweighted_leaves31 | sigmoid | 0.7050 | 0.2978 | 0.1364 | 0.2545 | 0.9048 | 0.3973 | 0.2059 | 3.1499 | 20.8146 |
| LightGBM | baseline | balanced_leaves63 | sigmoid | 0.7011 | 0.2886 | 0.1368 | 0.2531 | 0.9074 | 0.3958 | 0.2059 | 3.5337 | 22.6082 |
| CatBoost | baseline | unweighted_depth5 | sigmoid | 0.7064 | 0.3016 | 0.1363 | 0.2529 | 0.9061 | 0.3954 | 0.2059 | 6.2206 | 8.5129 |
| CatBoost | baseline | balanced_depth7 | sigmoid | 0.7039 | 0.2962 | 0.1366 | 0.2525 | 0.9048 | 0.3948 | 0.2206 | 8.1140 | 8.4962 |
| Logistic Regression | booking_context | balanced_C0.1 | sigmoid | 0.7000 | 0.3011 | 0.1373 | 0.2446 | 0.9048 | 0.3851 | 0.0882 | 1.8589 | 9.3226 |
| Logistic Regression | booking_context | unweighted_C1 | sigmoid | 0.7002 | 0.2989 | 0.1372 | 0.2450 | 0.9035 | 0.3854 | 0.0882 | 1.6782 | 8.4795 |
| Random Forest | booking_context | balanced_depth12 | sigmoid | 0.7024 | 0.2923 | 0.1370 | 0.2518 | 0.9022 | 0.3937 | 0.0882 | 9.0621 | 42.0281 |
| Random Forest | booking_context | unweighted_depth20 | sigmoid | 0.7029 | 0.2961 | 0.1368 | 0.2496 | 0.9009 | 0.3909 | 0.1029 | 10.5178 | 36.0516 |
| HistGradientBoosting | booking_context | weighted_baseline_params | sigmoid | 0.7048 | 0.2913 | 0.1364 | 0.2532 | 0.9035 | 0.3955 | 0.1912 | 4.5902 | 16.6444 |
| HistGradientBoosting | booking_context | unweighted_regularized | sigmoid | 0.7063 | 0.2955 | 0.1363 | 0.2515 | 0.9048 | 0.3936 | 0.1765 | 9.7288 | 49.5806 |
| XGBoost | booking_context | unweighted_depth3 | sigmoid | 0.7071 | 0.3006 | 0.1362 | 0.2549 | 0.9035 | 0.3976 | 0.2059 | 10.1547 | 16.0146 |
| XGBoost | booking_context | balanced_depth5 | sigmoid | 0.7076 | 0.2941 | 0.1362 | 0.2546 | 0.9022 | 0.3971 | 0.1765 | 6.4736 | 13.8083 |
| LightGBM | booking_context | unweighted_leaves31 | sigmoid | 0.7052 | 0.2999 | 0.1363 | 0.2513 | 0.9022 | 0.3931 | 0.1765 | 3.1372 | 20.6595 |
| LightGBM | booking_context | balanced_leaves63 | sigmoid | 0.7008 | 0.2867 | 0.1369 | 0.2542 | 0.9009 | 0.3966 | 0.2059 | 3.3495 | 21.7076 |
| CatBoost | booking_context | unweighted_depth5 | sigmoid | 0.7059 | 0.2987 | 0.1363 | 0.2514 | 0.9048 | 0.3934 | 0.1765 | 8.2606 | 10.5448 |
| CatBoost | booking_context | balanced_depth7 | sigmoid | 0.7043 | 0.2907 | 0.1365 | 0.2542 | 0.9035 | 0.3968 | 0.1765 | 9.2877 | 8.4114 |
| Logistic Regression | same_day_context | balanced_C0.1 | sigmoid | 0.6994 | 0.2956 | 0.1375 | 0.2457 | 0.9022 | 0.3862 | 0.0882 | 2.8912 | 6.2556 |
| Logistic Regression | same_day_context | unweighted_C1 | sigmoid | 0.6997 | 0.2937 | 0.1373 | 0.2472 | 0.9009 | 0.3880 | 0.0882 | 3.4855 | 9.6689 |
| Random Forest | same_day_context | balanced_depth12 | sigmoid | 0.6998 | 0.2892 | 0.1371 | 0.2519 | 0.9022 | 0.3939 | 0.0882 | 12.8285 | 26.7954 |
| Random Forest | same_day_context | unweighted_depth20 | sigmoid | 0.7021 | 0.2925 | 0.1368 | 0.2488 | 0.9048 | 0.3903 | 0.1176 | 13.3405 | 41.8426 |
| HistGradientBoosting | same_day_context | weighted_baseline_params | sigmoid | 0.7053 | 0.2930 | 0.1363 | 0.2538 | 0.9035 | 0.3962 | 0.1765 | 6.3315 | 18.6504 |
| HistGradientBoosting | same_day_context | unweighted_regularized | sigmoid | 0.7046 | 0.2932 | 0.1363 | 0.2516 | 0.9035 | 0.3936 | 0.1618 | 10.6806 | 30.4509 |
| XGBoost | same_day_context | unweighted_depth3 | sigmoid | 0.7079 | 0.3043 | 0.1360 | 0.2554 | 0.9035 | 0.3983 | 0.2059 | 7.1781 | 14.7371 |
| XGBoost | same_day_context | balanced_depth5 | sigmoid | 0.7066 | 0.2962 | 0.1361 | 0.2544 | 0.9022 | 0.3969 | 0.1765 | 8.1590 | 15.9654 |
| LightGBM | same_day_context | unweighted_leaves31 | sigmoid | 0.7037 | 0.2958 | 0.1366 | 0.2543 | 0.9009 | 0.3967 | 0.1765 | 2.0249 | 19.5215 |
| LightGBM | same_day_context | balanced_leaves63 | sigmoid | 0.7044 | 0.2904 | 0.1365 | 0.2541 | 0.9087 | 0.3972 | 0.2059 | 2.6189 | 21.4134 |
| CatBoost | same_day_context | unweighted_depth5 | sigmoid | 0.7075 | 0.3027 | 0.1361 | 0.2529 | 0.9022 | 0.3951 | 0.1176 | 6.2282 | 11.5006 |
| CatBoost | same_day_context | balanced_depth7 | sigmoid | 0.7082 | 0.2979 | 0.1359 | 0.2559 | 0.9022 | 0.3987 | 0.1912 | 6.7231 | 12.7006 |
| Existing HistGradientBoosting | existing_baseline | preserved_saved_baseline | sigmoid | 0.7054 | 0.2926 | 0.1363 | 0.2543 | 0.9009 | 0.3967 | 0.2206 | nan | 25.5110 |

## Same-day investigation

Same-day appointments are compared with later appointments using lead-time, age, prior-history, missingness, observed base-rate, and mean predicted-probability summaries. The selected model's same-day precision/recall at its May 31 90%-recall operating point and its June 2 audit metrics are recorded in `subgroup_validation.csv`. Performance is interpreted at the overall validation-selected threshold; no separate subgroup model or threshold is deployed.

## Calibration and threshold artifacts

`calibration_comparison.csv` contains uncalibrated, isotonic, and sigmoid Brier score, log loss, ECE, ROC-AUC, and PR-AUC on May 31. `threshold_sweep.csv` contains June 1 precision, recall, F1, confusion counts, and alert rate across thresholds. The high-recall, max-F1, and top-30-per-day strategies are reported separately.

## Caveats

The source cohort is historical 2016 data from Vitória, Brazil. The original model and its serialized preprocessing configuration are preserved as a comparison candidate; exact fit-time sample weights cannot be recovered from its estimator object. The locked test set was accessed in earlier workflow versions. Permutation importance is global, validation-only, and not a per-patient causal explanation. Predictions require human review and do not initiate communications or alter appointments.
