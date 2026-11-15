# CarePredict

CarePredict is a historical hospital appointment no-show decision-support prototype. It is not connected to a live hospital system and must not be used as live scheduling, outreach, or cancellation guidance.

## Data and safeguards

The project expects chronological parquet splits and model artifacts in sibling folders:

- `..\data\processed\train.parquet`
- `..\data\processed\val.parquet`
- `..\data\processed\test.parquet` — the June 3–8, 2016 locked cohort
- `..\models\model.joblib` — preserved original baseline

The binary `no_show` target is unchanged. The workflow retains only booking-time features and prior-visit aggregates that use outcomes known before the current booking. SMS, clinical-condition, scholarship, and other sensitive attributes are excluded. The dashboard uses the historical test cohort as a clearly labelled simulation; it has no live patient or schedule connection and makes no calls, cancellations, or scheduling changes. Appointment times are not present in the source dataset and are not invented for display.

The locked test set was evaluated by earlier workflow versions. It is therefore not an untouched research holdout. The new workflow nevertheless excludes its labels from all model, feature, calibration, and threshold selection and records the source hashes.

## Environment

CatBoost does not provide a compatible wheel for the workspace's Python 3.15 interpreter. Use Python 3.14 for the complete experiment and dashboard environment:

```powershell
uv venv --python 3.14 .venv-carepredict
uv pip install --python .\.venv-carepredict\Scripts\python.exe -r requirements-model-experiments.txt
```

`requirements.txt` pins the application and core ML dependencies. `requirements-model-experiments.txt` adds pinned XGBoost, LightGBM, and CatBoost versions.

## Train and select

From the project directory:

```powershell
.\.venv-carepredict\Scripts\python.exe train_model.py
```

The workflow compares HistGradientBoosting, Random Forest, Logistic Regression, XGBoost, LightGBM, and CatBoost. It uses a common training-only preprocessor and evaluates baseline, booking-context, and same-day interaction feature sets. Two fixed configurations per model family cover capacity and class-weighting variants; these are finite experiments, not exhaustive hyperparameter searches. CatBoost uses the same one-hot encoded inputs as the other candidates.

Validation dates are reserved chronologically:

- May 30: fit calibration methods.
- May 31: select model, feature set, configuration, and calibration by validation metrics.
- June 1: choose the high-recall, max-F1, and top-30-per-day operating strategies.
- June 2: fixed-policy validation audit; do not use this audit to retune.

Preprocessing is fit on training data only. Calibration methods are uncalibrated, isotonic, and sigmoid. Selection prioritizes precision at a 90% recall operating point when attainable, then PR-AUC and supporting metrics. The threshold is chosen on June 1, never on the locked test. Same-day interactions are compared as features; there is no separate same-day classifier.

The script writes `..\models\model_v3.joblib`, `..\models\model_v3_metadata.json`, a validation-only model-selection report, and CSV/JSON diagnostics under `reports\`. It does **not** decode test rows or labels; it hashes the test file to detect source changes, and does not score the test set. It refuses to retrain after a test-finalization lock exists.

## Finalize the test once

Review the model-selection report and frozen candidate, then run:

```powershell
.\.venv-carepredict\Scripts\python.exe finalize_test.py
```

The finalizer verifies frozen model and data hashes, scores the locked test once, and saves the row-level probabilities to `reports\historical_predictions.parquet` for the dashboard. It also writes `reports\locked_test_evaluation.json`, final metrics and subgroup reports, and updates `model_improvement_report.md`. It refuses to score again for that artifact. Do not select or modify the model in response to test results; a new research cycle requires a new untouched holdout.

## Dashboard

After finalization:

```powershell
.\.venv-carepredict\Scripts\python.exe -m streamlit run app.py
```

The dashboard loads the versioned model, exact preprocessor, calibration object, feature columns, and threshold together for hypothetical inputs. Historical appointment probabilities are loaded from the hash-verified one-time finalization artifact rather than re-scored on dashboard reruns. The historical simulation has no fabricated appointment times or heuristic patient-specific feature explanations. It displays measured input context separately from validation-only global permutation importance. Model performance is shown only when generated evaluation metadata and source hashes match.

## Notebook

`01_preprocessing_cleaning_and_eda.ipynb` remains a historical exploration artifact. Its overlapping validation calibration/threshold cells, locked-test diagnostics, old model export, and dashboard generation/deployment cells are marked or disabled. Use `train_model.py` and `finalize_test.py` as the reproducible workflow.

## Tests

```powershell
.\.venv-carepredict\Scripts\python.exe -m unittest discover -s tests -v
```

Reports distinguish observed validation/test metrics from simulated planning calculations. All outputs are decision support only and require human review.
