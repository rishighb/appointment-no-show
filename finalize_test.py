from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from model_pipeline import (
    add_booking_features,
    calibrate_probabilities,
    evaluate_probabilities,
    validate_splits,
)
from train_model import decision_metrics, subgroup_metrics

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data" / "processed"
MODEL_DIR = PROJECT_ROOT.parent / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "model_v3.joblib"
METADATA_PATH = MODEL_DIR / "model_v3_metadata.json"
TEST_PATH = DATA_DIR / "test.parquet"
LOCK_PATH = REPORT_DIR / "locked_test_evaluation.json"
PREDICTIONS_PATH = REPORT_DIR / "historical_predictions.parquet"
DECISION_LOG_PATH = REPORT_DIR / "decision_log.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top_k_predictions(
    frame: pd.DataFrame, scores: np.ndarray, capacity: int
) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=np.int8)
    days = pd.to_datetime(frame["appointment_day"]).dt.normalize()
    for _, positions in frame.groupby(days, sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        order = positions[np.argsort(scores[positions])[::-1]]
        predictions[order[: min(capacity, len(order))]] = 1
    return predictions


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(
            f"Locked-test evaluation has already been initiated; refusing to score again: {LOCK_PATH}"
        )
    for path in (MODEL_PATH, METADATA_PATH, TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required final-evaluation input is missing: {path}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != 3:
        raise ValueError("Only a frozen version-3 candidate can be finalized.")
    model_hash = sha256_file(MODEL_PATH)
    if metadata.get("model_hash") != model_hash:
        raise RuntimeError("Frozen model hash does not match its validation metadata.")
    test_hash = sha256_file(TEST_PATH)
    expected_test_hash = metadata.get("source_hashes", {}).get(str(TEST_PATH))
    if not expected_test_hash or expected_test_hash != test_hash:
        raise RuntimeError("Locked test data hash differs from the pre-selection snapshot.")
    source_paths = [
        DATA_DIR / "train.parquet",
        DATA_DIR / "val.parquet",
        MODEL_DIR / "model.joblib",
    ]
    for split_path in source_paths:
        expected_hash = metadata.get("source_hashes", {}).get(str(split_path))
        if not split_path.is_file() or expected_hash != sha256_file(split_path):
            raise RuntimeError(
                f"The frozen input differs from the model-selection run: {split_path}"
            )

    lock = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": model_hash,
        "test_hash": test_hash,
        "note": "The locked test is not to be re-scored for this frozen artifact.",
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    train_integrity = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=["appointment_id", "appointment_day", "no_show"],
    )
    validation_integrity = pd.read_parquet(
        DATA_DIR / "val.parquet",
        columns=["appointment_id", "appointment_day", "no_show"],
    )
    raw_test = pd.read_parquet(TEST_PATH)
    validate_splits(train_integrity, validation_integrity, raw_test)
    test = add_booking_features(raw_test).reset_index(drop=True)
    labels = test["no_show"].to_numpy(dtype=np.int8)
    bundle = joblib.load(MODEL_PATH)
    matrix = bundle["preprocessor"].transform(
        test[bundle["modeling_features"]]
    )
    raw_scores = bundle["model"].predict_proba(matrix)[:, 1]
    scores = calibrate_probabilities(raw_scores, bundle["calibrator"])
    threshold = float(bundle["optimal_threshold"])
    predictions = (scores >= threshold).astype(np.int8)
    high_recall = decision_metrics(labels, scores, predictions, threshold)
    high_recall["strategy"] = "Validation-selected high-recall threshold"

    threshold_strategies = metadata["validation"]["threshold_strategies"]
    balanced_row = next(
        row
        for row in threshold_strategies
        if row["strategy"] == "Balanced precision-recall (max F1)"
    )
    balanced_threshold = float(balanced_row["threshold"])
    balanced_predictions = (scores >= balanced_threshold).astype(np.int8)
    balanced = decision_metrics(
        labels, scores, balanced_predictions, balanced_threshold
    )
    balanced["strategy"] = "Validation-selected balanced (max F1)"

    resource_predictions = top_k_predictions(test, scores, capacity=30)
    resource = decision_metrics(labels, scores, resource_predictions)
    resource.update(
        {
            "strategy": "Resource-efficient (top 30/day)",
            "daily_capacity": 30,
        }
    )
    strategies = [high_recall, balanced, resource]

    same_day_mask = test["is_same_day"].to_numpy() == 1
    same_day = decision_metrics(
        labels[same_day_mask],
        scores[same_day_mask],
        predictions[same_day_mask],
        threshold,
    )
    same_day.update(
        {
            "no_shows": int(labels[same_day_mask].sum()),
            "no_show_rate": float(labels[same_day_mask].mean()),
        }
    )
    subgroups = subgroup_metrics(test, labels, scores, predictions)
    observed, predicted = calibration_curve(
        labels, scores, n_bins=10, strategy="quantile"
    )
    test_result = {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "n": int(len(test)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "threshold": threshold,
        "high_recall": high_recall,
        "strategies": strategies,
        "same_day": same_day,
        "subgroups": subgroups,
        "calibration_curve": {
            "mean_predicted": np.asarray(predicted, dtype=float).tolist(),
            "observed_fraction": np.asarray(observed, dtype=float).tolist(),
        },
        "brier_score": float(evaluate_probabilities(labels, scores, threshold)["brier_score"]),
        "log_loss": float(evaluate_probabilities(labels, scores, threshold)["log_loss"]),
        "test_data_hash": test_hash,
        "model_hash": model_hash,
    }
    prediction_frame = pd.DataFrame(
        {
            "appointment_id": test["appointment_id"].to_numpy(),
            "no_show_prob": scores,
            "predicted_no_show": predictions,
        }
    )
    temp_predictions = PREDICTIONS_PATH.with_suffix(".parquet.tmp")
    prediction_frame.to_parquet(temp_predictions, index=False)
    temp_predictions.replace(PREDICTIONS_PATH)
    test_result["prediction_artifact_hash"] = sha256_file(PREDICTIONS_PATH)
    if sha256_file(TEST_PATH) != test_hash or sha256_file(MODEL_PATH) != model_hash:
        raise RuntimeError("The locked test or frozen model changed during final evaluation.")

    metadata["test"] = test_result
    temp_metadata = METADATA_PATH.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp_metadata.replace(METADATA_PATH)
    (REPORT_DIR / "test_metrics.json").write_text(
        json.dumps(test_result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(strategies).drop(
        columns=["confusion_matrix"], errors="ignore"
    ).to_csv(REPORT_DIR / "threshold_strategies_test.csv", index=False)
    pd.DataFrame(subgroups).to_csv(REPORT_DIR / "subgroup_test.csv", index=False)

    report_path = REPORT_DIR / "model_improvement_report.md"
    report_path.write_text(
        f"""# CarePredict model improvement report

## Frozen model and validation-selected policy

- Model: **{metadata['selected_model']['model']}** ({metadata['selected_model']['configuration']})
- Feature set: **{metadata['selected_model']['feature_set']}**
- Calibration: **{metadata['selected_model']['calibration_method']}**
- Decision threshold: **{threshold:.3f}**, selected on June 1 validation data.
- Calibration fit: May 30; model/configuration selection: May 31; threshold selection: June 1; validation audit: June 2.
- The locked June 3–8, 2016 test set was accessed only after the candidate, calibrator, and policy were frozen in this run. It was evaluated in prior workflow versions, so it is not an untouched research holdout.
- The test set did not influence model, feature, calibration, or threshold selection.

## Final locked-test metrics

| Metric | Value |
|---|---:|
| Appointments | {test_result['n']:,} |
| Observed no-shows | {test_result['positive_count']:,} ({test_result['positive_rate']:.1%}) |
| Precision | {high_recall['precision']:.1%} |
| Recall | {high_recall['recall']:.1%} |
| F1-score | {high_recall['f1']:.4f} |
| ROC-AUC | {high_recall['roc_auc']:.4f} |
| PR-AUC | {high_recall['pr_auc']:.4f} |
| Brier score | {high_recall['brier_score']:.4f} |
| Log loss | {high_recall['log_loss']:.4f} |
| Confusion matrix (TN, FP, FN, TP) | {high_recall['tn']:,}, {high_recall['fp']:,}, {high_recall['fn']:,}, {high_recall['tp']:,} |

## Operating strategy trade-offs

| Strategy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| {high_recall['strategy']} | {high_recall['precision']:.1%} | {high_recall['recall']:.1%} | {high_recall['f1']:.4f} | {high_recall['tn']:,} | {high_recall['fp']:,} | {high_recall['fn']:,} | {high_recall['tp']:,} |
| {balanced['strategy']} | {balanced['precision']:.1%} | {balanced['recall']:.1%} | {balanced['f1']:.4f} | {balanced['tn']:,} | {balanced['fp']:,} | {balanced['fn']:,} | {balanced['tp']:,} |
| {resource['strategy']} | {resource['precision']:.1%} | {resource['recall']:.1%} | {resource['f1']:.4f} | {resource['tn']:,} | {resource['fp']:,} | {resource['fn']:,} | {resource['tp']:,} |

High-recall favors detecting more no-shows and increases false-positive alerts. Balanced maximizes validation F1. Resource-efficient prioritizes up to 30 appointments per day; that capacity is illustrative, and its dynamic ranking is not a fixed threshold.

## Same-day and subgroup performance

Same-day: {same_day['n']:,} appointments; {same_day['no_shows']:,} no-shows ({same_day['no_show_rate']:.1%}); precision {same_day['precision']:.1%}; recall {same_day['recall']:.1%}; TN/FP/FN/TP {same_day['tn']:,}/{same_day['fp']:,}/{same_day['fn']:,}/{same_day['tp']:,}.

Detailed age, lead-time, and neighbourhood metrics are in `subgroup_test.csv`. They are descriptive and should not be interpreted as causal or as evidence of equal performance across groups.

## Calibration and feature interpretation

The selected calibration method was chosen on the May 31 validation window using Brier score, log loss, and ECE. The calibration comparison and genuine validation reliability curve are in `calibration_comparison.csv` and `validation_curves.json`. The locked-test reliability curve is included in model metadata for final evaluation only.

Global permutation importance is measured on the May 31 model-selection validation window using average-precision drop. It is not a per-patient explanation. The system does not automatically contact patients, cancel appointments, or modify schedules.

## Limitations

Historical 2016 data from Vitória, Brazil may not represent current operations or another hospital. The binary outcome does not distinguish cancellations from no-shows. The locked test had historical prior use in earlier iterations. Same-day, age, lead-time, and neighbourhood performance require ongoing human review. SMS, clinical, scholarship, and other sensitive attributes were not added.
""",
        encoding="utf-8",
    )
    with DECISION_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            f"""\n\n## Final locked-test evaluation

- Frozen model hash: `{model_hash}`; locked test hash: `{test_hash}`.
- Test use: one finalized evaluation in this workflow; earlier versions had prior historical evaluation.
- Selected policy threshold: **{threshold:.3f}**.
- Precision/recall/F1: **{high_recall['precision']:.1%} / {high_recall['recall']:.1%} / {high_recall['f1']:.4f}**.
- ROC-AUC / PR-AUC / Brier: **{high_recall['roc_auc']:.4f} / {high_recall['pr_auc']:.4f} / {high_recall['brier_score']:.4f}**.
- Confusion counts TN/FP/FN/TP: **{high_recall['tn']:,}/{high_recall['fp']:,}/{high_recall['fn']:,}/{high_recall['tp']:,}**.
- Same-day n/precision/recall: **{same_day['n']:,}/{same_day['precision']:.1%}/{same_day['recall']:.1%}**.
- No model or threshold was changed in response to test results.
"""
        )
    lock.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "report": str(report_path),
        }
    )
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print("Final test metrics:", high_recall)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
