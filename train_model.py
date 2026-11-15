from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve,
)

from model_pipeline import (
    FEATURE_SETS,
    RANDOM_SEED,
    add_booking_features,
    build_classifier,
    build_preprocessor,
    calibrate_probabilities,
    evaluate_probabilities,
    expected_calibration_error,
    fit_calibrator,
    validate_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data" / "processed"
MODEL_DIR = PROJECT_ROOT.parent / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
BASELINE_MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_PATH = MODEL_DIR / "model_v3.joblib"
METADATA_PATH = MODEL_DIR / "model_v3_metadata.json"
DECISION_LOG_PATH = REPORT_DIR / "decision_log.md"

CALIBRATION_DATE = pd.Timestamp("2016-05-30")
MODEL_SELECTION_DATE = pd.Timestamp("2016-05-31")
THRESHOLD_SELECTION_DATE = pd.Timestamp("2016-06-01")
VALIDATION_AUDIT_DATE = pd.Timestamp("2016-06-02")
RECALL_TARGET = 0.90
DAILY_REVIEW_CAPACITY = 30
THRESHOLD_GRID = np.round(np.arange(0.0, 0.951, 0.005), 3)
MODEL_CONFIGURATIONS = {
    "Logistic Regression": [
        ("balanced_C0.1", {"C": 0.1, "class_weight": "balanced"}, None),
        ("unweighted_C1", {"C": 1.0, "class_weight": None}, None),
    ],
    "Random Forest": [
        (
            "balanced_depth12",
            {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 10,
             "class_weight": "balanced_subsample"},
            None,
        ),
        (
            "unweighted_depth20",
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 10,
             "class_weight": None},
            None,
        ),
    ],
    "HistGradientBoosting": [
        (
            "weighted_baseline_params",
            {"max_iter": 120, "max_depth": 6, "learning_rate": 0.08,
             "min_samples_leaf": 30},
            "balanced",
        ),
        (
            "unweighted_regularized",
            {"max_iter": 200, "max_depth": 6, "learning_rate": 0.05,
             "min_samples_leaf": 40, "l2_regularization": 1.0},
            None,
        ),
    ],
    "XGBoost": [
        ("unweighted_depth3", {"max_depth": 3, "scale_pos_weight": 1.0}, None),
        ("balanced_depth5", {"max_depth": 5}, "balanced"),
    ],
    "LightGBM": [
        ("unweighted_leaves31", {"num_leaves": 31, "class_weight": None}, None),
        ("balanced_leaves63", {"num_leaves": 63, "class_weight": "balanced"}, None),
    ],
    "CatBoost": [
        ("unweighted_depth5", {"depth": 5, "auto_class_weights": None}, None),
        ("balanced_depth7", {"depth": 7, "auto_class_weights": "Balanced"}, None),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for split_name in ("train", "val"):
        path = DATA_DIR / f"{split_name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Required chronological split is missing: {path}")
        frame = add_booking_features(pd.read_parquet(path)).reset_index(drop=True)
        frames.append(frame)

    train, validation = frames
    validate_splits(train, validation)
    return train, validation


def frame_for_date(frame: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame["appointment_day"]).dt.normalize()
    result = frame.loc[dates.eq(day)].reset_index(drop=True)
    if result.empty or result["no_show"].nunique() != 2:
        raise ValueError(f"Validation date {day.date()} is empty or lacks both labels.")
    return result


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    negatives = ~positives
    rows = []
    for threshold in THRESHOLD_GRID:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & positives))
        fp = int(np.sum(predicted & negatives))
        fn = int(np.sum(~predicted & positives))
        tn = int(np.sum(~predicted & negatives))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": (
                    float(2 * tp / (2 * tp + fp + fn))
                    if 2 * tp + fp + fn else 0.0
                ),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "alert_rate": float((tp + fp) / len(labels)),
            }
        )
    return pd.DataFrame(rows)


def choose_high_recall(sweep: pd.DataFrame) -> dict:
    eligible = sweep.loc[sweep["recall"] >= RECALL_TARGET]
    if not eligible.empty:
        chosen = eligible.sort_values("threshold").iloc[-1].to_dict()
        chosen["recall_target_met"] = True
        return chosen
    chosen = sweep.sort_values(
        ["recall", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0].to_dict()
    chosen["recall_target_met"] = False
    return chosen


def decision_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions, dtype=np.int8)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    if np.unique(labels).size == 2:
        result = evaluate_probabilities(labels, scores, 0.5)
    else:
        result = {
            "n": int(len(labels)),
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": float(brier_score_loss(labels, scores)),
            "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
            "ece_10": expected_calibration_error(labels, scores),
            "accuracy": float(accuracy_score(labels, predictions)),
        }
    result.update(
        {
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "alert_rate": float(predictions.mean()),
        }
    )
    if threshold is not None:
        result["threshold"] = float(threshold)
    return result


def strategy_results(
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    sweep: pd.DataFrame,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    high = choose_high_recall(sweep)
    balanced = sweep.sort_values(
        ["f1", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0].to_dict()
    policies = [
        (
            "High recall",
            float(high["threshold"]),
            (scores >= float(high["threshold"])).astype(np.int8),
        ),
        (
            "Balanced precision-recall (max F1)",
            float(balanced["threshold"]),
            (scores >= float(balanced["threshold"])).astype(np.int8),
        ),
    ]
    capacity_predictions = np.zeros(len(frame), dtype=np.int8)
    daily = pd.to_datetime(frame["appointment_day"]).dt.normalize()
    capacity_details = []
    for day, positions in frame.groupby(daily, sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        order = positions[np.argsort(scores[positions])[::-1]]
        chosen = order[: min(DAILY_REVIEW_CAPACITY, len(order))]
        capacity_predictions[chosen] = 1
        capacity_details.append(
            {
                "date": pd.Timestamp(day).date().isoformat(),
                "appointments": int(len(positions)),
                "selected": int(len(chosen)),
                "capacity": DAILY_REVIEW_CAPACITY,
                "lowest_selected_probability": (
                    float(scores[chosen[-1]]) if len(chosen) else None
                ),
            }
        )
    policies.append(("Resource-efficient (top 30/day)", None, capacity_predictions))

    rows, predictions_by_policy = [], {}
    for name, threshold, predictions in policies:
        row = decision_metrics(labels, scores, predictions, threshold)
        row["strategy"] = name
        if name == "High recall":
            row["recall_target"] = RECALL_TARGET
            row["recall_target_met"] = bool(high["recall_target_met"])
        if name.startswith("Resource-efficient"):
            row["daily_capacity"] = DAILY_REVIEW_CAPACITY
            row["capacity_details"] = capacity_details
        rows.append(row)
        predictions_by_policy[name] = predictions
    return rows, predictions_by_policy


def subgroup_metrics(
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> list[dict]:
    rows = []
    for column in ("age_group", "lead_bucket", "neighbourhood"):
        for value, positions in frame.groupby(
            column, observed=True, sort=True
        ).indices.items():
            positions = np.asarray(positions, dtype=int)
            row = decision_metrics(
                labels[positions], scores[positions], predictions[positions]
            )
            row.update({"group": str(value), "group_column": column})
            rows.append(row)
    return rows


def same_day_summary(
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    features: tuple[str, ...],
) -> list[dict]:
    rows = []
    for is_same_day, positions in frame.groupby(
        "is_same_day", observed=True, sort=True
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        selected = frame.iloc[positions]
        rows.append(
            {
                "booking_group": "Same-day" if is_same_day else "More than 0 days",
                "appointments": int(len(positions)),
                "no_shows": int(labels[positions].sum()),
                "no_show_rate": float(labels[positions].mean()),
                "median_age": float(selected["age"].median()),
                "mean_lead_time_days": float(selected["lead_time_days"].mean()),
                "mean_prior_appointments": float(
                    selected["prev_appointment_count"].mean()
                ),
                "mean_prior_no_show_rate": float(
                    selected["prev_no_show_rate"].mean()
                ),
                "mean_predicted_probability": float(scores[positions].mean()),
                "missing_feature_cells": int(selected[list(features)].isna().sum().sum()),
            }
        )
    return rows


def calibration_curves(
    labels: np.ndarray, scores_by_method: dict[str, np.ndarray]
) -> dict:
    curves = {}
    for method, scores in scores_by_method.items():
        observed, predicted = calibration_curve(
            labels, scores, n_bins=10, strategy="quantile"
        )
        curves[method] = {
            "mean_predicted": np.asarray(predicted, dtype=float).tolist(),
            "observed_fraction": np.asarray(observed, dtype=float).tolist(),
        }
    return curves


def feature_importance(
    frame: pd.DataFrame, labels: np.ndarray, candidate: dict
) -> list[dict]:
    sample_count = min(3000, len(frame))
    sample_positions = frame.sample(
        n=sample_count, random_state=RANDOM_SEED
    ).index.to_numpy(dtype=int)
    sample = frame.iloc[sample_positions].reset_index(drop=True)
    sample_labels = labels[sample_positions]
    features = list(candidate["features"])
    baseline_matrix = candidate["preprocessor"].transform(sample[features])
    baseline_scores = candidate["model"].predict_proba(baseline_matrix)[:, 1]
    baseline_ap = float(average_precision_score(sample_labels, baseline_scores))
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for feature in features:
        drops = []
        for _ in range(3):
            shuffled = sample[features].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            transformed = candidate["preprocessor"].transform(shuffled)
            scores = candidate["model"].predict_proba(transformed)[:, 1]
            drops.append(
                baseline_ap - average_precision_score(sample_labels, scores)
            )
        rows.append(
            {
                "feature": feature,
                "mean_average_precision_drop": float(np.mean(drops)),
                "std_average_precision_drop": float(np.std(drops)),
                "validation_sample_size": int(sample_count),
                "repeats": 3,
            }
        )
    return sorted(rows, key=lambda row: row["mean_average_precision_drop"], reverse=True)


def package_versions() -> dict:
    result = {}
    for package in (
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost"
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def candidate_selection_result(
    model: object,
    preprocessor: object,
    feature_names: tuple[str, ...],
    calibrators: dict,
    selection_frame: pd.DataFrame,
    y_selection: np.ndarray,
    raw_scores: np.ndarray,
) -> tuple[dict, list[dict], dict]:
    calibration_results = {}
    for method, calibrator in calibrators.items():
        scores = calibrate_probabilities(raw_scores, calibrator)
        metrics = evaluate_probabilities(y_selection, scores, 0.5)
        calibration_results[method] = {
            "calibrator": calibrator,
            "scores": scores,
            "metrics": metrics,
        }

    chosen_method = min(
        calibration_results,
        key=lambda method: (
            calibration_results[method]["metrics"]["brier_score"],
            calibration_results[method]["metrics"]["log_loss"],
            calibration_results[method]["metrics"]["ece_10"],
        ),
    )
    chosen_scores = calibration_results[chosen_method]["scores"]
    sweep = threshold_sweep(y_selection, chosen_scores)
    high = choose_high_recall(sweep)
    threshold = float(high["threshold"])
    predictions = (chosen_scores >= threshold).astype(np.int8)
    metrics = decision_metrics(y_selection, chosen_scores, predictions, threshold)
    same_day_mask = selection_frame["is_same_day"].to_numpy() == 1
    same_day_metrics = decision_metrics(
        y_selection[same_day_mask],
        chosen_scores[same_day_mask],
        predictions[same_day_mask],
        threshold,
    )

    calibration_rows = []
    for method, result in calibration_results.items():
        calibration_rows.append(
            {
                "method": method,
                "brier_score": result["metrics"]["brier_score"],
                "log_loss": result["metrics"]["log_loss"],
                "ece_10": result["metrics"]["ece_10"],
                "roc_auc": result["metrics"]["roc_auc"],
                "pr_auc": result["metrics"]["pr_auc"],
            }
        )
    return (
        {
            "calibration_method": chosen_method,
            "calibrator": calibration_results[chosen_method]["calibrator"],
            "selection_scores": chosen_scores,
            "selection_sweep": sweep,
            "selection_threshold": threshold,
            "selection_target_met": bool(high["recall_target_met"]),
            "selection_metrics": metrics,
            "same_day_metrics": same_day_metrics,
            "calibration_results": calibration_results,
            "calibration_rows": calibration_rows,
        },
        calibration_rows,
        calibration_curves(
            y_selection,
            {method: value["scores"] for method, value in calibration_results.items()},
        ),
    )


def train_candidates(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    original_bundle: dict,
    candidate_dir: Path,
) -> tuple[list[dict], dict]:
    y_train = train["no_show"].to_numpy(dtype=np.int8)
    y_calibration = calibration["no_show"].to_numpy(dtype=np.int8)
    y_selection = selection["no_show"].to_numpy(dtype=np.int8)
    class_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    rows = []
    calibrations_by_candidate = {}
    candidate_number = 0

    experiments = [
        (name, tuple(features), False)
        for name, features in FEATURE_SETS.items()
    ]
    experiments.append(
        (
            "existing_baseline",
            tuple(original_bundle["modeling_features"]),
            True,
        )
    )

    for feature_set, features, is_existing in experiments:
        configurations = (
            [("preserved_saved_baseline", {}, None)]
            if is_existing
            else [
                config
                for family in MODEL_CONFIGURATIONS.values()
                for config in family
            ]
        )
        for config_name, params, weight_mode in configurations:
            model_name = (
                "Existing HistGradientBoosting"
                if is_existing
                else next(
                    family
                    for family, options in MODEL_CONFIGURATIONS.items()
                    if any(option[0] == config_name for option in options)
                )
            )
            if is_existing:
                preprocessor = original_bundle["preprocessor"]
                model = original_bundle["model"]
                fit_seconds = None
                candidate_features = features
            else:
                preprocessor = build_preprocessor(features)
                model = build_classifier(model_name, params)
                started = time.perf_counter()
                x_train = preprocessor.fit_transform(train[list(features)])
                fit_kwargs = {}
                if weight_mode == "balanced":
                    fit_kwargs["sample_weight"] = np.where(
                        y_train == 1, class_ratio, 1.0
                    )
                model.fit(x_train, y_train, **fit_kwargs)
                fit_seconds = time.perf_counter() - started
                candidate_features = features

            x_calibration = preprocessor.transform(calibration[list(candidate_features)])
            raw_calibration_scores = model.predict_proba(x_calibration)[:, 1]
            calibrators = {
                method: fit_calibrator(method, raw_calibration_scores, y_calibration)
                for method in ("uncalibrated", "isotonic", "sigmoid")
            }
            started = time.perf_counter()
            for _ in range(3):
                x_selection = preprocessor.transform(selection[list(candidate_features)])
                raw_selection_scores = model.predict_proba(x_selection)[:, 1]
            inference_seconds = (time.perf_counter() - started) / 3
            inference_ms_per_1000 = inference_seconds / len(selection) * 1_000_000

            selected, calibration_rows, curves = candidate_selection_result(
                model,
                preprocessor,
                candidate_features,
                calibrators,
                selection,
                y_selection,
                raw_selection_scores,
            )
            candidate_number += 1
            artifact_path = candidate_dir / f"candidate_{candidate_number:03d}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "preprocessor": preprocessor,
                    "features": list(candidate_features),
                    "calibrators": calibrators,
                    "chosen_calibrator": selected["calibrator"],
                    "chosen_method": selected["calibration_method"],
                },
                artifact_path,
            )
            calibration_key = f"{model_name}|{feature_set}|{config_name}"
            calibrations_by_candidate[calibration_key] = {
                "rows": calibration_rows,
                "curves": curves,
                "selected_method": selected["calibration_method"],
            }
            model_params = (
                model.get_params()
                if hasattr(model, "get_params")
                else {}
            )
            row = {
                "model": model_name,
                "feature_set": feature_set,
                "configuration": config_name,
                "calibration_method": selected["calibration_method"],
                "fit_seconds": fit_seconds,
                "inference_ms_per_1000": inference_ms_per_1000,
                "roc_auc": selected["selection_metrics"]["roc_auc"],
                "pr_auc": selected["selection_metrics"]["pr_auc"],
                "brier_score": selected["selection_metrics"]["brier_score"],
                "log_loss": selected["selection_metrics"]["log_loss"],
                "ece_10": selected["selection_metrics"]["ece_10"],
                "selection_threshold_at_90_recall": selected["selection_threshold"],
                "selection_recall_target_met": selected["selection_target_met"],
                "high_recall_precision": selected["selection_metrics"]["precision"],
                "high_recall_recall": selected["selection_metrics"]["recall"],
                "high_recall_f1": selected["selection_metrics"]["f1"],
                "high_recall_tn": selected["selection_metrics"]["tn"],
                "high_recall_fp": selected["selection_metrics"]["fp"],
                "high_recall_fn": selected["selection_metrics"]["fn"],
                "high_recall_tp": selected["selection_metrics"]["tp"],
                "same_day_precision": selected["same_day_metrics"]["precision"],
                "same_day_recall": selected["same_day_metrics"]["recall"],
                "same_day_n": selected["same_day_metrics"]["n"],
                "model_params": json.dumps(model_params, sort_keys=True, default=str),
                "fit_time_measured": fit_seconds is not None,
                "candidate_path": str(artifact_path),
            }
            rows.append(row)
            del x_calibration, x_selection
            if not is_existing:
                del x_train

    return rows, calibrations_by_candidate


def choose_candidate(comparison: pd.DataFrame) -> pd.Series:
    eligible = comparison.loc[comparison["selection_recall_target_met"]]
    pool = eligible if not eligible.empty else comparison
    if eligible.empty:
        return pool.sort_values(
            ["high_recall_recall", "pr_auc", "high_recall_precision", "brier_score"],
            ascending=[False, False, False, True],
        ).iloc[0]
    return pool.sort_values(
        [
            "high_recall_precision",
            "pr_auc",
            "same_day_recall",
            "brier_score",
            "inference_ms_per_1000",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def write_selection_report(metadata: dict) -> None:
    selected = metadata["selected_model"]
    validation = metadata["validation"]
    comparison = pd.DataFrame(validation["model_comparison"])
    columns = [
        "model", "feature_set", "configuration", "calibration_method",
        "roc_auc", "pr_auc", "brier_score", "high_recall_precision",
        "high_recall_recall", "high_recall_f1", "same_day_recall",
        "fit_seconds", "inference_ms_per_1000",
    ]
    table_header = "| " + " | ".join(columns) + " |\n"
    table_header += "|" + "|".join("---" for _ in columns) + "|\n"
    table_rows = []
    for row in comparison[columns].itertuples(index=False, name=None):
        cells = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        table_rows.append("| " + " | ".join(cells) + " |")
    table = table_header + "\n".join(table_rows)
    report = f"""# CarePredict model-selection report

## Selection protocol

The existing binary `no_show` target and chronological splits are unchanged. Preprocessing is fit on the training split only. May 30 is reserved for calibrator fitting, May 31 for model/configuration/feature-set selection, June 1 for operating-threshold selection, and June 2 for a one-time validation audit. The locked June 3–8 test records and labels are not decoded or scored during this run; only the file hash is checked. The test was evaluated in prior workflow versions, so it is not described as an untouched holdout.

All six requested model families were compared with the same chronological data and the same one-hot feature encoding. Each family has two prespecified configurations; weighting and capacity settings are reported rather than tuned on the test data. `same_day_context` tests log lead time and same-day-by-age/neighbourhood interactions. These are booking-time features; no separate same-day classifier is used.

The selected calibration method minimizes May 31 Brier score, then log loss and ECE. Model selection prioritizes precision at a 90% recall operating point when that target is reached, followed by PR-AUC, same-day recall, Brier score, and inference time. The deployed threshold is selected separately on June 1. June 2 is an audit, not a selection set.

## Selected candidate

- Model: **{selected['model']}** ({selected['configuration']})
- Feature set: **{selected['feature_set']}**
- Calibration: **{selected['calibration_method']}**
- June 1 high-recall threshold: **{selected['threshold']:.3f}**
- Recall target met during May 31 model selection: **{selected['selection_recall_target_met']}**
- June 1 precision / recall / F1: **{selected['threshold_selection_precision']:.1%} / {selected['threshold_selection_recall']:.1%} / {selected['threshold_selection_f1']:.3f}**
- June 2 audit precision / recall / F1: **{selected['audit_precision']:.1%} / {selected['audit_recall']:.1%} / {selected['audit_f1']:.3f}**

## Candidate comparison (May 31 selection window)

{table}

## Same-day investigation

Same-day appointments are compared with later appointments using lead-time, age, prior-history, missingness, observed base-rate, and mean predicted-probability summaries. The selected model's same-day precision/recall at its May 31 90%-recall operating point and its June 2 audit metrics are recorded in `subgroup_validation.csv`. Performance is interpreted at the overall validation-selected threshold; no separate subgroup model or threshold is deployed.

## Calibration and threshold artifacts

`calibration_comparison.csv` contains uncalibrated, isotonic, and sigmoid Brier score, log loss, ECE, ROC-AUC, and PR-AUC on May 31. `threshold_sweep.csv` contains June 1 precision, recall, F1, confusion counts, and alert rate across thresholds. The high-recall, max-F1, and top-30-per-day strategies are reported separately.

## Caveats

The source cohort is historical 2016 data from Vitória, Brazil. The original model and its serialized preprocessing configuration are preserved as a comparison candidate; exact fit-time sample weights cannot be recovered from its estimator object. The locked test set was accessed in earlier workflow versions. Permutation importance is global, validation-only, and not a per-patient causal explanation. Predictions require human review and do not initiate communications or alter appointments.
"""
    (REPORT_DIR / "model_selection_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    if (REPORT_DIR / "locked_test_evaluation.json").exists():
        raise RuntimeError(
            "The locked test has already been finalized for a model. "
            "Do not retrain over the evaluated artifact."
        )
    required = [
        DATA_DIR / "train.parquet",
        DATA_DIR / "val.parquet",
        DATA_DIR / "test.parquet",
        BASELINE_MODEL_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required CarePredict input is missing: {path}")
    input_hashes = {str(path): sha256_file(path) for path in required}
    train, validation = load_splits()
    calibration = frame_for_date(validation, CALIBRATION_DATE)
    selection = frame_for_date(validation, MODEL_SELECTION_DATE)
    threshold_frame = frame_for_date(validation, THRESHOLD_SELECTION_DATE)
    audit = frame_for_date(validation, VALIDATION_AUDIT_DATE)
    original_bundle = joblib.load(BASELINE_MODEL_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carepredict_candidates_") as temp_dir:
        candidate_rows, calibrations_by_candidate = train_candidates(
            train,
            calibration,
            selection,
            original_bundle,
            Path(temp_dir),
        )
        comparison = pd.DataFrame(candidate_rows)
        chosen_row = choose_candidate(comparison)
        chosen = joblib.load(chosen_row["candidate_path"])

        y_threshold = threshold_frame["no_show"].to_numpy(dtype=np.int8)
        x_threshold = chosen["preprocessor"].transform(
            threshold_frame[chosen["features"]]
        )
        raw_threshold_scores = chosen["model"].predict_proba(x_threshold)[:, 1]
        threshold_scores_by_method = {
            method: calibrate_probabilities(raw_threshold_scores, calibrator)
            for method, calibrator in chosen["calibrators"].items()
        }
        calibrated_threshold_scores = threshold_scores_by_method[
            chosen["chosen_method"]
        ]
        sweep = threshold_sweep(y_threshold, calibrated_threshold_scores)
        policies, policy_predictions = strategy_results(
            threshold_frame,
            y_threshold,
            calibrated_threshold_scores,
            sweep,
        )
        high_policy = next(row for row in policies if row["strategy"] == "High recall")
        selected_threshold = float(high_policy["threshold"])

        y_audit = audit["no_show"].to_numpy(dtype=np.int8)
        x_audit = chosen["preprocessor"].transform(audit[chosen["features"]])
        audit_raw_scores = chosen["model"].predict_proba(x_audit)[:, 1]
        audit_scores = calibrate_probabilities(
            audit_raw_scores, chosen["chosen_calibrator"]
        )
        audit_predictions = (audit_scores >= selected_threshold).astype(np.int8)
        audit_metrics = decision_metrics(
            y_audit, audit_scores, audit_predictions, selected_threshold
        )
        audit_subgroups = subgroup_metrics(audit, y_audit, audit_scores, audit_predictions)
        audit_context = same_day_summary(
            audit, y_audit, audit_scores, tuple(chosen["features"])
        )

        selected_calibration = calibrations_by_candidate[
            f"{chosen_row['model']}|{chosen_row['feature_set']}|{chosen_row['configuration']}"
        ]
        calibration_rows = [
            {
                "model": chosen_row["model"],
                "feature_set": chosen_row["feature_set"],
                **row,
            }
            for row in selected_calibration["rows"]
        ]
        selection_frame = frame_for_date(validation, MODEL_SELECTION_DATE)
        y_selection = selection_frame["no_show"].to_numpy(dtype=np.int8)
        x_selection = chosen["preprocessor"].transform(
            selection_frame[chosen["features"]]
        )
        selection_raw_scores = chosen["model"].predict_proba(x_selection)[:, 1]
        selection_scores = calibrate_probabilities(
            selection_raw_scores, chosen["chosen_calibrator"]
        )
        selection_sweep = threshold_sweep(y_selection, selection_scores)
        selection_high = choose_high_recall(selection_sweep)
        selection_preds = (selection_scores >= float(selection_high["threshold"])).astype(np.int8)
        selection_subgroups = subgroup_metrics(
            selection_frame, y_selection, selection_scores, selection_preds
        )
        selection_context = same_day_summary(
            selection_frame, y_selection, selection_scores, tuple(chosen["features"])
        )
        selected_importance = feature_importance(
            selection_frame, y_selection, chosen
        )

        precision, recall, pr_thresholds = precision_recall_curve(
            y_threshold, calibrated_threshold_scores
        )
        fpr, tpr, roc_thresholds = roc_curve(y_threshold, calibrated_threshold_scores)
        selected_details = {
            "model": str(chosen_row["model"]),
            "configuration": str(chosen_row["configuration"]),
            "feature_set": str(chosen_row["feature_set"]),
            "features": list(chosen["features"]),
            "calibration_method": str(chosen["chosen_method"]),
            "threshold": selected_threshold,
            "selection_recall_target_met": bool(
                chosen_row["selection_recall_target_met"]
            ),
            "selection_validation_date": MODEL_SELECTION_DATE.date().isoformat(),
            "selection_threshold_at_90_recall": float(
                chosen_row["selection_threshold_at_90_recall"]
            ),
            "selection_precision": float(chosen_row["high_recall_precision"]),
            "selection_recall": float(chosen_row["high_recall_recall"]),
            "selection_f1": float(chosen_row["high_recall_f1"]),
            "threshold_selection_precision": float(high_policy["precision"]),
            "threshold_selection_recall": float(high_policy["recall"]),
            "threshold_selection_f1": float(high_policy["f1"]),
            "audit_precision": float(audit_metrics["precision"]),
            "audit_recall": float(audit_metrics["recall"]),
            "audit_f1": float(audit_metrics["f1"]),
        }
        model_bundle = {
            "artifact_version": 3,
            "preprocessor": chosen["preprocessor"],
            "model": chosen["model"],
            "calibrator": chosen["chosen_calibrator"],
            "calibration_method": chosen["chosen_method"],
            "optimal_threshold": selected_threshold,
            "model_name": chosen_row["model"],
            "model_configuration": chosen_row["configuration"],
            "feature_set": chosen_row["feature_set"],
            "modeling_features": list(chosen["features"]),
            "random_seed": RANDOM_SEED,
            "train_end": str(train["appointment_day"].max().date()),
            "calibration_date": CALIBRATION_DATE.date().isoformat(),
            "model_selection_date": MODEL_SELECTION_DATE.date().isoformat(),
            "threshold_selection_date": THRESHOLD_SELECTION_DATE.date().isoformat(),
            "validation_audit_date": VALIDATION_AUDIT_DATE.date().isoformat(),
            "decision_policy": "highest validation threshold meeting 90% recall, if attainable",
            "model_parameters": json.loads(str(chosen_row["model_params"])),
        }
        temp_model = MODEL_PATH.with_suffix(".joblib.tmp")
        joblib.dump(model_bundle, temp_model)
        temp_model.replace(MODEL_PATH)

        source_hashes_after = {str(path): sha256_file(path) for path in required}
        if source_hashes_after != input_hashes:
            raise RuntimeError("A source or locked split artifact changed during training.")

        metadata = {
            "artifact_version": 3,
            "model_path": str(MODEL_PATH),
            "model_hash": sha256_file(MODEL_PATH),
            "model_name": chosen_row["model"],
            "feature_set": chosen_row["feature_set"],
            "calibration_method": chosen["chosen_method"],
            "decision_threshold": selected_threshold,
            "random_seed": RANDOM_SEED,
            "package_versions": package_versions(),
            "selected_model": selected_details,
            "source_hashes": input_hashes,
            "validation_protocol": {
                "train_end": str(train["appointment_day"].max().date()),
                "calibration_period": [CALIBRATION_DATE.date().isoformat()] * 2,
                "model_selection_period": [MODEL_SELECTION_DATE.date().isoformat()] * 2,
                "threshold_selection_period": [THRESHOLD_SELECTION_DATE.date().isoformat()] * 2,
                "validation_audit_period": [VALIDATION_AUDIT_DATE.date().isoformat()] * 2,
                "recall_target": RECALL_TARGET,
                "daily_review_capacity": DAILY_REVIEW_CAPACITY,
                "locked_test_used_for_selection": False,
                "locked_test_historical_prior_use": True,
            },
            "validation": {
                "model_comparison": json_records(comparison.drop(columns=["candidate_path"])),
                "calibration_comparison": calibration_rows,
                "threshold_sweep": json_records(sweep),
                "threshold_strategies": policies,
                "subgroups": audit_subgroups,
                "model_selection_subgroups": selection_subgroups,
                "same_day_analysis": audit_context,
                "model_selection_same_day_analysis": selection_context,
                "feature_importance": selected_importance,
                "curves": {
                    "precision_recall": {
                        "precision": precision.astype(float).tolist(),
                        "recall": recall.astype(float).tolist(),
                        "thresholds": pr_thresholds.astype(float).tolist(),
                    },
                    "roc": {
                        "false_positive_rate": fpr.astype(float).tolist(),
                        "true_positive_rate": tpr.astype(float).tolist(),
                        "thresholds": roc_thresholds.astype(float).tolist(),
                    },
                    "calibration": selected_calibration["curves"],
                },
                "audit": {
                    "date": VALIDATION_AUDIT_DATE.date().isoformat(),
                    "high_recall": audit_metrics,
                    "subgroups": audit_subgroups,
                    "same_day_analysis": audit_context,
                },
            },
            "limitations": [
                "Historical 2016 cohort from Vitória, Brazil; performance may not generalize to another time or hospital.",
                "The binary no_show target does not distinguish cancellations from no-shows.",
                "The original saved baseline's serialized estimator cannot verify fit-time sample weights.",
                "The locked test set was evaluated in prior workflow versions and is not an untouched research holdout.",
                "Permutation importance is global and validation-only; it is not a causal or per-patient explanation.",
                "No separate same-day classifier is used; subgroup differences still require human review.",
                "SMS, clinical-condition, scholarship, and other sensitive attributes were not added.",
                "The top-30-per-day capacity is an illustrative planning assumption.",
                "Predictions do not cancel appointments or initiate patient communications.",
            ],
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        comparison.drop(columns=["candidate_path"]).to_csv(
            REPORT_DIR / "model_comparison.csv", index=False
        )
        pd.DataFrame(calibration_rows).to_csv(
            REPORT_DIR / "calibration_comparison.csv", index=False
        )
        sweep.to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)
        pd.DataFrame(audit_subgroups).to_csv(
            REPORT_DIR / "subgroup_validation.csv", index=False
        )
        pd.DataFrame(selected_importance).to_csv(
            REPORT_DIR / "feature_importance_validation.csv", index=False
        )
        pd.DataFrame(audit_context).to_csv(
            REPORT_DIR / "same_day_validation.csv", index=False
        )
        (REPORT_DIR / "validation_curves.json").write_text(
            json.dumps(metadata["validation"]["curves"], indent=2), encoding="utf-8"
        )

    write_selection_report(metadata)
    log_text = f"""# CarePredict experiment decision log

## Audit and protocol

- The original saved artifact is HistGradientBoosting (`max_iter=120`, `max_depth=6`, `learning_rate=0.08`). Its estimator has no class-weight setting; the legacy notebook trained with positive-class sample weights, but the serialized estimator cannot verify those fit-time weights.
- The legacy notebook calibrated and selected a threshold on the same full validation data, used test labels for permutation importance, and exported to the original baseline path. It is not the selection workflow.
- The original baseline and locked test parquet remain unchanged. The test data has prior evaluation in earlier workflow versions and is not an untouched holdout.
- The new chronological protocol reserves May 30 for calibration, May 31 for model selection, June 1 for threshold selection, and June 2 for a fixed-policy validation audit. Preprocessing is fit only on training data.
- Model families/configurations: {len(comparison)} candidate rows spanning Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM, CatBoost, the baseline/booking-context/same-day-context feature sets, and the saved original model.
- Candidate selection rule: precision at a 90% recall operating point where attainable; otherwise recall first, then PR-AUC. Calibration is selected by validation Brier score, then log loss and ECE. The final threshold is chosen on June 1 and is not tuned on the June 2 audit or locked test.

## Results

- Selected: **{selected_details['model']} / {selected_details['configuration']} / {selected_details['feature_set']}**
- Calibration: **{selected_details['calibration_method']}**
- Frozen threshold: **{selected_threshold:.3f}**
- May 31 selection precision/recall/F1: **{selected_details['selection_precision']:.1%} / {selected_details['selection_recall']:.1%} / {selected_details['selection_f1']:.3f}**
- June 1 threshold-selection precision/recall/F1: **{selected_details['threshold_selection_precision']:.1%} / {selected_details['threshold_selection_recall']:.1%} / {selected_details['threshold_selection_f1']:.3f}**
- June 2 audit precision/recall/F1: **{selected_details['audit_precision']:.1%} / {selected_details['audit_recall']:.1%} / {selected_details['audit_f1']:.3f}**
- Locked test: **not decoded or scored in this stage**. Run `python finalize_test.py` once only after reviewing the frozen artifact.

See `model_selection_report.md` and the generated CSV/JSON files for the complete candidate table, calibration, thresholds, audit subgroups, and global validation permutation importance.
"""
    DECISION_LOG_PATH.write_text(log_text, encoding="utf-8")
    print("Selected:", selected_details["model"], "/", selected_details["configuration"])
    print("Feature set:", selected_details["feature_set"])
    print("Calibration:", selected_details["calibration_method"])
    print("Threshold selected on June 1:", selected_threshold)
    print("June 2 validation audit:", audit_metrics)
    print("Locked test rows and labels were not decoded or scored. Run finalize_test.py once.")
    print("Saved model:", MODEL_PATH)
    print("Saved metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()
