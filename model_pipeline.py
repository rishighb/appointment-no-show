from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

BASE_FEATURES = (
    "lead_time_days",
    "scheduled_hour",
    "age",
    "prev_appointment_count",
    "prev_no_show_rate",
    "is_same_day",
    "has_history",
    "appointment_weekday",
    "age_group",
    "neighbourhood",
)

ENGINEERED_FEATURES = (
    "scheduled_weekday",
    "lead_bucket",
    "same_day_has_history",
    "same_day_prior_no_show_rate",
)

FEATURE_SETS = {
    "baseline": BASE_FEATURES,
    "booking_context": BASE_FEATURES + ENGINEERED_FEATURES,
    "same_day_context": BASE_FEATURES
    + ENGINEERED_FEATURES
    + (
        "lead_time_log1p",
        "same_day_age_group",
        "same_day_neighbourhood",
    ),
}

NUMERIC_FEATURES = (
    "lead_time_days",
    "scheduled_hour",
    "age",
    "prev_appointment_count",
    "prev_no_show_rate",
    "same_day_prior_no_show_rate",
    "lead_time_log1p",
)

BINARY_FEATURES = (
    "is_same_day",
    "has_history",
    "same_day_has_history",
)

CATEGORICAL_FEATURES = (
    "appointment_weekday",
    "age_group",
    "neighbourhood",
    "scheduled_weekday",
    "lead_bucket",
    "same_day_age_group",
    "same_day_neighbourhood",
)

LEAD_TIME_BINS = [-1, 0, 3, 7, 14, 30, np.inf]
LEAD_TIME_LABELS = (
    "Same-day (0d)",
    "1-3 days",
    "4-7 days",
    "8-14 days",
    "15-30 days",
    "31+ days",
)


def add_booking_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive features that are available when an appointment is booked."""
    required = {
        "scheduled_day",
        "appointment_day",
        "age",
        "neighbourhood",
        "prev_appointment_count",
        "prev_no_show_rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns required for booking features: {sorted(missing)}")

    result = frame.copy()
    scheduled = pd.to_datetime(result["scheduled_day"], utc=True).dt.tz_convert(None)
    appointment = pd.to_datetime(result["appointment_day"], utc=True).dt.tz_convert(None)
    result["lead_time_days"] = (
        appointment.dt.normalize() - scheduled.dt.normalize()
    ).dt.days
    result["is_same_day"] = result["lead_time_days"].eq(0).astype("int8")
    result["scheduled_hour"] = scheduled.dt.hour.astype("int16")
    result["scheduled_weekday"] = scheduled.dt.day_name()
    result["appointment_weekday"] = appointment.dt.day_name()
    result["age_group"] = pd.cut(
        result["age"],
        bins=[-1, 12, 24, 45, 65, np.inf],
        labels=["0-12", "13-24", "25-45", "46-65", "66+"],
    ).astype(object)
    result["age_group"] = result["age_group"].where(
        result["age_group"].notna(), "Unknown"
    )
    result["lead_bucket"] = pd.cut(
        result["lead_time_days"],
        bins=LEAD_TIME_BINS,
        labels=LEAD_TIME_LABELS,
    ).astype(object)
    result["lead_bucket"] = result["lead_bucket"].where(
        result["lead_bucket"].notna(), "Unknown"
    )
    result["has_history"] = result["prev_appointment_count"].gt(0).astype("int8")
    result["same_day_has_history"] = (
        result["is_same_day"] * result["has_history"]
    ).astype("int8")
    result["same_day_prior_no_show_rate"] = (
        result["is_same_day"] * result["prev_no_show_rate"]
    )
    result["lead_time_log1p"] = np.log1p(
        result["lead_time_days"].clip(lower=0)
    )
    result["same_day_age_group"] = result["age_group"].where(
        result["is_same_day"].eq(1), "Not same-day"
    )
    result["same_day_neighbourhood"] = result["neighbourhood"].where(
        result["is_same_day"].eq(1), "Not same-day"
    )
    return result


def build_preprocessor(features: tuple[str, ...]) -> ColumnTransformer:
    numeric = [name for name in NUMERIC_FEATURES if name in features]
    binary = [name for name in BINARY_FEATURES if name in features]
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "bin",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_classifier(name: str, params: dict | None = None):
    params = dict(params or {})
    if name == "Logistic Regression":
        defaults = {
            "C": 0.1,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)
    if name == "Random Forest":
        defaults = {
            "n_estimators": 100,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if name == "HistGradientBoosting":
        defaults = {
            "max_iter": 120,
            "max_depth": 6,
            "learning_rate": 0.08,
            "min_samples_leaf": 30,
            "early_stopping": False,
            "random_state": RANDOM_SEED,
        }
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for this experiment. Install "
                "requirements-model-experiments.txt."
            ) from exc
        defaults = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for this experiment. Install "
                "requirements-model-experiments.txt."
            ) from exc
        defaults = {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "class_weight": None,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
            "verbosity": -1,
        }
        defaults.update(params)
        return LGBMClassifier(**defaults)
    if name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for this experiment. Install "
                "requirements-model-experiments.txt."
            ) from exc
        defaults = {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "auto_class_weights": None,
            "loss_function": "Logloss",
            "random_seed": RANDOM_SEED,
            "verbose": False,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        defaults.update(params)
        return CatBoostClassifier(**defaults)
    raise ValueError(f"Unsupported model family: {name}")


def fit_calibrator(method: str, scores: np.ndarray, labels: np.ndarray) -> dict | None:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if method == "uncalibrated":
        return None
    if method == "isotonic":
        estimator = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0
        ).fit(scores, labels)
    elif method == "sigmoid":
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        estimator = LogisticRegression(
            C=1e6, max_iter=1000, random_state=RANDOM_SEED
        ).fit(logits, labels)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    return {"method": method, "estimator": estimator}


def calibrate_probabilities(
    scores: np.ndarray, calibrator: dict | object | None
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if calibrator is None:
        return np.clip(scores, 0.0, 1.0)
    if isinstance(calibrator, dict):
        method = calibrator.get("method")
        estimator = calibrator.get("estimator")
        if method == "isotonic":
            result = estimator.predict(scores)
        elif method == "sigmoid":
            clipped = np.clip(scores, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            result = estimator.predict_proba(logits)[:, 1]
        else:
            raise ValueError(f"Unsupported saved calibration method: {method}")
    elif hasattr(calibrator, "predict"):
        result = calibrator.predict(scores)
    elif hasattr(calibrator, "predict_proba"):
        result = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    else:
        raise TypeError("Saved calibrator must provide predict or predict_proba.")
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    bin_ids = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    total = len(labels)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.sum() / total * abs(
                float(labels[mask].mean()) - float(scores[mask].mean())
            )
    return float(error)


def evaluate_probabilities(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int | list[list[int]]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "ece_10": expected_calibration_error(labels, scores),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "alert_rate": float(predictions.mean()),
    }


def validate_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> None:
    required = {"appointment_id", "appointment_day", "no_show"}
    splits = [("train", train), ("validation", validation)]
    if test is not None:
        splits.append(("test", test))
    for name, frame in splits:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing required columns: {sorted(missing)}")
        if frame["appointment_id"].duplicated().any():
            raise ValueError(f"{name} split contains duplicate appointment IDs.")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} split has missing IDs, dates, or targets.")
        if not frame["no_show"].isin([0, 1]).all():
            raise ValueError(f"{name} split target must contain only 0 and 1.")

    if set(train["appointment_id"]) & set(validation["appointment_id"]):
        raise ValueError("Train and validation appointment IDs overlap.")
    if train["appointment_day"].max() >= validation["appointment_day"].min():
        raise ValueError("Train and validation dates are not strictly chronological.")
    if test is not None:
        if set(train["appointment_id"]) & set(test["appointment_id"]):
            raise ValueError("Train and test appointment IDs overlap.")
        if set(validation["appointment_id"]) & set(test["appointment_id"]):
            raise ValueError("Validation and test appointment IDs overlap.")
        if validation["appointment_day"].max() >= test["appointment_day"].min():
            raise ValueError("Validation and test dates are not strictly chronological.")
