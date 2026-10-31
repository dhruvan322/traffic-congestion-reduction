"""
Multi-Horizon Traffic Congestion Forecasting.

Forecasts future congestion for:
    +1 hour
    +2 hours
    +3 hours
    +6 hours

Important:
- The existing project `future_congestion_level` is treated as the +1h target.
- Longer horizons are created by shifting that target forward within each
  road segment.
- The existing production/deployed XGBoost predictor is not modified.
- Each horizon gets its own preprocessing pipeline and XGBoost model.
- Splitting is chronological to avoid future-data leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "ml_traffic_dataset.csv"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "trained_models"
    / "multihorizon"
)

HORIZONS = {
    1: 0,
    2: 1,
    3: 2,
    6: 5,
}

CONGESTION_LABELS = [
    "Free Flow",
    "Low",
    "Moderate",
    "High",
    "Severe",
]

RANDOM_STATE = 42

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_final_dataset() -> pd.DataFrame:
    """Load and validate the final ML traffic dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Final ML dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = {
        "road_segment_id",
        "timestamp",
        "future_congestion_level",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Final ML dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps.")

    df = df.sort_values(
        ["road_segment_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# TARGET CREATION
# ---------------------------------------------------------------------

def build_horizon_target(
    df: pd.DataFrame,
    horizon_hours: int,
) -> pd.Series:
    """
    Build the target for a requested forecast horizon.

    +1h:
        Uses the existing project future_congestion_level.

    +2h:
        Shifts the existing target by one hourly observation.

    +3h:
        Shifts by two hourly observations.

    +6h:
        Shifts by five hourly observations.
    """

    if horizon_hours not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon_hours}. "
            f"Supported horizons: {sorted(HORIZONS)}"
        )

    shift_amount = HORIZONS[horizon_hours]

    base_target = (
        df["future_congestion_level"]
        .astype("string")
        .str.strip()
    )

    if shift_amount == 0:
        return base_target

    return (
        base_target
        .groupby(df["road_segment_id"])
        .shift(-shift_amount)
    )


# ---------------------------------------------------------------------
# DATASET PREPARATION
# ---------------------------------------------------------------------

def prepare_horizon_dataset(
    df: pd.DataFrame,
    horizon_hours: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target vector for one horizon."""

    data = df.copy()

    data["forecast_target"] = build_horizon_target(
        data,
        horizon_hours,
    )

    data = data.dropna(
        subset=["forecast_target"]
    ).copy()

    invalid_targets = sorted(
        set(data["forecast_target"]) - set(CONGESTION_LABELS)
    )

    if invalid_targets:
        raise ValueError(
            "Unexpected congestion labels found: "
            + ", ".join(map(str, invalid_targets))
        )

    # Remove identifiers, timestamps and all future-target columns.
    drop_columns = {
        "future_congestion_level",
        "forecast_target",
        "timestamp",
        "traffic_id",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in drop_columns
    ]

    X = data[feature_columns].copy()

    y = data["forecast_target"].copy()

    return X, y


# ---------------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ---------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
) -> dict:
    """
    Split data using chronological timestamps.

    Approximate:
        70% train
        15% validation
        15% test
    """

    unique_times = (
        pd.Series(timestamps)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_times) < 10:
        raise ValueError(
            "Not enough unique timestamps for chronological splitting."
        )

    train_end = int(len(unique_times) * 0.70)
    val_end = int(len(unique_times) * 0.85)

    train_times = set(unique_times.iloc[:train_end])
    val_times = set(unique_times.iloc[train_end:val_end])
    test_times = set(unique_times.iloc[val_end:])

    timestamps_series = pd.Series(
        timestamps,
        index=X.index,
    )

    train_mask = timestamps_series.isin(train_times)
    val_mask = timestamps_series.isin(val_times)
    test_mask = timestamps_series.isin(test_times)

    return {
        "X_train": X.loc[train_mask].copy(),
        "y_train": y.loc[train_mask].copy(),
        "X_val": X.loc[val_mask].copy(),
        "y_val": y.loc[val_mask].copy(),
        "X_test": X.loc[test_mask].copy(),
        "y_test": y.loc[test_mask].copy(),
        "train_start": str(unique_times.iloc[0]),
        "train_end": str(unique_times.iloc[train_end - 1]),
        "val_start": str(unique_times.iloc[train_end]),
        "val_end": str(unique_times.iloc[val_end - 1]),
        "test_start": str(unique_times.iloc[val_end]),
        "test_end": str(unique_times.iloc[-1]),
    }


# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------

def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Create a leakage-safe preprocessing pipeline."""

    categorical_columns = [
        column
        for column in X_train.columns
        if (
            pd.api.types.is_object_dtype(X_train[column])
            or pd.api.types.is_string_dtype(X_train[column])
            or pd.api.types.is_categorical_dtype(X_train[column])
        )
    ]

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                encoder,
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def build_model() -> XGBClassifier:
    """Build the independent multi-horizon XGBoost classifier."""

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=180,
        max_depth=8,
        learning_rate=0.10,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate classification metrics."""

    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4,
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)),
            4,
        ),
        "macro_f1": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                )
            ),
            4,
        ),
        "weighted_f1": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                )
            ),
            4,
        ),
    }


# ---------------------------------------------------------------------
# SINGLE HORIZON TRAINING
# ---------------------------------------------------------------------

def train_horizon(
    df: pd.DataFrame,
    horizon_hours: int,
) -> dict:
    """
    Train and evaluate one forecast horizon.

    Returns metrics and saved-artifact information.
    """

    LOGGER.info(
        "Preparing +%dh forecasting dataset...",
        horizon_hours,
    )

    data = df.copy()

    data["forecast_target"] = build_horizon_target(
        data,
        horizon_hours,
    )

    data = data.dropna(
        subset=["forecast_target"]
    ).copy()

    drop_columns = {
        "future_congestion_level",
        "forecast_target",
        "timestamp",
        "traffic_id",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in drop_columns
    ]

    X = data[feature_columns].copy()

    y_text = data["forecast_target"].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(CONGESTION_LABELS)

    y = label_encoder.transform(y_text)

    split = chronological_split(
        X,
        pd.Series(y, index=X.index),
        data["timestamp"],
    )

    X_train = split["X_train"]
    y_train = split["y_train"].to_numpy()

    X_val = split["X_val"]
    y_val = split["y_val"].to_numpy()

    X_test = split["X_test"]
    y_test = split["y_test"].to_numpy()

    train_classes = sorted(np.unique(y_train).tolist())

    if len(train_classes) < len(CONGESTION_LABELS):
        missing_classes = sorted(
            set(range(len(CONGESTION_LABELS)))
            - set(train_classes)
        )

        missing_names = [
            CONGESTION_LABELS[index]
            for index in missing_classes
        ]

        raise ValueError(
            f"+{horizon_hours}h training split is missing "
            f"classes: {missing_names}"
        )

    LOGGER.info(
        "+%dh split sizes -> train=%d, val=%d, test=%d",
        horizon_hours,
        len(X_train),
        len(X_val),
        len(X_test),
    )

    preprocessor = build_preprocessor(
        X_train
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_val_transformed = preprocessor.transform(
        X_val
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    LOGGER.info(
        "+%dh transformed feature count: %d",
        horizon_hours,
        X_train_transformed.shape[1],
    )

    model = build_model()

    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[
            (
                X_val_transformed,
                y_val,
            )
        ],
        verbose=False,
    )

    test_predictions = model.predict(
        X_test_transformed
    ).astype(int)

    metrics = evaluate_predictions(
        y_test,
        test_predictions,
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon_dir = MODEL_ROOT / f"{horizon_hours}h"

    horizon_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        horizon_dir
        / "xgb_congestion.json"
    )

    preprocessor_path = (
        horizon_dir
        / "traffic_preprocessor.joblib"
    )

    label_encoder_path = (
        horizon_dir
        / "label_encoder.joblib"
    )

    metadata_path = (
        horizon_dir
        / "metadata.json"
    )

    model.save_model(
        model_path
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    joblib.dump(
        label_encoder,
        label_encoder_path,
    )

    metadata = {
        "horizon_hours": horizon_hours,
        "target_source": (
            "future_congestion_level"
            if horizon_hours == 1
            else (
                "future_congestion_level shifted "
                f"by {HORIZONS[horizon_hours]} hourly records "
                "within road_segment_id"
            )
        ),
        "records": int(len(data)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
        "feature_count_before_encoding": int(
            len(feature_columns)
        ),
        "transformed_feature_count": int(
            X_train_transformed.shape[1]
        ),
        "split": {
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "validation_start": split["val_start"],
            "validation_end": split["val_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
        },
        "metrics": metrics,
        "classes": CONGESTION_LABELS,
        "artifacts": {
            "model": str(model_path),
            "preprocessor": str(preprocessor_path),
            "label_encoder": str(label_encoder_path),
        },
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    LOGGER.info(
        "+%dh metrics: %s",
        horizon_hours,
        metrics,
    )

    return metadata


# ---------------------------------------------------------------------
# ALL HORIZONS
# ---------------------------------------------------------------------

def train_all_horizons() -> Dict[int, dict]:
    """Train +1h, +2h, +3h and +6h models."""

    df = load_final_dataset()

    results = {}

    for horizon_hours in HORIZONS:
        results[horizon_hours] = train_horizon(
            df,
            horizon_hours,
        )

    summary_path = (
        MODEL_ROOT
        / "multihorizon_summary.json"
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    results = train_all_horizons()

    print("\n" + "=" * 70)
    print("MULTI-HORIZON FORECASTING RESULTS")
    print("=" * 70)

    for horizon, result in results.items():
        metrics = result["metrics"]

        print(
            f"+{horizon}h | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Balanced Accuracy={metrics['balanced_accuracy']:.4f} | "
            f"Macro F1={metrics['macro_f1']:.4f} | "
            f"Weighted F1={metrics['weighted_f1']:.4f}"
        )

    print("=" * 70)
