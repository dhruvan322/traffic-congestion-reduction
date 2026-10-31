"""
Multi-Horizon Forecast Inference.

Loads the independently trained +1h, +2h, +3h and +6h
congestion forecasting models.

This module does not retrain models and does not modify
the existing congestion prediction engine.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    PROJECT_ROOT
    / "trained_models"
    / "multihorizon"
)

SUPPORTED_HORIZONS = (1, 2, 3, 6)

DROP_COLUMNS = {
    "future_congestion_level",
    "timestamp",
    "traffic_id",
}


@lru_cache(maxsize=4)
def load_horizon_artifacts(
    horizon_hours: int,
) -> tuple[XGBClassifier, object, object]:
    """
    Load model, preprocessor and label encoder for one horizon.
    """

    if horizon_hours not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon_hours}. "
            f"Supported horizons: {SUPPORTED_HORIZONS}"
        )

    horizon_dir = (
        MODEL_ROOT
        / f"{horizon_hours}h"
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

    for path in (
        model_path,
        preprocessor_path,
        label_encoder_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Forecast artifact not found: {path}"
            )

    model = XGBClassifier()
    model.load_model(model_path)

    preprocessor = joblib.load(
        preprocessor_path
    )

    label_encoder = joblib.load(
        label_encoder_path
    )

    return (
        model,
        preprocessor,
        label_encoder,
    )


def _prepare_row(
    row: pd.Series | dict,
) -> pd.DataFrame:
    """
    Convert one project traffic observation into the
    feature dataframe expected by the forecasting pipeline.
    """

    if isinstance(row, pd.Series):
        data = row.to_frame().T
    elif isinstance(row, dict):
        data = pd.DataFrame([row])
    else:
        raise TypeError(
            "row must be a pandas Series or dictionary."
        )

    data = data.drop(
        columns=[
            column
            for column in DROP_COLUMNS
            if column in data.columns
        ],
        errors="ignore",
    )

    return data


def predict_horizon(
    row: pd.Series | dict,
    horizon_hours: int,
) -> dict:
    """
    Forecast congestion for one future horizon.
    """

    (
        model,
        preprocessor,
        label_encoder,
    ) = load_horizon_artifacts(
        horizon_hours
    )

    features = _prepare_row(row)

    transformed = preprocessor.transform(
        features
    )

    predicted_id = int(
        np.asarray(
            model.predict(transformed)
        ).reshape(-1)[0]
    )

    probabilities = model.predict_proba(
        transformed
    )[0]

    predicted_class = str(
        label_encoder.inverse_transform(
            [predicted_id]
        )[0]
    )

    confidence = float(
        np.max(probabilities)
    )

    probability_map = {}

    for class_id, probability in zip(
        label_encoder.classes_,
        probabilities,
    ):
        probability_map[str(class_id)] = round(
            float(probability),
            4,
        )

    return {
        "horizon_hours": horizon_hours,
        "predicted_class": predicted_class,
        "predicted_class_id": predicted_id,
        "confidence": round(
            confidence,
            4,
        ),
        "probabilities": probability_map,
    }


def forecast_all_horizons(
    row: pd.Series | dict,
) -> dict:
    """
    Forecast congestion for +1h, +2h, +3h and +6h.
    """

    timestamp = None
    road_segment_id = None
    area = None

    if isinstance(row, pd.Series):
        timestamp = row.get("timestamp")
        road_segment_id = row.get(
            "road_segment_id"
        )
        area = row.get("area")

    elif isinstance(row, dict):
        timestamp = row.get("timestamp")
        road_segment_id = row.get(
            "road_segment_id"
        )
        area = row.get("area")

    forecasts = {}

    for horizon_hours in SUPPORTED_HORIZONS:
        result = predict_horizon(
            row,
            horizon_hours,
        )

        forecasts[f"{horizon_hours}h"] = result

    return {
        "timestamp": (
            str(timestamp)
            if timestamp is not None
            else None
        ),
        "road_segment_id": (
            str(road_segment_id)
            if road_segment_id is not None
            else None
        ),
        "area": (
            str(area)
            if area is not None
            else None
        ),
        "forecasts": forecasts,
    }


def get_latest_row_for_segment(
    df: pd.DataFrame,
    road_segment_id: str,
) -> pd.Series:
    """
    Get the latest available observation for a road segment.
    """

    if "road_segment_id" not in df.columns:
        raise ValueError(
            "Dataset is missing road_segment_id."
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            "Dataset is missing timestamp."
        )

    segment = df[
        df["road_segment_id"].astype(str)
        == str(road_segment_id)
    ].copy()

    if segment.empty:
        raise ValueError(
            f"No traffic data found for "
            f"road segment {road_segment_id}."
        )

    segment["timestamp"] = pd.to_datetime(
        segment["timestamp"],
        errors="coerce",
    )

    segment = segment.sort_values(
        "timestamp"
    )

    return segment.iloc[-1].copy()
