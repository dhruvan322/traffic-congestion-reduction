"""
Multi-Horizon Forecasting API.

Endpoint:
    POST /api/v1/forecast

Provides +1h, +2h, +3h and +6h congestion forecasts
using the independently trained multi-horizon models.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from engine.multihorizon_predictor import forecast_all_horizons


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "ml_traffic_dataset.csv"
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Forecasting"],
)


class ForecastRequest(BaseModel):
    """Forecast request."""

    area: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Project area such as Hitech City.",
    )

    road_segment_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Specific road segment such as RS0031.",
    )

    @model_validator(mode="after")
    def validate_target(self):
        if not self.area and not self.road_segment_id:
            raise ValueError(
                "Provide either area or road_segment_id."
            )

        return self


class ForecastHorizonResponse(BaseModel):
    """Forecast result for one horizon."""

    horizon_hours: int
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: Dict[str, float]


class ForecastResponse(BaseModel):
    """Multi-horizon forecast response."""

    area: Optional[str]
    road_segment_id: str
    timestamp: str
    forecasts: Dict[str, ForecastHorizonResponse]


@lru_cache(maxsize=1)
def load_forecast_dataset() -> pd.DataFrame:
    """Load the final traffic dataset for forecast requests."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Final traffic dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = {
        "road_segment_id",
        "timestamp",
        "area",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Forecast dataset is missing columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    return df


def resolve_forecast_row(
    request: ForecastRequest,
) -> pd.Series:
    """Resolve the traffic observation used for forecasting."""

    df = load_forecast_dataset()

    data = df.copy()

    if request.road_segment_id:
        data = data[
            data["road_segment_id"].astype(str)
            == str(request.road_segment_id)
        ].copy()

    elif request.area:
        requested_area = request.area.strip().lower()

        data = data[
            data["area"]
            .astype(str)
            .str.strip()
            .str.lower()
            == requested_area
        ].copy()

    if data.empty:
        target = (
            request.road_segment_id
            if request.road_segment_id
            else request.area
        )

        raise HTTPException(
            status_code=404,
            detail=(
                f"No traffic observation found for '{target}'."
            ),
        )

    latest_timestamp = data["timestamp"].max()

    latest = data[
        data["timestamp"] == latest_timestamp
    ].copy()

    if request.road_segment_id:
        row = latest.iloc[0].copy()

    else:
        latest = latest.sort_values(
            "road_segment_id"
        )

        row = latest.iloc[0].copy()

    return row


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Forecast traffic congestion",
)
def forecast_traffic(
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Forecast congestion +1h, +2h, +3h and +6h.

    For an area request, the first available road segment
    at the latest timestamp is used, matching the existing
    prediction API's area-level selection behavior.
    """

    try:
        row = resolve_forecast_row(request)

        result = forecast_all_horizons(row)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}",
        ) from exc

    return ForecastResponse(
        area=(
            str(row["area"])
            if "area" in row and pd.notna(row["area"])
            else None
        ),
        road_segment_id=str(
            result["road_segment_id"]
        ),
        timestamp=str(
            result["timestamp"]
        ),
        forecasts={
            horizon: ForecastHorizonResponse(
                **forecast
            )
            for horizon, forecast
            in result["forecasts"].items()
        },
    )
