import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from engine.osm_traffic_router import load_traffic_dataset
from engine.prediction.predictor import predict_from_row


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict_traffic(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Predict congestion for a selected historical observation
    in a project area.

    If timestamp is supplied, that timestamp is used.
    Otherwise, the latest available timestamp is used.

    If road_segment_id is supplied, that segment is used.
    Otherwise, the first available segment is selected.
    """

    df = load_traffic_dataset()

    normalized_area = (
        request.area
        .strip()
        .lower()
    )

    area_df = df[
        df["area"]
        .astype(str)
        .str.strip()
        .str.lower()
        == normalized_area
    ].copy()

    if area_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Traffic area '{request.area}' "
                "was not found."
            ),
        )

    # Select the requested timestamp or latest available timestamp.
    if request.timestamp:
        try:
            selected_timestamp = pd.to_datetime(
                request.timestamp,
                errors="raise",
            )
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Invalid timestamp format.",
            )

        selected_rows = area_df[
            area_df["timestamp"] == selected_timestamp
        ].copy()

        if selected_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No traffic record found for timestamp "
                    f"'{request.timestamp}' in area '{request.area}'."
                ),
            )

    else:
        selected_timestamp = area_df["timestamp"].max()

        selected_rows = area_df[
            area_df["timestamp"] == selected_timestamp
        ].copy()

    # Filter by road segment if supplied.
    if request.road_segment_id:
        selected_rows = selected_rows[
            selected_rows["road_segment_id"]
            .astype(str)
            .str.strip()
            .str.upper()
            == request.road_segment_id.strip().upper()
        ]

        if selected_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Road segment '{request.road_segment_id}' "
                    f"was not found for the selected timestamp "
                    f"in area '{request.area}'."
                ),
            )

    # Select the first matching road segment.
    row = (
        selected_rows
        .sort_values("road_segment_id")
        .iloc[0]
    )

    # Run the existing trained ML model.
    prediction = predict_from_row(row)

    return PredictionResponse(
        area=str(row["area"]),
        timestamp=str(selected_timestamp),
        road_segment_id=str(row["road_segment_id"]),
        predicted_class=str(
            prediction["predicted_class"]
        ),
        predicted_class_id=int(
            prediction["predicted_class_id"]
        ),
        probabilities={
            str(key): float(value)
            for key, value in prediction["probabilities"].items()
        },
        confidence=float(prediction["confidence"]),
        vehicle_count=int(
            round(float(row["vehicle_count"]))
        ),
        average_speed_kmh=round(
            float(row["average_speed_kmh"]),
            2,
        ),
    )