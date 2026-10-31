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
    Predict congestion for the latest available observation
    in a project area.

    If road_segment_id is supplied, that segment is used.
    Otherwise, the first available segment for the latest
    timestamp is used.
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

    latest_timestamp = area_df[
        "timestamp"
    ].max()

    latest_rows = area_df[
        area_df["timestamp"]
        == latest_timestamp
    ].copy()

    if request.road_segment_id:

        latest_rows = latest_rows[
            latest_rows["road_segment_id"]
            .astype(str)
            .str.upper()
            == request.road_segment_id
            .strip()
            .upper()
        ]

        if latest_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Road segment '{request.road_segment_id}' "
                    f"was not found in area '{request.area}'."
                ),
            )

    row = (
        latest_rows
        .sort_values("road_segment_id")
        .iloc[0]
    )

    prediction = predict_from_row(row)

    return PredictionResponse(
        area=str(row["area"]),
        timestamp=str(latest_timestamp),
        road_segment_id=str(
            row["road_segment_id"]
        ),
        predicted_class=str(
            prediction["predicted_class"]
        ),
        predicted_class_id=int(
            prediction["predicted_class_id"]
        ),
        probabilities={
            str(key): float(value)
            for key, value in prediction[
                "probabilities"
            ].items()
        },
        confidence=float(
            prediction["confidence"]
        ),
        vehicle_count=int(
            round(
                float(row["vehicle_count"])
            )
        ),
        average_speed_kmh=round(
            float(
                row["average_speed_kmh"]
            ),
            2,
        ),
    )
