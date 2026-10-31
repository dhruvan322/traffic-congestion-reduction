from fastapi import APIRouter, HTTPException

from api.schemas import TrafficResponse
from engine.osm_traffic_router import load_traffic_dataset


router = APIRouter(
    prefix="/traffic",
    tags=["Traffic"],
)


_CONGESTION_ORDER = {
    "Free Flow": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Severe": 4,
}


@router.get(
    "/{area}",
    response_model=TrafficResponse,
)
def get_traffic_by_area(area: str) -> TrafficResponse:
    """
    Return the latest available traffic state for a project area.
    """

    df = load_traffic_dataset()

    normalized_area = (
        str(area)
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
            detail=f"Traffic area '{area}' was not found.",
        )

    latest_timestamp = area_df["timestamp"].max()

    latest_rows = area_df[
        area_df["timestamp"]
        == latest_timestamp
    ].copy()

    congestion_level = max(
        latest_rows["future_congestion_level"].astype(str).tolist(),
        key=lambda value: _CONGESTION_ORDER.get(
            value,
            -1,
        ),
    )

    return TrafficResponse(
        area=str(area).strip(),
        timestamp=str(latest_timestamp),
        road_segment_ids=sorted(
            latest_rows["road_segment_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        vehicle_count=int(
            round(
                latest_rows["vehicle_count"].mean()
            )
        ),
        average_speed_kmh=round(
            float(
                latest_rows["average_speed_kmh"].mean()
            ),
            2,
        ),
        congestion_level=congestion_level,
    )
