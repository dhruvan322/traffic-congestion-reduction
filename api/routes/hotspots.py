from fastapi import APIRouter

from api.schemas import HotspotResponse
from engine.osm_traffic_router import load_traffic_dataset
from models.hotspot_detector import get_hotspot_summary


router = APIRouter(
    prefix="/hotspots",
    tags=["Hotspots"],
)


@router.get(
    "",
    response_model=list[HotspotResponse],
)
def get_hotspots() -> list[HotspotResponse]:
    """
    Return detected traffic hotspot clusters.
    """

    df = load_traffic_dataset()

    summaries = get_hotspot_summary(df)

    return [
        HotspotResponse(**summary)
        for summary in summaries
    ]
