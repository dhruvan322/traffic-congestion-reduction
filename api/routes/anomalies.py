from fastapi import APIRouter, Query

from api.schemas import AnomalyResponse
from engine.osm_traffic_router import load_traffic_dataset
from models.anomaly_detector import get_anomaly_report


router = APIRouter(
    prefix="/anomalies",
    tags=["Anomalies"],
)


@router.get(
    "",
    response_model=list[AnomalyResponse],
)
def get_anomalies(
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
) -> list[AnomalyResponse]:
    """
    Return the highest-priority detected traffic anomalies.

    The underlying project detector is used directly.
    """

    df = load_traffic_dataset()

    anomalies = get_anomaly_report(df)

    risk_order = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    anomalies = sorted(
        anomalies,
        key=lambda item: (
            risk_order.get(
                str(item.get("risk_level")),
                0,
            ),
            float(
                item.get("anomaly_score", 0.0)
            ),
        ),
        reverse=True,
    )

    return [
        AnomalyResponse(**item)
        for item in anomalies[:limit]
    ]
