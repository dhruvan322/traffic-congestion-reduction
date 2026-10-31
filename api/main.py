from fastapi import FastAPI

from api.routes.anomalies import router as anomalies_router
from api.routes.hotspots import router as hotspots_router
from api.routes.prediction import router as prediction_router
from api.routes.routing import router as routing_router
from api.routes.traffic import router as traffic_router
from api.schemas import HealthResponse


app = FastAPI(
    title="Smart AI Traffic Intelligence Platform API",
    description=(
        "Backend API for traffic prediction, hotspot detection, "
        "anomaly detection, route optimization, and decision support."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Basic service health check.
    """

    return HealthResponse(
        status="healthy",
        service="ai-traffic-intelligence-api",
        version="1.0.0",
    )


app.include_router(
    traffic_router,
    prefix="/api/v1",
)

app.include_router(
    hotspots_router,
    prefix="/api/v1",
)

app.include_router(
    anomalies_router,
    prefix="/api/v1",
)

app.include_router(
    prediction_router,
    prefix="/api/v1",
)

app.include_router(
    routing_router,
    prefix="/api/v1",
)


from api.routes.forecast import router as forecast_router
app.include_router(forecast_router)
