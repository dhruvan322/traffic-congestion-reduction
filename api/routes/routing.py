from fastapi import APIRouter, HTTPException

from api.schemas import (
    RouteMetricsResponse,
    RouteRequest,
    RouteResponse,
)
from engine.osm_traffic_router import (
    compare_osm_routes,
    load_traffic_dataset,
)


router = APIRouter(
    prefix="/route",
    tags=["Routing"],
)


def _resolve_area_coordinates(
    df,
    area_name: str,
):
    normalized = (
        area_name
        .strip()
        .lower()
    )

    area_rows = df[
        df["area"]
        .astype(str)
        .str.strip()
        .str.lower()
        == normalized
    ]

    if area_rows.empty:
        return None

    return (
        float(area_rows["latitude"].mean()),
        float(area_rows["longitude"].mean()),
    )


def _metrics(data: dict) -> RouteMetricsResponse:
    return RouteMetricsResponse(
        distance_km=float(
            data["distance_km"]
        ),
        estimated_time_minutes=float(
            data["estimated_time_minutes"]
        ),
        average_speed_kmh=float(
            data["average_speed_kmh"]
        ),
        average_traffic_speed_kmh=(
            float(
                data["average_traffic_speed_kmh"]
            )
            if data["average_traffic_speed_kmh"]
            is not None
            else None
        ),
        average_occupancy_percent=(
            float(
                data["average_occupancy_percent"]
            )
            if data["average_occupancy_percent"]
            is not None
            else None
        ),
        congested_edges=int(
            data["congested_edges"]
        ),
        traffic_covered_edges=int(
            data["traffic_covered_edges"]
        ),
        total_edges=int(
            data["total_edges"]
        ),
        traffic_coverage_percent=float(
            data["traffic_coverage_percent"]
        ),
    )


@router.post(
    "",
    response_model=RouteResponse,
)
def calculate_route(
    request: RouteRequest,
) -> RouteResponse:
    """
    Compare the shortest-distance and traffic-aware
    routes between two project areas.
    """

    df = load_traffic_dataset()

    origin_coords = _resolve_area_coordinates(
        df,
        request.origin,
    )

    destination_coords = _resolve_area_coordinates(
        df,
        request.destination,
    )

    if origin_coords is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Origin area '{request.origin}' "
                "was not found."
            ),
        )

    if destination_coords is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Destination area '{request.destination}' "
                "was not found."
            ),
        )

    latest_timestamp = df["timestamp"].max()

    timestamp = (
        request.timestamp
        if request.timestamp
        else latest_timestamp
    )

    result = compare_osm_routes(
        origin=origin_coords,
        destination=destination_coords,
        timestamp=timestamp,
    )

    shortest_route = result[
        "shortest_route"
    ]

    traffic_route = result[
        "traffic_aware_route"
    ]

    route_changed = (
        shortest_route != traffic_route
    )

    if route_changed:
        recommendation = (
            "The traffic-aware route is recommended "
            "because the traffic-weighted network "
            "selected a different route."
        )
    else:
        recommendation = (
            "The shortest and traffic-aware routes "
            "are the same for the selected traffic "
            "conditions."
        )

    return RouteResponse(
        origin=request.origin.strip(),
        destination=request.destination.strip(),
        timestamp=str(
            result["timestamp_requested"]
        ),
        shortest_route=_metrics(
            result["shortest_metrics"]
        ),
        traffic_aware_route=_metrics(
            result["traffic_aware_metrics"]
        ),
        route_changed=route_changed,
        shortest_route_nodes=len(
            shortest_route
        ),
        traffic_aware_route_nodes=len(
            traffic_route
        ),
        recommendation=recommendation,
    )
