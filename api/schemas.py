from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class TrafficResponse(BaseModel):
    area: str
    timestamp: str
    road_segment_ids: list[str]
    vehicle_count: int
    average_speed_kmh: float
    congestion_level: str


class HotspotResponse(BaseModel):
    cluster_id: int
    center_lat: float
    center_lon: float
    size: int
    records: int
    severity: str
    peak_hours: list[Any]
    avg_vehicles: float
    avg_speed: float
    avg_occupancy: float
    congestion_rate: float
    high_or_severe_rate: float
    severe_rate: float
    road_segments: list[str]
    detected_at: str


class AnomalyResponse(BaseModel):
    road_segment_id: str
    timestamp: str
    vehicle_count: int
    average_speed_kmh: float
    occupancy_percent: float
    traffic_flow_vph: float
    future_congestion_level: str
    isolation_forest_anomaly: bool
    zscore_anomaly: bool
    anomaly_score: float
    risk_level: str
    anomaly_type: str
    detected_at: str
    description: str


class AnomalyAnalysisResponse(BaseModel):
    summary: dict[str, int | float]
    risk_counts: dict[str, int]
    method_counts: dict[str, int]
    monthly: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    top_records: list[dict[str, Any]]
    scatter_records: list[dict[str, Any]]
    score_records: list[dict[str, float]]
    recommendations: list[dict[str, Any]]
    peak_month: int
    peak_count: int
    affected_segments: int
    total_segments: int

class PredictionRequest(BaseModel):
    area: str = Field(
        ...,
        min_length=1,
        description="Project traffic area, e.g. Hitech City",
    )
    road_segment_id: str | None = Field(
        default=None,
        description="Optional project road segment, e.g. RS0004",
    )
    timestamp: str | None = Field(
        default=None,
        description=(
            "Optional historical traffic timestamp. "
            "If omitted, the latest available observation is used."
        ),
    )


class PredictionResponse(BaseModel):
    area: str
    timestamp: str
    road_segment_id: str
    predicted_class: str
    predicted_class_id: int
    probabilities: dict[str, float]
    confidence: float
    vehicle_count: int
    average_speed_kmh: float


class RouteRequest(BaseModel):
    origin: str = Field(
        ...,
        min_length=1,
        description="Origin project area",
    )
    destination: str = Field(
        ...,
        min_length=1,
        description="Destination project area",
    )
    timestamp: str | None = Field(
        default=None,
        description=(
            "Optional traffic timestamp. If omitted, the latest "
            "available project timestamp is used."
        ),
    )


class RouteMetricsResponse(BaseModel):
    distance_km: float
    estimated_time_minutes: float
    average_speed_kmh: float
    average_traffic_speed_kmh: float | None
    average_occupancy_percent: float | None
    congested_edges: int
    traffic_covered_edges: int
    total_edges: int
    traffic_coverage_percent: float


class RouteResponse(BaseModel):
    origin: str
    destination: str
    timestamp: str
    shortest_route: RouteMetricsResponse
    traffic_aware_route: RouteMetricsResponse
    route_changed: bool
    shortest_route_nodes: int
    traffic_aware_route_nodes: int
    recommendation: str


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User's traffic-related question",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique chat session identifier",
    )


class ChatResponse(BaseModel):
    response: str
    intent: str
    entities: dict[str, Any]
    detected_language: str
    turn_number: int
