from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Query

from api.schemas import AnomalyAnalysisResponse, AnomalyResponse
from engine.osm_traffic_router import load_traffic_dataset
from models.anomaly_detector import (
    analyze_traffic_anomalies,
    get_anomaly_report,
    get_anomaly_summary,
)


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
            risk_order.get(str(item.get("risk_level")), 0),
            float(item.get("anomaly_score", 0.0)),
        ),
        reverse=True,
    )

    return [
        AnomalyResponse(**item)
        for item in anomalies[:limit]
    ]


def _clean_records(records: list[dict]) -> list[dict]:
    """Convert pandas and NumPy values into JSON-safe Python values."""

    cleaned = []

    for record in records:
        clean_record = {}

        for key, value in record.items():
            if pd.isna(value):
                clean_record[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                clean_record[key] = value.isoformat()
            elif hasattr(value, "item"):
                clean_record[key] = value.item()
            else:
                clean_record[key] = value

        cleaned.append(clean_record)

    return cleaned


@router.post(
    "/analyze",
    response_model=AnomalyAnalysisResponse,
)
def analyze_anomalies() -> AnomalyAnalysisResponse:
    """
    Run anomaly detection once and return dashboard-ready aggregates
    and representative records.
    """

    df = load_traffic_dataset()

    detected = analyze_traffic_anomalies(df)

    summary = get_anomaly_summary(detected)

    anomalies = detected[
        detected["combined_anomaly"] == 1
    ].copy()

    risk_order = ["Critical", "High", "Medium", "Low"]

    risk_counts = (
        anomalies["risk_level"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .astype(int)
        .to_dict()
    )

    method_counts = {
        "Isolation Forest only": int(
            (
                (anomalies["is_anomaly"] == 1)
                & (anomalies["zscore_anomaly"] == 0)
            ).sum()
        ),
        "Z-score only": int(
            (
                (anomalies["is_anomaly"] == 0)
                & (anomalies["zscore_anomaly"] == 1)
            ).sum()
        ),
        "Both methods": int(
            (
                (anomalies["is_anomaly"] == 1)
                & (anomalies["zscore_anomaly"] == 1)
            ).sum()
        ),
    }

    # Monthly anomaly counts, using the project's 2025 dataset period.
    anomalies["timestamp"] = pd.to_datetime(
        anomalies["timestamp"],
        errors="coerce",
    )

    monthly_source = anomalies[
        anomalies["timestamp"].dt.year == 2025
    ].copy()

    monthly_counts = (
        monthly_source
        .groupby([
            monthly_source["timestamp"].dt.month,
            "risk_level",
        ])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 13), columns=risk_order, fill_value=0)
    )

    monthly = []

    for month in range(1, 13):
        item = {
            "month": month,
            "month_label": pd.Timestamp(
                year=2025,
                month=month,
                day=1,
            ).strftime("%b"),
        }

        for risk in risk_order:
            item[risk] = int(monthly_counts.loc[month, risk])

        item["total"] = sum(item[risk] for risk in risk_order)
        monthly.append(item)

    peak_item = max(monthly, key=lambda item: item["total"])

    # Aggregate affected road segments.
    segments = []

    for segment_id, group in anomalies.groupby("road_segment_id"):
        dominant_risk = (
            group["risk_level"].value_counts().idxmax()
        )

        first = group.iloc[0]

        segments.append({
            "road_segment_id": str(segment_id),
            "area": first.get("area"),
            "latitude": (
                float(first["latitude"])
                if pd.notna(first.get("latitude"))
                else None
            ),
            "longitude": (
                float(first["longitude"])
                if pd.notna(first.get("longitude"))
                else None
            ),
            "anomaly_records": int(len(group)),
            "dominant_risk": str(dominant_risk),
        })

    segments.sort(
        key=lambda item: item["anomaly_records"],
        reverse=True,
    )

    # Highest-scoring anomaly records for the dashboard table.
    top_source = anomalies.sort_values(
        "anomaly_score",
        ascending=False,
        kind="stable",
    ).head(5)

    top_records = []

    for _, row in top_source.iterrows():
        risk = str(row["risk_level"])
        segment = str(row["road_segment_id"])
        future_class = str(row["future_congestion_level"])

        description = (
            f"{risk} risk anomaly detected on {segment}. "
            f"Future congestion classification: {future_class}."
        )

        top_records.append({
            "road_segment_id": segment,
            "area": row.get("area"),
            "timestamp": row["timestamp"],
            "risk_level": risk,
            "future_congestion_level": future_class,
            "average_speed_kmh": row.get("average_speed_kmh"),
            "occupancy_percent": row.get("occupancy_percent"),
            "vehicle_count": row.get("vehicle_count"),
            "traffic_flow_vph": row.get("traffic_flow_vph"),
            "anomaly_score": row.get("anomaly_score"),
            "is_anomaly": row.get("is_anomaly"),
            "zscore_anomaly": row.get("zscore_anomaly"),
            "description": description,
        })

    # Scatter chart sample: preserve anomaly representation.
    scatter_source = detected.copy()

    scatter_source["anomaly_label"] = (
        scatter_source["combined_anomaly"]
        .map({1: "Anomaly", 0: "Normal"})
    )

    if len(scatter_source) > 8000:
        anomaly_part = scatter_source[
            scatter_source["combined_anomaly"] == 1
        ]

        normal_part = scatter_source[
            scatter_source["combined_anomaly"] == 0
        ]

        anomaly_limit = min(3000, len(anomaly_part))
        normal_limit = 8000 - anomaly_limit

        if len(anomaly_part) > anomaly_limit:
            anomaly_part = anomaly_part.sample(
                n=anomaly_limit,
                random_state=42,
            )

        if len(normal_part) > normal_limit:
            normal_part = normal_part.sample(
                n=normal_limit,
                random_state=42,
            )

        scatter_source = pd.concat(
            [anomaly_part, normal_part],
            ignore_index=True,
        )

    scatter_columns = [
        "vehicle_count",
        "average_speed_kmh",
        "road_segment_id",
        "future_congestion_level",
        "hour",
        "anomaly_label",
    ]

    scatter_records = _clean_records(
        scatter_source[scatter_columns].to_dict("records")
    )

    # Anomaly-score histogram sample.
    score_source = anomalies["anomaly_score"]

    if len(score_source) > 10000:
        score_source = score_source.sample(
            n=10000,
            random_state=42,
        )

    score_records = [
        {"anomaly_score": float(score)}
        for score in score_source
    ]

    # The dashboard creates its recommendation text locally.
    recommendation_records = _clean_records(top_records)

    return AnomalyAnalysisResponse(
        summary=summary,
        risk_counts=risk_counts,
        method_counts=method_counts,
        monthly=monthly,
        segments=_clean_records(segments),
        top_records=_clean_records(top_records),
        scatter_records=scatter_records,
        score_records=score_records,
        recommendations=recommendation_records,
        peak_month=int(peak_item["month"]),
        peak_count=int(peak_item["total"]),
        affected_segments=int(anomalies["road_segment_id"].nunique()),
        total_segments=int(detected["road_segment_id"].nunique()),
    )