import pandas as pd
import pytest

import dashboard.command_center as command_center
from engine.reduction_engine import generate_integrated_recommendations


def test_build_prediction_uses_latest_valid_timestamp(monkeypatch):
    seen_rows = []
    monkeypatch.setattr(
        command_center,
        "predict_from_row",
        lambda row: seen_rows.append(row.copy())
        or {
            "predicted_class": "High",
            "confidence": 0.82,
            "probabilities": {"High": 0.82},
        },
    )
    df = pd.DataFrame(
        {
            "timestamp": [
                "2025-03-01 10:00:00",
                "invalid",
                "2025-03-01 12:00:00",
            ],
            "road_segment_id": ["RS0001", "RS0002", "RS0003"],
        },
        index=[8, 2, 8],
    )

    result = command_center._build_prediction(df)

    assert len(seen_rows) == 1
    assert seen_rows[0]["road_segment_id"] == "RS0003"
    assert result == {
        "road_segment_id": "RS0003",
        "congestion_level": "High",
        "confidence": 82.0,
        "probabilities": {"High": 0.82},
        "timestamp": "2025-03-01 12:00:00",
    }


def test_build_prediction_rejects_dataset_without_valid_timestamps(monkeypatch):
    monkeypatch.setattr(
        command_center,
        "predict_from_row",
        lambda row: pytest.fail("Prediction should not run without a valid timestamp"),
    )

    with pytest.raises(ValueError, match="no valid timestamps"):
        command_center._build_prediction(
            pd.DataFrame(
                {
                    "timestamp": ["invalid"],
                    "road_segment_id": ["RS0001"],
                }
            )
        )


def test_hotspot_context_reuses_one_summary_result(monkeypatch):
    summaries = [
        {"cluster_id": 1, "severity": "Medium", "congestion_rate": 0.9},
        {
            "cluster_id": 2,
            "severity": "Critical",
            "congestion_rate": 0.4,
            "road_segments": ["RS0002"],
        },
    ]
    calls = []
    monkeypatch.setattr(
        command_center,
        "get_hotspot_summary",
        lambda df: calls.append(df) or summaries,
    )

    context = command_center._build_hotspot_context(pd.DataFrame())

    assert len(calls) == 1
    assert context["cluster_id"] == 2
    assert context["cluster_count"] == 2
    assert context["summary"] == [summaries[1], summaries[0]]


def test_anomaly_context_reuses_report_for_count_and_priority_alerts(monkeypatch):
    report = [
        {"risk_level": "Low", "anomaly_score": 0.9, "road_segment_id": "RS0001"},
        {"risk_level": "High", "anomaly_score": 0.2, "road_segment_id": "RS0002"},
        {
            "risk_level": "Critical",
            "anomaly_score": 0.1,
            "road_segment_id": "RS0003",
        },
    ]
    calls = []
    monkeypatch.setattr(
        command_center,
        "get_anomaly_report",
        lambda df: calls.append(df) or report,
    )

    context = command_center._build_anomaly_context(pd.DataFrame())

    assert len(calls) == 1
    assert context["count"] == 3
    assert context["road_segment_id"] == "RS0003"
    assert [item["road_segment_id"] for item in context["priority_alerts"]] == [
        "RS0003",
        "RS0002",
    ]
    assert context["records"][0]["road_segment_id"] == "RS0003"


def test_route_failure_keeps_times_unavailable_and_recommendations_safe(monkeypatch):
    def fail_route(**kwargs):
        raise RuntimeError("OSM route unavailable")

    monkeypatch.setattr(command_center, "compare_osm_routes", fail_route)

    route_context = command_center._build_route_context(
        pd.DataFrame(),
        {"timestamp": "2025-03-01 12:00:00"},
    )
    engine_route_context = command_center._route_conditions_for_recommendations(
        route_context
    )
    result = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0001",
            "congestion_level": "Low",
            "confidence": 82.0,
        },
        route_conditions=engine_route_context,
    )

    assert route_context["shortest_time_minutes"] is None
    assert route_context["traffic_aware_time_minutes"] is None
    assert engine_route_context == {
        "alternative_available": False,
        "shortest_time_minutes": None,
        "traffic_aware_time_minutes": None,
    }
    assert result["route"] == {
        "alternative_available": False,
        "shortest_time_minutes": None,
        "traffic_aware_time_minutes": None,
        "time_difference_minutes": None,
    }


def test_route_context_handles_unavailable_metrics(monkeypatch):
    monkeypatch.setattr(
        command_center,
        "compare_osm_routes",
        lambda **kwargs: {
            "shortest_metrics": {"estimated_time_minutes": None},
            "traffic_aware_metrics": {"estimated_time_minutes": None},
        },
    )

    route_context = command_center._build_route_context(
        pd.DataFrame(),
        {"timestamp": "2025-03-01 12:00:00"},
    )

    assert route_context == {
        "alternative_available": False,
        "shortest_time_minutes": None,
        "traffic_aware_time_minutes": None,
    }


def test_recommendation_engine_does_not_accept_alternative_without_times():
    result = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0001",
            "congestion_level": "High",
            "confidence": 80.0,
        },
        route_conditions={
            "alternative_available": True,
            "shortest_time_minutes": None,
            "traffic_aware_time_minutes": None,
        },
    )

    assert result["route"]["alternative_available"] is False
    assert result["route"]["shortest_time_minutes"] is None
    assert not any(
        recommendation["category"] == "Route Diversion"
        for recommendation in result["recommendations"]
    )
