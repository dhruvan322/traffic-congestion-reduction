"""
Unit tests for engine modules.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "junction_id": np.random.choice(["J001", "J002", "J003", "J004", "J005"], n),
        "latitude": np.random.uniform(17.35, 17.45, n),
        "longitude": np.random.uniform(78.40, 78.55, n),
        "vehicle_count": np.random.randint(50, 400, n),
        "avg_speed": np.random.uniform(10, 55, n).round(1),
        "congestion_level": np.random.choice(["Low", "Medium", "High"], n),
        "weather": np.random.choice(["Clear", "Rain"], n),
        "hour": np.random.randint(0, 24, n),
        "timestamp": pd.date_range("2026-05-01", periods=n, freq="h").astype(str),
    })


class TestRouteOptimizer:
    def test_build_graph(self, sample_df):
        from engine.route_optimizer import build_junction_graph
        G = build_junction_graph(sample_df)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_shortest_path(self, sample_df):
        from engine.route_optimizer import build_junction_graph, find_shortest_path
        G = build_junction_graph(sample_df)
        nodes = list(G.nodes())
        result = find_shortest_path(G, nodes[0], nodes[-1])
        assert "path" in result or "error" in result

    def test_least_congested_path(self, sample_df):
        from engine.route_optimizer import build_junction_graph, find_least_congested_path
        G = build_junction_graph(sample_df)
        nodes = list(G.nodes())
        result = find_least_congested_path(G, nodes[0], nodes[-1])
        assert "path" in result or "error" in result

    def test_compare_routes(self, sample_df):
        from engine.route_optimizer import build_junction_graph, compare_routes
        G = build_junction_graph(sample_df)
        nodes = list(G.nodes())
        result = compare_routes(G, nodes[0], nodes[-1])
        assert "shortest" in result
        assert "least_congested" in result


class TestAlertManager:
    def test_congestion_alerts(self, sample_df):
        from engine.alert_manager import check_congestion_alerts
        alerts = check_congestion_alerts(sample_df)
        assert isinstance(alerts, list)
        if alerts:
            assert "alert_type" in alerts[0]
            assert alerts[0]["alert_type"] == "congestion"

    def test_incident_alerts(self):
        from engine.alert_manager import check_incident_alerts
        records = [
            {"junction_id": "J001", "has_incident": True,
             "vehicle_count": 400, "avg_speed": 10},
            {"junction_id": "J002", "has_incident": False,
             "vehicle_count": 100, "avg_speed": 40},
        ]
        alerts = check_incident_alerts(records)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "Critical"
