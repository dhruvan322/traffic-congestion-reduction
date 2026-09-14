"""
Unit tests for ML models.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_PATH


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=n, freq="h").astype(str),
            "junction_id": np.random.choice(
                ["J001", "J002", "J003", "J004", "J005"], n
            ),
            "latitude": np.random.uniform(17.35, 17.45, n),
            "longitude": np.random.uniform(78.40, 78.55, n),
            "vehicle_count": np.random.randint(50, 400, n),
            "avg_speed": np.random.uniform(5, 60, n).round(1),
            "weather": np.random.choice(["Clear", "Rain", "Fog", "Cloudy"], n),
            "day_of_week": np.random.randint(0, 7, n),
            "hour": np.random.randint(0, 24, n),
            "is_peak": np.random.randint(0, 2, n),
            "is_event": np.random.randint(0, 2, n),
            "is_construction": np.random.randint(0, 2, n),
            "congestion_level": np.random.choice(["Low", "Medium", "High"], n),
        }
    )


class TestHotspotDetector:
    def test_hotspot_summary_returns_list(self, sample_df):
        from models.hotspot_detector import get_hotspot_summary

        result = get_hotspot_summary(sample_df)
        assert isinstance(result, list)

    def test_hotspot_fields(self, sample_df):
        from models.hotspot_detector import get_hotspot_summary

        result = get_hotspot_summary(sample_df)
        if result:
            spot = result[0]
            assert "cluster_id" in spot
            assert "severity" in spot
            assert "center_lat" in spot


class TestAnomalyDetector:
    def test_detect_anomalies_adds_column(self, sample_df):
        from models.anomaly_detector import detect_anomalies

        result = detect_anomalies(sample_df)
        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns

    def test_zscore_anomalies_adds_column(self, sample_df):
        from models.anomaly_detector import zscore_anomalies

        result = zscore_anomalies(sample_df)
        assert "zscore_anomaly" in result.columns

    def test_anomaly_report_returns_list(self, sample_df):
        from models.anomaly_detector import get_anomaly_report

        result = get_anomaly_report(sample_df)
        assert isinstance(result, list)
        if result:
            assert "risk_level" in result[0]
            assert "junction_id" in result[0]


class TestPredictor:
    def test_predict_returns_dict(self, sample_df):
        """Test prediction output format (may auto-train)."""
        # Skip if dataset not available
        if not os.path.exists(DATASET_PATH):
            pytest.skip("Dataset not generated yet")

        from engine.prediction.predictor import predict_congestion

        result = predict_congestion(
            {
                "hour": 9,
                "day_of_week": 1,
                "vehicle_count": 250,
                "avg_speed": 25,
                "is_peak": 1,
                "is_event": 0,
                "is_construction": 0,
                "weather": "Clear",
                "junction_id": "J001",
            }
        )
        assert "predicted_level" in result
        assert "confidence" in result
        assert result["predicted_level"] in ["Low", "Medium", "High"]
