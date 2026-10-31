"""
Unit tests for ML models.
"""

import os
from pathlib import Path
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
            # Legacy fields retained for backward compatibility.
            "congestion_level": np.random.choice(["Low", "Medium", "High"], n),

            # Fields required by the current traffic-intelligence models.
            "road_segment_id": np.random.choice(
                [f"RS{i:04d}" for i in range(1, 21)], n
            ),
            "average_speed_kmh": np.random.uniform(5, 60, n).round(1),
            "occupancy_percent": np.random.uniform(5, 95, n).round(1),
            "traffic_flow_vph": np.random.randint(30, 1000, n),
            "future_congestion_level": np.random.choice(
                ["Free Flow", "Low", "Moderate", "High", "Severe"], n
            ),
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
            assert "road_segment_id" in result[0]


class TestPredictor:
    def test_predict_returns_dict(self):
        """Verify predictions using the current trained model and dataset."""

        project_root = Path(__file__).resolve().parents[1]
        final_dataset_path = (
            project_root
            / "data"
            / "final_dataset"
            / "ml_traffic_dataset.csv"
        )

        if not final_dataset_path.exists():
            pytest.skip("Final ML dataset not generated yet")

        from engine.prediction.predictor import predict_congestion

        final_df = pd.read_csv(final_dataset_path)

        if final_df.empty:
            pytest.skip("Final ML dataset is empty")

        # Pass a real record containing the model's current feature schema.
        result = predict_congestion(final_df.iloc[[0]])

        assert isinstance(result, dict)

        assert {
            "predicted_class",
            "predicted_class_id",
            "probabilities",
            "confidence",
        }.issubset(result)

        expected_classes = {
            "Free Flow", "Low", "Moderate", "High", "Severe"
        }

        assert result["predicted_class"] in expected_classes
        assert isinstance(result["probabilities"], dict)
        assert set(result["probabilities"]) == expected_classes

        assert np.isclose(
            sum(result["probabilities"].values()),
            1.0,
            atol=0.005,
        )

        assert 0.0 <= result["confidence"] <= 1.0
