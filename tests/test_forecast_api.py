"""API regression tests for multi-horizon traffic forecasting."""

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)

EXPECTED_HORIZONS = {"1h", "2h", "3h", "6h"}
EXPECTED_CLASSES = {
    "Free Flow",
    "Low",
    "Moderate",
    "High",
    "Severe",
}


def assert_valid_forecast_response(payload):
    """Validate the forecast response structure and probability values."""

    assert payload["timestamp"]
    assert payload["road_segment_id"]
    assert payload["area"]

    forecasts = payload["forecasts"]
    assert set(forecasts) == EXPECTED_HORIZONS

    for horizon, result in forecasts.items():
        assert result["horizon_hours"] == int(horizon[:-1])
        assert result["predicted_class"] in EXPECTED_CLASSES
        assert 0 <= result["confidence"] <= 1

        probabilities = result["probabilities"]
        assert set(probabilities) == EXPECTED_CLASSES

        assert all(
            0 <= value <= 1
            for value in probabilities.values()
        )

        assert abs(sum(probabilities.values()) - 1.0) <= 0.005

        assert result["predicted_class"] == max(
            probabilities,
            key=probabilities.get,
        )


def test_forecast_by_road_segment():
    response = client.post(
        "/api/v1/forecast",
        json={"road_segment_id": "RS0031"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["road_segment_id"] == "RS0031"

    assert_valid_forecast_response(payload)


def test_forecast_by_area():
    response = client.post(
        "/api/v1/forecast",
        json={"area": "Hitech City"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["area"] == "Hitech City"

    assert_valid_forecast_response(payload)


def test_forecast_requires_location():
    response = client.post(
        "/api/v1/forecast",
        json={},
    )

    assert response.status_code == 422
    assert "Provide either area or road_segment_id." in str(
        response.json()
    )
