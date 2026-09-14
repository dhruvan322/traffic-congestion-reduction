import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH
from dashboard.command_center import (
    _build_prediction,
    _build_hotspot_context,
    _build_anomaly_context,
    _build_route_context,
)
from engine.reduction_engine import generate_integrated_recommendations


def main():

    print("Loading final dataset...")

    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("DATASET:", df.shape)

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    prediction = _build_prediction(df)

    print(
        "PREDICTION:",
        prediction["congestion_level"],
        f"{prediction['confidence']:.1f}%",
        prediction["road_segment_id"],
    )

    # ---------------------------------------------------------------
    # Hotspot detection
    # ---------------------------------------------------------------

    hotspot = _build_hotspot_context(df)

    print(
        "HOTSPOT:",
        hotspot["detected"],
        hotspot["severity"],
        hotspot.get("cluster_id"),
    )

    # ---------------------------------------------------------------
    # Anomaly detection
    # ---------------------------------------------------------------

    anomaly = _build_anomaly_context(df)

    print(
        "ANOMALY:",
        anomaly["detected"],
        anomaly["risk_level"],
        anomaly.get("road_segment_id"),
    )

    # ---------------------------------------------------------------
    # OSM route optimization
    # ---------------------------------------------------------------

    route_conditions = _build_route_context(
        df,
        prediction,
    )

    print("ROUTE:", route_conditions)

    # ---------------------------------------------------------------
    # Integrated reduction engine
    # ---------------------------------------------------------------

    integrated = generate_integrated_recommendations(
        prediction=prediction,
        hotspot=hotspot,
        anomaly=anomaly,
        route_conditions=route_conditions,
    )

    print(
        "PRIORITY:",
        integrated["overall_priority"],
    )

    print(
        "RECOMMENDATIONS:",
        len(integrated["recommendations"]),
    )

    print()
    print("COMMAND CENTER INTEGRATION TEST: PASS")


if __name__ == "__main__":
    main()
