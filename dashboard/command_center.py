import streamlit as st
import pandas as pd

from engine.prediction.predictor import predict_from_row
from models.hotspot_detector import get_hotspot_summary
from models.anomaly_detector import get_anomaly_report
from engine.osm_traffic_router import compare_osm_routes
from engine.reduction_engine import generate_integrated_recommendations


def _normalize_confidence(confidence):
    """
    Convert prediction confidence to percentage form.

    predictor.py returns confidence as a fraction between 0 and 1.
    """
    confidence = float(confidence)

    if confidence <= 1.0:
        return confidence * 100.0

    return confidence


def _build_prediction(df):
    """Generate a real ML prediction from a representative traffic row."""

    row = df.iloc[0].copy()

    result = predict_from_row(row)

    return {
        "road_segment_id": str(row["road_segment_id"]),
        "congestion_level": str(result["predicted_class"]),
        "confidence": _normalize_confidence(result["confidence"]),
        "probabilities": result["probabilities"],
        "timestamp": str(row["timestamp"]),
    }


def _build_hotspot_context(df):
    """Run the verified hotspot detector and select the most important hotspot."""

    summaries = get_hotspot_summary(df)

    if not summaries:
        return {
            "detected": False,
            "severity": "None",
            "cluster_id": None,
        }

    # Highest severity first, then highest congestion rate.
    severity_rank = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    summaries = sorted(
        summaries,
        key=lambda item: (
            severity_rank.get(str(item.get("severity", "Low")), 0),
            float(item.get("congestion_rate", 0)),
        ),
        reverse=True,
    )

    hotspot = summaries[0]

    return {
        "detected": True,
        "severity": str(hotspot.get("severity", "Medium")),
        "cluster_id": hotspot.get("cluster_id"),
        "congestion_rate": float(hotspot.get("congestion_rate", 0)),
        "road_segments": hotspot.get("road_segments", []),
        "summary": summaries,
    }


def _build_anomaly_context(df):
    """Run the verified combined anomaly detector."""

    anomalies = get_anomaly_report(df)

    if not anomalies:
        return {
            "detected": False,
            "risk_level": "Normal",
            "records": [],
        }

    risk_rank = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
        "Normal": 0,
    }

    anomalies = sorted(
        anomalies,
        key=lambda item: (
            risk_rank.get(str(item.get("risk_level", "Low")), 0),
            float(item.get("anomaly_score", 0)),
        ),
        reverse=True,
    )

    anomaly = anomalies[0]

    return {
        "detected": True,
        "risk_level": str(anomaly.get("risk_level", "Low")),
        "road_segment_id": str(anomaly.get("road_segment_id", "Unknown")),
        "anomaly_score": float(anomaly.get("anomaly_score", 0)),
        "description": str(anomaly.get("description", "")),
        "records": anomalies,
    }


def _build_route_context(df, prediction):
    """
    Run the verified OSM traffic-aware router.

    The selected prediction row determines the traffic timestamp.
    A project road segment is used as the origin/destination pair
    through the verified route comparison module.
    """

    try:
        timestamp = pd.to_datetime(prediction["timestamp"])

        result = compare_osm_routes(
            origin=(17.3850, 78.4867),
            destination=(17.4399, 78.4983),
            timestamp=timestamp,
        )

        if not result:
            return {
                "alternative_available": False,
                "shortest_time_minutes": None,
                "traffic_aware_time_minutes": None,
            }

        shortest = result.get("shortest_metrics", {})
        traffic_aware = result.get("traffic_aware_metrics", {})

        shortest_time = shortest.get("estimated_time_minutes")
        traffic_time = traffic_aware.get("estimated_time_minutes")

        alternative_available = (
            shortest_time is not None
            and traffic_time is not None
            and abs(float(shortest_time) - float(traffic_time)) > 0.01
        )

        return {
            "alternative_available": alternative_available,
            "shortest_time_minutes": (
                float(shortest_time) if shortest_time is not None else None
            ),
            "traffic_aware_time_minutes": (
                float(traffic_time) if traffic_time is not None else None
            ),
        }

    except Exception as exc:
        return {
            "alternative_available": False,
            "shortest_time_minutes": None,
            "traffic_aware_time_minutes": None,
            "error": str(exc),
        }


def render_command_center(data_context):

    st.title("Executive Traffic Command Center")

    st.markdown(
        "Unified traffic intelligence combining prediction, hotspots, "
        "anomalies, route optimization, and reduction recommendations."
    )

    st.markdown("---")

    df = data_context["traffic_data"].copy()

    if df.empty:
        st.warning("Traffic dataset is empty.")
        return

    # ---------------------------------------------------------------
    # Core traffic KPIs
    # ---------------------------------------------------------------

    total_vehicles = int(df["vehicle_count"].sum())
    avg_speed = float(df["average_speed_kmh"].mean())
    avg_occupancy = float(df["occupancy_percent"].mean())

    if "future_congestion_level" in df.columns:
        severe_high = int(df["future_congestion_level"].isin(["High", "Severe"]).sum())
    else:
        severe_high = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Vehicles",
        f"{total_vehicles:,}",
    )

    col2.metric(
        "Average Speed",
        f"{avg_speed:.1f} km/h",
    )

    col3.metric(
        "Avg Occupancy",
        f"{avg_occupancy:.1f}%",
    )

    col4.metric(
        "High / Severe Records",
        f"{severe_high:,}",
    )

    st.markdown("---")

    # ---------------------------------------------------------------
    # Run integrated intelligence
    # ---------------------------------------------------------------

    with st.spinner("Running traffic intelligence..."):

        prediction = _build_prediction(df)

        hotspot = _build_hotspot_context(df)

        anomaly = _build_anomaly_context(df)

        route_conditions = _build_route_context(
            df,
            prediction,
        )

        integrated = generate_integrated_recommendations(
            prediction=prediction,
            hotspot=hotspot,
            anomaly=anomaly,
            route_conditions=route_conditions,
        )

    # ---------------------------------------------------------------
    # Intelligence summary
    # ---------------------------------------------------------------

    st.subheader("Current Traffic Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted Congestion",
        prediction["congestion_level"],
    )

    col2.metric(
        "Prediction Confidence",
        f"{prediction['confidence']:.1f}%",
    )

    col3.metric(
        "Hotspot Status",
        hotspot["severity"] if hotspot["detected"] else "None",
    )

    col4.metric(
        "Anomaly Risk",
        anomaly["risk_level"],
    )

    st.markdown("---")

    # ---------------------------------------------------------------
    # Operational priority
    # ---------------------------------------------------------------

    priority = integrated["overall_priority"]

    if priority == "Critical":
        st.error(f"Operational Priority: {priority}")
    elif priority == "High":
        st.warning(f"Operational Priority: {priority}")
    elif priority == "Medium":
        st.info(f"Operational Priority: {priority}")
    else:
        st.success(f"Operational Priority: {priority}")

    # ---------------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------------

    st.subheader("Decision Evidence")

    for evidence in integrated["evidence"]:
        st.write(f"• {evidence}")

    # ---------------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------------

    st.subheader("Recommended Actions")

    recommendations = integrated["recommendations"]

    if recommendations:

        for recommendation in recommendations:

            category = recommendation.get(
                "category",
                "Traffic Management",
            )

            action = recommendation.get(
                "action",
                "Monitor traffic conditions.",
            )

            reason = recommendation.get(
                "reason",
                "",
            )

            st.markdown(f"**{category}:** {action}")

            if reason:
                st.caption(reason)

    else:
        st.info("No operational recommendations generated.")

    st.markdown("---")

    # ---------------------------------------------------------------
    # Hotspot details
    # ---------------------------------------------------------------

    st.subheader("Top Hotspot")

    if hotspot["detected"]:

        st.write(f"**Cluster:** {hotspot['cluster_id']}")

        st.write(f"**Severity:** {hotspot['severity']}")

        st.write(f"**Congestion Rate:** " f"{hotspot['congestion_rate']:.1%}")

        segments = hotspot.get("road_segments", [])

        if segments:
            st.write("**Road Segments:** " + ", ".join(map(str, segments)))

    else:
        st.success("No hotspot detected.")

    # ---------------------------------------------------------------
    # Anomaly details
    # ---------------------------------------------------------------

    st.subheader("Highest-Risk Anomaly")

    if anomaly["detected"]:

        st.write(f"**Road Segment:** " f"{anomaly['road_segment_id']}")

        st.write(f"**Risk Level:** " f"{anomaly['risk_level']}")

        st.write(f"**Anomaly Score:** " f"{anomaly['anomaly_score']:.4f}")

        st.caption(anomaly["description"])

    else:
        st.success("No combined anomaly detected.")

    # ---------------------------------------------------------------
    # Route intelligence
    # ---------------------------------------------------------------

    st.subheader("Route Intelligence")

    if "error" in route_conditions:

        st.warning("Route optimization could not be evaluated for this scenario.")

        st.caption(route_conditions["error"])

    else:

        col1, col2, col3 = st.columns(3)

        shortest_time = route_conditions["shortest_time_minutes"]
        traffic_time = route_conditions["traffic_aware_time_minutes"]

        col1.metric(
            "Shortest Route",
            (f"{shortest_time:.2f} min" if shortest_time is not None else "N/A"),
        )

        col2.metric(
            "Traffic-Aware Route",
            (f"{traffic_time:.2f} min" if traffic_time is not None else "N/A"),
        )

        if shortest_time is not None and traffic_time is not None:

            difference = traffic_time - shortest_time

            col3.metric(
                "Time Difference",
                f"{difference:+.2f} min",
            )

        else:

            col3.metric(
                "Time Difference",
                "N/A",
            )

    # ---------------------------------------------------------------
    # Decision summary
    # ---------------------------------------------------------------

    st.markdown("---")

    st.subheader("Executive Decision Summary")

    st.info(integrated["decision_summary"])
