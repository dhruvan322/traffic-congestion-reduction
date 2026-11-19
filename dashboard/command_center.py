import math
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dashboard.ui_theme import (
    COLOR_TEXT_DIM,
    apply_plotly_theme,
    render_header,
    render_kpi_card,
)
from engine.multihorizon_predictor import (
    forecast_all_horizons,
    get_latest_row_for_segment,
)
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
    """Predict from the latest row with a valid historical timestamp."""
    if df.empty:
        raise ValueError("Cannot predict from an empty traffic dataset.")
    if "timestamp" not in df.columns:
        raise ValueError("Traffic dataset is missing the timestamp column.")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    valid_timestamps = timestamps.dropna()
    if valid_timestamps.empty:
        raise ValueError("Traffic dataset has no valid timestamps.")

    latest_position = int(timestamps.reset_index(drop=True).idxmax())
    row = df.iloc[latest_position].copy()

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
            "cluster_count": 0,
            "summary": [],
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
        "cluster_count": len(summaries),
        "summary": summaries,
    }


def _build_anomaly_context(df):
    """Run the verified combined anomaly detector."""

    anomalies = get_anomaly_report(df)

    if not anomalies:
        return {
            "detected": False,
            "risk_level": "Normal",
            "count": 0,
            "priority_alerts": [],
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
    priority_alerts = [
        item
        for item in anomalies
        if str(item.get("risk_level", "")).lower() in {"high", "critical"}
    ]

    return {
        "detected": True,
        "risk_level": str(anomaly.get("risk_level", "Low")),
        "road_segment_id": str(anomaly.get("road_segment_id", "Unknown")),
        "anomaly_score": float(anomaly.get("anomaly_score", 0)),
        "description": str(anomaly.get("description", "")),
        "count": len(anomalies),
        "priority_alerts": priority_alerts,
        "records": anomalies,
    }


def _route_conditions_for_recommendations(route_context):
    """Pass only available route metrics to the recommendation engine."""

    def valid_time(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    shortest_time = valid_time(route_context.get("shortest_time_minutes"))
    traffic_time = valid_time(route_context.get("traffic_aware_time_minutes"))
    alternative_available = (
        bool(route_context.get("alternative_available"))
        and shortest_time is not None
        and traffic_time is not None
    )

    return {
        "alternative_available": alternative_available,
        "shortest_time_minutes": shortest_time,
        "traffic_aware_time_minutes": traffic_time,
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

        def valid_time(metrics):
            value = metrics.get("estimated_time_minutes")
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) and value >= 0 else None

        shortest_time = valid_time(shortest)
        traffic_time = valid_time(traffic_aware)

        alternative_available = (
            shortest_time is not None
            and traffic_time is not None
            and abs(shortest_time - traffic_time) > 0.01
        )

        return {
            "alternative_available": alternative_available,
            "shortest_time_minutes": (
                shortest_time
            ),
            "traffic_aware_time_minutes": (
                traffic_time
            ),
        }

    except Exception as exc:
        return {
            "alternative_available": False,
            "shortest_time_minutes": None,
            "traffic_aware_time_minutes": None,
            "error": str(exc),
        }


def _render_congestion_distribution(distribution):
    class_names = distribution["Congestion class"].tolist()
    counts = distribution["Records"].tolist()
    percentages = distribution["Percentage"].tolist()
    class_colors = {
        "Free Flow": "#10B981",
        "Low": "#00E5BC",
        "Moderate": "#00F0FF",
        "High": "#F59E0B",
        "Severe": "#EF4444",
    }
    marker_colors = [
        class_colors.get(class_name, COLOR_TEXT_DIM)
        for class_name in class_names
    ]

    figure = go.Figure(
        go.Bar(
            x=counts,
            y=class_names,
            orientation="h",
            marker={"color": marker_colors},
            text=[
                f"{count:,} · {percentage:.1f}%"
                for count, percentage in zip(counts, percentages)
            ],
            textposition="outside",
            customdata=percentages,
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Records: %{x:,}<br>"
                "Share: %{customdata:.1f}%<extra></extra>"
            ),
            showlegend=False,
        )
    )
    apply_plotly_theme(figure)
    figure.update_layout(
        height=280,
        barmode="overlay",
        margin=dict(l=12, r=64, t=12, b=36),
    )
    figure.update_xaxes(
        title_text="Historical records",
        range=[0, max(max(counts, default=0) * 1.3, 1)],
        rangemode="tozero",
    )
    figure.update_yaxes(
        title_text=None,
        categoryorder="array",
        categoryarray=list(reversed(class_names)),
    )
    st.plotly_chart(
        figure,
        theme=None,
        width="stretch",
        config={"responsive": True},
    )


def render_command_center(data_context):
    render_header(
        title="Executive Traffic Operations Command Center",
        subtitle=(
            "Historical traffic signals, model forecasts, and evidence-based "
            "operational recommendations."
        ),
        badge_text="Historical synthetic dataset",
        badge_type="teal",
    )
    df = data_context["traffic_data"].copy()

    if df.empty:
        st.warning("Traffic dataset is empty.")
        return

    with st.spinner("Evaluating the latest available historical record..."):
        prediction = _build_prediction(df)
        hotspot = _build_hotspot_context(df)
        anomaly = _build_anomaly_context(df)
        route_conditions = _build_route_context(df, prediction)
        integrated = generate_integrated_recommendations(
            prediction=prediction,
            hotspot=hotspot,
            anomaly=anomaly,
            route_conditions=_route_conditions_for_recommendations(route_conditions),
        )
        forecast_row = get_latest_row_for_segment(
            df,
            prediction["road_segment_id"],
        )
        forecast_payload = forecast_all_horizons(forecast_row)

    as_of = pd.to_datetime(prediction["timestamp"], errors="coerce")
    as_of_label = (
        as_of.strftime("%d %b %Y · %H:%M")
        if pd.notna(as_of)
        else prediction["timestamp"]
    )
    st.caption(
        f"As of {as_of_label} · Selected record {prediction['road_segment_id']} · "
        "Not live traffic"
    )

    if "road_segment_id" in df.columns:
        segment_count = f"{df['road_segment_id'].nunique():,}"
    else:
        segment_count = "Unavailable"

    if "future_congestion_level" in df.columns:
        congestion_levels = df["future_congestion_level"].astype("string").str.strip()
        high_severe_count = int(congestion_levels.isin(["High", "Severe"]).sum())
        class_counts = congestion_levels.value_counts()
        class_order = ["Free Flow", "Low", "Moderate", "High", "Severe"]
        class_order.extend(
            sorted(class_name for class_name in class_counts.index if class_name not in class_order)
        )
        class_counts = class_counts.reindex(class_order, fill_value=0)
        class_total = int(congestion_levels.notna().sum())
        congestion_distribution = pd.DataFrame(
            {
                "Congestion class": class_order,
                "Records": class_counts.to_numpy(),
                "Percentage": (
                    class_counts.to_numpy() / class_total * 100
                    if class_total
                    else 0
                ),
            }
        )
    else:
        high_severe_count = None
        congestion_distribution = None

    priority = integrated["overall_priority"]
    priority_status = {
        "Critical": "red",
        "High": "amber",
        "Medium": "cyan",
        "Low": "emerald",
    }.get(priority, "blue")
    priority_markdown_color = {
        "Critical": "red",
        "High": "orange",
        "Medium": "blue",
        "Low": "green",
    }.get(priority, "blue")
    kpi_columns = st.columns(4)
    kpis = (
        (
            "MONITORED ROAD SEGMENTS",
            segment_count,
            "Unique IDs in the historical dataset",
            "cyan",
        ),
        (
            "HIGH / SEVERE RECORDS",
            f"{high_severe_count:,}" if high_severe_count is not None else "N/A",
            "Historical congestion classifications",
            "amber",
        ),
        (
            "DETECTED ANOMALIES",
            f"{anomaly['count']:,}",
            "Combined historical detector results",
            "red" if anomaly["count"] else "teal",
        ),
        (
            "OPERATIONAL PRIORITY",
            priority.upper(),
            "From the latest historical analysis",
            priority_status,
        ),
    )
    for column, (title, value, subtitle, status) in zip(kpi_columns, kpis):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    st.subheader("Operational decisions")
    decision_column, recommendations_column = st.columns([0.85, 1.8])
    with decision_column:
        with st.container(border=True):
            st.markdown(f"### :{priority_markdown_color}[{priority} priority]")
            st.write(integrated["decision_summary"])

    recommendations = integrated["recommendations"]
    with recommendations_column:
        if recommendations:
            recommendation_columns = st.columns(2)
            for index, recommendation in enumerate(recommendations):
                with recommendation_columns[index % 2]:
                    with st.container(border=True):
                        st.markdown(
                            f"**{recommendation.get('category', 'Operational action')}**"
                        )
                        st.markdown(
                            f"Priority: **{recommendation.get('priority', priority)}**"
                        )
                        st.write(recommendation.get("action", ""))
                        reason = recommendation.get("reason")
                        if reason:
                            st.caption(f"Supporting reason: {reason}")
        else:
            with st.container(border=True):
                st.caption("No operational recommendations were returned.")

    st.subheader("Priority observations")
    st.caption("Historical detector observations · Not a live alert feed")
    anomaly_column, hotspot_column = st.columns(2)
    critical_anomalies = [
        record
        for record in anomaly["priority_alerts"]
        if str(record.get("risk_level", "")).lower() == "critical"
    ]
    hotspot_summaries = hotspot.get("summary", [])
    high_priority_hotspots = [
        summary
        for summary in hotspot_summaries
        if str(summary.get("severity", "")).lower() in {"high", "critical"}
    ]

    with anomaly_column:
        with st.container(border=True):
            st.markdown("#### Representative critical anomaly")
            st.caption(
                f"{anomaly['count']:,} combined anomaly records · "
                f"{len(critical_anomalies):,} critical observations"
            )
            if critical_anomalies:
                record = critical_anomalies[0]
                st.markdown(f"**{record['risk_level']}** · {record['road_segment_id']}")
                st.caption(
                    f"{record['timestamp']} · "
                    f"{record['future_congestion_level']} congestion"
                )
                st.write(record["description"])
                st.caption(
                    f"Score {record['anomaly_score']:.4f} · "
                    f"{record['vehicle_count']:,} vehicles · "
                    f"{record['average_speed_kmh']:.1f} km/h"
                )
            else:
                st.write("No critical anomaly observation was returned.")

    with hotspot_column:
        with st.container(border=True):
            st.markdown("#### Representative high-priority hotspot")
            st.caption(
                f"{len(hotspot_summaries):,} hotspot clusters in the detector summary"
            )
            if high_priority_hotspots:
                summary = high_priority_hotspots[0]
                segments = summary.get("road_segments", [])
                displayed_segments = ", ".join(map(str, segments[:4]))
                if len(segments) > 4:
                    displayed_segments += f", +{len(segments) - 4} more"
                st.markdown(
                    f"**Cluster {summary['cluster_id']}** · {summary['severity']} · "
                    f"{summary['size']} segments"
                )
                st.caption(f"Segments: {displayed_segments or 'Unavailable'}")
                st.write(
                    f"Congestion {summary['congestion_rate']:.1%} · "
                    f"High/Severe {summary['high_or_severe_rate']:.1%}"
                )
                st.caption(
                    f"Avg. {summary['avg_vehicles']:.1f} vehicles · "
                    f"{summary['avg_speed']:.1f} km/h · "
                    f"Peak hours: {', '.join(map(str, summary['peak_hours'])) or 'Unavailable'}"
                )
            else:
                st.write("No High/Critical hotspot observation was returned.")

        if hotspot_summaries:
            with st.expander(
                f"Full hotspot summary · {len(hotspot_summaries):,} clusters"
            ):
                st.dataframe(
                    pd.DataFrame(hotspot_summaries),
                    hide_index=True,
                )

    st.subheader("Traffic intelligence summary")
    summary_column, context_column = st.columns([1, 1.2])
    with summary_column:
        with st.container(border=True):
            st.markdown("#### Congestion-class distribution")
            if congestion_distribution is not None:
                _render_congestion_distribution(congestion_distribution)
                st.caption(
                    f"Percentages use {int(congestion_distribution['Records'].sum()):,} "
                    "records with a classified congestion level."
                )
            else:
                st.warning(
                    "Historical congestion distribution is unavailable because "
                    "future_congestion_level is missing."
                )
    with context_column:
        with st.container(border=True):
            st.markdown("#### Prediction summary")
            st.markdown(
                f"**{prediction['congestion_level']}** · "
                f"{prediction['confidence']:.1f}% model confidence"
            )
            st.write(f"Road segment: **{prediction['road_segment_id']}**")
            st.caption(f"Input record: {as_of_label}")

        with st.container(border=True):
            st.markdown("#### Historical route evaluation")
            shortest_time = route_conditions["shortest_time_minutes"]
            traffic_time = route_conditions["traffic_aware_time_minutes"]
            route_columns = st.columns(3)
            route_columns[0].metric(
                "Shortest",
                f"{shortest_time:.2f} min" if shortest_time is not None else "N/A",
            )
            route_columns[1].metric(
                "Traffic-aware",
                f"{traffic_time:.2f} min" if traffic_time is not None else "N/A",
            )
            if shortest_time is not None and traffic_time is not None:
                time_difference = traffic_time - shortest_time
                difference_label = f"{time_difference:+.2f} min"
            else:
                difference_label = "N/A"
            route_columns[2].metric("Difference", difference_label)
            if route_conditions.get("error"):
                st.caption(
                    f"Route evaluation unavailable: {route_conditions['error']}"
                )
            else:
                st.caption(
                    "Historical OSM route estimates; not live navigation ETAs."
                )

    st.subheader("Multi-horizon forecast")
    st.caption(
        "Model forecasts from the selected segment's latest available historical "
        "record; these are not live traffic readings."
    )
    st.caption(
        f"Input: {forecast_payload['road_segment_id']} · "
        f"{forecast_payload['timestamp']}"
    )
    forecast_columns = st.columns(4)
    congestion_status = {
        "Free Flow": "emerald",
        "Low": "teal",
        "Moderate": "cyan",
        "High": "amber",
        "Severe": "red",
    }
    for column, horizon in zip(forecast_columns, (1, 2, 3, 6)):
        result = forecast_payload["forecasts"][f"{horizon}h"]
        with column:
            render_kpi_card(
                title=f"+{horizon} HOUR FORECAST",
                value=result["predicted_class"],
                subtitle=f"{result['confidence']:.1%} model confidence",
                status=congestion_status.get(result["predicted_class"], "cyan"),
            )

    st.subheader("System intelligence status")
    st.caption(
        "File presence is checked separately from runtime execution in this view."
    )
    project_root = Path(__file__).resolve().parents[1]
    component_files = {
        "Congestion prediction": (
            "engine/prediction/predictor.py",
            "trained_models/xgb_congestion_final.json",
            "trained_models/traffic_preprocessor.joblib",
            "trained_models/congestion_label_encoder.joblib",
            "trained_models/feature_columns.joblib",
        ),
        "Multi-horizon forecasting": (
            "engine/multihorizon_predictor.py",
            *tuple(
                f"trained_models/multihorizon/{horizon}h/{artifact}"
                for horizon in (1, 2, 3, 6)
                for artifact in (
                    "xgb_congestion.json",
                    "traffic_preprocessor.joblib",
                    "label_encoder.joblib",
                )
            ),
        ),
        "Hotspot detection": ("models/hotspot_detector.py",),
        "Anomaly detection": ("models/anomaly_detector.py",),
        "Route optimization": ("engine/osm_traffic_router.py",),
        "AI chat assistant": ("dashboard/nlp_panel.py", "nlp/assistant.py"),
    }
    runtime_status = {
        "Congestion prediction": (
            f"Verified for {prediction['road_segment_id']} at {as_of_label}"
        ),
        "Multi-horizon forecasting": (
            f"Verified for {forecast_payload['road_segment_id']}"
        ),
        "Hotspot detection": (
            f"Completed · {hotspot['cluster_count']:,} clusters returned"
        ),
        "Anomaly detection": (
            f"Completed · {anomaly['count']:,} combined observations returned"
        ),
        "Route optimization": (
            "Failed"
            if route_conditions.get("error")
            else (
                "Completed with route estimates"
                if route_conditions["shortest_time_minutes"] is not None
                and route_conditions["traffic_aware_time_minutes"] is not None
                else "Completed without route estimates"
            )
        ),
        "AI chat assistant": "Not checked in this dashboard view",
    }
    status_columns = st.columns(3)
    for index, (component, paths) in enumerate(component_files.items()):
        files_available = all((project_root / path).is_file() for path in paths)
        with status_columns[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{component}**")
                st.write(
                    "Checked files: Present"
                    if files_available
                    else "Checked files: Missing"
                )
                st.caption(f"Runtime: {runtime_status[component]}")
                if component == "Route optimization" and route_conditions.get("error"):
                    st.caption(route_conditions["error"])

    st.subheader("Decision evidence")
    evidence = integrated.get("evidence", [])
    if evidence:
        evidence_columns = st.columns(2)
        for index, item in enumerate(evidence):
            with evidence_columns[index % 2]:
                with st.container(border=True):
                    st.write(item)
    else:
        st.caption("The recommendation engine returned no evidence items.")

    st.caption(
        "Historical synthetic observations only · No live traffic feed. "
        "Forecasts and recommendations are model-based decision support; "
        "they do not confirm deployed interventions or real-world congestion reduction."
    )
