"""
Overview Dashboard Panel.

Enterprise-grade traffic intelligence overview using:
- project traffic dataset
- hotspot detection
- anomaly detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from config import DATASET_PATH
from models.hotspot_detector import get_hotspot_summary
from models.anomaly_detector import get_anomaly_report
from dashboard.ui_theme import (
    render_header,
    render_kpi_card,
    apply_plotly_theme,
    CONGESTION_COLOR_MAP,
    COLOR_CYAN,
    COLOR_TEAL,
    COLOR_BLUE,
    COLOR_AMBER,
)


def render_overview():
    """Render the main traffic intelligence operations overview."""

    render_header(
        title="📊 Traffic Intelligence Overview",
        subtitle="Historical traffic analytics, congestion distribution, hotspot detection and anomaly intelligence.",
        badge_text="HISTORICAL DATASET",
        badge_type="teal",
    )

    # -------------------------------------------------------------
    # Load final dataset
    # -------------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    required_columns = {
        "vehicle_count",
        "average_speed_kmh",
        "future_congestion_level",
        "occupancy_percent",
        "weather_condition",
        "hour",
        "road_segment_id",
        "area",
    }

    missing = sorted(required_columns.difference(df.columns))

    if missing:
        st.error("The final traffic dataset is missing required columns.")
        st.write(missing)
        return

    # -------------------------------------------------------------
    # KPI Calculations
    # -------------------------------------------------------------

    total_records = len(df)
    monitored_segments = df["road_segment_id"].nunique()
    avg_vehicles = df["vehicle_count"].mean()
    avg_speed = df["average_speed_kmh"].mean()

    congested_mask = df["future_congestion_level"].isin(
        [
            "Moderate",
            "High",
            "Severe",
        ]
    )

    congested_pct = (
        congested_mask.mean() * 100 if total_records > 0 else 0.0
    )

    # -------------------------------------------------------------
    # Primary KPI Ribbon
    # -------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card(
            title="Total Records",
            value=f"{total_records:,}",
            status="cyan",
            icon="📦",
            subtitle="Historical observations",
        )

    with col2:
        render_kpi_card(
            title="Monitored Segments",
            value=f"{monitored_segments:,}",
            status="teal",
            icon="📍",
            subtitle="Active road network",
        )

    with col3:
        speed_status = "emerald" if avg_speed >= 35 else ("cyan" if avg_speed >= 25 else "amber")
        render_kpi_card(
            title="Average Speed",
            value=f"{avg_speed:.1f} km/h",
            status=speed_status,
            icon="⚡",
            subtitle=f"Flow: {avg_vehicles:.0f} veh/seg avg",
        )

    with col4:
        congested_status = "red" if congested_pct >= 40 else ("amber" if congested_pct >= 20 else "emerald")
        render_kpi_card(
            title="Moderate-to-Severe Congestion",
            value=f"{congested_pct:.1f}%",
            status=congested_status,
            icon="🔴",
            subtitle="Corridors under load",
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Traffic Summary & Distribution
    # -------------------------------------------------------------

    st.markdown(
        '<div class="ops-section-title">🚦 Traffic State Distribution</div>',
        unsafe_allow_html=True,
    )

    congestion_order = [
        "Free Flow",
        "Low",
        "Moderate",
        "High",
        "Severe",
    ]

    congestion_counts = (
        df["future_congestion_level"]
        .value_counts()
        .reindex(
            congestion_order,
            fill_value=0,
        )
    )

    congestion_percent = (
        congestion_counts / total_records * 100
        if total_records > 0
        else congestion_counts * 0
    )

    summary_df = pd.DataFrame(
        {
            "Congestion Level": congestion_counts.index,
            "Observations": congestion_counts.values,
            "Percentage": congestion_percent.round(1).values,
        }
    )

    dist_col1, dist_col2 = st.columns([5, 7])

    with dist_col1:
        display_summary = summary_df.copy()
        display_summary["Percentage"] = display_summary["Percentage"].map(
            lambda v: f"{v:.1f}%"
        )
        display_summary["Observations"] = display_summary["Observations"].map(
            lambda v: f"{v:,}"
        )
        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True,
        )

    with dist_col2:
        fig_distribution = px.bar(
            summary_df,
            x="Congestion Level",
            y="Observations",
            color="Congestion Level",
            color_discrete_map=CONGESTION_COLOR_MAP,
            text="Observations",
        )

        fig_distribution.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
        )

        fig_distribution.update_layout(
            height=320,
            showlegend=False,
            yaxis_title="Observations",
            xaxis_title="Congestion Level",
        )

        apply_plotly_theme(fig_distribution)

        st.plotly_chart(
            fig_distribution,
            use_container_width=True,
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Traffic speed vs vehicles correlation
    # -------------------------------------------------------------

    st.markdown(
        '<div class="ops-section-title">🔬 Vehicle Count vs Average Speed Correlation</div>',
        unsafe_allow_html=True,
    )

    sample_size = min(
        2000,
        len(df),
    )

    sample = df.sample(
        sample_size,
        random_state=42,
    )

    fig_scatter = px.scatter(
        sample,
        x="vehicle_count",
        y="average_speed_kmh",
        color="future_congestion_level",
        color_discrete_map=CONGESTION_COLOR_MAP,
        hover_data=[
            "road_segment_id",
            "area",
            "hour",
        ],
        opacity=0.7,
    )

    fig_scatter.update_layout(
        xaxis_title="Vehicle Count",
        yaxis_title="Average Speed (km/h)",
        height=420,
        legend_title="Congestion Level",
    )

    apply_plotly_theme(fig_scatter)

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Weather impact
    # -------------------------------------------------------------

    st.markdown(
        '<div class="ops-section-title">🌤️ Meteorological Impact Analysis</div>',
        unsafe_allow_html=True,
    )

    weather_stats = (
        df.groupby("weather_condition")
        .agg(
            average_vehicle_count=(
                "vehicle_count",
                "mean",
            ),
            average_speed_kmh=(
                "average_speed_kmh",
                "mean",
            ),
            observations=(
                "road_segment_id",
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "average_vehicle_count",
            ascending=False,
        )
    )

    w_col1, w_col2 = st.columns([6, 6])

    with w_col1:
        fig_weather = px.bar(
            weather_stats,
            x="weather_condition",
            y="average_vehicle_count",
            text="average_vehicle_count",
            color="weather_condition",
            color_discrete_sequence=[COLOR_CYAN, COLOR_TEAL, COLOR_BLUE, COLOR_AMBER],
        )

        fig_weather.update_traces(
            texttemplate="%{text:.0f}",
            textposition="outside",
        )

        fig_weather.update_layout(
            xaxis_title="Weather Condition",
            yaxis_title="Avg Vehicle Count",
            height=320,
            showlegend=False,
        )

        apply_plotly_theme(fig_weather)

        st.plotly_chart(
            fig_weather,
            use_container_width=True,
        )

    with w_col2:
        w_display = weather_stats.rename(
            columns={
                "weather_condition": "Weather",
                "average_vehicle_count": "Avg Vehicles",
                "average_speed_kmh": "Avg Speed (km/h)",
                "observations": "Observations",
            }
        ).copy()

        w_display["Avg Vehicles"] = w_display["Avg Vehicles"].round(1)
        w_display["Avg Speed (km/h)"] = w_display["Avg Speed (km/h)"].round(1)
        w_display["Observations"] = w_display["Observations"].map(
            lambda v: f"{v:,}"
        )

        st.dataframe(
            w_display,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Hotspots
    # -------------------------------------------------------------

    hotspot_summaries = get_hotspot_summary(df)
    hotspot_count = len(hotspot_summaries) if hotspot_summaries else 0

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.85rem;">
            <div class="ops-section-title" style="margin: 0;">🔥 Spatial Hotspot Clustering</div>
            <span class="ops-badge ops-badge-{'red' if hotspot_count > 0 else 'emerald'}">
                {hotspot_count} Detected Clusters
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hotspot_summaries:
        hotspot_rows = []

        for hotspot in hotspot_summaries:
            hotspot_rows.append(
                {
                    "Cluster": hotspot["cluster_id"],
                    "Severity": hotspot["severity"],
                    "Road Segments": hotspot["size"],
                    "Congestion Rate (%)": round(
                        hotspot["congestion_rate"] * 100,
                        1,
                    ),
                    "High/Severe Rate (%)": round(
                        hotspot["high_or_severe_rate"] * 100,
                        1,
                    ),
                    "Severe Rate (%)": round(
                        hotspot["severe_rate"] * 100,
                        1,
                    ),
                    "Avg Speed (km/h)": round(
                        hotspot["avg_speed"],
                        1,
                    ),
                    "Peak Hours": ", ".join(
                        str(hour) for hour in hotspot["peak_hours"]
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(hotspot_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No traffic hotspots were detected across the network.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Anomaly Intelligence
    # -------------------------------------------------------------

    st.markdown(
        '<div class="ops-section-title">⚠️ Traffic Anomaly Intelligence</div>',
        unsafe_allow_html=True,
    )

    anomaly_rows = get_anomaly_report(df)

    if anomaly_rows:
        anomaly_df = pd.DataFrame(anomaly_rows)
        total_anomalies = len(anomaly_df)
        high_critical = int(
            anomaly_df[
                anomaly_df["risk_level"].isin(["High", "Critical"])
            ].shape[0]
        )
        critical = int(
            anomaly_df[anomaly_df["risk_level"] == "Critical"].shape[0]
        )
        unique_segments = anomaly_df["road_segment_id"].nunique()

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            render_kpi_card(
                title="Total Anomalies",
                value=f"{total_anomalies:,}",
                status="amber",
                icon="⚠️",
                subtitle="Combined detections",
            )

        with a2:
            render_kpi_card(
                title="High / Critical",
                value=f"{high_critical:,}",
                status="red" if high_critical > 0 else "teal",
                icon="🚨",
                subtitle="Immediate attention",
            )

        with a3:
            render_kpi_card(
                title="Critical Severity",
                value=f"{critical:,}",
                status="red" if critical > 0 else "emerald",
                icon="🔴",
                subtitle="Highest risk events",
            )

        with a4:
            render_kpi_card(
                title="Affected Segments",
                value=f"{unique_segments:,}",
                status="cyan",
                icon="📍",
                subtitle="Spatial distribution",
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        display_columns = [
            "road_segment_id",
            "timestamp",
            "future_congestion_level",
            "risk_level",
            "vehicle_count",
            "average_speed_kmh",
            "occupancy_percent",
            "anomaly_type",
        ]

        available_display_columns = [
            column for column in display_columns if column in anomaly_df.columns
        ]

        display_labels = {
            "road_segment_id": "Road Segment",
            "timestamp": "Timestamp",
            "future_congestion_level": "Predicted Congestion",
            "risk_level": "Risk Level",
            "vehicle_count": "Vehicle Count",
            "average_speed_kmh": "Average Speed (km/h)",
            "occupancy_percent": "Occupancy (%)",
            "anomaly_type": "Anomaly Type",
        }

        display_anomalies = (
            anomaly_df[available_display_columns]
            .sort_values(
                [
                    "risk_level",
                    "timestamp",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(10)
            .rename(columns=display_labels)
        )

        st.dataframe(
            display_anomalies,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No traffic anomalies were detected across current observations.")

    # -------------------------------------------------------------
    # Interpretation note
    # -------------------------------------------------------------

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    st.caption(
        "Overview metrics are calculated from the project's final "
        "historical traffic dataset. Hotspot and anomaly sections "
        "use verified project detection engines. Congestion "
        "reduction figures are intentionally not presented here "
        "as measured outcomes because the project provides "
        "decision-support recommendations rather than observed "
        "real-world intervention results."
    )
