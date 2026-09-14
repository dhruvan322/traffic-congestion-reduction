"""
Overview Dashboard Panel.

Final-schema traffic intelligence overview using:
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


def render_overview():
    """Render the main traffic intelligence overview."""

    st.markdown("# 📊 Traffic Intelligence Overview")

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

    missing = sorted(
        required_columns.difference(df.columns)
    )

    if missing:

        st.error(
            "The final traffic dataset is missing required columns."
        )

        st.write(missing)

        return

    # -------------------------------------------------------------
    # KPI Cards
    # -------------------------------------------------------------

    total_records = len(df)

    monitored_segments = (
        df["road_segment_id"]
        .nunique()
    )

    avg_vehicles = (
        df["vehicle_count"]
        .mean()
    )

    avg_speed = (
        df["average_speed_kmh"]
        .mean()
    )

    congested_mask = df[
        "future_congestion_level"
    ].isin(
        [
            "Moderate",
            "High",
            "Severe",
        ]
    )

    congested_pct = (
        congested_mask.mean() * 100
        if total_records > 0
        else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📦 Total Records",
            f"{total_records:,}",
        )

    with col2:

        st.metric(
            "🛣️ Monitored Segments",
            f"{monitored_segments:,}",
        )

    with col3:

        st.metric(
            "⚡ Average Speed",
            f"{avg_speed:.1f} km/h",
        )

    with col4:

        st.metric(
            "🔴 Moderate+ Congestion",
            f"{congested_pct:.1f}%",
        )

    st.divider()

    # -------------------------------------------------------------
    # Traffic Summary
    # -------------------------------------------------------------

    st.subheader(
        "🚦 Traffic State Distribution"
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
        congestion_counts
        / total_records
        * 100
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

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    fig_distribution = px.bar(
        summary_df,
        x="Congestion Level",
        y="Observations",
        text="Observations",
    )

    fig_distribution.update_layout(
        height=350,
        template="plotly_dark",
        yaxis_title="Observations",
        xaxis_title="Congestion Level",
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True,
    )

    # -------------------------------------------------------------
    # Traffic speed vs vehicles
    # -------------------------------------------------------------

    st.subheader(
        "🔬 Vehicle Count vs Average Speed"
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
        hover_data=[
            "road_segment_id",
            "area",
            "hour",
        ],
        opacity=0.6,
    )

    fig_scatter.update_layout(
        xaxis_title="Vehicle Count",
        yaxis_title="Average Speed (km/h)",
        height=420,
        template="plotly_dark",
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )

    # -------------------------------------------------------------
    # Weather impact
    # -------------------------------------------------------------

    st.subheader(
        "🌤️ Weather Impact"
    )

    weather_stats = (
        df.groupby(
            "weather_condition"
        )
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

    fig_weather = px.bar(
        weather_stats,
        x="weather_condition",
        y="average_vehicle_count",
        text="average_vehicle_count",
    )

    fig_weather.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
    )

    fig_weather.update_layout(
        xaxis_title="Weather Condition",
        yaxis_title="Average Vehicles",
        height=350,
        template="plotly_dark",
    )

    st.plotly_chart(
        fig_weather,
        use_container_width=True,
    )

    st.dataframe(
        weather_stats.rename(
            columns={
                "weather_condition": "Weather",
                "average_vehicle_count": "Avg Vehicles",
                "average_speed_kmh": "Avg Speed (km/h)",
                "observations": "Observations",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------
    # Hotspots
    # -------------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔥 Detected Traffic Hotspots"
    )

    hotspot_summaries = get_hotspot_summary(
        df
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
                        str(hour)
                        for hour in hotspot[
                            "peak_hours"
                        ]
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(hotspot_rows),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No traffic hotspots were detected."
        )

    # -------------------------------------------------------------
    # Anomaly intelligence
    # -------------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "⚠️ Traffic Anomaly Intelligence"
    )

    anomaly_rows = get_anomaly_report(
        df
    )

    if anomaly_rows:

        anomaly_df = pd.DataFrame(
            anomaly_rows
        )

        total_anomalies = len(
            anomaly_df
        )

        high_critical = int(
            anomaly_df[
                anomaly_df["risk_level"].isin(
                    [
                        "High",
                        "Critical",
                    ]
                )
            ].shape[0]
        )

        critical = int(
            anomaly_df[
                anomaly_df["risk_level"]
                == "Critical"
            ].shape[0]
        )

        unique_segments = anomaly_df[
            "road_segment_id"
        ].nunique()

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.metric(
                "Total Anomalies",
                f"{total_anomalies:,}",
            )

        with a2:

            st.metric(
                "High/Critical",
                f"{high_critical:,}",
            )

        with a3:

            st.metric(
                "Critical",
                f"{critical:,}",
            )

        with a4:

            st.metric(
                "Affected Segments",
                f"{unique_segments:,}",
            )

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
            column
            for column in display_columns
            if column in anomaly_df.columns
        ]

        st.dataframe(
            anomaly_df[
                available_display_columns
            ]
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
            .head(10),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No traffic anomalies were detected."
        )

    # -------------------------------------------------------------
    # Interpretation note
    # -------------------------------------------------------------

    st.markdown("---")

    st.caption(
        "Overview metrics are calculated from the project's final "
        "historical traffic dataset. Hotspot and anomaly sections "
        "use the verified project detection engines. Congestion "
        "reduction figures are intentionally not presented here "
        "as measured outcomes because the project provides "
        "decision-support recommendations rather than observed "
        "real-world intervention results."
    )
