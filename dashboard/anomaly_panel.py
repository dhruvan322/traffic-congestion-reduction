"""
Anomaly Detection Dashboard Panel.

Uses the final integrated ML traffic dataset and provides:

    - Isolation Forest anomaly detection
    - Z-score anomaly detection
    - Combined anomaly analysis
    - Risk classification
    - Traffic anomaly visualization
    - Road-segment anomaly analysis
    - Decision-support recommendations
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from models.anomaly_detector import (
    detect_anomalies,
    zscore_anomalies,
    get_anomaly_report,
    get_anomaly_summary,
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"

# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------


@st.cache_data
def load_final_dataset():
    """
    Load the final integrated ML traffic dataset.
    """

    if not FINAL_DATASET_PATH.exists():

        raise FileNotFoundError("Final ML dataset not found: " f"{FINAL_DATASET_PATH}")

    df = pd.read_csv(FINAL_DATASET_PATH)

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

    return df


# -------------------------------------------------------------------
# Recommendation engine
# -------------------------------------------------------------------


def generate_anomaly_recommendations(
    anomaly: dict,
) -> list[dict]:
    """
    Generate deterministic decision-support recommendations.

    These are recommendations only. They are not claims of measured
    traffic reduction.
    """

    risk = anomaly.get(
        "risk_level",
        "Low",
    )

    road_segment = anomaly.get(
        "road_segment_id",
        "Unknown",
    )

    congestion = anomaly.get(
        "future_congestion_level",
        "Unknown",
    )

    speed = float(
        anomaly.get(
            "average_speed_kmh",
            0,
        )
    )

    occupancy = float(
        anomaly.get(
            "occupancy_percent",
            0,
        )
    )

    recommendations = []

    if risk == "Critical":

        recommendations.append(
            {
                "priority": "Critical",
                "action": (
                    f"Prioritize immediate traffic monitoring " f"on {road_segment}"
                ),
                "explanation": (
                    f"The segment shows an anomalous pattern "
                    f"combined with {congestion} future congestion."
                ),
            }
        )

        recommendations.append(
            {
                "priority": "High",
                "action": ("Prepare diversion and incident-response " "procedures"),
                "explanation": (
                    "A severe anomalous condition may propagate "
                    "to nearby road segments."
                ),
            }
        )

    elif risk == "High":

        recommendations.append(
            {
                "priority": "High",
                "action": (f"Increase monitoring frequency on " f"{road_segment}"),
                "explanation": (
                    f"An unusual traffic pattern coincides with "
                    f"{congestion} future congestion."
                ),
            }
        )

        recommendations.append(
            {
                "priority": "Medium",
                "action": ("Prepare an alternate-route advisory"),
                "explanation": (
                    "An alternate route can be considered if "
                    "congestion continues to increase."
                ),
            }
        )

    elif risk == "Medium":

        recommendations.append(
            {
                "priority": "Medium",
                "action": (f"Monitor {road_segment} for congestion escalation"),
                "explanation": (
                    f"The anomaly occurs with {congestion} " f"future congestion."
                ),
            }
        )

        recommendations.append(
            {
                "priority": "Low",
                "action": ("Review nearby traffic conditions before intervention"),
                "explanation": (
                    f"Current average speed is {speed:.1f} km/h "
                    f"with {occupancy:.1f}% occupancy."
                ),
            }
        )

    else:

        recommendations.append(
            {
                "priority": "Low",
                "action": (f"Continue routine monitoring of {road_segment}"),
                "explanation": (
                    "The observation is unusual but does not "
                    "indicate a high operational risk."
                ),
            }
        )

    return recommendations


# -------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------


def render_anomaly_panel():
    """
    Render the final-dataset anomaly detection dashboard.
    """

    st.markdown("# ⚠️ Anomaly Detection")

    st.markdown(
        "Detect unusual traffic patterns using "
        "**Isolation Forest** and **Z-score** analysis "
        "on the final integrated traffic dataset."
    )

    # ---------------------------------------------------------------
    # Load final dataset
    # ---------------------------------------------------------------

    try:

        df = load_final_dataset()

    except Exception as exc:

        st.error(f"Unable to load the final ML dataset: {exc}")

        return

    st.success(f"Final ML dataset loaded: " f"{len(df):,} traffic records")

    st.caption(
        "Anomaly detection uses the complete final traffic "
        "dataset. Detection features include vehicle count, "
        "average speed, occupancy, and traffic flow."
    )

    st.divider()

    # ---------------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Road Segments",
            df["road_segment_id"].nunique(),
        )

    with col2:

        st.metric(
            "Traffic Records",
            f"{len(df):,}",
        )

    with col3:

        st.metric(
            "Congestion Classes",
            df["future_congestion_level"].nunique(),
        )

    with col4:

        st.metric(
            "Detection Features",
            "4",
        )

    st.divider()

    # ---------------------------------------------------------------
    # Controls
    # ---------------------------------------------------------------

    run_col, clear_col = st.columns([3, 1])

    with run_col:

        run_clicked = st.button(
            "🔍 Run Anomaly Detection",
            type="primary",
            use_container_width=True,
        )

    with clear_col:

        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True,
        )

    if clear_clicked:

        st.session_state.pop(
            "anomaly_results",
            None,
        )

        st.session_state.pop(
            "anomaly_report",
            None,
        )

        st.rerun()

    # ---------------------------------------------------------------
    # Run detection
    # ---------------------------------------------------------------

    if run_clicked:

        try:

            with st.spinner(
                "Running Isolation Forest and Z-score analysis "
                "on the complete dataset..."
            ):

                df_detected = detect_anomalies(df)

                df_detected = zscore_anomalies(df_detected)

                df_detected["combined_anomaly"] = (
                    (df_detected["is_anomaly"] == 1)
                    | (df_detected["zscore_anomaly"] == 1)
                ).astype(int)

                summary = get_anomaly_summary(df_detected)

                anomaly_records = get_anomaly_report(df)

            st.session_state["anomaly_results"] = df_detected

            st.session_state["anomaly_report"] = anomaly_records

            st.session_state["anomaly_summary"] = summary

        except Exception as exc:

            st.error(f"Anomaly detection failed: {exc}")

            return

    # ---------------------------------------------------------------
    # Retrieve results
    # ---------------------------------------------------------------

    df_detected = st.session_state.get("anomaly_results")

    anomaly_records = st.session_state.get("anomaly_report")

    summary = st.session_state.get("anomaly_summary")

    if df_detected is None:

        st.info(
            "Click **Run Anomaly Detection** to analyze "
            "the complete final traffic dataset."
        )

        return

    # ---------------------------------------------------------------
    # Summary KPIs
    # ---------------------------------------------------------------

    st.success("✅ Anomaly detection completed successfully.")

    st.markdown("---")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:

        st.metric(
            "Total Records",
            f"{summary['total_records']:,}",
        )

    with kpi2:

        st.metric(
            "Isolation Forest",
            f"{summary['isolation_forest_anomalies']:,}",
        )

    with kpi3:

        st.metric(
            "Z-Score",
            f"{summary['zscore_anomalies']:,}",
        )

    with kpi4:

        st.metric(
            "Combined Anomalies",
            f"{summary['combined_anomalies']:,}",
        )

    with kpi5:

        st.metric(
            "Anomaly Rate",
            f"{summary['anomaly_rate_percent']:.2f}%",
        )

    # ---------------------------------------------------------------
    # Detection methodology
    # ---------------------------------------------------------------

    with st.expander(
        "ℹ️ Anomaly Detection Methodology",
        expanded=False,
    ):

        st.markdown("""
### Isolation Forest

Isolation Forest identifies observations that are
unusual compared with the overall traffic distribution.

The model uses:

- Vehicle count
- Average speed
- Occupancy percentage
- Traffic flow

### Z-Score Analysis

For each detection feature, the absolute Z-score is
calculated.

An observation is flagged when at least one feature
exceeds the configured Z-score threshold.

### Combined Detection

An observation is considered anomalous when:

**Isolation Forest OR Z-score identifies it as anomalous.**

### Risk Classification

The combined anomaly result is evaluated together with


- **Critical:** anomalous + Severe congestion
- **High:** anomalous + High congestion
- **Medium:** anomalous + Moderate congestion
- **Low:** unusual pattern without high congestion severity

Recommendations shown by the dashboard are
**decision-support suggestions**, not measured traffic
reductions.
""")

    # ---------------------------------------------------------------
    # Scatter visualization
    # ---------------------------------------------------------------

    st.subheader("🔬 Vehicle Count vs Average Speed")

    plot_columns = [
        "vehicle_count",
        "average_speed_kmh",
        "road_segment_id",
        "future_congestion_level",
        "hour",
    ]

    available_plot_columns = [
        column for column in plot_columns if column in df_detected.columns
    ]

    df_plot = df_detected[available_plot_columns].copy()

    df_plot["anomaly_label"] = (
        df_detected["combined_anomaly"]
        .map(
            {
                0: "Normal",
                1: "Anomaly",
            }
        )
        .values
    )

    # Keep the browser visualization responsive.
    # Detection itself was performed on the complete dataset.
    max_plot_points = 8000

    if len(df_plot) > max_plot_points:

        anomaly_part = df_plot[df_plot["anomaly_label"] == "Anomaly"]

        normal_part = df_plot[df_plot["anomaly_label"] == "Normal"]

        anomaly_limit = min(
            3000,
            len(anomaly_part),
        )

        normal_limit = max_plot_points - anomaly_limit

        if len(anomaly_part) > anomaly_limit:

            anomaly_part = anomaly_part.sample(
                n=anomaly_limit,
                random_state=42,
            )

        if len(normal_part) > normal_limit:

            normal_part = normal_part.sample(
                n=normal_limit,
                random_state=42,
            )

        df_plot = pd.concat(
            [
                anomaly_part,
                normal_part,
            ],
            ignore_index=True,
        )

    fig1 = px.scatter(
        df_plot,
        x="vehicle_count",
        y="average_speed_kmh",
        color="anomaly_label",
        hover_data=[
            column
            for column in [
                "road_segment_id",
                "future_congestion_level",
                "hour",
            ]
            if column in df_plot.columns
        ],
        opacity=0.65,
        color_discrete_map={
            "Normal": "#4CAF50",
            "Anomaly": "#f44336",
        },
    )

    fig1.update_layout(
        xaxis_title="Vehicle Count",
        yaxis_title="Average Speed (km/h)",
        height=500,
        template="plotly_dark",
        legend_title="Status",
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Risk distribution + anomaly methods
    # ---------------------------------------------------------------

    if anomaly_records:

        st.subheader("🎯 Anomaly Risk Analysis")

        risk_df = pd.DataFrame(anomaly_records)

        col1, col2 = st.columns(2)

        with col1:

            risk_counts = (
                risk_df["risk_level"]
                .value_counts()
                .reindex(
                    [
                        "Critical",
                        "High",
                        "Medium",
                        "Low",
                    ],
                    fill_value=0,
                )
            )

            fig2 = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={
                    "Critical": "#d32f2f",
                    "High": "#f44336",
                    "Medium": "#FF9800",
                    "Low": "#4CAF50",
                },
                hole=0.35,
            )

            fig2.update_layout(
                height=350,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )

        with col2:

            method_counts = pd.DataFrame(
                {
                    "Method": [
                        "Isolation Forest only",
                        "Z-Score only",
                        "Both methods",
                    ],
                    "Count": [
                        int(
                            (
                                (df_detected["is_anomaly"] == 1)
                                & (df_detected["zscore_anomaly"] == 0)
                            ).sum()
                        ),
                        int(
                            (
                                (df_detected["is_anomaly"] == 0)
                                & (df_detected["zscore_anomaly"] == 1)
                            ).sum()
                        ),
                        int(
                            (
                                (df_detected["is_anomaly"] == 1)
                                & (df_detected["zscore_anomaly"] == 1)
                            ).sum()
                        ),
                    ],
                }
            )

            fig3 = px.bar(
                method_counts,
                x="Method",
                y="Count",
                text_auto=True,
            )

            fig3.update_layout(
                xaxis_title="Detection Method",
                yaxis_title="Anomaly Count",
                height=350,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig3,
                use_container_width=True,
            )

        # -----------------------------------------------------------
        # Road-segment anomaly concentration
        # -----------------------------------------------------------

        st.subheader("🛣️ Anomalies by Road Segment")

        segment_counts = (
            risk_df["road_segment_id"].value_counts().head(15).reset_index()
        )

        segment_counts.columns = [
            "road_segment_id",
            "anomaly_count",
        ]

        fig4 = px.bar(
            segment_counts,
            x="road_segment_id",
            y="anomaly_count",
            text_auto=True,
        )

        fig4.update_layout(
            xaxis_title="Road Segment",
            yaxis_title="Anomaly Count",
            height=400,
            template="plotly_dark",
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
        )

        # -----------------------------------------------------------
        # Anomaly records
        # -----------------------------------------------------------

        st.subheader("📋 Top Anomaly Records")

        display_columns = [
            "road_segment_id",
            "timestamp",
            "risk_level",
            "vehicle_count",
            "average_speed_kmh",
            "occupancy_percent",
            "traffic_flow_vph",
            "future_congestion_level",
            "anomaly_score",
            "isolation_forest_anomaly",
            "zscore_anomaly",
            "description",
        ]

        available_columns = [
            column for column in display_columns if column in risk_df.columns
        ]

        display_df = risk_df[available_columns].head(50).copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------------
        # Decision-support recommendations
        # -----------------------------------------------------------

        st.markdown("---")

        st.subheader("🚨 Decision-Support Recommendations")

        top_anomalies = anomaly_records[:5]

        for index, anomaly in enumerate(
            top_anomalies,
            start=1,
        ):

            road_segment = anomaly.get(
                "road_segment_id",
                "Unknown",
            )

            risk = anomaly.get(
                "risk_level",
                "Low",
            )

            title = f"Anomaly #{index} — " f"{road_segment} " f"({risk} Risk)"

            with st.expander(
                title,
                expanded=(index == 1),
            ):

                st.markdown(f"**Description:** " f"{anomaly.get('description', 'N/A')}")

                st.markdown(
                    f"**Future Congestion:** "
                    f"{anomaly.get('future_congestion_level', 'N/A')}"
                )

                st.markdown(
                    f"**Anomaly Score:** " f"{anomaly.get('anomaly_score', 0):.4f}"
                )

                st.markdown("---")

                recommendations = generate_anomaly_recommendations(anomaly)

                for recommendation in recommendations:

                    priority = recommendation["priority"]

                    icon = {
                        "Critical": "🔴",
                        "High": "🟠",
                        "Medium": "🟡",
                        "Low": "🟢",
                    }.get(
                        priority,
                        "⚪",
                    )

                    st.markdown(f"{icon} **{recommendation['action']}**")

                    st.markdown(f"- Why: " f"{recommendation['explanation']}")

    else:

        st.info("No combined anomalies were detected.")

    # ---------------------------------------------------------------
    # Anomaly score distribution
    # ---------------------------------------------------------------

    st.subheader("📊 Anomaly Score Distribution")

    anomalous = df_detected[df_detected["combined_anomaly"] == 1]

    if not anomalous.empty:

        score_plot = anomalous.copy()

        max_score_points = 10000

        if len(score_plot) > max_score_points:

            score_plot = score_plot.sample(
                n=max_score_points,
                random_state=42,
            )

        fig5 = px.histogram(
            score_plot,
            x="anomaly_score",
            nbins=40,
        )

        fig5.update_layout(
            xaxis_title=("Anomaly Score " "(higher = more anomalous)"),
            yaxis_title="Count",
            height=350,
            template="plotly_dark",
        )

        st.plotly_chart(
            fig5,
            use_container_width=True,
        )
