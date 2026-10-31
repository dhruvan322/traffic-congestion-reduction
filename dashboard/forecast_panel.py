"""
Multi-Horizon Forecast Dashboard.

Uses the trained 1h, 2h, 3h and 6h XGBoost forecasting models.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.multihorizon_predictor import (
    forecast_all_horizons,
    get_latest_row_for_segment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "ml_traffic_dataset.csv"
)


@st.cache_data
def load_forecast_dataset():
    """Load the final hourly traffic dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def render_forecast_panel():
    """Render the multi-horizon traffic forecasting dashboard."""

    st.markdown("# 🔮 Multi-Horizon Traffic Forecast")
    st.markdown(
        "Forecast congestion for the next **1, 2, 3 and 6 hours** "
        "using independently trained XGBoost models."
    )

    try:
        df = load_forecast_dataset()
    except Exception as exc:
        st.error(f"Unable to load forecast dataset: {exc}")
        return

    # ---------------------------------------------------------------
    # Selection
    # ---------------------------------------------------------------

    st.subheader("📍 Select Traffic Location")

    selection_mode = st.radio(
        "Forecast by",
        ["Road Segment", "Area"],
        horizontal=True,
    )

    if selection_mode == "Road Segment":

        road_segments = sorted(
            df["road_segment_id"].dropna().unique().tolist()
        )

        selected_segment = st.selectbox(
            "🛣️ Road Segment",
            road_segments,
        )

    else:

        areas = sorted(
            df["area"].dropna().unique().tolist()
        )

        selected_area = st.selectbox(
            "📍 Area",
            areas,
        )

        area_segments = sorted(
            df.loc[
                df["area"] == selected_area,
                "road_segment_id"
            ].dropna().unique().tolist()
        )

        if not area_segments:
            st.warning("No road segments were found for this area.")
            return

        selected_segment = area_segments[0]

        st.caption(
            f"Using representative road segment **{selected_segment}** "
            f"for the selected area."
        )

    # ---------------------------------------------------------------
    # Latest traffic record
    # ---------------------------------------------------------------

    try:
        row = get_latest_row_for_segment(df, selected_segment)

        if row is None:
            st.warning("No traffic record is available for this selection.")
            return

        forecast_payload = forecast_all_horizons(row)
        forecasts = forecast_payload["forecasts"]

    except Exception as exc:
        st.error(f"Forecast generation failed: {exc}")
        return

    area = str(row.get("area", "Unknown"))
    timestamp = row.get("timestamp")

    st.divider()

    # ---------------------------------------------------------------
    # Current traffic context
    # ---------------------------------------------------------------

    st.subheader("📊 Current Traffic Context")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Area", area)

    with col2:
        st.metric("Road Segment", selected_segment)

    with col3:
        vehicle_count = row.get("vehicle_count")
        st.metric(
            "Vehicles",
            f"{vehicle_count:.0f}" if pd.notna(vehicle_count) else "N/A",
        )

    with col4:
        speed = row.get("average_speed_kmh")
        st.metric(
            "Avg Speed",
            f"{speed:.1f} km/h" if pd.notna(speed) else "N/A",
        )

    st.caption(f"Forecast base timestamp: {timestamp}")

    st.divider()

    # ---------------------------------------------------------------
    # Forecast cards
    # ---------------------------------------------------------------

    st.subheader("🔮 Forecast Results")

    horizon_order = [1, 2, 3, 6]

    columns = st.columns(4)

    for column, horizon in zip(columns, horizon_order):

        result = forecasts[f"{horizon}h"]

        with column:
            st.metric(
                f"+{horizon} Hour",
                result["predicted_class"],
                f"{result['confidence']:.1%} confidence",
            )

    st.divider()

    # ---------------------------------------------------------------
    # Forecast table
    # ---------------------------------------------------------------

    table_rows = []

    for horizon in horizon_order:

        result = forecasts[f"{horizon}h"]

        table_rows.append(
            {
                "Horizon": f"+{horizon}h",
                "Prediction": result["predicted_class"],
                "Confidence": f"{result['confidence']:.1%}",
                "Free Flow": result["probabilities"]["Free Flow"],
                "Low": result["probabilities"]["Low"],
                "Moderate": result["probabilities"]["Moderate"],
                "High": result["probabilities"]["High"],
                "Severe": result["probabilities"]["Severe"],
            }
        )

    forecast_table = pd.DataFrame(table_rows)

    display_table = forecast_table.copy()

    for column in [
        "Free Flow",
        "Low",
        "Moderate",
        "High",
        "Severe",
    ]:
        display_table[column] = display_table[column].map(
            lambda value: f"{value:.1%}"
        )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Probability chart
    # ---------------------------------------------------------------

    st.subheader("📈 Forecast Probability Trends")

    fig = go.Figure()

    class_labels = [
        "Free Flow",
        "Low",
        "Moderate",
        "High",
        "Severe",
    ]

    for label in class_labels:

        values = [
            forecasts[f"{horizon}h"]["probabilities"][label]
            for horizon in horizon_order
        ]

        fig.add_trace(
            go.Scatter(
                x=[f"+{h}h" for h in horizon_order],
                y=values,
                mode="lines+markers",
                name=label,
            )
        )

    fig.update_layout(
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=420,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------

    predicted_classes = [
        forecasts[f"{horizon}h"]["predicted_class"]
        for horizon in horizon_order
    ]

    combined_risk = [
        (
            horizon,
            forecasts[f"{horizon}h"]["probabilities"]["High"]
            + forecasts[f"{horizon}h"]["probabilities"]["Severe"],
        )
        for horizon in horizon_order
    ]

    highest_risk_horizon, highest_risk_probability = max(
        combined_risk,
        key=lambda item: item[1],
    )

    severe_count = predicted_classes.count("Severe")
    high_count = predicted_classes.count("High")

    st.subheader("Forecast Interpretation")

    if severe_count > 0:
        st.error(
            f"Severe congestion is the most likely class at "
            f"{severe_count} forecast horizon(s)."
        )
    elif high_count > 0:
        st.warning(
            f"High congestion is the most likely class at "
            f"{high_count} forecast horizon(s)."
        )
    else:
        st.info(
            "High or Severe congestion is not the most likely class "
            "at any displayed horizon."
        )

    st.warning(
        f"Highest combined High + Severe probability: "
        f"{highest_risk_probability:.1%} at +{highest_risk_horizon}h."
    )

    st.caption(
        "The forecast class is the highest-probability class for each horizon. "
        "The High + Severe probability combines the probabilities of both "
        "higher-severity classes. Forecasts use the project's historical "
        "hourly traffic dataset and are intended for decision-support, "
        "not live navigation."
    )
