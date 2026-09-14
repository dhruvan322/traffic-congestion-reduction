"""
Prediction Panel.

Uses the final trained XGBoost model and the final ML dataset.
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.prediction.predictor import (
    predict_from_row,
    get_feature_importance,
)

from engine.prediction.explainer import explain_prediction

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------


@st.cache_data
def load_final_dataset():
    """Load the final ML dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Final ML dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# -------------------------------------------------------------------
# SHAP feature-name formatter
# -------------------------------------------------------------------


def format_shap_feature(name):
    """
    Convert transformed preprocessing feature names into
    readable names for the dashboard.
    """

    # Numerical feature
    if name.startswith("num__"):
        feature = name.replace("num__", "", 1)
        return feature.replace("_", " ").title()

    # Categorical / one-hot encoded feature
    if name.startswith("cat__"):
        feature = name.replace("cat__", "", 1)

        categorical_features = [
            "road_segment_id",
            "traffic_signal",
            "weather_condition",
            "accident_severity",
            "road_type",
            "area",
        ]

        for column in categorical_features:
            prefix = column + "_"

            if feature.startswith(prefix):
                value = feature[len(prefix) :]

                return f"{column.replace('_', ' ').title()} = " f"{value}"

        return feature.replace("_", " ").title()

    return name.replace("_", " ").title()


# -------------------------------------------------------------------
# Prediction panel
# -------------------------------------------------------------------


def render_prediction_panel():
    """Render the final ML congestion prediction interface."""

    st.markdown("# 🔮 Congestion Prediction Engine")

    st.markdown(
        "Predict future traffic congestion using the " "final trained XGBoost model."
    )

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    try:
        df = load_final_dataset()

    except Exception as exc:
        st.error(f"Unable to load final ML dataset: {exc}")
        return

    st.success(f"Final ML dataset loaded: {len(df):,} traffic records")

    st.divider()

    # ---------------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------------

    st.subheader("📊 Final ML Dataset")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("Road Segments", df["road_segment_id"].nunique())

    with info_col2:
        st.metric("Records", f"{len(df):,}")

    with info_col3:
        st.metric("Prediction Classes", "5")

    st.divider()

    # ---------------------------------------------------------------
    # Prediction input
    # ---------------------------------------------------------------

    st.subheader("📝 Select Traffic Record")

    road_segments = sorted(df["road_segment_id"].dropna().unique().tolist())

    selected_segment = st.selectbox("🛣️ Road Segment", road_segments)

    segment_df = df[df["road_segment_id"] == selected_segment].copy()

    segment_df = segment_df.sort_values("timestamp")

    min_date = segment_df["timestamp"].min().date()
    max_date = segment_df["timestamp"].max().date()

    selected_date = st.date_input(
        "📅 Date", value=min_date, min_value=min_date, max_value=max_date
    )

    available_hours = sorted(
        segment_df[segment_df["timestamp"].dt.date == selected_date]["timestamp"]
        .dt.hour.dropna()
        .unique()
        .tolist()
    )

    if not available_hours:
        st.warning("No traffic record is available for this road segment " "and date.")
        return

    selected_hour = st.selectbox("🕐 Hour", available_hours)

    selected_timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)

    matching_rows = df[
        (df["road_segment_id"] == selected_segment)
        & (df["timestamp"] == selected_timestamp)
    ]

    if matching_rows.empty:
        st.warning("No matching traffic record was found.")
        return

    row = matching_rows.iloc[0]

    # ---------------------------------------------------------------
    # Show selected traffic conditions
    # ---------------------------------------------------------------

    st.subheader("🚦 Traffic Conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Vehicles", f"{row['vehicle_count']:.0f}")

    with col2:
        st.metric("Average Speed", f"{row['average_speed_kmh']:.1f} km/h")

    with col3:
        st.metric("Occupancy", f"{row['occupancy_percent']:.1f}%")

    with col4:
        st.metric("Traffic Flow", f"{row['traffic_flow_vph']:.0f} vph")

    # ---------------------------------------------------------------
    # Accident information
    # ---------------------------------------------------------------

    accident_value = int(row["accident"])

    if accident_value == 1:

        accident_severity = row["accident_severity"]

        st.warning(f"⚠️ Accident detected — Severity: " f"{accident_severity}")

    else:

        st.info("No accident recorded for this traffic record.")

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    if st.button("🔮 Predict Congestion", type="primary", use_container_width=True):

        try:

            with st.spinner("Running the final XGBoost prediction pipeline..."):

                result = predict_from_row(row)

        except Exception as exc:

            st.error(f"Prediction failed: {exc}")

            return

        # -----------------------------------------------------------
        # Result
        # -----------------------------------------------------------

        level = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        emoji_map = {
            "Free Flow": "🟢",
            "Low": "🟢",
            "Moderate": "🟡",
            "High": "🟠",
            "Severe": "🔴",
        }

        emoji = emoji_map.get(level, "⚪")

        st.markdown("---")

        st.markdown(f"## {emoji} Predicted: **{level}** Congestion")

        st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

        # -----------------------------------------------------------
        # Prediction details
        # -----------------------------------------------------------

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Predicted Class", level)

        with result_col2:
            st.metric("Confidence", f"{confidence:.1%}")

        # -----------------------------------------------------------
        # Probability chart
        # -----------------------------------------------------------

        st.subheader("📊 Class Probabilities")

        labels = list(probabilities.keys())
        values = list(probabilities.values())

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                text=[f"{value:.1%}" for value in values],
                textposition="outside",
            )
        )

        fig.update_layout(
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=350,
            showlegend=False,
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------
        # Actual class from dataset
        # -----------------------------------------------------------

        actual_class = row["future_congestion_level"]

        st.subheader("🔎 Dataset Comparison")

        if level == actual_class:

            st.success(f"Prediction matches the recorded class: " f"**{actual_class}**")

        else:

            st.info(
                f"Recorded class: **{actual_class}**  \n"
                f"Predicted class: **{level}**"
            )

        # -----------------------------------------------------------
        # SHAP Explanation
        # -----------------------------------------------------------

        st.subheader("🧠 Prediction Explanation (SHAP)")

        st.markdown(
            f"SHAP explains which input features influenced the "
            f"model's prediction of **{level}** congestion."
        )

        st.caption(
            "Positive SHAP values push the prediction toward the "
            "predicted class, while negative values push it away."
        )

        try:

            with st.spinner("Calculating SHAP feature contributions..."):

                shap_result = explain_prediction(row, top_n=15)

            if "error" in shap_result:

                st.warning("SHAP explanation unavailable: " f"{shap_result['error']}")

            else:

                contributions = shap_result["contributions"]

                if contributions:

                    shap_df = pd.DataFrame(
                        {
                            "Feature": list(contributions.keys()),
                            "SHAP Value": list(contributions.values()),
                        }
                    )

                    shap_df["Feature"] = shap_df["Feature"].apply(format_shap_feature)

                    # Sort so the strongest positive/negative
                    # contributions appear clearly.
                    shap_df = shap_df.sort_values("SHAP Value")

                    # Positive = supports predicted class
                    # Negative = moves away from predicted class
                    bar_values = shap_df["SHAP Value"].tolist()

                    bar_labels = [
                        "Positive" if value >= 0 else "Negative" for value in bar_values
                    ]

                    fig_shap = go.Figure()

                    fig_shap.add_trace(
                        go.Bar(
                            x=bar_values,
                            y=shap_df["Feature"],
                            orientation="h",
                            text=[f"{value:+.3f}" for value in bar_values],
                            textposition="outside",
                            customdata=bar_labels,
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "SHAP value: %{x:+.4f}<br>"
                                "Effect: %{customdata}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig_shap.add_vline(x=0, line_width=1)

                    fig_shap.update_layout(
                        title=("Top 15 Features Influencing " f"{level} Prediction"),
                        xaxis_title=("SHAP Value " "(Impact on Prediction)"),
                        yaxis_title="Feature",
                        height=650,
                        showlegend=False,
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig_shap, use_container_width=True)

                    st.caption(
                        "Larger absolute SHAP values indicate a "
                        "stronger influence on this individual "
                        "prediction."
                    )

                    # ------------------------------------------------
                    # SHAP contribution table
                    # ------------------------------------------------

                    with st.expander(
                        "📋 View SHAP Contribution Values", expanded=False
                    ):

                        display_shap_df = shap_df.copy()

                        display_shap_df["Direction"] = display_shap_df[
                            "SHAP Value"
                        ].apply(
                            lambda value: (
                                "Supports prediction"
                                if value >= 0
                                else "Opposes prediction"
                            )
                        )

                        display_shap_df["SHAP Value"] = display_shap_df[
                            "SHAP Value"
                        ].round(6)

                        st.dataframe(
                            display_shap_df, use_container_width=True, hide_index=True
                        )

                else:

                    st.info(
                        "No SHAP contributions were returned " "for this prediction."
                    )

        except Exception as exc:

            st.warning(f"Unable to generate SHAP explanation: {exc}")

        # -----------------------------------------------------------
        # Traffic record details
        # -----------------------------------------------------------

        with st.expander("📋 View Selected Record", expanded=False):

            display_row = row.to_frame(name="Value")

            st.dataframe(display_row, use_container_width=True)

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    st.divider()

    with st.expander("📊 XGBoost Feature Importance", expanded=False):

        try:

            importance = get_feature_importance()

            if importance:

                importance_df = pd.DataFrame(
                    importance, columns=["Feature", "Importance"]
                )

                importance_df = importance_df.sort_values(
                    "Importance", ascending=False
                ).head(20)

                fig = go.Figure(
                    go.Bar(
                        x=importance_df["Importance"],
                        y=importance_df["Feature"],
                        orientation="h",
                    )
                )

                fig.update_layout(
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=600,
                    template="plotly_dark",
                )

                fig.update_yaxes(autorange="reversed")

                st.plotly_chart(fig, use_container_width=True)

            else:

                st.info("Feature importance is unavailable.")

        except Exception as exc:

            st.warning(f"Unable to display feature importance: {exc}")
