"""
Prediction Panel.

Uses the final trained XGBoost model through the FastAPI backend
and the final ML dataset.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import API_BASE_URL, API_TIMEOUT_SECONDS

from dashboard.ui_theme import (
    COLOR_CYAN,
    COLOR_RED,
    CONGESTION_COLOR_MAP,
    apply_plotly_theme,
    render_header,
    render_kpi_card,
)

from engine.prediction.explainer import explain_prediction
from engine.prediction.predictor import get_feature_importance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"


@st.cache_data
def load_final_dataset():
    """Load the final ML dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Final ML dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def format_shap_feature(name):
    """Convert transformed preprocessing feature names into readable names."""

    if name.startswith("num__"):
        feature = name.replace("num__", "", 1)
        return feature.replace("_", " ").title()

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
                value = feature[len(prefix):]
                return f"{column.replace('_', ' ').title()} = {value}"

        return feature.replace("_", " ").title()

    return name.replace("_", " ").title()


def render_prediction_panel():
    """Render the final ML congestion prediction interface."""

    try:
        df = load_final_dataset()
    except Exception as exc:
        st.error(f"Unable to load final ML dataset: {exc}")
        return

    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()

    if pd.notna(dataset_start) and pd.notna(dataset_end):
        if dataset_start.year == dataset_end.year:
            dataset_period = str(dataset_start.year)
            dataset_period_caption = (
                f"{dataset_start:%b}–{dataset_end:%b} · historical observations"
            )
        else:
            dataset_period = f"{dataset_start:%Y}–{dataset_end:%Y}"
            dataset_period_caption = (
                f"{dataset_start:%b %Y} to {dataset_end:%b %Y}"
            )
    else:
        dataset_period = "Unavailable"
        dataset_period_caption = "Historical observations"

    render_header(
        title="Traffic Predictions",
        subtitle=(
            "Run the trained congestion model against a selected historical "
            "traffic record."
        ),
        badge_text="Historical dataset",
        badge_type="teal",
    )

    summary_columns = st.columns(4)

    summary_cards = (
        ("HISTORICAL RECORDS", f"{len(df):,}", "Final ML dataset", "cyan"),
        (
            "ROAD SEGMENTS",
            f"{df['road_segment_id'].nunique():,}",
            "Represented in this dataset",
            "teal",
        ),
        (
            "TARGET CLASSES",
            f"{df['future_congestion_level'].nunique():,}",
            "Recorded future congestion labels",
            "emerald",
        ),
        ("DATASET PERIOD", dataset_period, dataset_period_caption, "blue"),
    )

    for column, (title, value, subtitle, status) in zip(
        summary_columns, summary_cards
    ):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    with st.container(border=True):
        st.markdown("#### Select a historical traffic record")
        st.caption(
            "Choose a road segment, date, and available hour from the final ML dataset."
        )

        road_segments = sorted(
            df["road_segment_id"].dropna().unique().tolist()
        )

        segment_column, date_column = st.columns([1.2, 1])

        with segment_column:
            selected_segment = st.selectbox("Road segment", road_segments)

        segment_df = df[
            df["road_segment_id"] == selected_segment
        ].copy()

        segment_df = segment_df.sort_values("timestamp")

        min_date = segment_df["timestamp"].min().date()
        max_date = segment_df["timestamp"].max().date()

        with date_column:
            selected_date = st.date_input(
                "Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
            )

        available_hours = sorted(
            segment_df[
                segment_df["timestamp"].dt.date == selected_date
            ]["timestamp"]
            .dt.hour.dropna()
            .unique()
            .tolist()
        )

        if not available_hours:
            st.warning(
                "No traffic record is available for this road segment and date."
            )
            return

        hour_column, _ = st.columns([1, 2])

        with hour_column:
            selected_hour = st.selectbox("Hour", available_hours)

        selected_timestamp = pd.Timestamp(selected_date) + pd.Timedelta(
            hours=selected_hour
        )

        matching_rows = df[
            (df["road_segment_id"] == selected_segment)
            & (df["timestamp"] == selected_timestamp)
        ]

        if matching_rows.empty:
            st.warning("No matching traffic record was found.")
            return

        row = matching_rows.iloc[0]

        st.markdown("#### Selected traffic conditions")

        condition_columns = st.columns(4)

        condition_metrics = (
            ("Vehicles", f"{row['vehicle_count']:.0f}"),
            ("Average speed", f"{row['average_speed_kmh']:.1f} km/h"),
            ("Occupancy", f"{row['occupancy_percent']:.1f}%"),
            ("Traffic flow", f"{row['traffic_flow_vph']:.0f} vph"),
        )

        for column, (label, value) in zip(
            condition_columns, condition_metrics
        ):
            with column:
                st.metric(label, value)

        accident_value = int(row["accident"])

        if accident_value == 1:
            st.warning(
                f"Accident detected — severity: {row['accident_severity']}"
            )
        else:
            st.info("No accident recorded for this traffic record.")

        run_prediction = st.button(
            "Predict congestion",
            type="primary",
            width="stretch",
        )

    st.caption(
        "Predictions use historical synthetic traffic data. This page does not "
        "provide live traffic monitoring."
    )

    if not run_prediction:
        return

    # ---------------------------------------------------------------
    # FastAPI prediction request
    # ---------------------------------------------------------------

    try:
        with st.spinner("Requesting prediction from FastAPI..."):

            response = requests.post(
                f"{API_BASE_URL.rstrip('/')}/api/v1/predict",
                json={
                    "area": str(row["area"]),
                    "road_segment_id": str(row["road_segment_id"]),
                    "timestamp": pd.Timestamp(
                        row["timestamp"]
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                },
                timeout=API_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            result = response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the FastAPI backend. "
            "Please make sure the API server is running."
        )
        return

    except requests.exceptions.Timeout:
        st.error("The FastAPI prediction request timed out.")
        return

    except requests.exceptions.HTTPError:
        try:
            error_detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            error_detail = response.text

        st.error(f"Prediction API error: {error_detail}")
        return

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    # ---------------------------------------------------------------
    # Prediction result
    # ---------------------------------------------------------------

    level = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    class_status = {
        "Free Flow": "emerald",
        "Low": "teal",
        "Moderate": "cyan",
        "High": "amber",
        "Severe": "red",
    }.get(level, "cyan")

    st.markdown("### Prediction result")

    result_columns = st.columns(3)

    result_cards = (
        (
            "PREDICTED CONGESTION",
            level,
            "Model output for the selected record",
            class_status,
        ),
        (
            "MODEL CONFIDENCE",
            f"{confidence:.1%}",
            "Probability of the predicted class",
            "cyan",
        ),
        (
            "RECORDED TARGET CLASS",
            str(row["future_congestion_level"]),
            "Historical dataset label for comparison",
            "blue",
        ),
    )

    for column, (title, value, subtitle, status) in zip(
        result_columns, result_cards
    ):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    probability_column, explanation_column = st.columns([0.9, 1.1])

    # ---------------------------------------------------------------
    # Class probabilities
    # ---------------------------------------------------------------

    with probability_column:
        with st.container(border=True):
            st.markdown("#### Class probabilities")
            st.caption("Model output for the selected historical record.")

            class_order = list(CONGESTION_COLOR_MAP)

            labels = [
                label for label in class_order
                if label in probabilities
            ]

            labels.extend(
                label for label in probabilities
                if label not in labels
            )

            values = [
                probabilities[label]
                for label in labels
            ]

            probability_figure = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    text=[
                        f"{value:.1%}"
                        for value in values
                    ],
                    textposition="outside",
                    marker_color=[
                        CONGESTION_COLOR_MAP.get(
                            label,
                            COLOR_CYAN,
                        )
                        for label in labels
                    ],
                    hovertemplate=(
                        "%{y}<br>"
                        "Probability: %{x:.1%}"
                        "<extra></extra>"
                    ),
                )
            )

            probability_figure.update_layout(
                height=300,
                showlegend=False,
                margin=dict(l=12, r=42, t=16, b=36),
            )

            apply_plotly_theme(probability_figure)

            probability_figure.update_xaxes(
                title_text="Probability",
                range=[0, 1],
                tickformat=".0%",
            )

            probability_figure.update_yaxes(
                title_text=None,
                autorange="reversed",
                showgrid=False,
            )

            st.plotly_chart(
                probability_figure,
                width="stretch",
                config={"responsive": True},
            )

    # ---------------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------------

    with explanation_column:
        with st.container(border=True):
            st.markdown("#### Prediction explanation (SHAP)")

            st.caption(
                "Positive values support the predicted class; negative values "
                "move away from it."
            )

            try:
                with st.spinner(
                    "Calculating SHAP feature contributions..."
                ):
                    shap_result = explain_prediction(
                        row,
                        top_n=15,
                    )

                if "error" in shap_result:
                    st.warning(
                        f"SHAP explanation unavailable: {shap_result['error']}"
                    )

                else:
                    contributions = shap_result["contributions"]

                    if contributions:
                        shap_df = pd.DataFrame(
                            {
                                "Feature": list(contributions.keys()),
                                "SHAP Value": list(contributions.values()),
                            }
                        )

                        shap_df["Feature"] = shap_df["Feature"].apply(
                            format_shap_feature
                        )

                        shap_df = shap_df.sort_values("SHAP Value")

                        shap_figure = go.Figure(
                            go.Bar(
                                x=shap_df["SHAP Value"],
                                y=shap_df["Feature"],
                                orientation="h",
                                text=[
                                    f"{value:+.3f}"
                                    for value in shap_df["SHAP Value"]
                                ],
                                textposition="outside",
                                marker_color=[
                                    COLOR_CYAN if value >= 0 else COLOR_RED
                                    for value in shap_df["SHAP Value"]
                                ],
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "SHAP value: %{x:+.4f}<br>"
                                    "Effect: %{customdata}<extra></extra>"
                                ),
                                customdata=[
                                    "Supports prediction"
                                    if value >= 0
                                    else "Opposes prediction"
                                    for value in shap_df["SHAP Value"]
                                ],
                            )
                        )

                        shap_figure.add_vline(
                            x=0,
                            line_width=1,
                            line_color="rgba(148, 163, 184, 0.65)",
                        )

                        shap_figure.update_layout(
                            height=max(
                                360,
                                28 * len(shap_df) + 100,
                            ),
                            showlegend=False,
                            margin=dict(l=12, r=52, t=16, b=48),
                        )

                        apply_plotly_theme(shap_figure)

                        shap_figure.update_xaxes(
                            title_text="SHAP value (impact on prediction)"
                        )

                        shap_figure.update_yaxes(
                            title_text=None,
                            automargin=True,
                            showgrid=False,
                        )

                        st.plotly_chart(
                            shap_figure,
                            width="stretch",
                            config={"responsive": True},
                        )

                        st.caption(
                            "Larger absolute SHAP values indicate a stronger "
                            "influence on this individual prediction."
                        )

                        with st.expander(
                            "View SHAP contribution values"
                        ):
                            display_shap_df = shap_df.copy()

                            display_shap_df["Direction"] = (
                                display_shap_df["SHAP Value"].apply(
                                    lambda value: (
                                        "Supports prediction"
                                        if value >= 0
                                        else "Opposes prediction"
                                    )
                                )
                            )

                            display_shap_df["SHAP Value"] = (
                                display_shap_df["SHAP Value"].round(6)
                            )

                            st.dataframe(
                                display_shap_df,
                                width="stretch",
                                hide_index=True,
                            )

                    else:
                        st.info(
                            "No SHAP contributions were returned for this prediction."
                        )

            except Exception as exc:
                st.warning(
                    f"Unable to generate SHAP explanation: {exc}"
                )

    # ---------------------------------------------------------------
    # Dataset comparison
    # ---------------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Dataset comparison")

        actual_class = row["future_congestion_level"]

        if level == actual_class:
            st.success(
                f"Prediction matches the recorded class: **{actual_class}**"
            )
        else:
            st.info(
                f"Recorded class: **{actual_class}**  \n"
                f"Predicted class: **{level}**"
            )

    # ---------------------------------------------------------------
    # Selected record
    # ---------------------------------------------------------------

    with st.expander("View selected record"):
        st.dataframe(
            row.to_frame(name="Value"),
            width="stretch",
        )

    # ---------------------------------------------------------------
    # XGBoost feature importance
    # ---------------------------------------------------------------

    with st.expander("XGBoost feature importance"):
        try:
            importance = get_feature_importance()

            if importance:
                importance_df = pd.DataFrame(
                    importance,
                    columns=["Feature", "Importance"],
                )

                importance_df = importance_df.sort_values(
                    "Importance",
                    ascending=True,
                ).tail(20)

                importance_figure = go.Figure(
                    go.Bar(
                        x=importance_df["Importance"],
                        y=importance_df["Feature"],
                        orientation="h",
                        marker_color=COLOR_CYAN,
                    )
                )

                importance_figure.update_layout(
                    height=max(
                        360,
                        26 * len(importance_df) + 100,
                    ),
                    showlegend=False,
                )

                apply_plotly_theme(importance_figure)

                importance_figure.update_xaxes(
                    title_text="Importance"
                )

                importance_figure.update_yaxes(
                    title_text=None,
                    automargin=True,
                    showgrid=False,
                )

                st.plotly_chart(
                    importance_figure,
                    width="stretch",
                    config={"responsive": True},
                )

            else:
                st.info("Feature importance is unavailable.")

        except Exception as exc:
            st.warning(
                f"Unable to display feature importance: {exc}"
            )