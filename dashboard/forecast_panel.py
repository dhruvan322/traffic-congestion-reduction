"""
Multi-Horizon Forecast Dashboard.

Uses the trained 1h, 2h, 3h and 6h XGBoost forecasting models.
"""

import json
import requests
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import API_BASE_URL, API_TIMEOUT_SECONDS

from dashboard.ui_theme import (
    CONGESTION_COLOR_MAP,
    apply_plotly_theme,
    render_header,
    render_kpi_card,
)
from engine.multihorizon_predictor import get_latest_row_for_segment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"
VALIDATION_REPORT_PATH = (
    PROJECT_ROOT
    / "trained_models"
    / "multihorizon"
    / "validation_report.json"
)

HORIZON_ORDER = (1, 2, 3, 6)
CLASS_ORDER = ("Free Flow", "Low", "Moderate", "High", "Severe")


@st.cache_data
def load_forecast_dataset():
    """Load the final hourly traffic dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


@st.cache_data
def load_validation_report():
    """Load saved held-out evaluation metrics for the four forecast models."""

    if not VALIDATION_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Forecast validation report not found: {VALIDATION_REPORT_PATH}"
        )

    with VALIDATION_REPORT_PATH.open(encoding="utf-8") as report_file:
        report = json.load(report_file)

    required_metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "test_records",
        "test_period",
    )
    for horizon in HORIZON_ORDER:
        result = report.get(str(horizon))
        if not isinstance(result, dict):
            raise ValueError(f"Missing held-out metrics for the {horizon}h model.")
        missing = [key for key in required_metrics if key not in result]
        if missing:
            raise ValueError(
                f"Incomplete {horizon}h validation metrics: {', '.join(missing)}"
            )
        if not isinstance(result["test_period"], dict) or not {
            "start",
            "end",
        }.issubset(result["test_period"]):
            raise ValueError(f"Incomplete test period for the {horizon}h model.")

    return report


def _format_timestamp(value):
    """Format a dataset timestamp without implying that it is current."""

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Unavailable"
    return timestamp.strftime("%d %b %Y Â· %H:%M")


def _render_forecast_trend(forecasts):
    """Render actual class probabilities across the four model horizons."""

    figure = go.Figure()
    x_values = [f"+{horizon}h" for horizon in HORIZON_ORDER]

    for class_name in CLASS_ORDER:
        values = [
            forecasts[f"{horizon}h"]["probabilities"][class_name]
            for horizon in HORIZON_ORDER
        ]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode="lines+markers",
                name=class_name,
                line=dict(color=CONGESTION_COLOR_MAP[class_name], width=2),
                marker=dict(size=7),
                hovertemplate=(
                    f"<b>{class_name}</b><br>"
                    "Horizon: %{x}<br>"
                    "Probability: %{y:.1%}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        height=330,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=12, r=18, t=48, b=36),
    )
    apply_plotly_theme(figure)
    figure.update_xaxes(title_text="Forecast horizon", showgrid=False)
    figure.update_yaxes(
        title_text="Model probability",
        range=[0, 1],
        tickformat=".0%",
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={"responsive": True},
    )


def _render_evaluation_metrics(report):
    """Render offline held-out evaluation results, not live traffic metrics."""

    rows = []
    for horizon in HORIZON_ORDER:
        result = report[str(horizon)]
        period = result["test_period"]
        rows.append(
            {
                "Horizon": f"+{horizon}h",
                "Accuracy": result["accuracy"],
                "Balanced accuracy": result["balanced_accuracy"],
                "Macro F1": result["macro_f1"],
                "Weighted F1": result["weighted_f1"],
                "Test records": result["test_records"],
                "Held-out period": (
                    f"{str(period['start'])[:10]} â€“ {str(period['end'])[:10]}"
                ),
            }
        )

    evaluation_df = pd.DataFrame(rows)
    for column in ("Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1"):
        evaluation_df[column] = evaluation_df[column].map(
            lambda value: f"{value:.2%}"
        )
    evaluation_df["Test records"] = evaluation_df["Test records"].map(
        lambda value: f"{value:,}"
    )

    st.dataframe(
        evaluation_df,
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Offline held-out evaluation on the synthetic historical dataset. "
        "These metrics are not live traffic performance."
    )


def render_forecast_panel():
    """Render the multi-horizon traffic forecasting dashboard."""

    try:
        df = load_forecast_dataset()
    except Exception as exc:
        st.error(f"Unable to load forecast dataset: {exc}")
        return

    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()
    if pd.notna(dataset_start) and pd.notna(dataset_end):
        archive_period = f"{dataset_start:%Y} ARCHIVE"
        dataset_description = (
            f"Historical records Â· {dataset_start:%d %b %Y} to "
            f"{dataset_end:%d %b %Y}"
        )
    else:
        archive_period = "ARCHIVE"
        dataset_description = "Historical traffic records"

    render_header(
        title="Multi-Horizon Traffic Forecast",
        subtitle=(
            "Historical synthetic-data forecasts for one selected road segment "
            "across four supported horizons."
        ),
        badge_text="Historical dataset",
        badge_type="teal",
    )
    st.caption(dataset_description)

    with st.container(border=True):
        st.markdown("#### Forecast selection")
        selection_mode = st.radio(
            "Forecast by",
            ["Road Segment", "Area"],
            horizontal=True,
        )

        if selection_mode == "Road Segment":
            road_segments = sorted(
                df["road_segment_id"].dropna().astype(str).unique().tolist()
            )
            selected_segment = st.selectbox("Road segment", road_segments)
        else:
            areas = sorted(df["area"].dropna().astype(str).unique().tolist())
            selected_area = st.selectbox("Area", areas)
            area_segments = sorted(
                df.loc[df["area"] == selected_area, "road_segment_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if not area_segments:
                st.warning("No road segments were found for this area.")
                return
            selected_segment = area_segments[0]
            st.caption(
                f"Using representative road segment **{selected_segment}** "
                f"for {selected_area}."
            )

        st.caption(
            "Each forecast uses the latest available historical record for the "
            "selected segment; it is not a live traffic reading."
        )

    try:
        row = get_latest_row_for_segment(df, selected_segment)
        if row is None:
            st.warning("No traffic record is available for this selection.")
            return

        response = requests.post(
            f"{API_BASE_URL.rstrip('/')}/api/v1/forecast",
            json={
                "road_segment_id": selected_segment,
            },
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        forecast_payload = response.json()
        forecasts = forecast_payload["forecasts"]
    except Exception as exc:
        st.error(f"Forecast generation failed: {exc}")
        return

    area = str(row.get("area", "Unknown"))
    base_timestamp = row.get("timestamp")
    vehicle_count = row.get("vehicle_count")
    speed = row.get("average_speed_kmh")

    st.markdown("### Forecast snapshot")
    context_columns = st.columns(4)
    context_values = (
        ("SELECTED AREA", area, "Dataset area", "blue"),
        ("ROAD SEGMENT", selected_segment, "Single-segment inference", "cyan"),
        (
            "VEHICLES",
            f"{vehicle_count:.0f}" if pd.notna(vehicle_count) else "N/A",
            "At selected historical record",
            "teal",
        ),
        (
            "AVERAGE SPEED",
            f"{speed:.1f} km/h" if pd.notna(speed) else "N/A",
            f"Record time Â· {_format_timestamp(base_timestamp)}",
            "emerald",
        ),
    )
    for column, (title, value, subtitle, status) in zip(
        context_columns, context_values
    ):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    st.markdown("### Forecast by horizon")
    forecast_columns = st.columns(4)
    for column, horizon in zip(forecast_columns, HORIZON_ORDER):
        result = forecasts[f"{horizon}h"]
        with column:
            render_kpi_card(
                title=f"+{horizon} HOUR FORECAST",
                value=result["predicted_class"],
                subtitle=f"{result['confidence']:.1%} model confidence",
                status={
                    "Free Flow": "emerald",
                    "Low": "teal",
                    "Moderate": "cyan",
                    "High": "amber",
                    "Severe": "red",
                }.get(result["predicted_class"], "cyan"),
            )

    chart_column, insight_column = st.columns([1.7, 0.9])
    with chart_column:
        with st.container(border=True):
            st.markdown("#### Multi-horizon congestion forecast")
            st.caption(
                "Model class probabilities for the selected segment and "
                "historical input record."
            )
            _render_forecast_trend(forecasts)
            st.caption(
                "Each point is a separate horizon model's prediction for the "
                "same selected input record; this is not a continuously observed "
                "traffic trend."
            )

    predicted_classes = [
        forecasts[f"{horizon}h"]["predicted_class"] for horizon in HORIZON_ORDER
    ]
    combined_risk = {
        horizon: (
            forecasts[f"{horizon}h"]["probabilities"]["High"]
            + forecasts[f"{horizon}h"]["probabilities"]["Severe"]
        )
        for horizon in HORIZON_ORDER
    }
    highest_risk_horizon = max(combined_risk, key=combined_risk.get)
    highest_risk_probability = combined_risk[highest_risk_horizon]
    severe_count = predicted_classes.count("Severe")
    high_count = predicted_classes.count("High")
    class_changes = [
        (previous, current)
        for previous, current in zip(predicted_classes, predicted_classes[1:])
        if previous != current
    ]

    with insight_column:
        with st.container(border=True):
            st.markdown("#### Forecast insights")
            st.caption(f"One selected segment Â· {archive_period.lower()}")

            if severe_count:
                st.error(
                    f"Severe is the predicted class at {severe_count} "
                    f"of {len(HORIZON_ORDER)} horizons."
                )
            elif high_count:
                st.warning(
                    f"High is the predicted class at {high_count} "
                    f"of {len(HORIZON_ORDER)} horizons."
                )
            else:
                st.info(
                    "High or Severe is not the predicted class at any "
                    "displayed horizon."
                )

            st.metric(
                "Highest combined High + Severe probability",
                f"{highest_risk_probability:.1%}",
                f"At +{highest_risk_horizon}h",
            )
            if class_changes:
                transitions = " Â· ".join(
                    f"{previous} â†’ {current}"
                    for previous, current in class_changes
                )
                st.caption(f"Predicted class changes across horizons: {transitions}")
            else:
                st.caption("Predicted class is consistent across all four horizons.")

            st.caption(
                "These are model outputs from historical data, not current "
                "conditions or guaranteed future outcomes."
            )

    with st.container(border=True):
        st.markdown("#### Forecast probabilities")
        st.caption(
            f"Predicted classes and class probabilities for {selected_segment} "
            f"({area}); input record: {_format_timestamp(base_timestamp)}."
        )

        probability_rows = []
        for horizon in HORIZON_ORDER:
            result = forecasts[f"{horizon}h"]
            probability_rows.append(
                {
                    "Horizon": f"+{horizon}h",
                    "Predicted class": result["predicted_class"],
                    "Confidence": f"{result['confidence']:.1%}",
                    **{
                        class_name: (
                            f"{result['probabilities'][class_name]:.1%}"
                        )
                        for class_name in CLASS_ORDER
                    },
                }
            )
        st.dataframe(
            pd.DataFrame(probability_rows),
            width="stretch",
            hide_index=True,
        )

    try:
        validation_report = load_validation_report()
    except Exception as exc:
        st.warning(f"Held-out model evaluation metrics unavailable: {exc}")
    else:
        with st.container(border=True):
            st.markdown("#### Model performance")
            st.caption(
                "Held-out synthetic-dataset evaluation results for the saved "
                "forecast models."
            )
            _render_evaluation_metrics(validation_report)

    st.caption(
        "Historical synthetic traffic data Â· forecasts use the selected "
        "segment's latest dataset record Â· no live traffic feed Â· "
        "forecasting provides decision support and does not demonstrate "
        "measured traffic reduction."
    )
