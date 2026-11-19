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

from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from dashboard.ui_theme import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_EMERALD,
    COLOR_RED,
    COLOR_TEAL,
    COLOR_TEXT_MUTED,
    apply_plotly_theme,
    render_header,
    render_kpi_card,
)
import requests

from config import API_BASE_URL, API_TIMEOUT_SECONDS
from models.anomaly_detector import classify_risk

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"

RISK_ORDER = ("Critical", "High", "Medium", "Low")
RISK_COLORS = {
    "Critical": COLOR_RED,
    "High": COLOR_AMBER,
    "Medium": COLOR_CYAN,
    "Low": COLOR_EMERALD,
}

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
                    f"Review {road_segment} for historically observed severe "
                    "congestion patterns."
                ),
                "explanation": (
                    f"The historical record pairs an anomalous pattern with a "
                    f"{congestion} future-congestion classification."
                ),
            }
        )

        recommendations.append(
            {
                "priority": "High",
                "action": (
                    "Evaluate alternative-route strategies in a simulated "
                    "traffic scenario."
                ),
                "explanation": (
                    "Use the severe future-congestion classification to evaluate "
                    "nearby-segment and diversion scenarios; no incident is confirmed."
                ),
            }
        )

    elif risk == "High":

        recommendations.append(
            {
                "priority": "High",
                "action": (
                    f"Review {road_segment} for unusual historical traffic patterns."
                ),
                "explanation": (
                    f"The historical record pairs an unusual traffic pattern with "
                    f"a {congestion} future-congestion classification."
                ),
            }
        )

        recommendations.append(
            {
                "priority": "Medium",
                "action": (
                    "Evaluate alternative-route options in a simulated traffic "
                    "scenario."
                ),
                "explanation": (
                    "Compare route options in scenarios modeled on this historical "
                    "traffic pattern."
                ),
            }
        )

    elif risk == "Medium":

        recommendations.append(
            {
                "priority": "Medium",
                "action": (
                    f"Review historical records for {road_segment} to assess "
                    "congestion escalation patterns."
                ),
                "explanation": (
                    f"The historical anomaly record is associated with a "
                    f"{congestion} future-congestion classification."
                ),
            }
        )

        recommendations.append(
            {
                "priority": "Low",
                "action": (
                    "Compare nearby-segment patterns in the historical dataset "
                    "before considering an intervention."
                ),
                "explanation": (
                    f"Recorded average speed is {speed:.1f} km/h with "
                    f"{occupancy:.1f}% occupancy."
                ),
            }
        )

    else:

        recommendations.append(
            {
                "priority": "Low",
                "action": (
                    f"Review historical records for unusual patterns on "
                    f"{road_segment}."
                ),
                "explanation": (
                    "The historical observation is unusual but is not classified "
                    "as high operational risk."
                ),
            }
        )

    return recommendations


def _build_anomaly_report(df_detected: pd.DataFrame) -> list[dict]:
    """Build the existing report shape from already-computed detector results."""
    anomalies = df_detected.loc[df_detected["combined_anomaly"] == 1].copy()
    if anomalies.empty:
        return []

    if "risk_level" not in anomalies.columns:
        anomalies["risk_level"] = anomalies.apply(classify_risk, axis=1)

    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False,
        kind="stable",
    )
    detected_at = datetime.now().isoformat()
    records = []

    for _, row in anomalies.iterrows():
        timestamp = row.get("timestamp", "")
        if pd.isna(timestamp):
            timestamp = ""

        road_segment = row.get("road_segment_id", "Unknown")
        vehicle_count = float(row.get("vehicle_count", 0))
        speed = float(row.get("average_speed_kmh", 0))
        occupancy = float(row.get("occupancy_percent", 0))
        flow = float(row.get("traffic_flow_vph", 0))
        congestion = row.get("future_congestion_level", "Unknown")

        records.append(
            {
                "road_segment_id": str(road_segment),
                "timestamp": str(timestamp),
                "vehicle_count": int(round(vehicle_count)),
                "average_speed_kmh": speed,
                "occupancy_percent": occupancy,
                "traffic_flow_vph": flow,
                "future_congestion_level": str(congestion),
                "isolation_forest_anomaly": int(row["is_anomaly"]),
                "zscore_anomaly": int(row["zscore_anomaly"]),
                "anomaly_score": float(row["anomaly_score"]),
                "risk_level": str(row["risk_level"]),
                "anomaly_type": "isolation_forest+zscore",
                "detected_at": detected_at,
                "description": (
                    f"Unusual traffic pattern detected on {road_segment}: "
                    f"{vehicle_count:.0f} vehicles, {speed:.1f} km/h average speed, "
                    f"{occupancy:.1f}% occupancy, {flow:.0f} vehicles/hour flow. "
                    f"Future congestion level: {congestion}."
                ),
            }
        )

    return records


def _prepare_anomaly_views(df_detected: pd.DataFrame) -> dict:
    """Aggregate dashboard views without changing detector outputs."""
    detected = df_detected.copy()
    if "timestamp" in detected.columns:
        detected["timestamp"] = pd.to_datetime(
            detected["timestamp"],
            errors="coerce",
        )

    if "risk_level" not in detected.columns:
        detected["risk_level"] = "Normal"
        anomaly_mask = detected["combined_anomaly"] == 1
        detected.loc[anomaly_mask, "risk_level"] = detected.loc[anomaly_mask].apply(
            classify_risk, axis=1
        )

    anomalies = detected.loc[detected["combined_anomaly"] == 1].copy()
    risk_counts = (
        anomalies["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .astype(int)
    )

    method_counts = pd.Series(
        {
            "Isolation Forest only": int(
                (
                    (detected["is_anomaly"] == 1) & (detected["zscore_anomaly"] == 0)
                ).sum()
            ),
            "Z-score only": int(
                (
                    (detected["is_anomaly"] == 0) & (detected["zscore_anomaly"] == 1)
                ).sum()
            ),
            "Both methods": int(
                (
                    (detected["is_anomaly"] == 1) & (detected["zscore_anomaly"] == 1)
                ).sum()
            ),
        },
        dtype="int64",
    )

    monthly = pd.DataFrame(
        0,
        index=pd.Index(range(1, 13), name="month"),
        columns=RISK_ORDER,
        dtype="int64",
    )
    if not anomalies.empty and "timestamp" in anomalies.columns:
        year_anomalies = anomalies.loc[anomalies["timestamp"].dt.year == 2025]
        if not year_anomalies.empty:
            monthly_counts = (
                year_anomalies.groupby(
                    [year_anomalies["timestamp"].dt.month, "risk_level"]
                )
                .size()
                .unstack(fill_value=0)
            )
            monthly.update(monthly_counts.reindex(columns=RISK_ORDER, fill_value=0))

    monthly["month_label"] = pd.date_range(
        "2025-01-01",
        periods=12,
        freq="MS",
    ).strftime("%b")
    monthly["total"] = monthly[list(RISK_ORDER)].sum(axis=1)
    peak_month = int(monthly["total"].idxmax()) if not monthly.empty else 1
    peak_count = int(monthly.loc[peak_month, "total"]) if not monthly.empty else 0

    segments = pd.DataFrame()
    if not anomalies.empty and {
        "road_segment_id",
        "area",
        "latitude",
        "longitude",
    }.issubset(anomalies.columns):
        segments = (
            anomalies.groupby("road_segment_id", sort=False)
            .agg(
                area=("area", "first"),
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
                anomaly_records=("road_segment_id", "size"),
            )
            .reset_index()
        )
        segment_risks = pd.crosstab(
            anomalies["road_segment_id"],
            anomalies["risk_level"],
        ).reindex(columns=RISK_ORDER, fill_value=0)
        dominant_risk = segment_risks.idxmax(axis=1)
        segments["dominant_risk"] = segments["road_segment_id"].map(dominant_risk)
        segments = segments.sort_values(
            "anomaly_records",
            ascending=False,
            kind="stable",
        )

    top_records = anomalies.sort_values(
        "anomaly_score",
        ascending=False,
        kind="stable",
    ).head(5)

    return {
        "detected": detected,
        "anomalies": anomalies,
        "risk_counts": risk_counts,
        "method_counts": method_counts,
        "monthly": monthly,
        "peak_month": peak_month,
        "peak_count": peak_count,
        "segments": segments,
        "top_records": top_records,
        "affected_segments": int(anomalies["road_segment_id"].nunique()),
        "total_segments": int(detected["road_segment_id"].nunique()),
    }


def _render_segment_map(segments: pd.DataFrame) -> None:
    """Render historical affected-segment markers using dataset coordinates."""
    if segments.empty:
        st.info("No geocoded anomalous road segments are available.")
        return

    center = [segments["latitude"].mean(), segments["longitude"].mean()]
    traffic_map = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    max_count = max(int(segments["anomaly_records"].max()), 1)

    for row in segments.itertuples(index=False):
        risk = row.dominant_risk
        color = RISK_COLORS.get(risk, COLOR_TEAL)
        radius = 5 + 9 * (row.anomaly_records / max_count) ** 0.5
        label = (
            f"{row.road_segment_id} Â· {row.area} Â· {risk} Â· "
            f"{row.anomaly_records:,} anomaly records"
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            weight=1,
            tooltip=label,
        ).add_to(traffic_map)

    st_folium(
        traffic_map,
        width=None,
        height=440,
        use_container_width=True,
        key="anomaly_segment_map",
    )


# -------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------


def render_anomaly_panel():
    """
    Render the final-dataset anomaly detection dashboard.
    """

    render_header(
        title="Anomaly Detection Intelligence",
        subtitle=(
            "Historical synthetic-data outlier analysis across 60 monitored "
            "Hyderabad road segments."
        ),
        badge_text="HISTORICAL DATASET",
        badge_type="cyan",
    )

    st.caption(
        "The dataset is historical and synthetic; this page does not represent "
        "live traffic monitoring."
    )

    try:
        df = load_final_dataset()
    except Exception as exc:
        st.error(f"Unable to load the final ML dataset: {exc}")
        return

    st.caption(
        f"Analysis source: {len(df):,} historical records Â· "
        f"{df['road_segment_id'].nunique()} road segments"
    )

    run_col, clear_col = st.columns([3, 1])
    with run_col:
        run_clicked = st.button(
            "Run Anomaly Detection",
            type="primary",
            use_container_width=True,
        )
    with clear_col:
        clear_clicked = st.button(
            "Clear Results",
            use_container_width=True,
        )


    if clear_clicked:
        for key in (
            "anomaly_results",
            "anomaly_report",
            "anomaly_summary",
            "anomaly_payload",
        ):
            st.session_state.pop(key, None)

        st.rerun()

    if run_clicked:
        try:
            with st.spinner("Requesting anomaly analysis from FastAPI..."):
                response = requests.post(
                    f"{API_BASE_URL.rstrip('/')}/api/v1/anomalies/analyze",
                    timeout=API_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()

            required_keys = {
                "summary",
                "risk_counts",
                "method_counts",
                "monthly",
                "segments",
                "top_records",
                "scatter_records",
                "score_records",
                "recommendations",
            }

            missing_keys = required_keys.difference(payload)

            if missing_keys:
                raise ValueError(
                    f"API response is missing fields: {sorted(missing_keys)}"
                )

            st.session_state["anomaly_payload"] = payload
            st.session_state["anomaly_report"] = payload["recommendations"]
            st.session_state["anomaly_summary"] = payload["summary"]

        except requests.exceptions.ConnectionError:
            st.error(
                "Unable to connect to FastAPI. " "Make sure the backend is running."
            )
            return
        except requests.exceptions.Timeout:
            st.error("The anomaly analysis request timed out. " "Please try again.")
            return
        except requests.exceptions.HTTPError as exc:
            st.error(f"FastAPI returned an error: {exc}")
            return
        except Exception as exc:
            st.error(f"Anomaly analysis failed: {exc}")
            return

    payload = st.session_state.get("anomaly_payload")

    if payload is None:
        st.info("Select **Run Anomaly Detection** to analyze the full dataset.")
        return

    summary = payload["summary"]
    anomaly_records = payload["recommendations"]
    anomaly_count = int(summary["combined_anomalies"])

    risk_counts = pd.Series(
        payload["risk_counts"],
        dtype="int64",
    ).reindex(RISK_ORDER, fill_value=0)

    method_counts = pd.Series(
        payload["method_counts"],
        dtype="int64",
    )

    monthly = pd.DataFrame(payload["monthly"]).set_index("month").reindex(range(1, 13))

    segments = pd.DataFrame(payload["segments"])
    top_records = pd.DataFrame(payload["top_records"])

    if not top_records.empty:
        top_records["timestamp"] = pd.to_datetime(
            top_records["timestamp"],
            errors="coerce",
        )

    views = {
        "risk_counts": risk_counts,
        "method_counts": method_counts,
        "monthly": monthly,
        "peak_month": int(payload["peak_month"]),
        "peak_count": int(payload["peak_count"]),
        "segments": segments,
        "top_records": top_records,
        "affected_segments": int(payload["affected_segments"]),
        "total_segments": int(payload["total_segments"]),
    }
    st.markdown("#### Detection summary")
    kpi_columns = st.columns(4)
    kpis = (
        (
            "COMBINED ANOMALY RECORDS",
            f"{anomaly_count:,}",
            "Isolation Forest OR Z-score",
            "red",
        ),
        (
            "CRITICAL RISK RECORDS",
            f"{int(views['risk_counts']['Critical']):,}",
            "Anomalous records with Severe future class",
            "amber",
        ),
        (
            "AFFECTED ROAD SEGMENTS",
            f"{views['affected_segments']} / {views['total_segments']}",
            "Segments with at least one detected anomaly",
            "cyan",
        ),
        (
            "PEAK ANOMALY MONTH",
            pd.Timestamp(year=2025, month=views["peak_month"], day=1).strftime("%b %Y"),
            f"{views['peak_count']:,} combined anomaly records",
            "teal",
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

    st.markdown("### Monthly anomaly trend by risk")
    st.caption(
        "Monthly counts are derived from historical timestamps and risk classes."
    )
    monthly_long = (
        views["monthly"]
        .reset_index()
        .melt(
            id_vars=["month", "month_label", "total"],
            value_vars=list(RISK_ORDER),
            var_name="Risk",
            value_name="Records",
        )
    )
    trend = px.line(
        monthly_long,
        x="month_label",
        y="Records",
        color="Risk",
        markers=True,
        category_orders={"month_label": list(views["monthly"]["month_label"])},
        color_discrete_map=RISK_COLORS,
        custom_data=["month", "total"],
    )
    trend.update_traces(
        hovertemplate=(
            "%{x} 2025<br>%{fullData.name}: %{y:,}"
            "<br>All risks: %{customdata[1]:,}<extra></extra>"
        )
    )
    peak_label = views["monthly"].loc[views["peak_month"], "month_label"]
    trend.add_vline(
        x=peak_label,
        line_dash="dot",
        line_color=COLOR_TEXT_MUTED,
        opacity=0.7,
    )
    trend.add_annotation(
        x=peak_label,
        y=1,
        yref="paper",
        text=f"Peak Â· {views['peak_count']:,}",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="#F8FAFC", size=11),
        bgcolor="rgba(19, 28, 46, 0.85)",
    )
    trend.update_layout(
        xaxis_title="Month Â· 2025",
        yaxis_title="Combined anomaly records",
        height=360,
        legend_title="Risk",
        hovermode="x unified",
    )
    st.plotly_chart(apply_plotly_theme(trend), use_container_width=True)

    distribution_col, methods_col = st.columns(2)
    with distribution_col:
        st.markdown("### Severity distribution")
        risk_distribution = pd.DataFrame(
            {
                "Risk": list(RISK_ORDER),
                "Records": [int(views["risk_counts"][risk]) for risk in RISK_ORDER],
            }
        )
        total_anomalies = max(anomaly_count, 1)
        risk_distribution["Percentage"] = (
            risk_distribution["Records"] / total_anomalies * 100
        )
        risk_distribution["Label"] = risk_distribution.apply(
            lambda row: f"{int(row['Records']):,}  Â·  {row['Percentage']:.1f}%",
            axis=1,
        )
        severity = px.bar(
            risk_distribution,
            x="Records",
            y="Risk",
            orientation="h",
            text="Label",
            color="Risk",
            color_discrete_map=RISK_COLORS,
            category_orders={"Risk": list(reversed(RISK_ORDER))},
            custom_data=["Percentage"],
        )
        severity.update_traces(
            textposition="inside",
            hovertemplate="%{y}: %{x:,} records (%{customdata[0]:.1f}%)<extra></extra>",
        )
        severity.update_layout(
            height=320,
            xaxis_title="Anomaly records",
            yaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(apply_plotly_theme(severity), use_container_width=True)

    with methods_col:
        st.markdown("### Detection-method breakdown")
        method_frame = (
            views["method_counts"].rename_axis("Method").reset_index(name="Records")
        )
        methods = px.bar(
            method_frame,
            x="Records",
            y="Method",
            orientation="h",
            text="Records",
            color="Method",
            color_discrete_map={
                "Isolation Forest only": COLOR_CYAN,
                "Z-score only": COLOR_TEAL,
                "Both methods": COLOR_AMBER,
            },
            category_orders={
                "Method": [
                    "Isolation Forest only",
                    "Z-score only",
                    "Both methods",
                ]
            },
        )
        methods.update_traces(
            texttemplate="%{x:,}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} records<extra></extra>",
        )
        methods.update_layout(
            height=320,
            xaxis_title="Flagged records",
            yaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(apply_plotly_theme(methods), use_container_width=True)

    st.markdown("### Affected road segments Â· Hyderabad")
    st.caption(
        "Historical dataset coordinates; marker size represents anomaly count and "
        "color shows the dominant risk. This is not live monitoring."
    )
    legend = " &nbsp; ".join(
        f'<span style="color:{RISK_COLORS[risk]}">â—</span> {risk}'
        for risk in RISK_ORDER
    )
    st.markdown(legend, unsafe_allow_html=True)
    map_col, ranking_col = st.columns([1.45, 1])
    with map_col:
        _render_segment_map(views["segments"])
    with ranking_col:
        st.markdown("#### Most affected segments")
        segment_table = (
            views["segments"]
            .head(5)
            .rename(
                columns={
                    "road_segment_id": "Road segment",
                    "area": "Area",
                    "anomaly_records": "Anomaly records",
                    "dominant_risk": "Dominant risk",
                }
            )
        )
        st.dataframe(
            segment_table[["Road segment", "Area", "Anomaly records", "Dominant risk"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Top historical anomaly records")
    if not views["top_records"].empty:
        top_table = views["top_records"][
            [
                "road_segment_id",
                "area",
                "timestamp",
                "risk_level",
                "future_congestion_level",
                "average_speed_kmh",
                "occupancy_percent",
                "anomaly_score",
                "is_anomaly",
                "zscore_anomaly",
            ]
        ].copy()
        top_table["timestamp"] = top_table["timestamp"].dt.strftime("%b %d, %Y %H:%M")
        top_table = top_table.rename(
            columns={
                "road_segment_id": "Road Segment",
                "area": "Area",
                "timestamp": "Observation Time",
                "risk_level": "Risk",
                "future_congestion_level": "Future Class",
                "average_speed_kmh": "Average Speed",
                "occupancy_percent": "Occupancy",
                "anomaly_score": "IF Score",
                "is_anomaly": "IF Flag",
                "zscore_anomaly": "Z Flag",
            }
        )
        top_table["Average Speed"] = top_table["Average Speed"].round(1)
        top_table["Occupancy"] = top_table["Occupancy"].round(1)
        st.dataframe(
            top_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No combined anomalies were detected.")

    with st.expander("Additional anomaly analysis", expanded=False):
        # Scatter chart using API-provided records
        st.markdown("#### Vehicle count vs average speed")

        scatter_data = pd.DataFrame(payload.get("scatter_records", []))

        if not scatter_data.empty:
            scatter = px.scatter(
                scatter_data,
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
                    if column in scatter_data.columns
                ],
                opacity=0.65,
                color_discrete_map={
                    "Normal": COLOR_EMERALD,
                    "Anomaly": COLOR_RED,
                },
            )

            scatter.update_layout(
                xaxis_title="Vehicle Count",
                yaxis_title="Average Speed (km/h)",
                height=420,
                legend_title="Status",
            )

            st.plotly_chart(
                apply_plotly_theme(scatter),
                use_container_width=True,
            )
        else:
            st.info("No scatter-plot data available.")

        # Anomaly score histogram using API-provided records
        score_data = pd.DataFrame(payload.get("score_records", []))

        if not score_data.empty:
            score_chart = px.histogram(
                score_data,
                x="anomaly_score",
                nbins=40,
            )

            score_chart.update_layout(
                xaxis_title="Anomaly score Â· higher means more unusual",
                yaxis_title="Count",
                height=320,
            )

            st.plotly_chart(
                apply_plotly_theme(score_chart),
                use_container_width=True,
            )

            st.caption(
                "Isolation Forest scores indicate relative outlierness; they are "
                "not probabilities or confidence scores."
            )
        else:
            st.info("No anomaly-score data available.")

    st.markdown("### Historical decision-support recommendations")
    st.caption(
        "Recommendations are scenario-evaluation suggestions based on historical "
        "synthetic records, not live monitoring or confirmed incidents."
    )
    if anomaly_records:
        for index, anomaly in enumerate(anomaly_records[:5], start=1):
            title = (
                f"Anomaly #{index} â€” {anomaly['road_segment_id']} "
                f"({anomaly['risk_level']} Risk)"
            )
            with st.expander(title, expanded=(index == 1)):
                st.markdown(f"#### {title}")
                st.markdown(f"**Description:** {anomaly['description']}")
                st.markdown(
                    f"**Future Congestion:** " f"{anomaly['future_congestion_level']}"
                )
                st.markdown(f"**Anomaly Score:** {anomaly['anomaly_score']:.4f}")
                for recommendation in generate_anomaly_recommendations(anomaly):
                    st.markdown(
                        f"**{recommendation['priority']}:** "
                        f"{recommendation['action']}"
                    )
                    st.markdown(f"- Why: {recommendation['explanation']}")
    else:
        st.info("No anomaly recommendations are available.")

    with st.expander("Detection methodology", expanded=False):
        st.markdown("""
**Isolation Forest:** 100 estimators with 5% contamination, using vehicle count,
average speed, occupancy percentage, and traffic flow.

**Z-score:** absolute Z-scores are calculated for those same four features. A row is
flagged when any value is greater than 3.

**Combined detection:** a record is anomalous when Isolation Forest **OR** Z-score
flags it. Risk classification uses the existing future congestion class rules.

The anomaly score is the negated Isolation Forest decision function: higher values
mean greater relative outlierness, not a probability or confidence score.
""")
