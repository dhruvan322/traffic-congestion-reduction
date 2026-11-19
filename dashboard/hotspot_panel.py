"""
Hotspot Detection Panel.

Uses the final ML traffic dataset and the DBSCAN-based
hotspot detector to identify spatial clusters of congested
road segments.
"""

from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from streamlit_folium import st_folium

from config import MAP_CENTER_LAT, MAP_CENTER_LON, MAP_ZOOM
from config import API_BASE_URL, API_TIMEOUT_SECONDS
from dashboard.ui_theme import (
    COLOR_AMBER,
    COLOR_RED,
    COLOR_TEAL,
    apply_plotly_theme,
    render_header,
    render_kpi_card,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"

SEVERITY_COLORS = {
    "Critical": COLOR_RED,
    "High": COLOR_AMBER,
    "Medium": COLOR_TEAL,
}


@st.cache_data
def load_final_dataset():
    """Load the final integrated ML traffic dataset."""

    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Final ML dataset not found: {FINAL_DATASET_PATH}")

    df = pd.read_csv(FINAL_DATASET_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def _format_peak_hours(hours):
    """Format detector-provided peak congestion hours."""

    return ", ".join(f"{int(hour):02d}:00" for hour in hours)


def _build_hotspot_map(summaries):
    """Build the existing OpenStreetMap-based map from detector summaries."""

    hotspot_map = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=MAP_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    hotspot_locations = []

    for spot in summaries:
        center_lat = float(spot["center_lat"])
        center_lon = float(spot["center_lon"])
        hotspot_locations.append([center_lat, center_lon])

        severity = str(spot["severity"])
        color = {
            "Critical": "red",
            "High": "orange",
            "Medium": "cadetblue",
        }.get(severity, "blue")
        radius = min(30, max(10, 8 + int(spot["size"]) * 3))
        peak_hours_text = _format_peak_hours(spot["peak_hours"])

        popup_html = (
            f"<b>Hotspot Cluster {spot['cluster_id']}</b><br>"
            f"Severity: {severity}<br>"
            f"Road Segments: {spot['size']}<br>"
            f"Traffic Records: {spot['records']:,}<br>"
            f"Congestion Rate: {spot['congestion_rate']:.1%}<br>"
            f"High/Severe Rate: {spot['high_or_severe_rate']:.1%}<br>"
            f"Severe Rate: {spot['severe_rate']:.1%}<br>"
            f"Avg Vehicles: {spot['avg_vehicles']:.0f}<br>"
            f"Avg Speed: {spot['avg_speed']:.1f} km/h<br>"
            f"Peak Congestion Hours: {peak_hours_text}"
        )

        folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"Cluster {spot['cluster_id']} · {severity}",
        ).add_to(hotspot_map)

    if hotspot_locations:
        min_lat = min(location[0] for location in hotspot_locations)
        max_lat = max(location[0] for location in hotspot_locations)
        min_lon = min(location[1] for location in hotspot_locations)
        max_lon = max(location[1] for location in hotspot_locations)
        hotspot_map.fit_bounds(
            [[min_lat, min_lon], [max_lat, max_lon]],
            padding=(30, 30),
        )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: #131C2E;
        color: #F8FAFC;
        padding: 10px 14px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        font: 12px Inter, sans-serif;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    ">
        <b>Detected cluster severity</b><br>
        <span style="color:{COLOR_RED};">●</span> Critical<br>
        <span style="color:{COLOR_AMBER};">●</span> High<br>
        <span style="color:{COLOR_TEAL};">●</span> Medium
    </div>
    """
    hotspot_map.get_root().html.add_child(folium.Element(legend_html))
    return hotspot_map


def _render_cluster_table(summaries):
    """Render the detector's actual cluster summaries and segment IDs."""

    cluster_df = pd.DataFrame(summaries)
    display_df = cluster_df[
        [
            "cluster_id",
            "severity",
            "size",
            "records",
            "congestion_rate",
            "high_or_severe_rate",
            "severe_rate",
            "avg_vehicles",
            "avg_speed",
            "avg_occupancy",
            "peak_hours",
        ]
    ].copy()
    display_df = display_df.rename(
        columns={
            "cluster_id": "Cluster ID",
            "severity": "Severity",
            "size": "Road segments",
            "records": "Historical records",
            "congestion_rate": "Congestion rate",
            "high_or_severe_rate": "High / severe rate",
            "severe_rate": "Severe rate",
            "avg_vehicles": "Avg. vehicles",
            "avg_speed": "Avg. speed (km/h)",
            "avg_occupancy": "Avg. occupancy (%)",
            "peak_hours": "Peak congestion hours",
        }
    )

    for column in ("Congestion rate", "High / severe rate", "Severe rate"):
        display_df[column] = display_df[column].map(lambda value: f"{value:.1%}")

    display_df["Avg. vehicles"] = display_df["Avg. vehicles"].map(
        lambda value: f"{value:.1f}"
    )
    display_df["Avg. speed (km/h)"] = display_df["Avg. speed (km/h)"].map(
        lambda value: f"{value:.1f}"
    )
    display_df["Avg. occupancy (%)"] = display_df["Avg. occupancy (%)"].map(
        lambda value: f"{value:.1f}"
    )
    display_df["Peak congestion hours"] = display_df[
        "Peak congestion hours"
    ].map(_format_peak_hours)

    st.dataframe(display_df, width="stretch", hide_index=True)

    for spot in summaries:
        segment_ids = [str(segment) for segment in spot["road_segments"]]
        with st.expander(
            f"Cluster {spot['cluster_id']} · {len(segment_ids)} road segment IDs"
        ):
            st.dataframe(
                pd.DataFrame({"Road segment ID": segment_ids}),
                width="stretch",
                height=min(300, 38 + 35 * len(segment_ids)),
                hide_index=True,
            )


def _render_cluster_charts(cluster_df):
    """Visualize only cluster severity and segment counts from detector results."""

    chart_columns = st.columns(2)
    severity_counts = (
        cluster_df["severity"]
        .value_counts()
        .reindex(["Critical", "High", "Medium"], fill_value=0)
    )

    with chart_columns[0]:
        with st.container(border=True):
            st.markdown("#### Detected cluster severity")
            figure = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                color=severity_counts.index,
                color_discrete_map=SEVERITY_COLORS,
                hole=0.55,
            )
            total_clusters = int(severity_counts.sum())
            figure.update_traces(
                text=[
                    f"{severity}<br>{count / total_clusters:.0%}"
                    if count > 0
                    else ""
                    for severity, count in severity_counts.items()
                ],
                textinfo="text",
                textposition="inside",
                insidetextorientation="horizontal",
                hovertemplate="%{label}: %{value} clusters<extra></extra>",
            )
            figure.update_layout(
                height=320,
                showlegend=True,
                uniformtext=dict(minsize=10, mode="hide"),
            )
            apply_plotly_theme(figure)
            st.plotly_chart(
                figure,
                width="stretch",
                config={"responsive": True},
            )

    with chart_columns[1]:
        with st.container(border=True):
            st.markdown("#### Road segments per cluster")
            cluster_plot_df = cluster_df.copy()
            cluster_plot_df["cluster_id"] = (
                pd.to_numeric(cluster_plot_df["cluster_id"], errors="raise")
                .astype(int)
                .astype(str)
            )
            cluster_ids = cluster_plot_df["cluster_id"].tolist()
            figure = px.bar(
                cluster_plot_df,
                x="cluster_id",
                y="size",
                color="severity",
                color_discrete_map=SEVERITY_COLORS,
                category_orders={"cluster_id": cluster_ids},
                text_auto=True,
            )
            figure.update_layout(height=320, showlegend=True)
            figure.update_xaxes(
                title_text="DBSCAN cluster ID",
                type="category",
                tickmode="array",
                tickvals=cluster_ids,
                ticktext=cluster_ids,
            )
            figure.update_yaxes(title_text="Road segments")
            apply_plotly_theme(figure)
            st.plotly_chart(
                figure,
                width="stretch",
                config={"responsive": True},
            )


def render_hotspot_panel():
    """Render the DBSCAN congestion hotspot dashboard."""

    render_header(
        title="Congestion Hotspot Detection",
        subtitle=(
            "Explore DBSCAN spatial clusters of road segments with historical "
            "Moderate, High, or Severe congestion observations."
        ),
        badge_text="Historical dataset",
        badge_type="teal",
    )

    try:
        df = load_final_dataset()
    except Exception as exc:
        st.error(f"Unable to load the final ML dataset: {exc}")
        return

    if df.empty:
        st.warning("The final ML dataset contains no traffic records.")
        return

    dataset_start = df["timestamp"].min() if "timestamp" in df.columns else pd.NaT
    dataset_end = df["timestamp"].max() if "timestamp" in df.columns else pd.NaT
    if pd.notna(dataset_start) and pd.notna(dataset_end):
        date_range = f"{dataset_start:%d %b %Y} – {dataset_end:%d %b %Y}"
    else:
        date_range = "Date range unavailable"

    st.caption(
        f"Historical synthetic traffic data · {len(df):,} records · {date_range}. "
        "Hourly observations are aggregated by road segment before spatial clustering."
    )

    dataset_columns = st.columns(3)
    dataset_stats = (
        ("DATASET RECORDS", f"{len(df):,}", "Historical traffic observations", "cyan"),
        (
            "ROAD SEGMENTS",
            f"{df['road_segment_id'].nunique():,}",
            "Represented in the dataset",
            "teal",
        ),
        (
            "CONGESTION CLASSES",
            f"{df['future_congestion_level'].nunique():,}",
            "Recorded future congestion labels",
            "blue",
        ),
    )
    for column, (title, value, subtitle, status) in zip(
        dataset_columns, dataset_stats
    ):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    with st.container(border=True):
        st.markdown("#### Detection controls")
        st.caption(
            "DBSCAN settings remain configured in config.py. Detection runs "
            "only when requested."
        )
        action_columns = st.columns([1, 1, 3])
        with action_columns[0]:
            detect_clicked = st.button(
                "Detect hotspots",
                type="primary",
                width="stretch",
            )
        with action_columns[1]:
            clear_clicked = st.button("Clear results", width="stretch")
        with action_columns[2]:
            st.caption(
                "Results describe spatial clusters in the historical dataset, "
                "not live or persistent hotspots."
            )

    if clear_clicked:
        st.session_state.pop("hotspot_summaries", None)
        st.rerun()

    if detect_clicked:
        try:
            with st.spinner("Detecting traffic hotspots through FastAPI..."):
                response = requests.get(
                    f"{API_BASE_URL.rstrip('/')}/api/v1/hotspots",
                    timeout=API_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                summaries = response.json()

            st.session_state["hotspot_summaries"] = summaries
        except Exception as exc:
            st.error(f"Hotspot detection failed: {exc}")
            return

    summaries = st.session_state.get("hotspot_summaries")
    if summaries is None:
        st.info("Select **Detect hotspots** to run spatial clustering.")
        return

    if not summaries:
        st.warning("No hotspot clusters were detected with the current DBSCAN settings.")
        st.caption("DBSCAN_EPS and DBSCAN_MIN_SAMPLES are configured in config.py.")
        return

    cluster_df = pd.DataFrame(summaries)
    total_segments = int(cluster_df["size"].sum())
    critical_count = int((cluster_df["severity"] == "Critical").sum())
    high_count = int((cluster_df["severity"] == "High").sum())
    medium_count = int((cluster_df["severity"] == "Medium").sum())

    st.markdown("### Detection summary")
    summary_columns = st.columns(4)
    summary_stats = (
        ("DETECTED CLUSTERS", f"{len(summaries):,}", "DBSCAN cluster groups", "cyan"),
        (
            "CLUSTERED ROAD SEGMENTS",
            f"{total_segments:,}",
            "Segments in detected clusters",
            "teal",
        ),
        ("CRITICAL / HIGH", f"{critical_count + high_count:,}", "Clusters", "amber"),
        ("MEDIUM", f"{medium_count:,}", "Clusters", "emerald"),
    )
    for column, (title, value, subtitle, status) in zip(
        summary_columns, summary_stats
    ):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    st.markdown("### Geographic distribution")
    st.caption(
        "Circle locations and sizes come from detector cluster centers and "
        "road-segment counts. The basemap uses OpenStreetMap tiles."
    )
    hotspot_map = _build_hotspot_map(summaries)
    st_folium(
        hotspot_map,
        width=None,
        height=500,
        use_container_width=True,
        key="hotspot_map",
    )

    with st.container(border=True):
        st.markdown("#### Cluster details")
        st.caption(
            "Cluster IDs are detector labels, not named neighborhoods. "
            "Rates and peak hours summarize the historical records used by the detector."
        )
        _render_cluster_table(summaries)

    _render_cluster_charts(cluster_df)

    with st.expander("DBSCAN hotspot detection methodology"):
        st.markdown(
            """
**Input:** Final integrated ML traffic dataset.

1. Records labelled Moderate, High, or Severe future congestion are considered congested.
2. Hourly records are aggregated to road-segment level to avoid repeatedly weighting
   the same physical segment.
3. DBSCAN groups candidate segment coordinates using `DBSCAN_EPS` and
   `DBSCAN_MIN_SAMPLES` from `config.py`.
4. DBSCAN noise points (`cluster_id = -1`) are excluded.
5. Cluster summaries report segment and record counts, congestion rates, average
   vehicle count, speed, occupancy, and the three most frequent congestion hours.
6. Critical severity is assigned for severe rate ≥ 25% or high/severe rate ≥ 70%;
   High for high/severe rate ≥ 40%; otherwise Medium.

This detector does not establish named-area rankings, hotspot persistence or
duration, temporal trends, or real-time monitoring.
            """
        )

    st.caption(
        "Historical synthetic traffic data · spatial cluster analysis · "
        "no live traffic feed or hotspot persistence tracking."
    )
