"""
Hotspot Detection Panel.

Uses the final ML traffic dataset and the DBSCAN-based
hotspot detector to identify spatial clusters of repeatedly
congested road segments.
"""

from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from config import (
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
)

from models.hotspot_detector import (
    get_hotspot_summary,
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
# Hotspot panel
# -------------------------------------------------------------------


def render_hotspot_panel():
    """
    Render the DBSCAN congestion hotspot dashboard.
    """

    st.markdown("# 🔥 Congestion Hotspot Detection")

    st.markdown(
        "Identify spatial clusters of road segments that "
        "repeatedly experience Moderate, High, or Severe "
        "future congestion using DBSCAN clustering."
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
        "Hotspot analysis uses the complete final traffic "
        "dataset and aggregates hourly observations to "
        "road-segment level before spatial clustering."
    )

    st.divider()

    # ---------------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------------

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Road Segments",
            df["road_segment_id"].nunique(),
        )

    with info_col2:

        st.metric(
            "Traffic Records",
            f"{len(df):,}",
        )

    with info_col3:

        st.metric(
            "Congestion Classes",
            "5",
        )

    st.divider()

    # ---------------------------------------------------------------
    # Detection controls
    # ---------------------------------------------------------------

    button_col1, button_col2 = st.columns([3, 1])

    with button_col1:

        detect_clicked = st.button(
            "🔍 Detect Hotspots",
            type="primary",
            use_container_width=True,
        )

    with button_col2:

        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # Clear previous results
    # ---------------------------------------------------------------

    if clear_clicked:

        st.session_state.pop(
            "hotspot_summaries",
            None,
        )

        st.rerun()

    # ---------------------------------------------------------------
    # Run hotspot detection
    # ---------------------------------------------------------------

    if detect_clicked:

        try:

            with st.spinner("Running DBSCAN hotspot detection..."):

                summaries = get_hotspot_summary(df)

            st.session_state["hotspot_summaries"] = summaries

        except Exception as exc:

            st.error(f"Hotspot detection failed: {exc}")

            return

    # ---------------------------------------------------------------
    # Retrieve stored results
    # ---------------------------------------------------------------

    summaries = st.session_state.get("hotspot_summaries")

    # Do not run the expensive detection automatically.
    if summaries is None:

        st.info(
            "Click **Detect Hotspots** to run DBSCAN "
            "spatial clustering on the final ML dataset."
        )

        return

    # ---------------------------------------------------------------
    # No hotspots
    # ---------------------------------------------------------------

    if not summaries:

        st.warning(
            "No hotspot clusters were detected using the " "current DBSCAN parameters."
        )

        st.caption(
            "DBSCAN settings are controlled by "
            "DBSCAN_EPS and DBSCAN_MIN_SAMPLES "
            "in config.py."
        )

        return

    # ---------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------

    st.success(f"✅ Found **{len(summaries)}** hotspot clusters.")

    total_segments = sum(spot["size"] for spot in summaries)

    critical_count = sum(spot["severity"] == "Critical" for spot in summaries)

    high_count = sum(spot["severity"] == "High" for spot in summaries)

    medium_count = sum(spot["severity"] == "Medium" for spot in summaries)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        st.metric(
            "Hotspot Clusters",
            len(summaries),
        )

    with metric_col2:

        st.metric(
            "Affected Road Segments",
            total_segments,
        )

    with metric_col3:

        st.metric(
            "Critical / High",
            critical_count + high_count,
        )

    with metric_col4:

        st.metric(
            "Medium",
            medium_count,
        )

    # ---------------------------------------------------------------
    # Hotspot map
    # ---------------------------------------------------------------

    st.subheader("🗺️ Hotspot Map")

    # Create the Folium map.
    # OpenStreetMap is used as the base layer for reliable rendering.
    m = folium.Map(
        location=[
            MAP_CENTER_LAT,
            MAP_CENTER_LON,
        ],
        zoom_start=MAP_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    severity_colors = {
        "Critical": "red",
        "High": "orange",
        "Medium": "yellow",
    }

    # Store hotspot centers so the map can automatically fit
    # all detected clusters into the visible area.
    hotspot_locations = []

    for spot in summaries:

        center_lat = float(spot["center_lat"])
        center_lon = float(spot["center_lon"])

        hotspot_locations.append([center_lat, center_lon])

        color = severity_colors.get(
            spot["severity"],
            "blue",
        )

        # Marker size represents the number
        # of physical road segments in the cluster.
        radius = min(
            30,
            max(
                10,
                8 + (spot["size"] * 3),
            ),
        )

        peak_hours_text = ", ".join(
            f"{int(hour):02d}:00" for hour in spot["peak_hours"]
        )

        popup_html = (
            f"<b>Hotspot Cluster "
            f"{spot['cluster_id']}</b><br>"
            f"Severity: "
            f"{spot['severity']}<br>"
            f"Road Segments: "
            f"{spot['size']}<br>"
            f"Traffic Records: "
            f"{spot['records']:,}<br>"
            f"Congestion Rate: "
            f"{spot['congestion_rate']:.1%}<br>"
            f"High/Severe Rate: "
            f"{spot['high_or_severe_rate']:.1%}<br>"
            f"Severe Rate: "
            f"{spot['severe_rate']:.1%}<br>"
            f"Avg Vehicles: "
            f"{spot['avg_vehicles']:.0f}<br>"
            f"Avg Speed: "
            f"{spot['avg_speed']:.1f} km/h<br>"
            f"Peak Hours: "
            f"{peak_hours_text}"
        )

        folium.CircleMarker(
            location=[
                center_lat,
                center_lon,
            ],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(
                popup_html,
                max_width=350,
            ),
            tooltip=(f"Cluster {spot['cluster_id']} — " f"{spot['severity']}"),
        ).add_to(m)

    # Automatically fit the map around all detected hotspots.
    if hotspot_locations:

        min_lat = min(location[0] for location in hotspot_locations)

        max_lat = max(location[0] for location in hotspot_locations)

        min_lon = min(location[1] for location in hotspot_locations)

        max_lon = max(location[1] for location in hotspot_locations)

        m.fit_bounds(
            [
                [min_lat, min_lon],
                [max_lat, max_lon],
            ],
            padding=(30, 30),
        )

    # Map legend
    legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background-color: white;
    padding: 10px 14px;
    border: 2px solid grey;
    border-radius: 6px;
    font-size: 13px;
">
    <b>Hotspot Severity</b><br>
    <span style="color:red;">●</span> Critical<br>
    <span style="color:orange;">●</span> High<br>
    <span style="color:#d4a900;">●</span> Medium
</div>
"""

    m.get_root().html.add_child(folium.Element(legend_html))

    # Explicit dimensions are more reliable than width=None
    # for the Streamlit-Folium component.
    st_folium(
        m,
        width=1000,
        height=550,
    )

    # ---------------------------------------------------------------
    # Cluster details
    # ---------------------------------------------------------------

    st.subheader("📊 Hotspot Cluster Details")

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
            "road_segments",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "cluster_id": "Cluster",
            "severity": "Severity",
            "size": "Road Segments",
            "records": "Traffic Records",
            "congestion_rate": "Congestion Rate",
            "high_or_severe_rate": ("High/Severe Rate"),
            "severe_rate": "Severe Rate",
            "avg_vehicles": "Avg Vehicles",
            "avg_speed": ("Avg Speed (km/h)"),
            "avg_occupancy": ("Avg Occupancy (%)"),
            "peak_hours": "Peak Hours",
            "road_segments": ("Road Segment IDs"),
        }
    )

    display_df["Congestion Rate"] = display_df["Congestion Rate"].map(
        lambda value: f"{value:.1%}"
    )

    display_df["High/Severe Rate"] = display_df["High/Severe Rate"].map(
        lambda value: f"{value:.1%}"
    )

    display_df["Severe Rate"] = display_df["Severe Rate"].map(
        lambda value: f"{value:.1%}"
    )

    display_df["Peak Hours"] = display_df["Peak Hours"].map(
        lambda hours: ", ".join(f"{int(hour):02d}:00" for hour in hours)
    )

    display_df["Road Segment IDs"] = display_df["Road Segment IDs"].map(
        lambda segments: ", ".join(str(segment) for segment in segments)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⚡ Severity Distribution")

        severity_counts = (
            cluster_df["severity"]
            .value_counts()
            .reindex(
                [
                    "Critical",
                    "High",
                    "Medium",
                ],
                fill_value=0,
            )
        )

        fig1 = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            color=severity_counts.index,
            color_discrete_map={
                "Critical": "#f44336",
                "High": "#FF9800",
                "Medium": "#FFC107",
            },
            hole=0.3,
        )

        fig1.update_layout(
            height=350,
            template="plotly_dark",
        )

        st.plotly_chart(
            fig1,
            use_container_width=True,
        )

    with col2:

        st.subheader("📈 Cluster Sizes")

        fig2 = px.bar(
            cluster_df,
            x="cluster_id",
            y="size",
            color="severity",
            color_discrete_map={
                "Critical": "#f44336",
                "High": "#FF9800",
                "Medium": "#FFC107",
            },
            text_auto=True,
        )

        fig2.update_layout(
            xaxis_title="Cluster ID",
            yaxis_title="Road Segments",
            height=350,
            template="plotly_dark",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # Methodology
    # ---------------------------------------------------------------

    with st.expander(
        "ℹ️ Hotspot Detection Methodology",
        expanded=False,
    ):

        st.markdown("""
### DBSCAN Spatial Hotspot Detection

**Input:** Final integrated ML traffic dataset.

**Step 1 — Congestion filtering**

Traffic observations with
`future_congestion_level` equal to **Moderate, High, or Severe**
are considered congestion observations.

**Step 2 — Road-segment aggregation**

The hourly records are aggregated to the physical
road-segment level so repeated observations at the same
location do not artificially dominate DBSCAN.

**Step 3 — Spatial clustering**

DBSCAN clusters nearby road-segment coordinates using:

- `DBSCAN_EPS`
- `DBSCAN_MIN_SAMPLES`

from `config.py`.

**Step 4 — Noise removal**

DBSCAN noise points (`cluster_id = -1`) are excluded.

**Step 5 — Hotspot characterization**

Each cluster is summarized using:

- Number of road segments
- Number of traffic records
- Congestion rate
- High/Severe rate
- Severe rate
- Average vehicle count
- Average speed
- Average occupancy
- Peak congestion hours

**Step 6 — Severity classification**

- **Critical:** Severe rate ≥ 25% OR High/Severe rate ≥ 70%
- **High:** High/Severe rate ≥ 40%
- **Medium:** Remaining detected clusters
            """)
