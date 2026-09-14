"""
Hotspot Detection Panel.
Heatmap visualization and cluster analysis.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px

from config import DATASET_PATH, MAP_CENTER_LAT, MAP_CENTER_LON, MAP_ZOOM
from models.hotspot_detector import get_hotspot_summary, detect_hotspots
from database import insert_hotspots


def render_hotspot_panel():
    """Render the hotspot detection panel."""
    st.markdown("# 🔥 Congestion Hotspot Detection")
    st.markdown("Identify traffic congestion clusters using DBSCAN spatial clustering")

    df = pd.read_csv(DATASET_PATH)

    # Run hotspot detection
    if st.button("🔍 Detect Hotspots", type="primary"):
        st.session_state["hotspots_detected"] = True

    if st.session_state.get("hotspots_detected", False) or True:
        with st.spinner("Running DBSCAN clustering..."):
            summaries = get_hotspot_summary(df)

        if not summaries:
            st.warning(
                "No hotspot clusters detected. Try adjusting DBSCAN parameters in config.py"
            )
            return

        st.success(f"✅ Found **{len(summaries)}** hotspot clusters!")

        # ─────────── Heatmap ───────────
        st.subheader("🗺️ Hotspot Heatmap")

        m = folium.Map(
            location=[MAP_CENTER_LAT, MAP_CENTER_LON],
            zoom_start=MAP_ZOOM,
            tiles="CartoDB dark_matter",
        )

        # Add heatmap markers for each cluster
        severity_colors = {"Critical": "red", "High": "orange", "Medium": "yellow"}

        for spot in summaries:
            color = severity_colors.get(spot["severity"], "blue")

            # Cluster circle
            folium.CircleMarker(
                location=[spot["center_lat"], spot["center_lon"]],
                radius=max(10, spot["size"] / 5),
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=(
                    f"<b>Cluster {spot['cluster_id']}</b><br>"
                    f"Severity: {spot['severity']}<br>"
                    f"Records: {spot['size']}<br>"
                    f"Avg Vehicles: {spot['avg_vehicles']:.0f}<br>"
                    f"Avg Speed: {spot['avg_speed']:.1f} km/h<br>"
                    f"Peak Hours: {spot['peak_hours']}"
                ),
                tooltip=f"Cluster {spot['cluster_id']} ({spot['severity']})",
            ).add_to(m)

        st_folium(m, width=None, height=500)

        # ─────────── Cluster Details ───────────
        st.subheader("📊 Cluster Details")

        cluster_df = pd.DataFrame(summaries)
        display_cols = [
            "cluster_id",
            "severity",
            "size",
            "avg_vehicles",
            "avg_speed",
            "peak_hours",
        ]
        available_cols = [c for c in display_cols if c in cluster_df.columns]
        st.dataframe(
            cluster_df[available_cols],
            use_container_width=True,
            hide_index=True,
        )

        # ─────────── Severity Chart ───────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Severity Distribution")
            sev_counts = cluster_df["severity"].value_counts()
            fig1 = px.pie(
                values=sev_counts.values,
                names=sev_counts.index,
                color=sev_counts.index,
                color_discrete_map={
                    "Critical": "#f44336",
                    "High": "#FF9800",
                    "Medium": "#FFC107",
                },
                hole=0.3,
            )
            fig1.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)

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
                yaxis_title="Records in Cluster",
                height=300,
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Save to DB
        try:
            db_hotspots = [
                {
                    "latitude": s["center_lat"],
                    "longitude": s["center_lon"],
                    "cluster_id": s["cluster_id"],
                    "severity": s["severity"],
                    "peak_hours": s["peak_hours"],
                    "detected_at": s["detected_at"],
                }
                for s in summaries
            ]
            insert_hotspots(db_hotspots)
        except Exception:
            pass
