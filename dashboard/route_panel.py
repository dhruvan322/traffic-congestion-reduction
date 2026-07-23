"""
Route Optimizer Panel.
Interactive map with Bidirectional Dijkstra for shortest and least-congested paths.
"""

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from config import DATASET_PATH
from engine.route_optimizer import (
    build_junction_graph, compare_routes, create_route_map, optimize_route,
)
from engine.osm_route_engine import (
    find_osm_route,
    create_osm_route_map
)
from database import insert_route
from engine.reduction_engine import generate_route_explanation
from data.generate_dataset import JUNCTION_NAMES

JUNCTION_COORDS = {
    "J001": (17.3850, 78.4867),
    "J002": (17.4374, 78.4482),
    "J003": (17.4448, 78.4660),
    "J004": (17.4399, 78.4983),
    "J005": (17.4435, 78.3772),
    "J006": (17.4483, 78.3915),
    "J007": (17.4401, 78.3489),
    "J008": (17.4239, 78.4738),
    "J009": (17.4126, 78.4482),
    "J010": (17.4239, 78.4738),
    "J011": (17.3950, 78.4390),
    "J012": (17.3457, 78.5520),
    "J013": (17.4058, 78.5591),
    "J014": (17.4849, 78.4138),
    "J015": (17.4948, 78.3560),
    "J016": (17.3616, 78.4747),
    "J017": (17.3948, 78.4678),
    "J018": (17.3924, 78.4738),
    "J019": (17.4430, 78.4875),
    "J020": (17.4620, 78.3565),
}


def render_route_panel():
    """Render the route optimization panel with map visualization."""
    st.markdown("# 🗺️ Route Optimizer")
    st.markdown(
        "Find the **shortest** and **least congested** paths using "
        "**Bidirectional Dijkstra's Algorithm** — visualized on an interactive map."
    )

    df = pd.read_csv(DATASET_PATH)

    # Persist route results across Streamlit reruns
    if "route_result" not in st.session_state:
        st.session_state.route_result = None

    junctions = sorted(df["junction_id"].unique())

    # Format junction labels
    junction_labels = {
        jid: f"{jid} — {JUNCTION_NAMES.get(jid, 'Unknown')}"
        for jid in junctions
    }

    # ─────────── Route Input ───────────
    st.subheader("📍 Select Origin & Destination")

    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox(
            "🟢 Origin Junction",
            junctions,
            format_func=lambda x: junction_labels[x],
            index=0,
        )
    with col2:
        dest_options = [j for j in junctions if j != origin]
        destination = st.selectbox(
            "🔴 Destination Junction",
            dest_options,
            format_func=lambda x: junction_labels[x],
            index=min(len(dest_options) - 1, 4),
        )

    st.divider()

    # ─────────── Route Calculation ───────────
    if st.button("🚀 Find Best Routes", type="primary", use_container_width=True):
        with st.spinner("Building traffic graph & running Bidirectional Dijkstra..."):
            st.session_state.route_result = optimize_route(
                df,
                origin,
                destination
            )
            origin_lat, origin_lon = JUNCTION_COORDS[origin]
            dest_lat, dest_lon = JUNCTION_COORDS[destination]
            
            print("ORIGIN:", origin, JUNCTION_NAMES[origin], JUNCTION_COORDS[origin])
            print("DEST:", destination, JUNCTION_NAMES[destination], JUNCTION_COORDS[destination])
            
            G_osm, osm_route = find_osm_route(
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon
            )

            st.session_state.route_map = create_osm_route_map(
                G_osm,
                osm_route,
                JUNCTION_NAMES.get(origin, origin),
                JUNCTION_NAMES.get(destination, destination)
            )

    result = st.session_state.route_result

    if result is not None:

        if "error" in result.get("shortest", {}):
            st.error(f"❌ {result['shortest']['error']}")
            return

        shortest = result["shortest"]
        least_congested = result["least_congested"]

        # AI Route Decision Support
        route_ai = generate_route_explanation(
            shortest,
            least_congested,
            result["graph"]
        )

        # ─────────── Results Summary ───────────
        st.markdown("---")
        st.subheader("📊 Route Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔵 Shortest Path")
            st.markdown(f"**Algorithm:** {shortest['algorithm']}")
            st.metric("Distance", f"{shortest['total_distance_km']} km")
            st.metric("Estimated Time", f"{shortest['estimated_minutes']} min")
            st.markdown(f"**Route:** {' → '.join(shortest['path'])}")

        with col2:
            st.markdown("### 🟢 Least Congested Path")
            st.markdown(f"**Algorithm:** {least_congested['algorithm']}")
            st.metric("Distance", f"{least_congested['total_distance_km']} km")
            st.metric("Estimated Time", f"{least_congested['estimated_minutes']} min")
            st.markdown(f"**Route:** {' → '.join(least_congested['path'])}")

        # Time savings
        time_saved = result.get("time_saved_minutes", 0)
        extra_dist = result.get("extra_distance_km", 0)

        st.markdown("---")
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        with rec_col1:
            delta_color = "normal" if time_saved > 0 else "inverse"
            st.metric("⏱️ Time Difference", f"{abs(time_saved):.1f} min",
                      delta=f"{'Saved' if time_saved > 0 else 'Extra'}")
        with rec_col2:
            st.metric("📏 Extra Distance", f"{extra_dist:.2f} km")
        with rec_col3:
            st.info(f"💡 **{result.get('recommendation', '')}**")

        st.markdown("---")
        st.subheader("🧠 AI Route Recommendation")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                "AI Route Score",
                f"{route_ai['ai_score']}/100"
            )

        with col2:
            st.success(route_ai["recommendation_text"])

        with st.expander("📖 Why This Route?", expanded=True):

            for factor in route_ai["factors"]:
                st.markdown(f"- {factor}")

            st.markdown("---")

            st.markdown(
                f"**Estimated Time Saved:** "
                f"{route_ai['time_saved_minutes']} min"
            )

            st.markdown(
                f"**Extra Distance:** "
                f"{route_ai['extra_distance_km']} km"
            )

        # ─────────── Interactive Map ───────────
        st.markdown("---")
        st.subheader("🗺️ Route Visualization")
        st.markdown(
            "🔵 **Blue** = Shortest Path | "
            "🟢 **Green (dashed)** = Least Congested Path"
        )

        route_map = st.session_state.route_map
        st_folium(
            route_map,
            height=550,
            width=1200,
            returned_objects=[]
        )

        # ─────────── Path Details ───────────
        with st.expander("🔍 Detailed Path Analysis"):
            G = result["graph"]

            st.markdown("#### Shortest Path — Edge Details")
            for i in range(len(shortest["path"]) - 1):
                n1, n2 = shortest["path"][i], shortest["path"][i + 1]
                edge = G[n1][n2]
                st.markdown(
                    f"- **{n1}** → **{n2}**: "
                    f"{edge['distance_km']:.2f} km, "
                    f"{edge['travel_time']:.1f} min, "
                    f"avg speed {edge['avg_speed']:.1f} km/h"
                )

            st.markdown("#### Least Congested Path — Edge Details")
            for i in range(len(least_congested["path"]) - 1):
                n1, n2 = least_congested["path"][i], least_congested["path"][i + 1]
                edge = G[n1][n2]
                st.markdown(
                    f"- **{n1}** → **{n2}**: "
                    f"{edge['distance_km']:.2f} km, "
                    f"{edge['travel_time']:.1f} min, "
                    f"avg speed {edge['avg_speed']:.1f} km/h"
                )

        # Store route in DB
        try:
            insert_route({
                "origin": origin,
                "destination": destination,
                "path": shortest["path"],
                "distance_km": shortest["total_distance_km"],
                "estimated_minutes": shortest["estimated_minutes"],
                "time_saved_minutes": time_saved,
            })
        except Exception:
            pass

    # ─────────── Algorithm Info ───────────
    with st.expander("ℹ️ About Bidirectional Dijkstra"):
        st.markdown("""
        **Bidirectional Dijkstra's Algorithm** searches simultaneously from both 
        the source and target nodes, meeting in the middle. This approach:
        
        - 🚀 **Faster** than standard Dijkstra — explores roughly half the nodes
        - 🎯 **Optimal** — guaranteed to find the shortest path
        - 🔄 **Two modes** used in this panel:
            - **Distance-based**: Pure shortest distance between junctions
            - **Traffic-aware**: Weights edges by congestion level, vehicle count, 
              and average speed to find the least congested route
        
        The traffic-aware weight formula penalizes:
        - 🐌 Low average speeds (speed penalty)
        - 🚗 High vehicle counts (congestion penalty)
        - 📏 Longer physical distances
        """)
