"""
AI Chat Assistant Panel.
Chat interface with intent detection and multilingual support.
"""

import streamlit as st
import pandas as pd
import uuid

from config import DATASET_PATH, SUPPORTED_LANGUAGES
from nlp.assistant import process_query, classify_intent, extract_entities
from nlp.multilingual import get_language_name
from database import get_chat_history
from engine.osm_route_engine import find_osm_route, create_osm_route_map
import folium
from streamlit_folium import st_folium
from models.hotspot_detector import get_hotspot_summary
from models.anomaly_detector import get_anomaly_report
from config import MAP_CENTER_LAT, MAP_CENTER_LON, MAP_ZOOM


def render_hotspot_map(df):
    """
    Render hotspot map inside AI Copilot.
    """

    summaries = get_hotspot_summary(df)

    if not summaries:
        st.info("No hotspot clusters detected.")
        return

    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=MAP_ZOOM,
        tiles="CartoDB dark_matter",
    )

    severity_colors = {
        "Critical": "#ff0000",  # Red
        "High": "#ff8800",  # Orange
        "Medium": "#ffd700",  # Gold
        "Low": "#00cc66",  # Green
    }

    for spot in summaries[:5]:

        color = severity_colors.get(spot["severity"], "blue")

        folium.CircleMarker(
            location=[spot["center_lat"], spot["center_lon"]],
            radius=18,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"Cluster {spot['cluster_id']}",
            popup=(f"""
                Cluster {spot['cluster_id']}

                Severity: {spot['severity']}

                Vehicles: {spot['avg_vehicles']:.0f}

                Avg Speed: {spot['avg_speed']:.1f} km/h
                """),
        ).add_to(m)

    st.markdown("### 🗺️ Hotspot Map")

    st_folium(
        m,
        width=None,
        height=550,
        key=f"hotspot_map_{st.session_state.chat_session_id}_{len(st.session_state.chat_messages)}",
    )


def render_route_map(df, origin, destination):

    junctions = (
        df.groupby("junction_id")
        .agg({"latitude": "first", "longitude": "first"})
        .reset_index()
    )

    origin_row = junctions[junctions["junction_id"] == origin]

    destination_row = junctions[junctions["junction_id"] == destination]

    if origin_row.empty or destination_row.empty:
        st.error("Unable to find junction coordinates.")
        return

    origin_lat = origin_row.iloc[0]["latitude"]
    origin_lon = origin_row.iloc[0]["longitude"]

    dest_lat = destination_row.iloc[0]["latitude"]
    dest_lon = destination_row.iloc[0]["longitude"]

    G, route = find_osm_route(origin_lat, origin_lon, dest_lat, dest_lon)

    route_map = create_osm_route_map(G, route, origin, destination)

    st.markdown("### 🗺️ Route Map")

    st_folium(
        route_map, height=550, width=None, key=f"osm_route_{origin}_{destination}"
    )


def render_nlp_panel():
    """Render the AI chat assistant panel."""
    st.markdown("# 🚦 AI Traffic Copilot")
    st.markdown(
        "Ask about traffic, congestion reduction, route optimization, "
        "traffic management plans, anomalies, and predictions."
    )

    # Session management
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Language selector
    col1, col2 = st.columns([3, 1])
    with col2:
        lang_display = {code: get_language_name(code) for code in SUPPORTED_LANGUAGES}
        selected_lang = st.selectbox(
            "🌐 Language",
            SUPPORTED_LANGUAGES,
            format_func=lambda x: lang_display[x],
        )

    # Load data context
    df = pd.read_csv(DATASET_PATH)
    data_context = {
        "traffic_data": df,
        "total_records": len(df),
        "avg_speed": df["avg_speed"].mean(),
        "hotspots": get_hotspot_summary(df),
        "anomalies": get_anomaly_report(df),
    }

    # Display chat history
    for msg in st.session_state.chat_messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("intent") == "hotspot_query":
                render_hotspot_map(df)

            if msg["role"] == "assistant" and msg.get("intent") == "route_query":
                entities = msg.get("entities", {})

                origin = entities.get("origin")
                destination = entities.get("destination")

                if origin and destination:
                    render_route_map(df, origin, destination)

    # Chat input
    user_input = st.chat_input("Ask about traffic... (e.g., 'Traffic at HITEC City?')")

    if user_input:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = process_query(
                    user_input,
                    st.session_state.chat_session_id,
                    data_context,
                )

            response = result["response"]

            # Better rendering for AI Copilot responses
            formatted_response = response.replace("• ", "\n• ")

            st.markdown(formatted_response)

            if result["intent"] == "hotspot_query":
                render_hotspot_map(df)

            elif result["intent"] == "route_query":

                origin = result["entities"].get("origin")
                destination = result["entities"].get("destination")

                if origin and destination:
                    render_route_map(df, origin, destination)

            # Debug info
            with st.expander("🔍 Debug Info", expanded=False):
                st.json(
                    {
                        "intent": result["intent"],
                        "entities": result["entities"],
                        "detected_language": result["detected_language"],
                        "turn": result["turn_number"],
                    }
                )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response,
                "intent": result["intent"],
                "entities": result["entities"],
            }
        )

    # Sidebar: example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
    ### 🚦 Traffic Status
    - What is the traffic at HITEC City?
    - Is Ameerpet congested right now?

    ### 🎯 Congestion Reduction
    - How can congestion at HITEC City be reduced?
    - Suggest actions for Gachibowli traffic.

    ### 📋 Traffic Management Plans
    - Generate a traffic management plan for Kukatpally.
    - What should traffic police do at Ameerpet?

    ### 🧠 Explanations
    - Why is congestion expected at HITEC City?
    - Explain the traffic prediction for Kukatpally.

    ### ⚠️ Anomalies
    - Any unusual traffic patterns detected?

    ### 🔥 Hotspots
    - Which areas are most congested?
    """)

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]
        st.rerun()
