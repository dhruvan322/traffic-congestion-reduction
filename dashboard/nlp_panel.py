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

    st.markdown("### ðŸ—ºï¸ Hotspot Map")

    st_folium(
        m,
        width=None,
        height=550,
        key=f"hotspot_map_{st.session_state.chat_session_id}_{len(st.session_state.chat_messages)}",
    )


def render_nlp_panel():
    """Render the AI chat assistant panel."""

    st.markdown("# 🚦 AI Traffic Copilot")
    st.markdown(
        "Ask about traffic, congestion reduction, route optimization, "
        "traffic management plans, anomalies, predictions, and routes."
    )

    # -------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # -------------------------------------------------------------
    # Load project traffic context once per Streamlit session
    # -------------------------------------------------------------

    if "nlp_data_context" not in st.session_state:

        try:
            df = pd.read_csv(DATASET_PATH)

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

            st.session_state.nlp_data_context = {
                "traffic_data": df,
                "total_records": len(df),
            }

        except Exception as e:
            st.error(
                "Unable to load the project traffic dataset."
            )
            st.exception(e)
            return

    data_context = st.session_state.nlp_data_context

    # -------------------------------------------------------------
    # Language selector / interface information
    # -------------------------------------------------------------

    col1, col2 = st.columns([3, 1])

    with col2:

        lang_display = {
            code: get_language_name(code)
            for code in SUPPORTED_LANGUAGES
        }

        st.selectbox(
            "🌐 Language",
            options=list(lang_display.keys()),
            format_func=lambda code: lang_display[code],
            key="nlp_language",
        )

    # -------------------------------------------------------------
    # Display existing conversation
    # -------------------------------------------------------------

    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                with st.expander("🔍 Debug Info"):

                    st.json(
                        {
                            "intent": message.get("intent"),
                            "entities": message.get("entities"),
                            "detected_language": message.get(
                                "detected_language"
                            ),
                            "turn": message.get(
                                "turn_number"
                            ),
                        }
                    )

    # -------------------------------------------------------------
    # Chat input
    # -------------------------------------------------------------

    query = st.chat_input(
        "Ask about Hyderabad traffic..."
    )

    if query:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        try:

            result = process_query(
                query=query,
                session_id=st.session_state.chat_session_id,
                data_context=data_context,
            )

            response = result.get(
                "response",
                "I could not generate a response.",
            )

        except Exception as e:

            result = {
                "intent": "error",
                "entities": {},
                "detected_language": "unknown",
                "turn_number": len(
                    st.session_state.chat_messages
                ),
            }

            response = (
                "⚠️ An error occurred while processing "
                "your request.\n\n"
                f"Error: {str(e)}"
            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response,
                "intent": result.get("intent"),
                "entities": result.get("entities", {}),
                "detected_language": result.get(
                    "detected_language"
                ),
                "turn_number": result.get(
                    "turn_number"
                ),
            }
        )

        st.rerun()

    # -------------------------------------------------------------
    # Example queries
    # -------------------------------------------------------------

    with st.expander("💡 Example Queries"):

        st.markdown(
            """
### 🚦 Traffic Status
- What is the traffic at HITEC City?
- Is Ameerpet congested right now?

### 🎯 Congestion Reduction
- How can congestion at HITEC City be reduced?
- Suggest actions for Gachibowli traffic.

### 📋 Traffic Management
- Generate a traffic management plan for Kukatpally.
- What should traffic police do at Ameerpet?

### 🧠 Explanations
- Why is congestion expected at HITEC City?
- Explain the traffic prediction for Kukatpally.

### ⚠️ Anomalies
- Any unusual traffic patterns detected?

### 🔥 Hotspots
- Which areas are most congested?

### 🗺️ Routes
- Best route from Ameerpet to Gachibowli.
- Route from HITEC City to Kukatpally.
"""
        )

    # -------------------------------------------------------------
    # Clear chat
    # -------------------------------------------------------------

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_messages = []
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]

        st.rerun()

