"""AI Traffic Assistant dashboard."""

import html
import uuid

import requests

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config import (
    API_BASE_URL,
    API_TIMEOUT_SECONDS,
    DATASET_PATH,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
    SUPPORTED_LANGUAGES,
)
from models.hotspot_detector import get_hotspot_summary
from nlp.multilingual import get_language_name


def render_hotspot_map(df):
    """Render the existing historical hotspot map."""
    summaries = get_hotspot_summary(df)

    if not summaries:
        st.info("No hotspot clusters detected.")
        return

    map_view = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=MAP_ZOOM,
        tiles="CartoDB dark_matter",
    )
    severity_colors = {
        "Critical": "#ef4444",
        "High": "#f59e0b",
        "Medium": "#00f0ff",
        "Low": "#10b981",
    }

    for spot in summaries[:5]:
        color = severity_colors.get(spot.get("severity"), "#00f0ff")
        folium.CircleMarker(
            location=[spot["center_lat"], spot["center_lon"]],
            radius=14,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=f"Cluster {spot['cluster_id']}",
            popup=(
                f"Cluster {spot['cluster_id']}<br>"
                f"Severity: {spot['severity']}<br>"
                f"Average vehicles: {spot['avg_vehicles']:.0f}<br>"
                f"Average speed: {spot['avg_speed']:.1f} km/h"
            ),
        ).add_to(map_view)

    st.markdown("### Hotspot map")
    st_folium(
        map_view,
        width=None,
        height=550,
        key=(
            f"hotspot_map_{st.session_state.chat_session_id}_"
            f"{len(st.session_state.chat_messages)}"
        ),
    )


def _render_badge(label: str, kind: str = "cyan") -> None:
    """Render a compact, static status badge."""
    st.markdown(
        f'<span class="assistant-badge assistant-badge-{kind}">'
        f"{html.escape(label)}</span>",
        unsafe_allow_html=True,
    )


def _render_stat_card(label: str, value: str, detail: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="assistant-stat-card assistant-stat-{accent}">
            <div class="assistant-stat-label">{html.escape(label)}</div>
            <div class="assistant-stat-value">{html.escape(value)}</div>
            <div class="assistant-stat-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric(value, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):,.{decimals}f}{suffix}"


def _render_conversation_message(message: dict) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        if role == "assistant":
            _render_badge("Historical analysis", "teal")
        st.markdown(message.get("content", ""))
        if role == "assistant":
            with st.expander("Response details"):
                st.json(
                    {
                        "intent": message.get("intent"),
                        "entities": message.get("entities"),
                        "detected_language": message.get("detected_language"),
                        "turn": message.get("turn_number"),
                    }
                )


def _process_submitted_query(query: str, data_context: dict) -> None:
    """Send chat questions to the FastAPI backend."""

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    try:
        with st.spinner("Analyzing historical traffic data..."):
            api_url = f"{API_BASE_URL.rstrip('/')}/api/v1/chat"

            api_response = requests.post(
                api_url,
                json={
                    "query": query,
                    "session_id": st.session_state.chat_session_id,
                },
                timeout=API_TIMEOUT_SECONDS,
            )

            api_response.raise_for_status()
            result = api_response.json()

        response = result.get(
            "response",
            "I could not generate a response.",
        )

    except requests.exceptions.ConnectionError:
        result = {
            "intent": "error",
            "entities": {},
            "detected_language": "unknown",
            "turn_number": len(st.session_state.chat_messages),
        }
        response = (
            "Unable to connect to the Traffic Assistant API. "
            "Please make sure the FastAPI server is running."
        )

    except requests.exceptions.Timeout:
        result = {
            "intent": "error",
            "entities": {},
            "detected_language": "unknown",
            "turn_number": len(st.session_state.chat_messages),
        }
        response = (
            "The assistant took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.HTTPError:
        result = {
            "intent": "error",
            "entities": {},
            "detected_language": "unknown",
            "turn_number": len(st.session_state.chat_messages),
        }
        response = (
            "The Traffic Assistant API returned an error. "
            "Please check the API server logs."
        )

    except (ValueError, requests.exceptions.RequestException):
        result = {
            "intent": "error",
            "entities": {},
            "detected_language": "unknown",
            "turn_number": len(st.session_state.chat_messages),
        }
        response = (
            "An unexpected communication error occurred. "
            "Please try again."
        )

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": response,
            "intent": result.get("intent"),
            "entities": result.get("entities", {}),
            "detected_language": result.get("detected_language"),
            "turn_number": result.get("turn_number"),
        }
    )

    st.rerun()


def render_nlp_panel():
    """Render the historical-data AI Traffic Assistant."""
    st.markdown(
        """
        <style>
        .assistant-badge {
            display: inline-block;
            border: 1px solid rgba(0, 240, 255, .24);
            border-radius: 999px;
            padding: .25rem .65rem;
            margin: 0 .35rem .65rem 0;
            color: #b6f8ff;
            background: rgba(0, 240, 255, .08);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .assistant-badge-teal {
            color: #a7f3df;
            border-color: rgba(0, 229, 188, .25);
            background: rgba(0, 229, 188, .08);
        }
        .assistant-badge-muted {
            color: #cbd5e1;
            border-color: rgba(148, 163, 184, .22);
            background: rgba(148, 163, 184, .08);
        }
        .assistant-stat-card {
            box-sizing: border-box;
            min-height: 122px;
            height: 100%;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(0, 240, 255, .15);
            border-top: 3px solid #00f0ff;
            border-radius: 10px;
            background: #131c2e;
        }
        .assistant-stat-teal { border-top-color: #00e5bc; }
        .assistant-stat-blue { border-top-color: #3b82f6; }
        .assistant-stat-label {
            color: #94a3b8;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .07em;
            text-transform: uppercase;
        }
        .assistant-stat-value {
            margin: .45rem 0 .25rem;
            color: #f8fafc;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .assistant-stat-detail {
            color: #64748b;
            font-size: .78rem;
        }
        .assistant-empty-state {
            padding: 1.5rem .75rem;
            color: #94a3b8;
            text-align: center;
        }
        .assistant-context-label {
            color: #94a3b8;
            font-size: .78rem;
        }
        .assistant-context-value {
            color: #f8fafc;
            font-size: .9rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# AI Traffic Assistant")
    st.markdown("Ask questions and explore traffic intelligence insights.")
    _render_badge("Historical dataset")
    _render_badge("Rule-based | Available", "teal")
    st.caption(
        "Rule-based intent and response orchestration integrates the project's "
        "prediction, hotspot, anomaly, routing, and recommendation modules."
    )

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

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
        except Exception as exc:
            st.error("Unable to load the project traffic dataset.")
            st.exception(exc)
            return

    data_context = st.session_state.nlp_data_context
    df = data_context["traffic_data"]
    total_records = len(df)
    area_count = int(df["area"].nunique()) if "area" in df.columns else 0
    segment_count = (
        int(df["road_segment_id"].nunique())
        if "road_segment_id" in df.columns
        else 0
    )
    latest_timestamp = (
        df["timestamp"].max()
        if "timestamp" in df.columns and not df.empty
        else pd.NaT
    )
    latest_label = (
        latest_timestamp.strftime("%b %d, %Y %H:%M")
        if pd.notna(latest_timestamp)
        else "Not available"
    )

    stat_columns = st.columns(3)
    stat_cards = [
        ("Archived observations", f"{total_records:,}", "Historical records", "cyan"),
        ("Project areas", f"{area_count:,}", "Unique dataset areas", "teal"),
        ("Road segments", f"{segment_count:,}", "Unique segment IDs", "blue"),
    ]
    for column, (label, value, detail, accent) in zip(stat_columns, stat_cards):
        with column:
            _render_stat_card(label, value, detail, accent)

    st.write("")
    conversation_column, context_column = st.columns([1.8, 1])
    submitted_query = None

    with conversation_column:
        with st.container(border=True):
            title_column, thread_column = st.columns([3, 1])
            with title_column:
                st.markdown("### Conversation")
                st.caption(
                    "Ask about archived observations, predictions, routes, "
                    "hotspots, anomalies, or traffic recommendations."
                )
            with thread_column:
                _render_badge(
                    "Current thread"
                    if st.session_state.chat_messages
                    else "New thread",
                    "muted",
                )

            if st.session_state.chat_messages:
                for message in st.session_state.chat_messages:
                    _render_conversation_message(message)
            else:
                st.markdown(
                    """
                    <div class="assistant-empty-state">
                        <strong>Your traffic analysis starts here</strong><br>
                        Ask a question or choose a suggested prompt below.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Suggested prompts")
            prompts = [
                "Show the archive's top hotspot clusters.",
                "Explain the Hitech City prediction.",
                "Summarize historical anomalies.",
                "Find a route from Ameerpet to Gachibowli.",
            ]
            prompt_columns = st.columns(2)
            for index, (column, prompt) in enumerate(
                zip(prompt_columns * 2, prompts)
            ):
                with column:
                    if st.button(
                        prompt,
                        key=f"nlp_suggested_prompt_{index}",
                    ):
                        submitted_query = prompt

            with st.form("nlp_chat_form", clear_on_submit=True):
                input_column, send_column = st.columns([5, 1])
                with input_column:
                    typed_query = st.text_input(
                        "Your question",
                        placeholder="Ask about historical Hyderabad traffic...",
                        label_visibility="collapsed",
                    )
                with send_column:
                    send_clicked = st.form_submit_button(
                        "Send",
                        type="primary",
                    )
                if send_clicked and typed_query.strip():
                    submitted_query = typed_query.strip()

    with context_column:
        with st.container(border=True):
            st.markdown("### Traffic context")
            _render_badge("Historical synthetic data", "teal")
            st.caption("Historical synthetic observations | No live feed")
            st.markdown(
                f'<div class="assistant-context-label">Total observations</div>'
                f'<div class="assistant-context-value">{total_records:,}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="assistant-context-label">Project areas</div>'
                f'<div class="assistant-context-value">{area_count:,}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="assistant-context-label">Road segments</div>'
                f'<div class="assistant-context-value">{segment_count:,}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="assistant-context-label">Latest dataset timestamp</div>'
                f'<div class="assistant-context-value">{html.escape(latest_label)}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("### Top listed hotspot cluster")
            if "nlp_hotspot_summaries" not in st.session_state:
                try:
                    st.session_state.nlp_hotspot_summaries = (
                        get_hotspot_summary(df)
                    )
                    st.session_state.nlp_hotspot_error = None
                except Exception as exc:
                    st.session_state.nlp_hotspot_summaries = []
                    st.session_state.nlp_hotspot_error = str(exc)

            hotspot_summaries = st.session_state.nlp_hotspot_summaries
            if hotspot_summaries:
                hotspot = hotspot_summaries[0]
                st.markdown(
                    f"**Cluster {html.escape(str(hotspot.get('cluster_id', 'Not available')))}**"
                )
                st.caption(
                    f"Severity: {hotspot.get('severity', 'Not available')}"
                )
                road_segments = hotspot.get("road_segments")
                road_segment_count = hotspot.get(
                    "size",
                    len(road_segments) if road_segments is not None else None,
                )
                congestion_rate = hotspot.get("congestion_rate")
                high_severe_rate = hotspot.get("high_or_severe_rate")
                congestion_percent = (
                    float(congestion_rate) * 100
                    if congestion_rate is not None
                    else None
                )
                high_severe_percent = (
                    float(high_severe_rate) * 100
                    if high_severe_rate is not None
                    else None
                )
                st.markdown(
                    f"- Road segments: {_format_metric(road_segment_count, decimals=0)}\n"
                    f"- Congestion rate: {_format_metric(congestion_percent, '%')}\n"
                    f"- High/severe share: {_format_metric(high_severe_percent, '%')}\n"
                    f"- Average vehicles: {_format_metric(hotspot.get('avg_vehicles'))}\n"
                    f"- Average speed: {_format_metric(hotspot.get('avg_speed'), ' km/h')}"
                )
                peak_hours = hotspot.get("peak_hours")
                peak_hours_text = (
                    ", ".join(str(hour) for hour in peak_hours)
                    if peak_hours
                    else "Not available"
                )
                st.caption(f"Peak hours: {peak_hours_text}")
            elif st.session_state.get("nlp_hotspot_error"):
                st.warning(
                    "Hotspot information is unavailable because the detector "
                    "could not complete."
                )
                st.caption(st.session_state.nlp_hotspot_error)
            else:
                st.info("No hotspot cluster is available for this dataset.")

        with st.container(border=True):
            st.markdown("### Supported query types")
            st.markdown(
                "- Traffic status by area\n"
                "- Predictions and explanations\n"
                "- Hotspot and anomaly summaries\n"
                "- Routes and management plans"
            )

        with st.container(border=True):
            st.markdown("### Languages")
            language_names = [
                get_language_name(code) for code in SUPPORTED_LANGUAGES
            ]
            st.markdown(" | ".join(language_names))
            st.selectbox(
                "Language preference (not applied to responses)",
                options=list(SUPPORTED_LANGUAGES),
                format_func=get_language_name,
                key="nlp_language",
            )
            st.caption(
                "Assistant responses use automatic language detection; "
                "this selector does not set the response language."
            )

    st.caption("Historical synthetic observations only | No live traffic feed.")

    if submitted_query:
        _process_submitted_query(submitted_query, data_context)

    if st.session_state.chat_messages and st.button(
        "Clear conversation",
        key="nlp_clear_chat",
    ):
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]
        st.rerun()
