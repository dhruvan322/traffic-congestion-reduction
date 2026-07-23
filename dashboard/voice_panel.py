"""
Voice Assistant Panel.
Browser-based speech recognition and text-to-speech.
"""

import streamlit as st
import pandas as pd
import uuid

from config import DATASET_PATH
from nlp.assistant import process_query
from nlp.voice import get_voice_input_component, text_to_speech_component


def render_voice_panel():
    """Render the voice assistant panel."""
    st.markdown("# 🎤 Voice Assistant")
    st.markdown("Speak your traffic queries — powered by Web Speech API")

    # Session management
    if "voice_session_id" not in st.session_state:
        st.session_state.voice_session_id = str(uuid.uuid4())[:8]
    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []

    # Load data context
    df = pd.read_csv(DATASET_PATH)
    data_context = {"traffic_data": df}

    # ─────────── Voice Input ───────────
    st.subheader("🎙️ Voice Input")
    st.info(
        "💡 Click the microphone button below and speak your query. "
        "Make sure your browser allows microphone access."
    )

    get_voice_input_component(height=150)

    st.markdown("---")

    # ─────────── Text Fallback ───────────
    st.subheader("⌨️ Or Type Your Query")
    voice_text = st.text_input(
        "Enter your traffic query:",
        placeholder="e.g., What is the traffic at Ameerpet?",
        key="voice_text_input",
    )

    if st.button("🚀 Process Query", type="primary") and voice_text:
        with st.spinner("Processing..."):
            result = process_query(
                voice_text,
                st.session_state.voice_session_id,
                data_context,
            )

        response = result["response"]

        # Display response
        st.markdown("### 📢 Response")
        st.markdown(response)

        # Play response with TTS
        st.markdown("🔊 *Playing audio response...*")
        text_to_speech_component(response, height=0)

        # Store in history
        st.session_state.voice_history.append({
            "query": voice_text,
            "response": response,
            "intent": result["intent"],
        })

    # ─────────── History ───────────
    if st.session_state.voice_history:
        st.markdown("---")
        st.subheader("📝 Conversation History")
        for i, entry in enumerate(reversed(st.session_state.voice_history), 1):
            with st.expander(f"Turn {i}: {entry['query'][:50]}..."):
                st.markdown(f"**You:** {entry['query']}")
                st.markdown(f"**AI:** {entry['response']}")
                st.caption(f"Intent: {entry['intent']}")

    # ─────────── Browser Compatibility ───────────
    with st.expander("ℹ️ Browser Compatibility"):
        st.markdown("""
        **Supported Browsers for Voice:**
        - ✅ Google Chrome (recommended)
        - ✅ Microsoft Edge
        - ⚠️ Firefox (limited support)
        - ❌ Safari (limited Web Speech API support)
        
        **Troubleshooting:**
        - Ensure microphone permissions are enabled
        - Use HTTPS or localhost for microphone access
        - Speak clearly and at a moderate pace
        """)
