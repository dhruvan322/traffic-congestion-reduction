"""
🚦 AI Traffic Intelligence Platform
Main Streamlit application entry point.
Handles navigation, initialization, and error isolation.
"""

import streamlit as st
import os
import sys
import logging
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard.command_center import render_command_center
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, DATASET_PATH, MODEL_DIR
from database import init_database, get_alert_count, log_error_to_db

# ──────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# ──────────────────────────── Error Isolation ────────────────────────────
def safe_render(panel_name: str, render_fn, **kwargs):
    """Safely render a dashboard panel with error isolation."""
    try:
        render_fn(**kwargs)
    except Exception as e:
        logger.error(f"Panel '{panel_name}' error: {e}", exc_info=True)
        st.error(f"⚠️ **{panel_name}** encountered an error: `{e}`")
        st.info(
            "💡 Other modules continue to function normally. "
            "Try refreshing the page or check the logs."
        )
        log_error_to_db(panel_name, str(e))


# ──────────────────────────── Initialization ────────────────────────────
@st.cache_resource
def initialize_app():
    """One-time initialization: DB, dataset check, model loading."""
    init_database()
    logger.info("Database initialized.")

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.warning(
            f"Dataset not found at {DATASET_PATH}. "
            "Run 'python data/generate_dataset.py' first."
        )
        return {"dataset_ready": False, "models_ready": False}

    # Check if models exist
    models_ready = (
        any(f.endswith(".joblib") for f in os.listdir(MODEL_DIR))
        if os.path.exists(MODEL_DIR)
        else False
    )

    return {"dataset_ready": True, "models_ready": models_ready}


@st.cache_data
def load_dataset():
    """Load the traffic dataset with caching."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


# ──────────────────────────── Sidebar ────────────────────────────
def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='font-size: 1.8rem; margin: 0;'>🚦</h1>
                <h3 style='margin: 0.2rem 0;'>Traffic Intelligence</h3>
                <p style='color: #888; font-size: 0.8rem; margin: 0;'>
                    AI-Powered Platform
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            options=[
                "📊 Overview",
                "🔮 Predictions",
                "🔥 Hotspots",
                "🗺️ Route Optimizer",
                "💬 AI Chat Assistant",
                "🚨 Command Center",
                "🎤 Voice Assistant",
                "📋 Reports",
                "⚠️ Anomaly Detection",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Alert badge
        alert_count = get_alert_count()
        if alert_count > 0:
            st.warning(f"🔔 **{alert_count}** active alerts")

        # System status
        app_state = initialize_app()
        st.markdown("#### System Status")
        col1, col2 = st.columns(2)
        with col1:
            if app_state["dataset_ready"]:
                st.success("Data ✓", icon="📦")
            else:
                st.error("No Data", icon="📦")
        with col2:
            if app_state["models_ready"]:
                st.success("Models ✓", icon="🧠")
            else:
                st.warning("No Models", icon="🧠")

        st.divider()
        st.caption("© 2026 Traffic Intelligence Platform")

    return page


# ──────────────────────────── Page Routing ────────────────────────────
def route_to_page(page: str, data_context):
    """Route to the selected dashboard page."""
    if page == "📊 Overview":
        from dashboard.overview import render_overview

        safe_render("Overview Dashboard", render_overview)

    elif page == "🔮 Predictions":
        from dashboard.prediction_panel import render_prediction_panel

        safe_render("Prediction Engine", render_prediction_panel)

    elif page == "🔥 Hotspots":
        from dashboard.hotspot_panel import render_hotspot_panel

        safe_render("Hotspot Detection", render_hotspot_panel)

    elif page == "🗺️ Route Optimizer":
        from dashboard.route_panel import render_route_panel

        safe_render("Route Optimizer", render_route_panel)

    elif page == "💬 AI Chat Assistant":
        from dashboard.nlp_panel import render_nlp_panel

        safe_render("AI Chat Assistant", render_nlp_panel)

    elif page == "🎤 Voice Assistant":
        from dashboard.voice_panel import render_voice_panel

        safe_render("Voice Assistant", render_voice_panel)

    elif page == "📋 Reports":
        from dashboard.report_panel import render_report_panel

        safe_render("Report Center", render_report_panel)

    elif page == "⚠️ Anomaly Detection":
        from dashboard.anomaly_panel import render_anomaly_panel

        safe_render("Anomaly Detection", render_anomaly_panel)

    elif page == "🚨 Command Center":
        render_command_center(data_context)


# ──────────────────────────── Main ────────────────────────────
def main():
    """Main application entry point."""
    # Initialize
    app_state = initialize_app()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Check if dataset exists
    if not app_state["dataset_ready"]:
        st.markdown("# 🚦 AI Traffic Intelligence Platform")
        st.warning(
            "⚠️ **Dataset not found.** Please generate the traffic dataset first:\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "python data/generate_dataset.py\n"
            "```"
        )
        st.info(
            "After generating the dataset, refresh this page. "
            "The ML models will be trained automatically on first use."
        )
        return
    # Load traffic dataset
    df = pd.read_csv(DATASET_PATH)

    data_context = {
        "traffic_data": df,
    }
    # Route to selected page
    route_to_page(page, data_context)


if __name__ == "__main__":
    main()
