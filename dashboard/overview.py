"""
Overview Dashboard Panel.
Displays KPI cards, congestion charts, and alerts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import DATASET_PATH
from database import get_traffic_stats, get_active_alerts
from engine.reduction_engine import (
    get_reduction_kpis,
    generate_proactive_alerts,
)


def render_overview():
    """Render the main overview dashboard."""
    st.markdown("# 📊 Traffic Intelligence Overview")

    # Load data
    df = pd.read_csv(DATASET_PATH)

    # ─────────── KPI Cards ───────────
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    avg_vehicles = df["vehicle_count"].mean()
    avg_speed = df["avg_speed"].mean()
    high_pct = (df["congestion_level"] == "High").mean() * 100

    with col1:
        st.metric("📦 Total Records", f"{total:,}")
    with col2:
        st.metric("🚗 Avg Vehicles", f"{avg_vehicles:.0f}")
    with col3:
        st.metric("⚡ Avg Speed", f"{avg_speed:.1f} km/h")
    with col4:
        st.metric("🔴 High Congestion", f"{high_pct:.1f}%")

    st.divider()

    # ─────────── AI Reduction KPIs ───────────

    kpis = get_reduction_kpis(df)

    st.subheader("🎯 Estimated Congestion Reduction Impact")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Delay Reduction",
            f"{kpis['estimated_delay_reduction_min']} min"
        )

    with k2:
        st.metric(
            "Congestion Reduction",
            f"{kpis['congestion_reduction_pct']}%"
        )

    with k3:
        st.metric(
            "Time Saved",
            f"{kpis['time_saved_hours']} hrs"
        )

    with k4:
        st.metric(
            "Vehicles Diverted",
            f"{kpis['vehicles_diverted']}"
        )

    st.divider()

    # ─────────── Charts Row 1 ───────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🕐 Congestion by Hour")
        hourly = df.groupby(["hour", "congestion_level"]).size().reset_index(name="count")
        fig1 = px.bar(
            hourly, x="hour", y="count", color="congestion_level",
            color_discrete_map={"Low": "#4CAF50", "Medium": "#FF9800", "High": "#f44336"},
            barmode="stack",
        )
        fig1.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Records",
            legend_title="Congestion", height=350,
            template="plotly_dark",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("📍 Top Junctions by Traffic")
        junction_avg = df.groupby("junction_id")["vehicle_count"].mean().sort_values(ascending=True).tail(10)
        fig2 = px.bar(
            x=junction_avg.values, y=junction_avg.index,
            orientation="h",
            color=junction_avg.values,
            color_continuous_scale="RdYlGn_r",
        )
        fig2.update_layout(
            xaxis_title="Avg Vehicle Count", yaxis_title="Junction",
            height=350, template="plotly_dark",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─────────── Charts Row 2 ───────────
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("🌤️ Weather Impact")
        weather_stats = df.groupby("weather").agg({
            "vehicle_count": "mean",
            "avg_speed": "mean"
        }).reset_index()
        fig3 = px.bar(
            weather_stats, x="weather", y="vehicle_count",
            color="avg_speed",
            color_continuous_scale="RdYlGn",
            text_auto=".0f",
        )
        fig3.update_layout(
            xaxis_title="Weather", yaxis_title="Avg Vehicles",
            height=300, template="plotly_dark",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        st.subheader("📈 Congestion Distribution")
        cong_counts = df["congestion_level"].value_counts()
        fig4 = px.pie(
            values=cong_counts.values,
            names=cong_counts.index,
            color=cong_counts.index,
            color_discrete_map={"Low": "#4CAF50", "Medium": "#FF9800", "High": "#f44336"},
            hole=0.4,
        )
        fig4.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    # ─────────── Speed vs Vehicles Scatter ───────────
    st.subheader("🔬 Vehicle Count vs Speed")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig5 = px.scatter(
        sample, x="vehicle_count", y="avg_speed",
        color="congestion_level",
        color_discrete_map={"Low": "#4CAF50", "Medium": "#FF9800", "High": "#f44336"},
        opacity=0.6, hover_data=["junction_id", "weather", "hour"],
    )
    fig5.update_layout(
        xaxis_title="Vehicle Count", yaxis_title="Avg Speed (km/h)",
        height=400, template="plotly_dark",
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ─────────── Active Alerts ───────────
    alerts = get_active_alerts(limit=5)
    if alerts:
        st.subheader("🔔 Recent Alerts")
        for a in alerts:
            severity_icon = {"Critical": "🔴", "High": "🟠", "Warning": "🟡"}.get(a["severity"], "🔵")
            st.warning(f"{severity_icon} **[{a['alert_type']}]** {a['message']}")

    # ─────────── Proactive AI Alerts ───────────

    st.markdown("---")
    st.subheader("🧠 AI Congestion Prevention Alerts")

    ai_alerts = generate_proactive_alerts(df, top_n=5)

    if ai_alerts:

        for alert in ai_alerts:

            with st.expander(
                f"🚨 {alert['junction_name']} "
                f"(Confidence {alert['confidence']:.0%})"
            ):

                st.markdown(alert["message"])

                st.markdown(
                    f"**Estimated Reduction:** "
                    f"{alert['estimated_reduction_pct']}%"
                )

                st.markdown("### Recommended Actions")

                for rec in alert["recommendations"]:

                    st.markdown(
                        f"- {rec['action']} "
                        f"(~{rec['expected_reduction']}%)"
                    )
