"""
Anomaly Detection Panel.
Interactive anomaly detection with scatter plots and risk analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import DATASET_PATH
from models.anomaly_detector import detect_anomalies, zscore_anomalies, get_anomaly_report
from database import insert_anomaly
from engine.reduction_engine import generate_anomaly_response


def render_anomaly_panel():
    """Render the anomaly detection panel."""
    st.markdown("# ⚠️ Anomaly Detection")
    st.markdown(
        "Detect unusual traffic patterns using **Isolation Forest** "
        "and **Z-score** analysis"
    )

    df = pd.read_csv(DATASET_PATH)

    # ─────────── Run Detection ───────────
    if st.button("🔍 Run Anomaly Detection", type="primary", use_container_width=True):
        st.session_state["anomaly_run"] = True

    if st.session_state.get("anomaly_run", False):
        with st.spinner("Running Isolation Forest + Z-score analysis..."):
            df_detected = detect_anomalies(df)
            df_detected = zscore_anomalies(df_detected)
            anomaly_records = get_anomaly_report(df)

        total_anomalies = df_detected["is_anomaly"].sum()
        zscore_count = df_detected["zscore_anomaly"].sum()

        # ─────────── Summary KPIs ───────────
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔬 Total Records", f"{len(df):,}")
        with col2:
            st.metric("⚠️ Isolation Forest", f"{total_anomalies}")
        with col3:
            st.metric("📊 Z-Score", f"{zscore_count}")
        with col4:
            pct = total_anomalies / len(df) * 100
            st.metric("📈 Anomaly Rate", f"{pct:.1f}%")

        # ─────────── Scatter Plot ───────────
        st.subheader("🔬 Vehicle Count vs Speed — Anomaly View")

        df_plot = df_detected.copy()
        df_plot["anomaly_label"] = df_plot["is_anomaly"].map({0: "Normal", 1: "Anomaly"})

        fig1 = px.scatter(
            df_plot,
            x="vehicle_count",
            y="avg_speed",
            color="anomaly_label",
            color_discrete_map={"Normal": "#4CAF50", "Anomaly": "#f44336"},
            opacity=0.6,
            hover_data=["junction_id", "weather", "hour", "congestion_level"],
        )
        fig1.update_layout(
            xaxis_title="Vehicle Count",
            yaxis_title="Avg Speed (km/h)",
            height=450,
            template="plotly_dark",
            legend_title="Status",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ─────────── Risk Distribution ───────────
        if anomaly_records:
            st.subheader("🎯 Risk Level Distribution")

            col1, col2 = st.columns(2)

            with col1:
                risk_counts = {}
                for a in anomaly_records:
                    r = a["risk_level"]
                    risk_counts[r] = risk_counts.get(r, 0) + 1

                fig2 = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    color=list(risk_counts.keys()),
                    color_discrete_map={
                        "Critical": "#d32f2f", "High": "#f44336",
                        "Medium": "#FF9800", "Low": "#4CAF50"
                    },
                    hole=0.35,
                )
                fig2.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                # Anomalies by junction
                junc_counts = {}
                for a in anomaly_records:
                    j = a["junction_id"]
                    junc_counts[j] = junc_counts.get(j, 0) + 1

                top_junctions = sorted(junc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                fig3 = px.bar(
                    x=[j[1] for j in top_junctions],
                    y=[j[0] for j in top_junctions],
                    orientation="h",
                    color=[j[1] for j in top_junctions],
                    color_continuous_scale="Reds",
                )
                fig3.update_layout(
                    xaxis_title="Anomaly Count",
                    yaxis_title="Junction",
                    height=300,
                    template="plotly_dark",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

            # ─────────── Anomaly Table ───────────
            st.subheader("📋 Anomaly Records")
            anom_df = pd.DataFrame(anomaly_records[:50])
            display_cols = [
                "junction_id", "risk_level", "vehicle_count",
                "avg_speed", "anomaly_score", "description",
            ]
            available = [c for c in display_cols if c in anom_df.columns]
            st.dataframe(anom_df[available], use_container_width=True, hide_index=True)

            # ─────────── AI Response Recommendations ───────────

            st.markdown("---")
            st.subheader("🚨 AI Response Recommendations")

            top_anomalies = anomaly_records[:5]

            for idx, anomaly in enumerate(top_anomalies, start=1):

                responses = generate_anomaly_response(anomaly)

                title = (
                    f"Anomaly #{idx} — "
                    f"{anomaly['junction_id']} "
                    f"({anomaly['risk_level']} Risk)"
                )

                with st.expander(title, expanded=(idx == 1)):

                    st.markdown(
                        f"**Description:** "
                        f"{anomaly.get('description', 'N/A')}"
                    )

                    st.markdown("---")

                    for response in responses:

                        priority_icon = {
                            "Critical": "🔴",
                            "High": "🟠",
                            "Medium": "🟡",
                            "Low": "🟢"
                        }.get(response["priority"], "⚪")

                        st.markdown(
                            f"{priority_icon} "
                            f"**{response['action']}**"
                        )

                        st.markdown(
                            f"- Expected Impact Reduction: "
                            f"{response['expected_impact_reduction']}%"
                        )

                        st.markdown(
                            f"- Why: "
                            f"{response['explanation']}"
                        )

                        st.markdown("")

            # Store top anomalies in DB
            for a in anomaly_records[:20]:
                try:
                    insert_anomaly(a)
                except Exception:
                    pass

        # ─────────── Anomaly Score Distribution ───────────
        st.subheader("📊 Anomaly Score Distribution")
        anomalous = df_detected[df_detected["is_anomaly"] == 1]
        if not anomalous.empty:
            fig4 = px.histogram(
                anomalous, x="anomaly_score", nbins=30,
                color_discrete_sequence=["#f44336"],
            )
            fig4.update_layout(
                xaxis_title="Anomaly Score (higher = more anomalous)",
                yaxis_title="Count",
                height=300,
                template="plotly_dark",
            )
            st.plotly_chart(fig4, use_container_width=True)
