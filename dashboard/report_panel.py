"""
Report Center Panel.
Generate and view daily, weekly, and anomaly reports.
"""

import streamlit as st
import pandas as pd

from config import DATASET_PATH
from reports.report_generator import (
    generate_daily_report, generate_weekly_report, generate_anomaly_report,
)
from models.anomaly_detector import get_anomaly_report
from database import get_reports


def render_report_panel():
    """Render the report generation panel."""
    st.markdown("# 📋 Report Center")
    st.markdown("Generate and download automated traffic intelligence reports")

    df = pd.read_csv(DATASET_PATH)

    # ─────────── Report Generator ───────────
    st.subheader("📝 Generate Report")

    report_type = st.selectbox(
        "Report Type",
        ["Daily Summary", "Weekly Summary", "Anomaly Report"],
    )

    if report_type == "Daily Summary":
        # Get available dates
        dates = sorted(df["timestamp"].str[:10].unique(), reverse=True)
        selected_date = st.selectbox("Select Date", dates)

        if st.button("📊 Generate Daily Report", type="primary"):
            with st.spinner("Generating report..."):
                report = generate_daily_report(df, selected_date)

            if "error" in report:
                st.error(report["error"])
            else:
                st.success("✅ Report generated!")
                st.markdown(report["content"])
                st.download_button(
                    "📥 Download Report",
                    report["content"],
                    file_name=f"daily_report_{selected_date}.md",
                    mime="text/markdown",
                )

    elif report_type == "Weekly Summary":
        if st.button("📊 Generate Weekly Report", type="primary"):
            with st.spinner("Generating report..."):
                report = generate_weekly_report(df)

            if "error" in report:
                st.error(report["error"])
            else:
                st.success("✅ Report generated!")
                st.markdown(report["content"])
                st.download_button(
                    "📥 Download Report",
                    report["content"],
                    file_name="weekly_report.md",
                    mime="text/markdown",
                )

    elif report_type == "Anomaly Report":
        if st.button("📊 Generate Anomaly Report", type="primary"):
            with st.spinner("Running anomaly detection and generating report..."):
                anomalies = get_anomaly_report(df)
                report = generate_anomaly_report(anomalies)

            st.success(f"✅ Report generated! ({len(anomalies)} anomalies found)")
            st.markdown(report["content"])
            st.download_button(
                "📥 Download Report",
                report["content"],
                file_name="anomaly_report.md",
                mime="text/markdown",
            )

    # ─────────── Report History ───────────
    st.divider()
    st.subheader("📚 Report History")

    history = get_reports(limit=10)
    if history:
        for r in history:
            with st.expander(
                f"📄 {r['report_type'].title()} — {r['generated_at'][:16]}"
            ):
                st.markdown(r["content"])
    else:
        st.info("No reports generated yet. Use the form above to create one.")
