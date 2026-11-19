"""
Reports and analytics panel for historical synthetic traffic data.
"""

import json
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DATASET_PATH
from dashboard.ui_theme import (
    CONGESTION_COLOR_MAP,
    COLOR_CYAN,
    COLOR_TEAL,
    apply_plotly_theme,
    render_header,
    render_kpi_card,
)
from database import get_reports
from models.anomaly_detector import get_anomaly_report
from reports.report_generator import (
    generate_anomaly_report,
    generate_daily_report,
    generate_weekly_report,
)

REPORT_TYPES = ["Daily Summary", "Weekly Summary", "Anomaly Report"]
CONGESTION_CLASSES = ["Free Flow", "Low", "Moderate", "High", "Severe"]
REPORT_SESSION_KEY = "reports_panel_selected_report"
OPEN_REPORT_SESSION_KEY = "reports_panel_open_report_id"


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Provide display aliases while leaving report-generator inputs intact."""
    data = df.copy()
    aliases = {
        "average_speed_kmh": "avg_speed",
        "future_congestion_level": "congestion_level",
        "road_segment_id": "junction_id",
        "weather_condition": "weather",
    }
    for project_name, legacy_name in aliases.items():
        if project_name not in data.columns and legacy_name in data.columns:
            data[project_name] = data[legacy_name]

    data["_report_timestamp"] = pd.to_datetime(
        data.get("timestamp", pd.Series(index=data.index, dtype="object")),
        errors="coerce",
    )
    data["_report_date"] = data["_report_timestamp"].dt.strftime("%Y-%m-%d")
    for column in ("vehicle_count", "average_speed_kmh", "hour"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _report_kind(report: dict) -> str:
    """Map stored report labels and older type aliases to a stable kind."""
    value = str(report.get("report_type", "")).strip().lower()
    if "daily" in value:
        return "daily"
    if "weekly" in value:
        return "weekly"
    if "anomal" in value:
        return "anomaly"
    return "unknown"


def _report_label(report: dict) -> str:
    kind = _report_kind(report)
    labels = {
        "daily": "Daily Summary",
        "weekly": "Weekly Summary",
        "anomaly": "Anomaly Report",
    }
    return labels.get(kind, str(report.get("report_type") or "Archived Report"))


def _safe_content(report: dict) -> str:
    """Render existing markdown and older JSON-like content without assumptions."""
    content = report.get("content", "")
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2, ensure_ascii=False)
    elif content is None:
        content = ""
    else:
        content = str(content)

    return re.sub(
        r"(Traffic Report)\s*\?\s*",
        r"\1 — ",
        content,
        flags=re.IGNORECASE,
    )


def _format_period(report: dict) -> str:
    start = str(report.get("period_start") or "").strip()
    end = str(report.get("period_end") or "").strip()
    if start and end and start != end:
        return f"{start} – {end}"
    return start or end or "Period not recorded"


def _format_generated(value) -> str:
    if not value:
        return "Unknown"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)


def _history_id(report: dict, index: int) -> str:
    value = report.get("id")
    if value is not None:
        return str(value)
    return f"{report.get('report_type', 'report')}:{report.get('generated_at', index)}"


def _scope_data(data: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Select the historical dataset rows represented by a saved report."""
    kind = _report_kind(report)
    if kind == "anomaly":
        return data

    start = str(report.get("period_start") or "").strip()
    end = str(report.get("period_end") or start).strip()
    if not start or not end:
        return data.iloc[0:0].copy()

    selected = data.loc[
        data["_report_date"].between(start, end, inclusive="both")
    ]
    return selected.copy()


def _report_metrics(data: pd.DataFrame) -> dict | None:
    required = {
        "_report_date",
        "vehicle_count",
        "average_speed_kmh",
        "future_congestion_level",
        "road_segment_id",
    }
    if data.empty or not required.issubset(data.columns):
        return None

    valid_hours = data.dropna(subset=["hour"]) if "hour" in data.columns else None
    peak_hour = (
        int(valid_hours.groupby("hour")["vehicle_count"].mean().idxmax())
        if valid_hours is not None and not valid_hours.empty
        else None
    )
    congestion = data["future_congestion_level"].fillna("Unknown").astype(str)
    distribution = congestion.value_counts().reindex(
        CONGESTION_CLASSES,
        fill_value=0,
    )
    total = len(data)

    segments = (
        data.groupby("road_segment_id")
        .agg(
            average_vehicles=("vehicle_count", "mean"),
            average_speed=("average_speed_kmh", "mean"),
        )
        .sort_values("average_vehicles", ascending=False)
        .head(5)
        .reset_index()
    )

    if "weather_condition" in data.columns:
        weather = (
            data.groupby("weather_condition", dropna=False)["vehicle_count"]
            .mean()
            .sort_values(ascending=False)
            .rename_axis("Weather")
            .reset_index(name="Average vehicles")
        )
        weather["Weather"] = weather["Weather"].fillna("Unknown").astype(str)
    else:
        weather = pd.DataFrame(columns=["Weather", "Average vehicles"])

    return {
        "total": total,
        "average_vehicles": float(data["vehicle_count"].mean()),
        "average_speed": float(data["average_speed_kmh"].mean()),
        "peak_hour": peak_hour,
        "distribution": distribution,
        "distribution_percent": distribution / total * 100,
        "segments": segments,
        "weather": weather,
    }


def _render_congestion_distribution(metrics: dict) -> None:
    distribution = pd.DataFrame(
        {
            "Congestion": CONGESTION_CLASSES,
            "Records": [int(metrics["distribution"][item]) for item in CONGESTION_CLASSES],
            "Percentage": [
                float(metrics["distribution_percent"][item])
                for item in CONGESTION_CLASSES
            ],
        }
    )
    distribution["Label"] = distribution.apply(
        lambda row: f"{int(row['Records']):,} · {row['Percentage']:.1f}%",
        axis=1,
    )
    chart = px.bar(
        distribution,
        x="Records",
        y="Congestion",
        orientation="h",
        color="Congestion",
        text="Label",
        color_discrete_map=CONGESTION_COLOR_MAP,
        category_orders={"Congestion": list(reversed(CONGESTION_CLASSES))},
        custom_data=["Percentage"],
    )
    chart.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,} records (%{customdata[0]:.1f}%)<extra></extra>",
    )
    chart.update_layout(
        height=300,
        xaxis_title="Records",
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=12, r=38, t=12, b=36),
    )
    st.plotly_chart(apply_plotly_theme(chart), use_container_width=True)


def _render_report_details(report: dict, data: pd.DataFrame) -> None:
    content = _safe_content(report)
    metrics = _report_metrics(_scope_data(data, report))

    st.markdown("#### Report summary")
    if metrics is not None:
        columns = st.columns(4)
        values = (
            ("TOTAL RECORDS", f"{metrics['total']:,}", "Records in report scope", "cyan"),
            (
                "AVERAGE VEHICLES",
                f"{metrics['average_vehicles']:.1f}",
                "Per observation",
                "teal",
            ),
            (
                "AVERAGE SPEED",
                f"{metrics['average_speed']:.1f} km/h",
                "Per observation",
                "emerald",
            ),
            (
                "PEAK HOUR",
                f"{metrics['peak_hour']:02d}:00" if metrics["peak_hour"] is not None else "N/A",
                "Highest average vehicle count",
                "amber",
            ),
        )
        for column, (title, value, subtitle, status) in zip(columns, values):
            with column:
                render_kpi_card(
                    title=title,
                    value=value,
                    subtitle=subtitle,
                    status=status,
                )

        congestion_col, segments_col, weather_col = st.columns(3)
        with congestion_col:
            with st.container(border=True):
                st.markdown("#### Congestion distribution")
                _render_congestion_distribution(metrics)
        with segments_col:
            with st.container(border=True):
                st.markdown("#### Top 5 busiest road segments")
                if metrics["segments"].empty:
                    st.info("Road-segment data is unavailable.")
                else:
                    segments = metrics["segments"].copy()
                    segments.insert(0, "Rank", range(1, len(segments) + 1))
                    segments["average_vehicles"] = segments[
                        "average_vehicles"
                    ].round(1)
                    segments["average_speed"] = segments["average_speed"].round(1)
                    st.dataframe(
                        segments.rename(
                            columns={
                                "road_segment_id": "Road segment",
                                "average_vehicles": "Avg vehicles",
                                "average_speed": "Avg speed (km/h)",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        with weather_col:
            with st.container(border=True):
                st.markdown("#### Weather summary")
                if metrics["weather"].empty:
                    st.info("Weather data is unavailable in this report.")
                else:
                    weather_chart = px.bar(
                        metrics["weather"],
                        x="Average vehicles",
                        y="Weather",
                        orientation="h",
                        text="Average vehicles",
                        color_discrete_sequence=[COLOR_CYAN],
                    )
                    weather_chart.update_traces(
                        texttemplate="%{x:.1f}",
                        textposition="outside",
                        cliponaxis=False,
                    )
                    weather_chart.update_layout(
                        height=300,
                        xaxis_title="Average vehicle count",
                        yaxis_title=None,
                        showlegend=False,
                        margin=dict(l=12, r=28, t=12, b=36),
                    )
                    st.plotly_chart(
                        apply_plotly_theme(weather_chart),
                        use_container_width=True,
                    )
    else:
        st.info(
            "This archived report does not include a usable date range or "
            "compatible scope metadata for reconstructing summary analytics."
        )

    if content:
        with st.expander("Open full report", expanded=False):
            st.markdown(content)
        report_type = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(report.get("report_type") or "report").lower(),
        ).strip("_")
        period = re.sub(r"[^0-9-]+", "_", _format_period(report)).strip("_")
        st.download_button(
            "Download report",
            data=content,
            file_name=f"{report_type or 'report'}_{period or 'archive'}.md",
            mime="text/markdown",
            key=f"download_report_{_history_id(report, 0)}",
        )
    else:
        st.info("No report content is available for this archive entry.")


def _render_history_row(report: dict, index: int) -> None:
    columns = st.columns([1.2, 1.6, 1.25, 0.8, 0.55])
    content = report.get("content")
    status = "Saved" if content else "Unavailable"
    report_id = _history_id(report, index)
    with columns[0]:
        st.write(_report_label(report))
    with columns[1]:
        st.write(_format_period(report))
    with columns[2]:
        st.write(_format_generated(report.get("generated_at")))
    with columns[3]:
        st.write(status)
    with columns[4]:
        if st.button("Open", key=f"open_report_{report_id}", disabled=not content):
            st.session_state[REPORT_SESSION_KEY] = report
            st.session_state[OPEN_REPORT_SESSION_KEY] = report_id


def _render_weekly_snapshot(data: pd.DataFrame) -> None:
    st.markdown("#### Weekly trend snapshot")
    dates = sorted(data["_report_date"].dropna().unique().tolist())[-7:]
    if not dates:
        st.info("No valid dataset dates are available for the weekly snapshot.")
        return

    week = data.loc[data["_report_date"].isin(dates)].copy()
    if week.empty:
        st.info("Insufficient historical records for the weekly snapshot.")
        return

    daily = (
        week.groupby("_report_date")
        .agg(
            records=("vehicle_count", "size"),
            average_vehicles=("vehicle_count", "mean"),
            average_speed=("average_speed_kmh", "mean"),
        )
        .reset_index()
    )
    high_severe = (
        week.assign(
            _high_severe=week["future_congestion_level"].isin(["High", "Severe"])
        )
        .groupby("_report_date")["_high_severe"]
        .mean()
        .mul(100)
    )
    daily["high_severe_percent"] = daily["_report_date"].map(high_severe)

    if len(dates) < 7:
        st.caption(f"Showing {len(dates)} available distinct date(s), fewer than seven.")

    kpis = (
        ("TOTAL RECORDS", f"{len(week):,}", f"{len(dates)} distinct dates", "cyan"),
        (
            "AVERAGE VEHICLES",
            f"{week['vehicle_count'].mean():.1f}",
            "Per observation",
            "teal",
        ),
        (
            "AVG HIGH / SEVERE",
            f"{daily['high_severe_percent'].mean():.1f}%",
            "Mean daily share",
            "emerald",
        ),
        (
            "AVG REPORT SPEED",
            f"{week['average_speed_kmh'].mean():.1f} km/h",
            "Per observation",
            "amber",
        ),
    )
    for row_start in range(0, len(kpis), 2):
        columns = st.columns(2, gap="small")
        for column, (title, value, subtitle, status) in zip(
            columns,
            kpis[row_start : row_start + 2],
        ):
            with column:
                render_kpi_card(
                    title=title,
                    value=value,
                    subtitle=subtitle,
                    status=status,
                )

    daily_table = daily.rename(
        columns={
            "_report_date": "Date",
            "records": "Records",
            "average_vehicles": "Avg vehicles",
            "average_speed": "Avg speed",
            "high_severe_percent": "High / Severe %",
        }
    )
    daily_table["Avg vehicles"] = daily_table["Avg vehicles"].round(1)
    daily_table["Avg speed"] = daily_table["Avg speed"].round(1)
    daily_table["High / Severe %"] = daily_table["High / Severe %"].round(1)
    st.dataframe(
        daily_table,
        width="stretch",
        height=max(100, 36 * (len(daily_table) + 1)),
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Records": st.column_config.NumberColumn("Records", width="small"),
            "Avg vehicles": st.column_config.NumberColumn(
                "Avg vehicles",
                width="small",
                help="Mean vehicle count per observation.",
            ),
            "Avg speed": st.column_config.NumberColumn(
                "Avg speed",
                width="small",
                help="Mean average speed in km/h.",
            ),
            "High / Severe %": st.column_config.NumberColumn(
                "High / Severe %",
                width="medium",
                format="%.1f%%",
                help="Percentage of daily records classified as High or Severe.",
            ),
        },
    )


def render_report_panel():
    """Render the reports dashboard while preserving existing generators."""
    render_header(
        "Reports & Analytics",
        "Generate daily, weekly and anomaly summaries from the historical synthetic traffic dataset.",
        badge_text="HISTORICAL DATASET",
        badge_type="cyan",
    )

    try:
        raw_data = pd.read_csv(DATASET_PATH)
        data = _normalize_dataset(raw_data)
        history = get_reports(limit=-1)
    except Exception as exc:
        st.error(f"Unable to load report data or history: {exc}")
        return

    report_counts = {"daily": 0, "weekly": 0, "anomaly": 0}
    for saved_report in history:
        kind = _report_kind(saved_report)
        if kind in report_counts:
            report_counts[kind] += 1

    kpi_columns = st.columns(4)
    kpis = (
        ("SAVED REPORTS", f"{len(history):,}", "Archive records", "cyan"),
        ("DAILY SUMMARIES", f"{report_counts['daily']:,}", "Saved reports", "teal"),
        ("WEEKLY SUMMARIES", f"{report_counts['weekly']:,}", "Saved reports", "emerald"),
        ("ANOMALY REPORTS", f"{report_counts['anomaly']:,}", "Saved reports", "amber"),
    )
    for column, (title, value, subtitle, status) in zip(kpi_columns, kpis):
        with column:
            render_kpi_card(
                title=title,
                value=value,
                subtitle=subtitle,
                status=status,
            )

    st.markdown("### Generate a report")
    with st.container(border=True):
        report_col, date_col, scope_col, button_col = st.columns(
            [1.25, 1.35, 1.45, 1]
        )
        with report_col:
            report_type = st.selectbox("Report Type", REPORT_TYPES)

        available_dates = sorted(
            data["_report_date"].dropna().unique().tolist(),
            reverse=True,
        )
        with date_col:
            selected_date = st.selectbox(
                "Available Dataset Date",
                available_dates or ["No valid dates"],
                disabled=(report_type != "Daily Summary" or not available_dates),
            )

        scopes = {
            "Daily Summary": (
                f"One historical date: {selected_date}"
                if available_dates
                else "No valid dates available"
            ),
            "Weekly Summary": "Latest seven distinct dataset dates",
            "Anomaly Report": "Full historical dataset",
        }
        with scope_col:
            st.markdown("**Scope**")
            st.caption(scopes[report_type])

        with button_col:
            st.markdown("<div style='height: 1.65rem'></div>", unsafe_allow_html=True)
            generate_clicked = st.button(
                "Generate Report",
                type="primary",
                use_container_width=True,
                disabled=(report_type == "Daily Summary" and not available_dates),
            )

        st.caption(
            "Daily reports use the selected date. Weekly reports use the latest "
            "seven distinct dates; anomaly reports analyze the full dataset."
        )

    if generate_clicked:
        try:
            with st.spinner("Generating report from historical dataset values..."):
                if report_type == "Daily Summary":
                    report = generate_daily_report(raw_data, selected_date)
                elif report_type == "Weekly Summary":
                    report = generate_weekly_report(raw_data)
                else:
                    anomalies = get_anomaly_report(raw_data)
                    report = generate_anomaly_report(anomalies)

            if "error" in report:
                st.error(report["error"])
            else:
                st.session_state[REPORT_SESSION_KEY] = report
                st.session_state.pop(OPEN_REPORT_SESSION_KEY, None)
                st.success(f"{report_type} generated and saved to report history.")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    st.markdown("### Report archive")
    history_col, snapshot_col = st.columns([1.15, 1])
    with history_col:
        with st.container(border=True):
            st.markdown("#### Recent report history")
            if not history:
                st.info("No saved reports yet. Generate a report to populate the archive.")
            else:
                header = st.columns([1.2, 1.6, 1.25, 0.8, 0.55])
                for column, label in zip(
                    header,
                    ["Report", "Period", "Generated", "Status", "Open"],
                ):
                    with column:
                        st.caption(f"**{label}**")
                for index, saved_report in enumerate(history[:3]):
                    _render_history_row(saved_report, index)

                if len(history) > 3:
                    with st.expander(f"View {len(history) - 3} older saved reports"):
                        for index, saved_report in enumerate(history[3:], start=3):
                            _render_history_row(saved_report, index)

    with snapshot_col:
        with st.container(border=True):
            _render_weekly_snapshot(data)

    selected_report = st.session_state.get(REPORT_SESSION_KEY)
    if selected_report:
        st.markdown("### Selected report")
        with st.container(border=True):
            st.markdown(
                f"#### {_report_label(selected_report)} · "
                f"{_format_period(selected_report)}"
            )
            st.caption(
                f"Generated {_format_generated(selected_report.get('generated_at'))} "
                "· Historical synthetic dataset"
            )
            _render_report_details(selected_report, data)

    st.markdown(
        "<div style='text-align:center; color:#64748B; font-size:0.72rem; "
        "letter-spacing:0.08em; padding:1.2rem 0 0.4rem;'>"
        "HISTORICAL SYNTHETIC DATA · ARCHIVE REPORTS · "
        "NO LIVE TRAFFIC MONITORING</div>",
        unsafe_allow_html=True,
    )
