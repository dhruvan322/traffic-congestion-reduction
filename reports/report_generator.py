"""
Report Generator Module.
Creates automated daily, weekly, and anomaly reports.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from database import insert_report, get_reports

logger = logging.getLogger(__name__)


def _prepare_report_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize current project fields for report generation."""
    data = df.copy()

    if "timestamp" not in data.columns:
        raise ValueError("Report data must contain a timestamp column.")

    data["_report_timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data["_report_date"] = (
        data["_report_timestamp"].dt.strftime("%Y-%m-%d")
    )

    aliases = {
        "average_speed_kmh": "avg_speed",
        "future_congestion_level": "congestion_level",
        "road_segment_id": "junction_id",
        "weather_condition": "weather",
    }

    for current_name, legacy_name in aliases.items():
        if current_name not in data.columns and legacy_name in data.columns:
            data[current_name] = data[legacy_name]

    if "hour" not in data.columns:
        data["hour"] = data["_report_timestamp"].dt.hour

    required = [
        "vehicle_count",
        "average_speed_kmh",
        "future_congestion_level",
        "road_segment_id",
    ]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Report data is missing required columns: "
            + ", ".join(missing)
        )

    data["vehicle_count"] = pd.to_numeric(
        data["vehicle_count"], errors="coerce"
    )
    data["average_speed_kmh"] = pd.to_numeric(
        data["average_speed_kmh"], errors="coerce"
    )
    data["hour"] = pd.to_numeric(data["hour"], errors="coerce")

    return data


def generate_daily_report(df: pd.DataFrame, date: str = None) -> dict:
    """Generate a daily report using the current congestion schema."""
    data = _prepare_report_data(df)

    if date is None:
        available_dates = data["_report_date"].dropna()
        if available_dates.empty:
            return {"error": "No valid timestamps are available."}
        date = available_dates.max()

    day_data = data[data["_report_date"] == date].copy()

    if day_data.empty:
        return {"error": f"No data found for {date}"}

    total = len(day_data)
    avg_vehicles = day_data["vehicle_count"].mean()
    avg_speed = day_data["average_speed_kmh"].mean()

    labels = (
        day_data["future_congestion_level"]
        .fillna("Unknown")
        .astype(str)
    )
    counts = labels.value_counts().to_dict()

    standard_classes = [
        "Free Flow", "Low", "Moderate", "High", "Severe"
    ]
    additional_classes = sorted(set(counts) - set(standard_classes))
    class_order = standard_classes + additional_classes

    busiest_segment = (
        day_data.groupby("road_segment_id")["vehicle_count"]
        .mean()
        .idxmax()
    )
    busiest_hour = (
        day_data.groupby("hour")["vehicle_count"]
        .mean()
        .idxmax()
    )

    lines = [
        f"# Daily Traffic Report - {date}",
        "",
        "Congestion categories reflect the dataset's "
        "future_congestion_level field.",
        "",
        "## Summary",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Records | {total} |",
        f"| Average Vehicle Count | {avg_vehicles:.1f} |",
        f"| Average Speed | {avg_speed:.1f} km/h |",
        f"| Busiest Road Segment | {busiest_segment} |",
        f"| Peak Hour | {int(busiest_hour):02d}:00 |",
        "",
        "## Congestion Distribution",
        "| Level | Count | Percentage |",
        "|---|---:|---:|",
    ]

    for level in class_order:
        count = int(counts.get(level, 0))
        lines.append(
            f"| {level} | {count} | {count / total * 100:.1f}% |"
        )

    top5 = (
        day_data.groupby("road_segment_id")
        .agg(
            average_vehicles=("vehicle_count", "mean"),
            average_speed=("average_speed_kmh", "mean"),
        )
        .sort_values("average_vehicles", ascending=False)
        .head(5)
    )

    lines.extend(["", "## Top 5 Busiest Road Segments"])

    for segment, row in top5.iterrows():
        lines.append(
            f"- **{segment}**: {row['average_vehicles']:.1f} "
            f"average vehicles, {row['average_speed']:.1f} km/h "
            "average speed"
        )

    lines.extend(["", "## Weather Summary"])

    if "weather_condition" in day_data.columns:
        weather_stats = (
            day_data.groupby("weather_condition")["vehicle_count"].mean()
        )
        for condition, value in weather_stats.items():
            lines.append(
                f"- {condition}: {value:.1f} average vehicles"
            )
    else:
        lines.append(
            "- Weather information is unavailable in this dataset."
        )

    report = {
        "report_type": "daily",
        "period_start": date,
        "period_end": date,
        "content": "\n".join(lines),
        "generated_at": datetime.now().isoformat(),
    }

    insert_report(report)
    logger.info("Daily report generated for %s", date)
    return report


def generate_weekly_report(df: pd.DataFrame) -> dict:
    """Generate a report for the latest seven dates in the dataset."""
    data = _prepare_report_data(df)
    dates = sorted(data["_report_date"].dropna().unique().tolist())

    if not dates:
        return {"error": "No valid dates are available for the weekly report."}

    dates = dates[-7:]
    week_data = data[data["_report_date"].isin(dates)].copy()

    total = len(week_data)
    avg_vehicles = week_data["vehicle_count"].mean()
    avg_speed = week_data["average_speed_kmh"].mean()

    lines = [
        f"# Weekly Traffic Report - {dates[0]} to {dates[-1]}",
        "",
        "Congestion percentages are calculated from the dataset's "
        "future_congestion_level field.",
        "",
        "## Summary",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Records | {total} |",
        f"| Dates Covered | {len(dates)} |",
        f"| Average Vehicle Count | {avg_vehicles:.1f} |",
        f"| Average Speed | {avg_speed:.1f} km/h |",
        "",
        "## Daily Trends",
        "| Date | Average Vehicles | Average Speed (km/h) | High/Severe (%) |",
        "|---|---:|---:|---:|",
    ]

    for report_date in dates:
        day = week_data[week_data["_report_date"] == report_date]
        high_severe_pct = (
            day["future_congestion_level"]
            .isin(["High", "Severe"])
            .mean() * 100
        )
        lines.append(
            f"| {report_date} | "
            f"{day['vehicle_count'].mean():.1f} | "
            f"{day['average_speed_kmh'].mean():.1f} | "
            f"{high_severe_pct:.1f}% |"
        )

    busiest = (
        week_data.groupby("road_segment_id")["vehicle_count"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    lines.extend([
        "",
        "## Road Segments with Highest Average Vehicle Count",
    ])

    for segment, value in busiest.items():
        lines.append(
            f"- **{segment}**: {value:.1f} average vehicles"
        )

    high_severe_overall = (
        week_data["future_congestion_level"]
        .isin(["High", "Severe"])
        .mean() * 100
    )

    lines.extend(["", "## Recommendations"])

    if high_severe_overall > 30:
        lines.append(
            "- The High/Severe share exceeds 30% in this dataset window. "
            "Review affected segments before considering interventions."
        )

    if avg_speed < 25:
        lines.append(
            "- Average speed is low. Review unusual observations, "
            "incident annotations, and road-segment data quality."
        )

    lines.append(
        "- Review peak-hour patterns and validate proposed actions "
        "against real traffic observations before operational use."
    )

    report = {
        "report_type": "weekly",
        "period_start": dates[0],
        "period_end": dates[-1],
        "content": "\n".join(lines),
        "generated_at": datetime.now().isoformat(),
    }

    insert_report(report)
    logger.info("Weekly report generated.")
    return report

def generate_anomaly_report(anomalies: list[dict]) -> dict:
    """Generate an anomaly summary report."""
    if not anomalies:
        content = "# ⚠️ Anomaly Report\n\nNo anomalies detected in the current data window."
    else:
        content = f"""# ⚠️ Anomaly Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total Anomalies:** {len(anomalies)}

## Risk Distribution
"""
        risk_counts = {}
        for a in anomalies:
            risk = a.get("risk_level", "Unknown")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        for risk, count in sorted(risk_counts.items()):
            content += f"- **{risk}**: {count} anomalies\n"

        content += "\n## Top Anomalies\n"
        for i, a in enumerate(anomalies[:10], 1):
            content += (
                f"{i}. **{a.get('road_segment_id', a.get('junction_id', 'Unknown'))}** — Risk: {a['risk_level']} | "
                f"Vehicles: {a.get('vehicle_count', 'N/A')} | "
                f"Speed: {a.get('average_speed_kmh', a.get('avg_speed', 'N/A'))} km/h | "
                f"Score: {a.get('anomaly_score', 0):.3f}\n"
            )

    report = {
        "report_type": "anomaly",
        "period_start": datetime.now().strftime('%Y-%m-%d'),
        "period_end": datetime.now().strftime('%Y-%m-%d'),
        "content": content,
        "generated_at": datetime.now().isoformat(),
    }

    insert_report(report)
    return report

