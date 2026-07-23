"""
Report Generator Module.
Creates automated daily, weekly, and anomaly reports.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from database import insert_report, get_reports

logger = logging.getLogger(__name__)


def generate_daily_report(df: pd.DataFrame, date: str = None) -> dict:
    """
    Generate a daily traffic summary report.
    date: 'YYYY-MM-DD' string, defaults to latest date in data.
    """
    if date is None:
        date = df["timestamp"].max()[:10]

    day_data = df[df["timestamp"].str.startswith(date)]

    if day_data.empty:
        return {"error": f"No data found for {date}"}

    total = len(day_data)
    avg_vehicles = day_data["vehicle_count"].mean()
    avg_speed = day_data["avg_speed"].mean()
    congestion_dist = day_data["congestion_level"].value_counts().to_dict()
    busiest_junction = day_data.groupby("junction_id")["vehicle_count"].mean().idxmax()
    busiest_hour = day_data.groupby("hour")["vehicle_count"].mean().idxmax()

    content = f"""# 📊 Daily Traffic Report — {date}

## Summary
| Metric | Value |
|--------|-------|
| Total Records | {total} |
| Avg Vehicle Count | {avg_vehicles:.0f} |
| Avg Speed | {avg_speed:.1f} km/h |
| Busiest Junction | {busiest_junction} |
| Peak Hour | {busiest_hour}:00 |

## Congestion Distribution
| Level | Count | Percentage |
|-------|-------|------------|
| High | {congestion_dist.get('High', 0)} | {congestion_dist.get('High', 0)/total*100:.1f}% |
| Medium | {congestion_dist.get('Medium', 0)} | {congestion_dist.get('Medium', 0)/total*100:.1f}% |
| Low | {congestion_dist.get('Low', 0)} | {congestion_dist.get('Low', 0)/total*100:.1f}% |

## Top 5 Busiest Junctions
"""
    top5 = day_data.groupby("junction_id").agg({
        "vehicle_count": "mean", "avg_speed": "mean"
    }).sort_values("vehicle_count", ascending=False).head(5)

    for jid, row in top5.iterrows():
        content += f"- **{jid}**: {row['vehicle_count']:.0f} avg vehicles, {row['avg_speed']:.1f} km/h\n"

    content += f"\n## Weather Impact\n"
    weather_stats = day_data.groupby("weather")["vehicle_count"].mean()
    for w, v in weather_stats.items():
        content += f"- {w}: {v:.0f} avg vehicles\n"

    report = {
        "report_type": "daily",
        "period_start": date,
        "period_end": date,
        "content": content,
        "generated_at": datetime.now().isoformat(),
    }

    insert_report(report)
    logger.info(f"Daily report generated for {date}")
    return report


def generate_weekly_report(df: pd.DataFrame) -> dict:
    """Generate a weekly traffic summary report."""
    # Get last 7 days of data
    df["date"] = df["timestamp"].str[:10]
    dates = sorted(df["date"].unique())
    if len(dates) > 7:
        dates = dates[-7:]
    week_data = df[df["date"].isin(dates)]

    total = len(week_data)
    avg_vehicles = week_data["vehicle_count"].mean()
    avg_speed = week_data["avg_speed"].mean()

    content = f"""# 📋 Weekly Traffic Report

**Period:** {dates[0]} to {dates[-1]}

## Overview
| Metric | Value |
|--------|-------|
| Total Records | {total} |
| Days Covered | {len(dates)} |
| Avg Vehicle Count | {avg_vehicles:.0f} |
| Avg Speed | {avg_speed:.1f} km/h |

## Daily Trends
| Date | Avg Vehicles | Avg Speed | High Congestion % |
|------|-------------|-----------|-------------------|
"""
    for date in dates:
        d = week_data[week_data["date"] == date]
        high_pct = (d["congestion_level"] == "High").mean() * 100
        content += f"| {date} | {d['vehicle_count'].mean():.0f} | {d['avg_speed'].mean():.1f} | {high_pct:.1f}% |\n"

    content += "\n## Worst Junctions (by avg vehicle count)\n"
    worst = week_data.groupby("junction_id")["vehicle_count"].mean().sort_values(ascending=False).head(5)
    for jid, v in worst.items():
        content += f"- **{jid}**: {v:.0f} avg vehicles\n"

    content += "\n## Recommendations\n"
    high_pct_overall = (week_data["congestion_level"] == "High").mean() * 100
    if high_pct_overall > 30:
        content += "- ⚠️ High congestion levels are significant. Consider traffic signal optimization.\n"
    if avg_speed < 25:
        content += "- 🐌 Average speeds are low. Review construction zones and incident patterns.\n"
    content += "- 📊 Monitor peak hours (8-10 AM, 5-7 PM) for intervention opportunities.\n"

    report = {
        "report_type": "weekly",
        "period_start": dates[0],
        "period_end": dates[-1],
        "content": content,
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
                f"{i}. **{a['junction_id']}** — Risk: {a['risk_level']} | "
                f"Vehicles: {a.get('vehicle_count', 'N/A')} | "
                f"Speed: {a.get('avg_speed', 'N/A')} km/h | "
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
