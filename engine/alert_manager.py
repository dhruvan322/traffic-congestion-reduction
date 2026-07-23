"""
Alert Manager Module.
Generates alerts for high congestion, anomalies, and incidents.
"""

import logging
from datetime import datetime
from database import insert_alert, get_active_alerts

from config import ALERT_CONGESTION_THRESHOLD, ALERT_ANOMALY_RISK_THRESHOLD

logger = logging.getLogger(__name__)


def check_congestion_alerts(df_or_records) -> list[dict]:
    """Generate alerts for junctions with high congestion."""
    import pandas as pd

    if isinstance(df_or_records, list):
        df = pd.DataFrame(df_or_records)
    else:
        df = df_or_records

    alerts = []
    high = df[df["congestion_level"] == ALERT_CONGESTION_THRESHOLD]

    for jid, group in high.groupby("junction_id"):
        avg_vehicles = group["vehicle_count"].mean()
        alert = {
            "alert_type": "congestion",
            "severity": "High",
            "message": (
                f"High congestion at {jid}: "
                f"avg {avg_vehicles:.0f} vehicles, "
                f"{len(group)} records in window"
            ),
            "junction_id": jid,
            "created_at": datetime.now().isoformat(),
        }
        alerts.append(alert)

    return alerts


def check_anomaly_alerts(anomalies: list[dict]) -> list[dict]:
    """Generate alerts for high-risk anomalies."""
    alerts = []
    for a in anomalies:
        if a.get("risk_level") in ("High", "Critical"):
            alert = {
                "alert_type": "anomaly",
                "severity": a["risk_level"],
                "message": a.get("description", f"Anomaly detected at {a['junction_id']}"),
                "junction_id": a["junction_id"],
                "created_at": datetime.now().isoformat(),
            }
            alerts.append(alert)

    return alerts


def check_incident_alerts(sim_records: list[dict]) -> list[dict]:
    """Generate alerts for simulated incidents."""
    alerts = []
    for r in sim_records:
        if r.get("has_incident"):
            alert = {
                "alert_type": "incident",
                "severity": "Critical",
                "message": (
                    f"Traffic incident at {r['junction_id']}! "
                    f"Vehicle count surged to {r['vehicle_count']}, "
                    f"speed dropped to {r['avg_speed']} km/h"
                ),
                "junction_id": r["junction_id"],
                "created_at": datetime.now().isoformat(),
            }
            alerts.append(alert)

    return alerts


def generate_and_store_alerts(df=None, anomalies=None, sim_records=None) -> int:
    """
    Run all alert checks and store new alerts in the database.
    Returns the number of new alerts generated.
    """
    all_alerts = []

    if df is not None:
        all_alerts.extend(check_congestion_alerts(df))

    if anomalies:
        all_alerts.extend(check_anomaly_alerts(anomalies))

    if sim_records:
        all_alerts.extend(check_incident_alerts(sim_records))

    # Store in database
    for alert in all_alerts:
        try:
            insert_alert(alert)
        except Exception as e:
            logger.error(f"Failed to insert alert: {e}")

    if all_alerts:
        logger.info(f"Generated {len(all_alerts)} new alerts.")

    return len(all_alerts)


def get_alert_summary() -> dict:
    """Get a summary of active alerts by type."""
    alerts = get_active_alerts(limit=100)
    summary = {
        "total": len(alerts),
        "by_type": {},
        "by_severity": {},
    }
    for a in alerts:
        atype = a.get("alert_type", "unknown")
        severity = a.get("severity", "unknown")
        summary["by_type"][atype] = summary["by_type"].get(atype, 0) + 1
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

    return summary
