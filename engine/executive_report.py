import pandas as pd

from models.hotspot_detector import get_hotspot_summary
from models.anomaly_detector import get_anomaly_report


def generate_executive_report(data_context):
    """
    Generate an executive-level traffic intelligence report
    using the final ML dataset and verified hotspot/anomaly modules.
    """

    df = data_context["traffic_data"]

    if df.empty:
        return "No traffic data available for executive analysis."

    # ------------------------------------------------------------
    # Network Status
    # ------------------------------------------------------------

    total_records = len(df)
    total_vehicles = int(df["vehicle_count"].sum())
    avg_speed = float(df["average_speed_kmh"].mean())

    congestion_counts = (
        df["future_congestion_level"].astype(str).value_counts().to_dict()
    )

    # ------------------------------------------------------------
    # Verified Hotspot Intelligence
    # ------------------------------------------------------------

    hotspots = get_hotspot_summary(df)

    # ------------------------------------------------------------
    # Verified Anomaly Intelligence
    # ------------------------------------------------------------

    anomalies = get_anomaly_report(df)

    high_risk_anomalies = [
        anomaly
        for anomaly in anomalies
        if anomaly.get("risk_level") in ["High", "Critical"]
    ]

    # Unique high-risk road segments
    high_risk_segments = []
    seen_segments = set()

    for anomaly in high_risk_anomalies:
        segment = str(anomaly.get("road_segment_id", "Unknown"))

        if segment in seen_segments:
            continue

        seen_segments.add(segment)
        high_risk_segments.append(anomaly)

        if len(high_risk_segments) >= 5:
            break

    # ------------------------------------------------------------
    # Executive Priority
    # ------------------------------------------------------------

    if any(
        hotspot.get("severity") in ["Critical", "High"] for hotspot in hotspots
    ) or any(
        anomaly.get("risk_level") == "Critical" for anomaly in high_risk_anomalies
    ):
        priority = "Critical"

    elif hotspots or high_risk_anomalies:
        priority = "High"

    elif congestion_counts.get("Severe", 0) > 0:
        priority = "High"

    elif congestion_counts.get("High", 0) > 0:
        priority = "Medium"

    else:
        priority = "Low"

    # ------------------------------------------------------------
    # Build Report
    # ------------------------------------------------------------

    report = []

    report.append("Executive Traffic Intelligence Report")
    report.append("")

    report.append("Network Status")
    report.append(f"• Monitored Records: {total_records}")
    report.append(f"• Total Vehicles: {total_vehicles}")
    report.append(f"• Average Speed: {avg_speed:.1f} km/h")
    report.append(f"• Operational Priority: {priority}")
    report.append("")

    # ------------------------------------------------------------
    # Congestion Distribution
    # ------------------------------------------------------------

    report.append("Congestion Distribution")

    congestion_order = [
        "Free Flow",
        "Low",
        "Moderate",
        "High",
        "Severe",
    ]

    for level in congestion_order:
        count = int(congestion_counts.get(level, 0))
        percentage = (count / total_records * 100) if total_records else 0

        report.append(f"• {level}: {count:,} records ({percentage:.1f}%)")

    report.append("")

    # ------------------------------------------------------------
    # Hotspots
    # ------------------------------------------------------------

    report.append("Major Congestion Hotspots")

    if hotspots:
        for hotspot in hotspots[:3]:
            cluster_id = hotspot.get("cluster_id", "Unknown")
            severity = hotspot.get("severity", "Unknown")
            size = hotspot.get("size", 0)
            congestion_rate = float(hotspot.get("congestion_rate", 0.0))
            peak_hours = hotspot.get("peak_hours", [])

            report.append(
                f"• Cluster {cluster_id}: {severity} severity, "
                f"{size} road segments, "
                f"{congestion_rate:.1%} congestion rate"
            )

            if peak_hours:
                report.append(f"  Peak hours: {', '.join(map(str, peak_hours))}")

    else:
        report.append("• No significant congestion hotspots detected.")

    report.append("")

    # ------------------------------------------------------------
    # High-Risk Anomalies
    # ------------------------------------------------------------

    report.append("High-Risk Traffic Anomalies")

    if high_risk_segments:
        for anomaly in high_risk_segments:
            segment = anomaly.get("road_segment_id", "Unknown")
            risk = anomaly.get("risk_level", "Unknown")
            score = float(anomaly.get("anomaly_score", 0.0))
            congestion = anomaly.get(
                "future_congestion_level",
                "Unknown",
            )

            report.append(
                f"• {segment}: {risk} risk, "
                f"{congestion} future congestion, "
                f"anomaly score {score:.3f}"
            )

    else:
        report.append("• No High or Critical anomalies detected.")

    report.append("")

    # ------------------------------------------------------------
    # Recommended Actions
    # ------------------------------------------------------------

    report.append("Recommended Operational Actions")

    if priority in ["Critical", "High"]:
        report.append(
            "• Prioritize high-risk road segments for operational monitoring."
        )

    if hotspots:
        report.append(
            "• Apply traffic management measures around identified hotspot clusters."
        )

    if high_risk_segments:
        report.append(
            "• Investigate High/Critical anomaly locations for abnormal traffic conditions."
        )

    if hotspots:
        report.append(
            "• Review signal timing and corridor management near hotspot regions."
        )

    report.append("• Prepare diversion strategies if congestion conditions escalate.")

    report.append("• Continue monitoring prediction, hotspot, and anomaly indicators.")

    report.append("")

    report.append(
        "Note: Recommendations are decision-support outputs based on "
        "historical/project traffic data. They do not represent measured "
        "real-world traffic reduction."
    )

    return "\n".join(report)
