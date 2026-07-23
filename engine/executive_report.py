from data.generate_dataset import JUNCTION_NAMES


def generate_executive_report(data_context):

    df = data_context["traffic_data"]

    hotspots = data_context.get("hotspots", [])
    anomalies = data_context.get("anomalies", [])

    total_vehicles = int(df["vehicle_count"].sum())
    avg_speed = round(df["avg_speed"].mean(), 1)

    report = []

    report.append("📋 Executive Traffic Report\n")

    report.append("🚦 Network Status")
    report.append(f"• Total Vehicles: {total_vehicles}")
    report.append(f"• Average Speed: {avg_speed} km/h")
    report.append("")

    # ----------------------------
    # Hotspots
    # ----------------------------

    report.append("🔥 Major Hotspots")

    for hotspot in hotspots[:3]:

        report.append(f"• Cluster {hotspot['cluster_id']} " f"({hotspot['severity']})")

    report.append("")

    # ----------------------------
    # Anomalies
    # ----------------------------

    report.append("⚠️ High-Risk Locations")

    shown = set()

    for anomaly in anomalies:

        if anomaly["risk_level"] not in ["High", "Critical"]:
            continue

        name = JUNCTION_NAMES.get(anomaly["junction_id"], anomaly["junction_id"])

        if name in shown:
            continue

        shown.add(name)

        report.append(f"• {name} " f"({anomaly['risk_level']} Risk)")

        if len(shown) >= 5:
            break

    report.append("")

    # ----------------------------
    # Recommendations
    # ----------------------------

    report.append("🎯 Recommended Actions")

    report.append("• Deploy personnel at high-risk locations")

    report.append("• Optimize signal timings near hotspot regions")

    report.append("• Monitor anomaly-prone junctions")

    report.append("• Prepare diversion plans during peak hours")

    return "\n".join(report)
