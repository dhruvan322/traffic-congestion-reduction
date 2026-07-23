def generate_management_plan(data_context):

    hotspots = data_context.get("hotspots", [])
    anomalies = data_context.get("anomalies", [])

    plan = []

    # High severity hotspots
    for hotspot in hotspots[:3]:

        severity = hotspot["severity"]

        if severity == "High":

            plan.append(
                f"Deploy traffic personnel near Cluster {hotspot['cluster_id']}."
            )

            plan.append(
                f"Increase green signal duration around Cluster {hotspot['cluster_id']}."
            )

        elif severity == "Medium":

            plan.append(
                f"Monitor Cluster {hotspot['cluster_id']} for congestion growth."
            )

    # High risk anomalies
    for anomaly in anomalies[:3]:

        if anomaly["risk_level"] in ["High", "Critical"]:

            plan.append(
                f"Investigate abnormal traffic patterns at {anomaly['junction_id']}."
            )

    # Long-term recommendations
    plan.append("Consider adaptive signal timing at high-congestion junctions.")

    plan.append("Prepare diversion routes during peak hours.")

    return plan
