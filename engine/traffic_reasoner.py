from data.generate_dataset import JUNCTION_NAMES
from engine.unified_risk_engine import calculate_unified_risk


def generate_operational_summary(data_context):

    hotspots = data_context.get("hotspots", [])
    anomalies = data_context.get("anomalies", [])
    traffic_df = data_context.get("traffic_data")

    summary = []

    # ---------------------------
    # Hotspots
    # ---------------------------

    seen_hotspots = set()

    for hotspot in hotspots:

        cluster_name = f"Cluster {hotspot['cluster_id']}"

        if cluster_name in seen_hotspots:
            continue

        seen_hotspots.add(cluster_name)

        score = calculate_unified_risk(
            congestion_level="Low",
            anomaly_risk=None,
            hotspot_severity=hotspot["severity"],
        )
        summary.append(
            {
                "type": "hotspot",
                "severity": hotspot["severity"],
                "location": cluster_name,
                "score": score,
            }
        )

    # ---------------------------
    # Anomalies
    # ---------------------------

    seen_locations = set()

    for anomaly in anomalies:

        if anomaly["risk_level"] not in ["High", "Critical"]:
            continue

        junction_id = anomaly["junction_id"]

        location = JUNCTION_NAMES.get(junction_id, junction_id)

        if location in seen_locations:
            continue

        seen_locations.add(location)

        congestion = "Low"

        if traffic_df is not None:
            junction_rows = traffic_df[traffic_df["junction_id"] == junction_id]
            if not junction_rows.empty:
                congestion_counts = (
                    junction_rows["congestion_level"].value_counts().to_dict()
                )
                if congestion_counts:
                    congestion = max(congestion_counts, key=congestion_counts.get)
                    score = calculate_unified_risk(
                        congestion_level=congestion,
                        anomaly_risk=anomaly["risk_level"],
                        hotspot_severity=None,
                    )

        summary.append(
            {
                "type": "anomaly",
                "severity": anomaly["risk_level"],
                "location": location,
                "score": score,
            }
        )

    # ---------------------------
    # Sort by score
    # ---------------------------

    summary.sort(key=lambda x: x["score"], reverse=True)

    return summary
