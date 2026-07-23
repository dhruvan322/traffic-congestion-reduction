from data.generate_dataset import JUNCTION_NAMES


def calculate_unified_risk(
    congestion_level,
    anomaly_risk=None,
    hotspot_severity=None,
):
    score = 0

    congestion_scores = {
        "Low": 10,
        "Medium": 25,
        "High": 40,
    }

    anomaly_scores = {
        None: 0,
        "Low": 10,
        "Medium": 20,
        "High": 30,
        "Critical": 40,
    }

    hotspot_scores = {
        None: 0,
        "Low": 10,
        "Medium": 20,
        "High": 30,
        "Critical": 40,
    }

    score += congestion_scores.get(congestion_level, 0)

    score += anomaly_scores.get(anomaly_risk, 0)

    score += hotspot_scores.get(hotspot_severity, 0)

    return min(score, 100)
