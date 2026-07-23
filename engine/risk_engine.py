def calculate_risk_score(
    congestion=None,
    anomaly_risk=None,
    hotspot_severity=None,
    predicted_congestion=None,
):

    score = 0

    # --------------------
    # Current Congestion
    # --------------------

    if congestion == "High":
        score += 35

    elif congestion == "Medium":
        score += 20

    elif congestion == "Low":
        score += 5

    # --------------------
    # Predicted Congestion
    # --------------------

    if predicted_congestion == "High":
        score += 25

    elif predicted_congestion == "Medium":
        score += 15

    elif predicted_congestion == "Low":
        score += 5

    # --------------------
    # Anomaly Risk
    # --------------------

    if anomaly_risk == "Critical":
        score += 25

    elif anomaly_risk == "High":
        score += 20

    elif anomaly_risk == "Medium":
        score += 10

    # --------------------
    # Hotspot Severity
    # --------------------

    if hotspot_severity == "Critical":
        score += 20

    elif hotspot_severity == "High":
        score += 15

    elif hotspot_severity == "Medium":
        score += 8

    return min(score, 100)
