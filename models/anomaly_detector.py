"""
Anomaly Detection Module.
Uses Isolation Forest and Z-score methods to detect traffic anomalies.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime

from config import ANOMALY_CONTAMINATION, ZSCORE_THRESHOLD

logger = logging.getLogger(__name__)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest on vehicle_count and avg_speed.
    Returns DataFrame with an 'is_anomaly' column.
    """
    df = df.copy()
    features = df[["vehicle_count", "avg_speed"]].values

    iso = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_estimators=100,
    )
    preds = iso.fit_predict(features)
    scores = iso.decision_function(features)

    # -1 = anomaly, 1 = normal in IsolationForest
    df["is_anomaly"] = (preds == -1).astype(int)
    df["anomaly_score"] = np.round(-scores, 4)  # Higher = more anomalous

    return df


def zscore_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag records with |z-score| > threshold for vehicle_count or avg_speed.
    """
    df = df.copy()
    for col in ["vehicle_count", "avg_speed"]:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f"{col}_zscore"] = np.abs((df[col] - mean) / std)
        else:
            df[f"{col}_zscore"] = 0.0

    df["zscore_anomaly"] = (
        (df["vehicle_count_zscore"] > ZSCORE_THRESHOLD) |
        (df["avg_speed_zscore"] > ZSCORE_THRESHOLD)
    ).astype(int)

    return df


def classify_risk(row: pd.Series) -> str:
    """Classify anomaly risk level based on anomaly score and congestion."""
    score = row.get("anomaly_score", 0)
    congestion = row.get("congestion_level", "Low")

    if score > 0.3 and congestion == "High":
        return "Critical"
    elif score > 0.2 or congestion == "High":
        return "High"
    elif score > 0.1:
        return "Medium"
    return "Low"


def get_anomaly_report(df: pd.DataFrame) -> list[dict]:
    """
    Run full anomaly detection and return structured anomaly records.
    """
    df_iso = detect_anomalies(df)
    df_z = zscore_anomalies(df_iso)

    # Combine: anomaly if either method flags it
    df_z["combined_anomaly"] = (
        (df_z["is_anomaly"] == 1) | (df_z["zscore_anomaly"] == 1)
    ).astype(int)

    anomalies_df = df_z[df_z["combined_anomaly"] == 1].copy()

    if anomalies_df.empty:
        return []

    anomalies_df["risk_level"] = anomalies_df.apply(classify_risk, axis=1)

    records = []
    for _, row in anomalies_df.iterrows():
        records.append({
            "junction_id": row["junction_id"],
            "timestamp": row.get("timestamp", ""),
            "vehicle_count": int(row["vehicle_count"]),
            "avg_speed": float(row["avg_speed"]),
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "risk_level": row["risk_level"],
            "anomaly_type": "isolation_forest+zscore",
            "detected_at": datetime.now().isoformat(),
            "description": (
                f"Unusual traffic at {row['junction_id']}: "
                f"{int(row['vehicle_count'])} vehicles, "
                f"{row['avg_speed']:.1f} km/h avg speed"
            ),
        })

    logger.info(f"Detected {len(records)} anomalies.")
    return sorted(records, key=lambda x: x["anomaly_score"], reverse=True)
