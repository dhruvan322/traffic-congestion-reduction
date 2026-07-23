"""
Hotspot Detection Module.
Uses DBSCAN clustering to identify traffic congestion hotspots.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from datetime import datetime

from config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, CONGESTION_THRESHOLDS

logger = logging.getLogger(__name__)


def detect_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect congestion hotspots using DBSCAN clustering on high-congestion points.
    Returns DataFrame with cluster_id, latitude, longitude, severity.
    """
    # Filter to high/medium congestion records
    congested = df[df["congestion_level"].isin(["High", "Medium"])].copy()

    if len(congested) < DBSCAN_MIN_SAMPLES:
        logger.warning("Not enough congested records for clustering.")
        return pd.DataFrame()

    # Cluster on lat/lon
    coords = congested[["latitude", "longitude"]].values
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    congested["cluster_id"] = db.fit_predict(coords)

    # Remove noise points (cluster_id == -1)
    clustered = congested[congested["cluster_id"] != -1].copy()

    if clustered.empty:
        logger.info("No clusters found.")
        return pd.DataFrame()

    logger.info(f"Found {clustered['cluster_id'].nunique()} hotspot clusters.")
    return clustered


def get_hotspot_summary(df: pd.DataFrame) -> list[dict]:
    """
    Get summary of each hotspot cluster.
    Returns list of {cluster_id, center_lat, center_lon, size, severity, peak_hours, avg_vehicles}.
    """
    clustered = detect_hotspots(df)
    if clustered.empty:
        return []

    summaries = []
    for cid, group in clustered.groupby("cluster_id"):
        # Peak hours for this cluster
        peak_hours = group["hour"].value_counts().head(3).index.tolist()

        # Severity based on proportion of High congestion
        high_ratio = (group["congestion_level"] == "High").mean()
        if high_ratio > 0.6:
            severity = "Critical"
        elif high_ratio > 0.3:
            severity = "High"
        else:
            severity = "Medium"

        summaries.append({
            "cluster_id": int(cid),
            "center_lat": round(group["latitude"].mean(), 6),
            "center_lon": round(group["longitude"].mean(), 6),
            "size": len(group),
            "severity": severity,
            "peak_hours": peak_hours,
            "avg_vehicles": round(group["vehicle_count"].mean(), 1),
            "avg_speed": round(group["avg_speed"].mean(), 1),
            "detected_at": datetime.now().isoformat(),
        })

    return sorted(summaries, key=lambda x: x["size"], reverse=True)
