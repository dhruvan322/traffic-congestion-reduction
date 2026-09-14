"""
Hotspot Detection Module.

Uses DBSCAN spatial clustering on road segments that show
repeated Moderate, High, or Severe future congestion in the
final ML traffic dataset.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Congestion classes used for hotspot detection
# -------------------------------------------------------------------

CONGESTED_LEVELS = {
    "Moderate",
    "High",
    "Severe",
}


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------


def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the minimum columns required by hotspot detection.
    """

    required_columns = {
        "road_segment_id",
        "latitude",
        "longitude",
        "hour",
        "vehicle_count",
        "average_speed_kmh",
        "occupancy_percent",
        "future_congestion_level",
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            "Final ML dataset is missing required hotspot "
            "columns: " + ", ".join(missing)
        )


# -------------------------------------------------------------------
# Road-segment aggregation
# -------------------------------------------------------------------


def _prepare_segment_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate hourly traffic records into one spatial
    observation per road segment.

    This prevents repeated hourly records from overweighting
    the same physical road segment during DBSCAN clustering.
    """

    data = df.copy()

    data["future_congestion_level"] = (
        data["future_congestion_level"].astype("string").str.strip()
    )

    # Keep only records with valid spatial information.
    data = data.dropna(
        subset=[
            "road_segment_id",
            "latitude",
            "longitude",
        ]
    )

    if data.empty:
        return pd.DataFrame()

    # ---------------------------------------------------------------
    # Congestion indicators
    # ---------------------------------------------------------------

    data["is_congested"] = (
        data["future_congestion_level"].isin(CONGESTED_LEVELS).astype(int)
    )

    data["is_high_or_severe"] = (
        data["future_congestion_level"].isin({"High", "Severe"}).astype(int)
    )

    data["is_severe"] = data["future_congestion_level"].eq("Severe").astype(int)

    # ---------------------------------------------------------------
    # Aggregate to road-segment level
    # ---------------------------------------------------------------

    segment_stats = data.groupby(
        "road_segment_id",
        as_index=False,
    ).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        records=("road_segment_id", "size"),
        congested_records=(
            "is_congested",
            "sum",
        ),
        high_or_severe_records=(
            "is_high_or_severe",
            "sum",
        ),
        severe_records=(
            "is_severe",
            "sum",
        ),
        avg_vehicles=(
            "vehicle_count",
            "mean",
        ),
        avg_speed=(
            "average_speed_kmh",
            "mean",
        ),
        avg_occupancy=(
            "occupancy_percent",
            "mean",
        ),
    )

    # ---------------------------------------------------------------
    # Congestion rates
    # ---------------------------------------------------------------

    segment_stats["congestion_rate"] = (
        segment_stats["congested_records"] / segment_stats["records"]
    )

    segment_stats["high_or_severe_rate"] = (
        segment_stats["high_or_severe_records"] / segment_stats["records"]
    )

    segment_stats["severe_rate"] = (
        segment_stats["severe_records"] / segment_stats["records"]
    )

    return segment_stats


# -------------------------------------------------------------------
# DBSCAN detection
# -------------------------------------------------------------------


def detect_hotspots(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect spatial congestion hotspots using DBSCAN.

    The final ML dataset contains repeated hourly observations
    for each road segment. Therefore, hotspot detection first
    aggregates the data to one observation per road segment.

    Road segments with Moderate, High, or Severe future
    congestion are then spatially clustered using DBSCAN.

    Returns:
        DataFrame containing clustered road segments and
        aggregate congestion statistics.
    """

    _validate_dataframe(df)

    segment_stats = _prepare_segment_statistics(df)

    if segment_stats.empty:

        logger.warning("No valid road-segment observations available.")

        return pd.DataFrame()

    # ---------------------------------------------------------------
    # Candidate hotspot segments
    # ---------------------------------------------------------------

    congested_segments = segment_stats[segment_stats["congested_records"] > 0].copy()

    if len(congested_segments) < DBSCAN_MIN_SAMPLES:

        logger.warning(
            "Not enough congested road segments for DBSCAN: " "%d found, %d required.",
            len(congested_segments),
            DBSCAN_MIN_SAMPLES,
        )

        return pd.DataFrame()

    # ---------------------------------------------------------------
    # Spatial coordinates
    # ---------------------------------------------------------------

    coords = congested_segments[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(dtype=float)

    # ---------------------------------------------------------------
    # DBSCAN
    # ---------------------------------------------------------------

    db = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
    )

    congested_segments["cluster_id"] = db.fit_predict(coords)

    # ---------------------------------------------------------------
    # Remove DBSCAN noise
    # ---------------------------------------------------------------

    clustered = congested_segments[congested_segments["cluster_id"] != -1].copy()

    if clustered.empty:

        logger.info("DBSCAN found no hotspot clusters.")

        return pd.DataFrame()

    logger.info(
        "Found %d hotspot clusters from %d " "congested road segments.",
        clustered["cluster_id"].nunique(),
        len(clustered),
    )

    return clustered


# -------------------------------------------------------------------
# Peak-hour calculation
# -------------------------------------------------------------------


def _get_peak_hours(
    df: pd.DataFrame,
) -> list[int]:
    """
    Return the three most common congestion hours.
    """

    if "hour" not in df.columns or df.empty:
        return []

    hours = df["hour"].dropna().astype(int).value_counts().head(3).index.tolist()

    return sorted(hours)


# -------------------------------------------------------------------
# Hotspot summary
# -------------------------------------------------------------------


def get_hotspot_summary(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Generate dashboard-ready summaries for each DBSCAN
    hotspot cluster.

    Severity rules:

    Critical:
        Severe rate >= 25%
        OR High/Severe rate >= 70%

    High:
        High/Severe rate >= 40%

    Medium:
        Otherwise.
    """

    _validate_dataframe(df)

    clustered = detect_hotspots(df)

    if clustered.empty:
        return []

    # ---------------------------------------------------------------
    # Identify clustered road segments
    # ---------------------------------------------------------------

    clustered_segment_ids = set(clustered["road_segment_id"].tolist())

    original = df[df["road_segment_id"].isin(clustered_segment_ids)].copy()

    original["future_congestion_level"] = (
        original["future_congestion_level"].astype("string").str.strip()
    )

    summaries = []

    # ---------------------------------------------------------------
    # Summarize every DBSCAN cluster
    # ---------------------------------------------------------------

    for cid, group in clustered.groupby("cluster_id"):

        segment_ids = group["road_segment_id"].tolist()

        original_group = original[original["road_segment_id"].isin(segment_ids)].copy()

        total_records = int(group["records"].sum())

        total_congested = int(group["congested_records"].sum())

        total_high_or_severe = int(group["high_or_severe_records"].sum())

        total_severe = int(group["severe_records"].sum())

        # -----------------------------------------------------------
        # Rates
        # -----------------------------------------------------------

        if total_records > 0:

            congestion_rate = total_congested / total_records

            high_or_severe_rate = total_high_or_severe / total_records

            severe_rate = total_severe / total_records

        else:

            congestion_rate = 0.0
            high_or_severe_rate = 0.0
            severe_rate = 0.0

        # -----------------------------------------------------------
        # Severity
        # -----------------------------------------------------------

        if severe_rate >= 0.25 or high_or_severe_rate >= 0.70:

            severity = "Critical"

        elif high_or_severe_rate >= 0.40:

            severity = "High"

        else:

            severity = "Medium"

        # -----------------------------------------------------------
        # Peak congestion hours
        # -----------------------------------------------------------

        congested_original = original_group[
            original_group["future_congestion_level"].isin(CONGESTED_LEVELS)
        ]

        peak_hours = _get_peak_hours(congested_original)

        # -----------------------------------------------------------
        # Weighted averages
        # -----------------------------------------------------------

        avg_vehicles = np.average(
            group["avg_vehicles"],
            weights=group["records"],
        )

        avg_speed = np.average(
            group["avg_speed"],
            weights=group["records"],
        )

        avg_occupancy = np.average(
            group["avg_occupancy"],
            weights=group["records"],
        )

        # -----------------------------------------------------------
        # Final summary
        # -----------------------------------------------------------

        summaries.append(
            {
                "cluster_id": int(cid),
                "center_lat": round(
                    group["latitude"].mean(),
                    6,
                ),
                "center_lon": round(
                    group["longitude"].mean(),
                    6,
                ),
                # Number of physical road segments
                "size": int(len(segment_ids)),
                # Total hourly observations represented
                "records": total_records,
                "severity": severity,
                "peak_hours": peak_hours,
                "avg_vehicles": round(
                    float(avg_vehicles),
                    1,
                ),
                "avg_speed": round(
                    float(avg_speed),
                    1,
                ),
                "avg_occupancy": round(
                    float(avg_occupancy),
                    1,
                ),
                "congestion_rate": round(
                    float(congestion_rate),
                    3,
                ),
                "high_or_severe_rate": round(
                    float(high_or_severe_rate),
                    3,
                ),
                "severe_rate": round(
                    float(severe_rate),
                    3,
                ),
                "road_segments": segment_ids,
                "detected_at": (datetime.now().isoformat()),
            }
        )

    # ---------------------------------------------------------------
    # Sort:
    # Critical → High → Medium
    # Then highest congestion rate
    # ---------------------------------------------------------------

    return sorted(
        summaries,
        key=lambda item: (
            item["severity"] != "Critical",
            item["severity"] != "High",
            -item["congestion_rate"],
            -item["size"],
        ),
    )
