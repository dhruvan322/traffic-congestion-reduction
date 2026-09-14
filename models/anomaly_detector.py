"""
Anomaly Detection Module.

Detects unusual traffic patterns in the final integrated ML dataset
using Isolation Forest and Z-score analysis.

Designed for the final project dataset:
    - road_segment_id
    - vehicle_count
    - average_speed_kmh
    - occupancy_percent
    - traffic_flow_vph
    - future_congestion_level
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import ANOMALY_CONTAMINATION, ZSCORE_THRESHOLD

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

ANOMALY_FEATURES = [
    "vehicle_count",
    "average_speed_kmh",
    "occupancy_percent",
    "traffic_flow_vph",
]

REQUIRED_COLUMNS = [
    "road_segment_id",
    "vehicle_count",
    "average_speed_kmh",
    "occupancy_percent",
    "traffic_flow_vph",
    "future_congestion_level",
]

# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------


def _validate_dataset(df: pd.DataFrame) -> None:
    """Validate that the final dataset contains required columns."""

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            "Final traffic dataset is missing required columns: " + ", ".join(missing)
        )


# -------------------------------------------------------------------
# Numeric preparation
# -------------------------------------------------------------------


def _prepare_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare numeric anomaly-detection features.

    Missing numeric values are replaced with the median calculated
    from the supplied dataset.
    """

    features = df[ANOMALY_FEATURES].copy()

    for column in ANOMALY_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        median_value = features[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        features[column] = features[column].fillna(median_value)

    return features


# -------------------------------------------------------------------
# Isolation Forest
# -------------------------------------------------------------------


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect traffic anomalies using Isolation Forest.

    The complete supplied dataset is analyzed.

    Output columns:
        is_anomaly
        anomaly_score

    Higher anomaly_score means the observation is more unusual.
    """

    _validate_dataset(df)

    result = df.copy()

    features = _prepare_numeric_features(result)

    iso = IsolationForest(
        n_estimators=100,
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
        max_samples="auto",
    )

    predictions = iso.fit_predict(features)

    # Isolation Forest:
    #   -1 = anomaly
    #    1 = normal
    result["is_anomaly"] = (predictions == -1).astype(int)

    # decision_function:
    # larger values = more normal
    # therefore negate it so that larger values = more anomalous
    scores = iso.decision_function(features)

    result["anomaly_score"] = np.round(
        -scores,
        6,
    )

    logger.info(
        "Isolation Forest detected %d anomalies out of %d records.",
        int(result["is_anomaly"].sum()),
        len(result),
    )

    return result


# -------------------------------------------------------------------
# Z-score detection
# -------------------------------------------------------------------


def zscore_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect statistical anomalies using absolute Z-scores.

    An observation is flagged when at least one anomaly feature has
    an absolute Z-score greater than ZSCORE_THRESHOLD.
    """

    _validate_dataset(df)

    result = df.copy()

    features = _prepare_numeric_features(result)

    for column in ANOMALY_FEATURES:

        values = features[column]

        mean_value = values.mean()
        std_value = values.std()

        if std_value > 0 and not pd.isna(std_value):

            zscores = np.abs((values - mean_value) / std_value)

        else:

            zscores = pd.Series(
                0.0,
                index=result.index,
            )

        result[f"{column}_zscore"] = np.round(
            zscores,
            6,
        )

    zscore_columns = [f"{column}_zscore" for column in ANOMALY_FEATURES]

    result["zscore_anomaly"] = (
        result[zscore_columns].gt(ZSCORE_THRESHOLD).any(axis=1).astype(int)
    )

    logger.info(
        "Z-score analysis detected %d anomalies out of %d records.",
        int(result["zscore_anomaly"].sum()),
        len(result),
    )

    return result


# -------------------------------------------------------------------
# Risk classification
# -------------------------------------------------------------------


def classify_risk(row: pd.Series) -> str:
    """
    Classify anomaly risk using anomaly status and predicted
    future congestion severity.

    Risk is intentionally tied to the final project's
    five-class congestion terminology.
    """

    congestion = str(
        row.get(
            "future_congestion_level",
            "Free Flow",
        )
    )

    anomaly_score = float(
        row.get(
            "anomaly_score",
            0.0,
        )
    )

    combined = int(
        row.get(
            "combined_anomaly",
            0,
        )
    )

    if combined == 0:
        return "Normal"

    # Severe congestion combined with an anomaly
    # represents the highest operational risk.
    if congestion == "Severe":
        return "Critical"

    if congestion == "High":
        return "High"

    if congestion == "Moderate":
        return "Medium"

    # Very unusual Free Flow / Low traffic observations
    # remain lower-risk anomalies.
    if anomaly_score >= 0.20:
        return "Medium"

    return "Low"


# -------------------------------------------------------------------
# Anomaly description
# -------------------------------------------------------------------


def _describe_anomaly(row: pd.Series) -> str:
    """Generate a deterministic explanation for an anomaly."""

    road_segment = row.get(
        "road_segment_id",
        "Unknown",
    )

    vehicle_count = float(row.get("vehicle_count", 0))

    speed = float(row.get("average_speed_kmh", 0))

    occupancy = float(row.get("occupancy_percent", 0))

    flow = float(row.get("traffic_flow_vph", 0))

    congestion = row.get(
        "future_congestion_level",
        "Unknown",
    )

    return (
        f"Unusual traffic pattern detected on "
        f"{road_segment}: {vehicle_count:.0f} vehicles, "
        f"{speed:.1f} km/h average speed, "
        f"{occupancy:.1f}% occupancy, "
        f"{flow:.0f} vehicles/hour flow. "
        f"Future congestion level: {congestion}."
    )


# -------------------------------------------------------------------
# Full anomaly report
# -------------------------------------------------------------------


def get_anomaly_report(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Run the complete anomaly-detection pipeline.

    Pipeline:

        Final Dataset
              ↓
        Isolation Forest
              +
           Z-score
              ↓
       Combined anomaly
              ↓
        Risk classification
              ↓
        Structured report

    Returns anomaly records sorted by anomaly score.
    """

    _validate_dataset(df)

    df_iso = detect_anomalies(df)

    df_detected = zscore_anomalies(df_iso)

    # An observation is considered anomalous if either
    # detection method flags it.
    df_detected["combined_anomaly"] = (
        (df_detected["is_anomaly"] == 1) | (df_detected["zscore_anomaly"] == 1)
    ).astype(int)

    anomalies_df = df_detected[df_detected["combined_anomaly"] == 1].copy()

    if anomalies_df.empty:
        logger.info("No combined anomalies detected.")
        return []

    anomalies_df["risk_level"] = anomalies_df.apply(
        classify_risk,
        axis=1,
    )

    records = []

    for _, row in anomalies_df.iterrows():

        timestamp = row.get(
            "timestamp",
            "",
        )

        if pd.isna(timestamp):
            timestamp = ""

        record = {
            "road_segment_id": str(row["road_segment_id"]),
            "timestamp": str(timestamp),
            "vehicle_count": int(round(float(row["vehicle_count"]))),
            "average_speed_kmh": float(row["average_speed_kmh"]),
            "occupancy_percent": float(row["occupancy_percent"]),
            "traffic_flow_vph": float(row["traffic_flow_vph"]),
            "future_congestion_level": str(row["future_congestion_level"]),
            "isolation_forest_anomaly": int(row["is_anomaly"]),
            "zscore_anomaly": int(row["zscore_anomaly"]),
            "anomaly_score": float(row["anomaly_score"]),
            "risk_level": str(row["risk_level"]),
            "anomaly_type": ("isolation_forest+zscore"),
            "detected_at": datetime.now().isoformat(),
            "description": _describe_anomaly(row),
        }

        records.append(record)

    records.sort(
        key=lambda record: record["anomaly_score"],
        reverse=True,
    )

    logger.info(
        "Combined anomaly detection produced %d anomaly records.",
        len(records),
    )

    return records


# -------------------------------------------------------------------
# Summary helper
# -------------------------------------------------------------------


def get_anomaly_summary(
    df_detected: pd.DataFrame,
) -> dict:
    """
    Calculate summary statistics from a dataframe that has already
    passed through detect_anomalies() and zscore_anomalies().
    """

    total_records = len(df_detected)

    isolation_count = int(df_detected["is_anomaly"].sum())

    zscore_count = int(df_detected["zscore_anomaly"].sum())

    if "combined_anomaly" in df_detected.columns:

        combined_count = int(df_detected["combined_anomaly"].sum())

    else:

        combined_count = int(
            (
                (df_detected["is_anomaly"] == 1) | (df_detected["zscore_anomaly"] == 1)
            ).sum()
        )

    anomaly_rate = combined_count / total_records * 100 if total_records > 0 else 0.0

    return {
        "total_records": total_records,
        "isolation_forest_anomalies": isolation_count,
        "zscore_anomalies": zscore_count,
        "combined_anomalies": combined_count,
        "anomaly_rate_percent": round(
            anomaly_rate,
            2,
        ),
    }
