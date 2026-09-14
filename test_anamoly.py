import pandas as pd

from models.anomaly_detector import (
    detect_anomalies,
    zscore_anomalies,
    get_anomaly_report,
    get_anomaly_summary,
)

DATASET_PATH = "data/final_dataset/ml_traffic_dataset.csv"


print("=" * 70)
print("LOADING FINAL TRAFFIC DATASET")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("DATASET SHAPE:", df.shape)
print("ROAD SEGMENTS:", df["road_segment_id"].nunique())
print(
    "TIMESTAMP RANGE:",
    df["timestamp"].min(),
    "->",
    df["timestamp"].max(),
)

print()


print("=" * 70)
print("RUNNING ISOLATION FOREST")
print("=" * 70)

df_detected = detect_anomalies(df)

print(
    "ISOLATION FOREST ANOMALIES:",
    int(df_detected["is_anomaly"].sum()),
)

print(
    "ISOLATION FOREST ANOMALY RATE:", f"{df_detected['is_anomaly'].mean() * 100:.2f}%"
)

print()


print("=" * 70)
print("RUNNING Z-SCORE ANALYSIS")
print("=" * 70)

df_detected = zscore_anomalies(df_detected)

print(
    "Z-SCORE ANOMALIES:",
    int(df_detected["zscore_anomaly"].sum()),
)

print("Z-SCORE ANOMALY RATE:", f"{df_detected['zscore_anomaly'].mean() * 100:.2f}%")

print()


print("=" * 70)
print("COMBINING DETECTION METHODS")
print("=" * 70)

df_detected["combined_anomaly"] = (
    (df_detected["is_anomaly"] == 1) | (df_detected["zscore_anomaly"] == 1)
).astype(int)

print(
    "COMBINED ANOMALIES:",
    int(df_detected["combined_anomaly"].sum()),
)

print("COMBINED ANOMALY RATE:", f"{df_detected['combined_anomaly'].mean() * 100:.2f}%")

print()


print("=" * 70)
print("ANOMALY SUMMARY")
print("=" * 70)

summary = get_anomaly_summary(df_detected)

for key, value in summary.items():
    print(f"{key}: {value}")

print()


print("=" * 70)
print("GENERATING ANOMALY REPORT")
print("=" * 70)

anomaly_records = get_anomaly_report(df)

print(
    "TOTAL ANOMALY RECORDS:",
    len(anomaly_records),
)

print()


print("=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

if anomaly_records:

    risk_counts = pd.Series(
        [record["risk_level"] for record in anomaly_records]
    ).value_counts()

    for risk, count in risk_counts.items():

        print(f"{risk}: {count}")

else:

    print("No anomalies detected.")

print()


print("=" * 70)
print("TOP 10 ANOMALIES")
print("=" * 70)

for index, record in enumerate(
    anomaly_records[:10],
    start=1,
):

    print(
        f"{index}. "
        f"Road={record['road_segment_id']} | "
        f"Risk={record['risk_level']} | "
        f"Congestion={record['future_congestion_level']} | "
        f"Vehicles={record['vehicle_count']} | "
        f"Speed={record['average_speed_kmh']:.1f} km/h | "
        f"Occupancy={record['occupancy_percent']:.1f}% | "
        f"Flow={record['traffic_flow_vph']:.0f} | "
        f"Score={record['anomaly_score']:.4f}"
    )

print()


print("=" * 70)
print("ANOMALY TYPE CROSS-CHECK")
print("=" * 70)

isolation_only = (
    (df_detected["is_anomaly"] == 1) & (df_detected["zscore_anomaly"] == 0)
).sum()

zscore_only = (
    (df_detected["is_anomaly"] == 0) & (df_detected["zscore_anomaly"] == 1)
).sum()

both_methods = (
    (df_detected["is_anomaly"] == 1) & (df_detected["zscore_anomaly"] == 1)
).sum()

print(
    "Isolation Forest only:",
    int(isolation_only),
)

print(
    "Z-score only:",
    int(zscore_only),
)

print(
    "Both methods:",
    int(both_methods),
)

print()

print("=" * 70)
print("ANOMALY DETECTION TEST COMPLETE")
print("=" * 70)
