import pandas as pd

from models.hotspot_detector import detect_hotspots

print("Loading final ML dataset...")

df = pd.read_csv("data/final_dataset/ml_traffic_dataset.csv")

print("FINAL DATASET SHAPE:", df.shape)
print("Running DBSCAN hotspot detection...")

clustered = detect_hotspots(df)

print()
print("CLUSTERED ROAD SEGMENTS:", len(clustered))
print("CLUSTERS:", clustered["cluster_id"].nunique())
print("=" * 100)

for cluster_id, group in clustered.groupby("cluster_id"):

    print()
    print(f"CLUSTER {cluster_id}")
    print("-" * 80)

    print("Road segments:", sorted(group["road_segment_id"].tolist()))

    print(
        "Latitude range:",
        round(group["latitude"].min(), 6),
        "to",
        round(group["latitude"].max(), 6),
    )

    print(
        "Longitude range:",
        round(group["longitude"].min(), 6),
        "to",
        round(group["longitude"].max(), 6),
    )

    print(
        "Center:",
        round(group["latitude"].mean(), 6),
        ",",
        round(group["longitude"].mean(), 6),
    )

    print("Road segments in cluster:", len(group))

    print("Congestion rate:", f"{group['congestion_rate'].mean():.1%}")

print()
print("=" * 100)
print("SPATIAL HOTSPOT VALIDATION COMPLETE")
