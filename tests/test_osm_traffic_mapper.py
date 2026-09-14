import os
import sys

import pandas as pd

# Allow imports from the project root when this test is run directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from engine.osm_traffic_mapper import (
    load_hyderabad_osm_graph,
    map_road_segments_to_osm,
    get_mapping_summary,
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "final_dataset",
    "road_segments.csv",
)


print("=" * 70)
print("OSM TRAFFIC SEGMENT MAPPING TEST")
print("=" * 70)

print("\nLoading project road segments...")

df = pd.read_csv(DATASET_PATH)

print("ROAD SEGMENTS:", len(df))
print("COLUMNS:", len(df.columns))

print("\nLoading Hyderabad OSM network...")

G = load_hyderabad_osm_graph()

print("OSM NODES:", len(G.nodes))
print("OSM EDGES:", len(G.edges))

print("\nMapping road segments to OSM edges...")

mapping_df = map_road_segments_to_osm(
    df,
    G,
)

summary = get_mapping_summary(mapping_df)

print("\n" + "=" * 70)
print("MAPPING SUMMARY")
print("=" * 70)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\n" + "=" * 70)
print("SEGMENT MAPPINGS")
print("=" * 70)

display_columns = [
    "road_segment_id",
    "road_name",
    "latitude",
    "longitude",
    "osm_u",
    "osm_v",
    "osm_key",
    "osm_distance_m",
    "mapping_confidence",
    "osm_name",
    "osm_highway",
]

print(mapping_df[display_columns].sort_values("osm_distance_m").to_string(index=False))

print("\n" + "=" * 70)
print("QUALITY CHECKS")
print("=" * 70)

assert len(mapping_df) == len(df)

assert mapping_df["osm_u"].notna().all()
assert mapping_df["osm_v"].notna().all()
assert mapping_df["osm_key"].notna().all()
assert mapping_df["osm_distance_m"].notna().all()

assert set(mapping_df["mapping_confidence"]).issubset({"High", "Medium", "Low"})

print("ROW COUNT CHECK: PASS")
print("OSM EDGE ID CHECK: PASS")
print("DISTANCE CHECK: PASS")
print("CONFIDENCE CHECK: PASS")

print("\nOSM TRAFFIC MAPPER TEST: PASS")
