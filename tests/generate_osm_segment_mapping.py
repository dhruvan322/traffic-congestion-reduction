"""
Generate and save the project road-segment → OSM-edge mapping.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.osm_traffic_mapper import (
    load_hyderabad_osm_graph,
    map_road_segments_to_osm,
    get_mapping_summary,
    save_osm_mapping,
)

DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "road_segments.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "final_dataset" / "osm_segment_mapping.csv"


def main():

    print("=" * 70)
    print("GENERATING OSM SEGMENT MAPPING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load project road segments
    # ---------------------------------------------------------------

    print("\nLoading project road segments...")

    road_segments = pd.read_csv(DATASET_PATH)

    print(
        "ROAD SEGMENTS:",
        len(road_segments),
    )

    # ---------------------------------------------------------------
    # Load OSM
    # ---------------------------------------------------------------

    print("\nLoading Hyderabad OSM network...")

    graph = load_hyderabad_osm_graph()

    print(
        "OSM NODES:",
        len(graph.nodes),
    )

    print(
        "OSM EDGES:",
        len(graph.edges),
    )

    # ---------------------------------------------------------------
    # Map
    # ---------------------------------------------------------------

    print("\nMapping road segments to OSM edges...")

    mapping = map_road_segments_to_osm(
        road_segments,
        graph,
    )

    print(
        "MAPPED SEGMENTS:",
        len(mapping),
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = get_mapping_summary(mapping)

    print("\n" + "=" * 70)
    print("MAPPING SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    print("\nSaving mapping...")

    save_osm_mapping(
        mapping,
        str(OUTPUT_PATH),
    )

    print(
        "SAVED TO:",
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------------
    # Verify
    # ---------------------------------------------------------------

    print("\nVerifying saved file...")

    saved = pd.read_csv(OUTPUT_PATH)

    print(
        "SAVED ROWS:",
        len(saved),
    )

    print(
        "SAVED COLUMNS:",
        len(saved.columns),
    )

    print(
        "UNIQUE ROAD SEGMENTS:",
        saved["road_segment_id"].nunique(),
    )

    assert len(saved) == 60
    assert saved["road_segment_id"].nunique() == 60
    assert saved["osm_u"].notna().all()
    assert saved["osm_v"].notna().all()
    assert saved["osm_key"].notna().all()

    print("\n" + "=" * 70)
    print("OSM SEGMENT MAPPING GENERATION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
