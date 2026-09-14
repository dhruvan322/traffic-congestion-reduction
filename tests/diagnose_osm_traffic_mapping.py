import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.osm_traffic_router import (
    load_osm_graph,
    load_segment_mapping,
    get_traffic_state,
    apply_traffic_weights,
)


def main():

    print("=" * 70)
    print("OSM TRAFFIC EDGE DIAGNOSTIC")
    print("=" * 70)

    mapping = load_segment_mapping()

    print("\nPROJECT SEGMENTS:", len(mapping))
    print(
        "UNIQUE OSM EDGES:",
        mapping[["osm_u", "osm_v", "osm_key"]].drop_duplicates().shape[0],
    )

    duplicate_edges = mapping.groupby(["osm_u", "osm_v", "osm_key"])[
        "road_segment_id"
    ].apply(list)

    duplicate_edges = duplicate_edges[duplicate_edges.apply(len) > 1]

    print(
        "\nOSM EDGES SHARED BY MULTIPLE PROJECT SEGMENTS:",
        len(duplicate_edges),
    )

    if len(duplicate_edges) > 0:

        print("\nSHARED EDGES:")

        for edge, segments in duplicate_edges.items():

            print(
                edge,
                "->",
                segments,
            )

    print("\nLoading OSM graph...")

    graph = load_osm_graph()

    traffic = get_traffic_state("2025-09-01 18:00:00")

    print(
        "\nTRAFFIC SEGMENTS:",
        len(traffic),
    )

    traffic_graph = apply_traffic_weights(
        graph,
        traffic,
        mapping,
    )

    mapped_edges = []

    for u, v, key, data in traffic_graph.edges(
        keys=True,
        data=True,
    ):

        if "traffic_weight" in data:

            mapped_edges.append((u, v, key))

    print(
        "TRAFFIC-WEIGHTED OSM EDGES:",
        len(mapped_edges),
    )

    mapping_edges = set(
        zip(
            mapping["osm_u"].astype(int),
            mapping["osm_v"].astype(int),
            mapping["osm_key"].astype(int),
        )
    )

    actual_edges = set(mapped_edges)

    missing_from_graph = mapping_edges - actual_edges

    print(
        "\nMAPPED EDGES NOT FOUND/WEIGHTED:",
        len(missing_from_graph),
    )

    if missing_from_graph:

        print("\nMISSING EDGES:")

        for edge in sorted(missing_from_graph):

            segments = mapping[
                (mapping["osm_u"].astype(int) == edge[0])
                & (mapping["osm_v"].astype(int) == edge[1])
                & (mapping["osm_key"].astype(int) == edge[2])
            ]["road_segment_id"].tolist()

            print(
                edge,
                "->",
                segments,
            )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
