"""
Backend test for Traffic-Aware OSM Routing.

This test verifies:

1. Final dataset loads.
2. OSM mapping loads.
3. Hyderabad OSM graph loads.
4. Traffic state can be selected.
5. Traffic penalties are generated.
6. Traffic weights are applied.
7. Shortest route works.
8. Traffic-aware route works.
9. Route metrics are generated.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

import pandas as pd

from engine.osm_traffic_router import (
    DATASET_PATH,
    MAPPING_PATH,
    load_osm_graph,
    load_traffic_dataset,
    load_segment_mapping,
    get_traffic_state,
    calculate_traffic_penalty,
    apply_traffic_weights,
    calculate_route,
    calculate_route_metrics,
    compare_osm_routes,
)


def main():

    print("=" * 70)
    print("TRAFFIC-AWARE OSM ROUTING TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. DATASET
    # ---------------------------------------------------------------

    print("\nLoading final traffic dataset...")

    df = load_traffic_dataset()

    print("DATASET SHAPE:", df.shape)
    print(
        "TIMESTAMP RANGE:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    assert len(df) > 0
    assert "road_segment_id" in df.columns
    assert "future_congestion_level" in df.columns

    print("DATASET CHECK: PASS")

    # ---------------------------------------------------------------
    # 2. OSM MAPPING
    # ---------------------------------------------------------------

    print("\nLoading OSM segment mapping...")

    mapping = load_segment_mapping()

    print("MAPPING ROWS:", len(mapping))
    print(
        "UNIQUE PROJECT SEGMENTS:",
        mapping["road_segment_id"].nunique(),
    )

    assert len(mapping) == 60
    assert mapping["road_segment_id"].nunique() == 60

    print("MAPPING CHECK: PASS")

    # ---------------------------------------------------------------
    # 3. OSM NETWORK
    # ---------------------------------------------------------------

    print("\nLoading Hyderabad OSM network...")

    graph = load_osm_graph()

    print("OSM NODES:", len(graph.nodes))
    print("OSM EDGES:", len(graph.edges))

    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

    print("OSM GRAPH CHECK: PASS")

    # ---------------------------------------------------------------
    # 4. TRAFFIC STATE
    # ---------------------------------------------------------------

    test_timestamp = "2025-09-01 18:00:00"

    print(
        "\nSelecting traffic state:",
        test_timestamp,
    )

    traffic = get_traffic_state(test_timestamp)

    print(
        "SELECTED TIMESTAMP:",
        traffic["timestamp"].iloc[0],
    )

    print(
        "TRAFFIC SEGMENTS:",
        len(traffic),
    )

    assert len(traffic) == 60
    assert traffic["road_segment_id"].nunique() == 60

    print("TRAFFIC STATE CHECK: PASS")

    # ---------------------------------------------------------------
    # 5. TRAFFIC PENALTY
    # ---------------------------------------------------------------

    print("\nTesting traffic penalty calculation...")

    sample = traffic.iloc[0]

    penalty = calculate_traffic_penalty(sample)

    print(
        "SAMPLE SEGMENT:",
        sample["road_segment_id"],
    )

    print(
        "CONGESTION:",
        sample["future_congestion_level"],
    )

    print(
        "AVERAGE SPEED:",
        sample["average_speed_kmh"],
    )

    print(
        "OCCUPANCY:",
        sample["occupancy_percent"],
    )

    print(
        "TRAFFIC PENALTY:",
        round(penalty, 4),
    )

    assert penalty > 0

    print("TRAFFIC PENALTY CHECK: PASS")

    # ---------------------------------------------------------------
    # 6. APPLY TRAFFIC WEIGHTS
    # ---------------------------------------------------------------

    print("\nApplying traffic weights to OSM network...")

    traffic_graph = apply_traffic_weights(
        graph,
        traffic,
        mapping,
    )

    weighted_edges = 0

    for u, v, key, data in traffic_graph.edges(
        keys=True,
        data=True,
    ):
        if "traffic_weight" in data:
            weighted_edges += 1

    print(
        "OSM EDGES WITH TRAFFIC WEIGHTS:",
        weighted_edges,
    )

    assert weighted_edges > 0

    print("TRAFFIC WEIGHT CHECK: PASS")

    # ---------------------------------------------------------------
    # 7. ROUTING
    # ---------------------------------------------------------------

    origin = (
        17.4374,
        78.4482,
    )

    destination = (
        17.4401,
        78.3489,
    )

    print("\nOrigin:", origin)
    print("Destination:", destination)

    print("\nCalculating shortest OSM route...")

    shortest_route = calculate_route(
        graph,
        origin,
        destination,
        weight="length",
    )

    print(
        "SHORTEST ROUTE NODES:",
        len(shortest_route),
    )

    assert len(shortest_route) > 1

    print("SHORTEST ROUTE CHECK: PASS")

    print("\nCalculating traffic-aware OSM route...")

    traffic_route = calculate_route(
        traffic_graph,
        origin,
        destination,
        weight="traffic_weight",
    )

    print(
        "TRAFFIC-AWARE ROUTE NODES:",
        len(traffic_route),
    )

    assert len(traffic_route) > 1

    print("TRAFFIC-AWARE ROUTE CHECK: PASS")

    # ---------------------------------------------------------------
    # 8. METRICS
    # ---------------------------------------------------------------

    print("\nCalculating route metrics...")

    shortest_metrics = calculate_route_metrics(
        traffic_graph,
        shortest_route,
    )

    traffic_metrics = calculate_route_metrics(
        traffic_graph,
        traffic_route,
    )

    print("\nSHORTEST ROUTE METRICS")
    print("-" * 40)

    for key, value in shortest_metrics.items():
        print(f"{key}: {value}")

    print("\nTRAFFIC-AWARE ROUTE METRICS")
    print("-" * 40)

    for key, value in traffic_metrics.items():
        print(f"{key}: {value}")

    assert shortest_metrics["distance_km"] > 0
    assert traffic_metrics["distance_km"] > 0

    print("\nROUTE METRICS CHECK: PASS")

    # ---------------------------------------------------------------
    # 9. END-TO-END
    # ---------------------------------------------------------------

    print("\nRunning end-to-end route comparison...")

    result = compare_osm_routes(
        origin,
        destination,
        test_timestamp,
    )

    assert "shortest_route" in result
    assert "traffic_aware_route" in result
    assert "shortest_metrics" in result
    assert "traffic_aware_metrics" in result

    print("END-TO-END CHECK: PASS")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAFFIC-AWARE OSM ROUTING TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
