"""
Traffic-aware route sensitivity test.

Purpose:
Verify that traffic weights actually influence route selection.

The test:
1. Calculates the baseline traffic-aware route.
2. Finds a project-mapped OSM edge that is actually used by that route.
3. Applies an extreme congestion penalty to that edge.
4. Recalculates the route.
5. Verifies that the stressed edge is avoided.
"""

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
    calculate_route,
)


def main():

    print("=" * 70)
    print("TRAFFIC ROUTE SENSITIVITY TEST")
    print("=" * 70)

    origin = (
        17.4374,
        78.4482,
    )

    destination = (
        17.4401,
        78.3489,
    )

    timestamp = "2025-09-01 18:00:00"

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------

    print("\nLoading OSM graph...")

    graph = load_osm_graph()

    print("\nLoading OSM mapping...")

    mapping = load_segment_mapping()

    print("\nLoading traffic state...")

    traffic = get_traffic_state(timestamp)

    # ---------------------------------------------------------------
    # 2. Build baseline traffic graph
    # ---------------------------------------------------------------

    print("\nCalculating baseline traffic-aware route...")

    baseline_graph = apply_traffic_weights(
        graph,
        traffic,
        mapping,
    )

    baseline_route = calculate_route(
        baseline_graph,
        origin,
        destination,
        weight="traffic_weight",
    )

    print(
        "BASELINE ROUTE NODES:",
        len(baseline_route),
    )

    assert len(baseline_route) > 1

    # ---------------------------------------------------------------
    # 3. Identify mapped edges on baseline route
    # ---------------------------------------------------------------

    baseline_edges = set(
        zip(
            baseline_route[:-1],
            baseline_route[1:],
        )
    )

    mapped_route_edges = []

    for _, row in mapping.iterrows():

        edge = (
            int(row["osm_u"]),
            int(row["osm_v"]),
            int(row["osm_key"]),
        )

        edge_pair = (
            edge[0],
            edge[1],
        )

        if edge_pair in baseline_edges:

            mapped_route_edges.append(
                {
                    "road_segment_id": row["road_segment_id"],
                    "osm_u": edge[0],
                    "osm_v": edge[1],
                    "osm_key": edge[2],
                }
            )

    print(
        "\nMAPPED PROJECT EDGES ON BASELINE ROUTE:",
        len(mapped_route_edges),
    )

    if not mapped_route_edges:

        print("\nSENSITIVITY RESULT: INCONCLUSIVE")

        print(
            "The baseline route contains no OSM edges "
            "that are mapped to project road segments."
        )

        print(
            "This means the current project traffic coverage "
            "is insufficient for an edge-avoidance test "
            "using this origin/destination."
        )

        return

    # ---------------------------------------------------------------
    # 4. Select a mapped route edge
    # ---------------------------------------------------------------

    test_edge = mapped_route_edges[0]

    test_segment = test_edge["road_segment_id"]

    test_u = test_edge["osm_u"]

    test_v = test_edge["osm_v"]

    test_key = test_edge["osm_key"]

    print(
        "\nSELECTED TEST SEGMENT:",
        test_segment,
    )

    print(
        "TEST OSM EDGE:",
        (test_u, test_v, test_key),
    )

    # ---------------------------------------------------------------
    # 5. Confirm edge exists
    # ---------------------------------------------------------------

    assert baseline_graph.has_edge(
        test_u,
        test_v,
        test_key,
    )

    edge_data = baseline_graph[test_u][test_v][test_key]

    original_weight = float(edge_data["traffic_weight"])

    edge_length = float(
        edge_data.get(
            "length",
            1.0,
        )
    )

    print(
        "ORIGINAL TRAFFIC WEIGHT:",
        original_weight,
    )

    # ---------------------------------------------------------------
    # 6. Apply extreme congestion
    # ---------------------------------------------------------------

    stress_graph = baseline_graph.copy()

    stressed_edge = stress_graph[test_u][test_v][test_key]

    stressed_edge["traffic_weight"] = edge_length * 100000.0

    print(
        "STRESSED TRAFFIC WEIGHT:",
        stressed_edge["traffic_weight"],
    )

    # ---------------------------------------------------------------
    # 7. Calculate stressed route
    # ---------------------------------------------------------------

    print("\nCalculating route after extreme congestion...")

    stressed_route = calculate_route(
        stress_graph,
        origin,
        destination,
        weight="traffic_weight",
    )

    print(
        "STRESSED ROUTE NODES:",
        len(stressed_route),
    )

    assert len(stressed_route) > 1

    # ---------------------------------------------------------------
    # 8. Compare routes
    # ---------------------------------------------------------------

    stressed_edges = set(
        zip(
            stressed_route[:-1],
            stressed_route[1:],
        )
    )

    test_edge_pair = (
        test_u,
        test_v,
    )

    route_changed = baseline_route != stressed_route

    stressed_edge_avoided = test_edge_pair not in stressed_edges

    print(
        "\nROUTE CHANGED:",
        route_changed,
    )

    print(
        "STRESSED EDGE AVOIDED:",
        stressed_edge_avoided,
    )

    # ---------------------------------------------------------------
    # 9. Final validation
    # ---------------------------------------------------------------

    assert route_changed, (
        "Route did not change after an extreme "
        "penalty was applied to an edge that was "
        "part of the baseline route."
    )

    assert stressed_edge_avoided, (
        "The extremely penalized OSM edge was still " "used by the stressed route."
    )

    print("\nSENSITIVITY RESULT: PASS")

    print("Traffic weighting successfully influenced " "route selection.")

    print("\n" + "=" * 70)
    print("TRAFFIC ROUTE SENSITIVITY TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
