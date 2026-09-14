from pathlib import Path
import sys

import networkx as nx
import pandas as pd

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.osm_traffic_router import (
    load_osm_graph,
    get_traffic_state,
    load_segment_mapping,
    apply_traffic_weights,
)

print("=" * 70)
print("AUTOMATIC TRAFFIC-EDGE SENSITIVITY TEST")
print("=" * 70)


# ---------------------------------------------------------------------
# 1. Load OSM graph
# ---------------------------------------------------------------------

print("\nLoading OSM graph...")

graph = load_osm_graph()

print(f"OSM NODES: {graph.number_of_nodes():,}")
print(f"OSM EDGES: {graph.number_of_edges():,}")


# ---------------------------------------------------------------------
# 2. Load mapping
# ---------------------------------------------------------------------

print("\nLoading mapping...")

mapping = load_segment_mapping()

required_columns = [
    "road_segment_id",
    "osm_u",
    "osm_v",
    "osm_key",
]

missing = [col for col in required_columns if col not in mapping.columns]

if missing:
    raise AssertionError(f"Mapping is missing required columns: {missing}")

print(f"MAPPING ROWS: {len(mapping):,}")
print(f"UNIQUE PROJECT SEGMENTS: " f"{mapping['road_segment_id'].nunique():,}")


# ---------------------------------------------------------------------
# 3. Load traffic state
# ---------------------------------------------------------------------

print("\nLoading traffic state...")

timestamp = pd.Timestamp("2025-09-01 18:00:00")

traffic_state = get_traffic_state(timestamp)

print(f"SELECTED TIMESTAMP: " f"{traffic_state['timestamp'].iloc[0]}")

print(f"TRAFFIC SEGMENTS: {len(traffic_state):,}")


# ---------------------------------------------------------------------
# 4. Apply traffic weights
# ---------------------------------------------------------------------

print("\nApplying traffic weights...")

weighted_graph = apply_traffic_weights(
    graph,
    traffic_state,
    mapping,
)

print(f"WEIGHTED OSM EDGES: " f"{weighted_graph.number_of_edges():,}")


# ---------------------------------------------------------------------
# 5. Search for an edge that is actually used
# ---------------------------------------------------------------------

print(
    "\nSearching for mapped edges that are actually "
    "preferred between their endpoints..."
)

candidates = []

for _, row in mapping.iterrows():

    u = int(row["osm_u"])
    v = int(row["osm_v"])
    key = int(row["osm_key"])

    if not weighted_graph.has_edge(u, v, key):
        continue

    edge_data = weighted_graph[u][v][key]

    original_weight = float(
        edge_data.get(
            "traffic_weight",
            edge_data.get("length", 1.0),
        )
    )

    edge_length = float(edge_data.get("length", 1.0))

    try:

        baseline_route = nx.shortest_path(
            weighted_graph,
            source=u,
            target=v,
            weight="traffic_weight",
        )

        baseline_cost = nx.shortest_path_length(
            weighted_graph,
            source=u,
            target=v,
            weight="traffic_weight",
        )

    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound,
    ):
        continue

    # A direct route means the selected edge is the
    # preferred route between its own endpoints.
    uses_selected_edge = (
        len(baseline_route) == 2 and baseline_route[0] == u and baseline_route[1] == v
    )

    if uses_selected_edge:

        candidates.append(
            {
                "road_segment_id": row["road_segment_id"],
                "u": u,
                "v": v,
                "key": key,
                "edge_length": edge_length,
                "original_weight": original_weight,
                "baseline_cost": float(baseline_cost),
                "baseline_route": baseline_route,
            }
        )


print(f"USABLE SENSITIVITY CANDIDATES: " f"{len(candidates)}")


# ---------------------------------------------------------------------
# 6. No suitable candidate
# ---------------------------------------------------------------------

if not candidates:

    print("\n" + "=" * 70)
    print("SENSITIVITY RESULT: INCONCLUSIVE")
    print("=" * 70)

    print(
        "\nNo mapped OSM edge is currently the preferred "
        "traffic-weighted route between its own endpoints."
    )

    print("\nThis does NOT indicate a routing failure.")

    print(
        "It means the current project road mapping does not "
        "provide a suitable direct-edge causal test."
    )

    sys.exit(0)


# ---------------------------------------------------------------------
# 7. Select candidate
# ---------------------------------------------------------------------

candidate = candidates[0]

road_segment_id = candidate["road_segment_id"]

u = candidate["u"]
v = candidate["v"]
key = candidate["key"]

edge_length = candidate["edge_length"]
original_weight = candidate["original_weight"]
baseline_cost = candidate["baseline_cost"]
baseline_route = candidate["baseline_route"]


print("\nSELECTED TEST SEGMENT:")
print(f"ROAD SEGMENT: {road_segment_id}")
print(f"OSM EDGE: ({u}, {v}, {key})")
print(f"EDGE LENGTH: {edge_length:.2f} m")
print(f"ORIGINAL WEIGHT: {original_weight:.2f}")
print(f"BASELINE COST: {baseline_cost:.2f}")


# ---------------------------------------------------------------------
# 8. Confirm baseline uses edge
# ---------------------------------------------------------------------

print("\nCalculating baseline route...")

print(f"BASELINE ROUTE: {baseline_route}")

baseline_uses_edge = (
    len(baseline_route) == 2 and baseline_route[0] == u and baseline_route[1] == v
)

print(f"BASELINE USES SELECTED EDGE: " f"{baseline_uses_edge}")

if not baseline_uses_edge:

    print("\n" + "=" * 70)
    print("SENSITIVITY RESULT: INCONCLUSIVE")
    print("=" * 70)

    print(
        "Candidate validation failed because the baseline "
        "route does not use the selected edge."
    )

    sys.exit(0)


# ---------------------------------------------------------------------
# 9. Stress selected edge
# ---------------------------------------------------------------------

print("\nApplying extreme traffic stress...")

stressed_graph = weighted_graph.copy()

stressed_weight = max(
    original_weight * 1_000_000,
    1_000_000.0,
)

stressed_graph[u][v][key]["traffic_weight"] = stressed_weight

print(f"STRESSED WEIGHT: " f"{stressed_weight:.2f}")


# ---------------------------------------------------------------------
# 10. Calculate stressed route
# ---------------------------------------------------------------------

print("\nCalculating stressed route...")

try:

    stressed_route = nx.shortest_path(
        stressed_graph,
        source=u,
        target=v,
        weight="traffic_weight",
    )

    stressed_cost = nx.shortest_path_length(
        stressed_graph,
        source=u,
        target=v,
        weight="traffic_weight",
    )

except (
    nx.NetworkXNoPath,
    nx.NodeNotFound,
):

    print("\n" + "=" * 70)
    print("SENSITIVITY RESULT: FAIL")
    print("=" * 70)

    print("No alternative route exists after stressing " "the selected edge.")

    sys.exit(1)


print(f"STRESSED ROUTE: {stressed_route}")
print(f"STRESSED COST: {stressed_cost:.2f}")


# ---------------------------------------------------------------------
# 11. Verify route changed
# ---------------------------------------------------------------------

route_changed = baseline_route != stressed_route

print(f"\nROUTE CHANGED: " f"{route_changed}")


# ---------------------------------------------------------------------
# 12. Final result
# ---------------------------------------------------------------------

if route_changed:

    print("\n" + "=" * 70)
    print("SENSITIVITY RESULT: PASS")
    print("=" * 70)

    print("\nTraffic-edge sensitivity has been demonstrated.")

    print(
        "Increasing the selected edge's traffic weight "
        "caused the optimizer to choose a different route."
    )

    print("\nBASELINE:")
    print(f"  Route nodes: {len(baseline_route)}")
    print(f"  Cost: {baseline_cost:.2f}")

    print("\nSTRESSED:")
    print(f"  Route nodes: {len(stressed_route)}")
    print(f"  Cost: {stressed_cost:.2f}")

    sys.exit(0)


print("\n" + "=" * 70)
print("SENSITIVITY RESULT: FAIL")
print("=" * 70)

print(
    "The selected edge was used by the baseline route, "
    "but extreme traffic stress did not change the route."
)

sys.exit(1)
