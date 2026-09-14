"""
Traffic-Aware OSM Route Engine

Layer 2 of the Traffic Diversion & OSM Route Optimization system.

This module:
1. Loads the Hyderabad OSM driving network.
2. Loads traffic conditions from the final project dataset.
3. Maps project road segments to nearby OSM edges.
4. Builds traffic-aware edge weights.
5. Computes:
   - Shortest OSM route
   - Traffic-aware OSM route
6. Returns comparable route metrics.

Important:
The project road segments are associated with OSM edges spatially.
The CSV does not contain official OSM way IDs, so this is an
association rather than an exact official road identity match.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

OSM_PLACE = "Hyderabad, Telangana, India"
OSM_NETWORK_TYPE = "drive"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset" / "ml_traffic_dataset.csv"

MAPPING_PATH = PROJECT_ROOT / "data" / "final_dataset" / "osm_segment_mapping.csv"

REQUIRED_TRAFFIC_COLUMNS = [
    "road_segment_id",
    "timestamp",
    "vehicle_count",
    "average_speed_kmh",
    "occupancy_percent",
    "traffic_flow_vph",
    "travel_time_minutes",
    "future_congestion_level",
    "number_of_lanes",
    "speed_limit_kmh",
    "road_length_km",
]

CONGESTION_PENALTY = {
    "Free Flow": 1.00,
    "Low": 1.10,
    "Moderate": 1.35,
    "High": 1.75,
    "Severe": 2.50,
}


# ---------------------------------------------------------------------
# OSM NETWORK
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_osm_graph() -> nx.MultiDiGraph:
    """
    Load the Hyderabad OSM driving network.

    Loading priority:
        1. In-memory lru_cache
        2. Local binary pickle cache
        3. Local GraphML cache
        4. Fresh OSM download

    The GraphML file remains the portable source-of-truth.
    """

    import pickle

    graph_dir = PROJECT_ROOT / "data" / "osm"
    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_path = graph_dir / "hyderabad_drive.graphml"
    pickle_path = graph_dir / "hyderabad_drive.pkl"

    # -------------------------------------------------------------
    # 1. Fast binary cache
    # -------------------------------------------------------------

    if pickle_path.exists():

        print(
            f"Loading binary-cached Hyderabad OSM network "
            f"from {pickle_path}..."
        )

        try:

            with open(
                pickle_path,
                "rb",
            ) as f:
                graph = pickle.load(f)

            if (
                len(graph.nodes) > 0
                and len(graph.edges) > 0
            ):
                print(
                    f"OSM NODES: {len(graph.nodes):,}"
                )

                print(
                    f"OSM EDGES: {len(graph.edges):,}"
                )

                return graph

        except Exception as exc:

            print(
                "Binary OSM cache could not be loaded. "
                f"Falling back to GraphML: {exc}"
            )

    # -------------------------------------------------------------
    # 2. GraphML cache
    # -------------------------------------------------------------

    if graph_path.exists():

        print(
            f"Loading cached Hyderabad OSM network "
            f"from {graph_path}..."
        )

        graph = ox.io.load_graphml(
            graph_path,
        )

    # -------------------------------------------------------------
    # 3. Fresh OSM download
    # -------------------------------------------------------------

    else:

        print(
            "Downloading Hyderabad OSM network..."
        )

        graph = ox.graph.graph_from_place(
            OSM_PLACE,
            network_type=OSM_NETWORK_TYPE,
        )

        print(
            f"Saving Hyderabad OSM network to "
            f"{graph_path}..."
        )

        ox.io.save_graphml(
            graph,
            graph_path,
        )

    print(
        f"OSM NODES: {len(graph.nodes):,}"
    )

    print(
        f"OSM EDGES: {len(graph.edges):,}"
    )

    # -------------------------------------------------------------
    # 4. Create binary cache for future fast loads
    # -------------------------------------------------------------

    try:

        print(
            f"Creating binary OSM cache at "
            f"{pickle_path}..."
        )

        with open(
            pickle_path,
            "wb",
        ) as f:
            pickle.dump(
                graph,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    except Exception as exc:

        print(
            "Warning: could not create binary OSM cache: "
            f"{exc}"
        )

    return graph

# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_traffic_dataset() -> pd.DataFrame:
    """
    Load the final traffic dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Traffic dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    missing = [col for col in REQUIRED_TRAFFIC_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Traffic dataset is missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# ---------------------------------------------------------------------
# OSM SEGMENT MAPPING
# ---------------------------------------------------------------------


def load_segment_mapping() -> pd.DataFrame:
    """
    Load the road-segment â†’ OSM-edge mapping created in Layer 1.
    """

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            "OSM mapping file not found.\n"
            f"Expected: {MAPPING_PATH}\n\n"
            "Run tests/test_osm_traffic_mapper.py first so the "
            "mapping file is generated."
        )

    mapping = pd.read_csv(MAPPING_PATH)

    required = [
        "road_segment_id",
        "osm_u",
        "osm_v",
        "osm_key",
        "osm_distance_m",
    ]

    missing = [col for col in required if col not in mapping.columns]

    if missing:
        raise ValueError(f"OSM mapping is missing columns: {missing}")

    return mapping


# ---------------------------------------------------------------------
# TRAFFIC STATE
# ---------------------------------------------------------------------


def get_traffic_state(
    timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Get the traffic state for the requested timestamp.

    The nearest available hourly observation is used if the exact
    timestamp does not exist.
    """

    df = load_traffic_dataset()

    requested_time = pd.Timestamp(timestamp)

    available_times = df["timestamp"].drop_duplicates().sort_values()

    if available_times.empty:
        raise ValueError("Traffic dataset contains no timestamps.")

    nearest_time = available_times.iloc[
        (available_times - requested_time).abs().argmin()
    ]

    traffic = df[df["timestamp"] == nearest_time].copy()

    if traffic.empty:
        raise ValueError(f"No traffic observations found near {requested_time}.")

    # One traffic observation per physical road segment.
    traffic = (
        traffic.sort_values("traffic_id").drop_duplicates("road_segment_id").copy()
    )

    return traffic


# ---------------------------------------------------------------------
# TRAFFIC SCORE
# ---------------------------------------------------------------------


def calculate_traffic_penalty(row: pd.Series) -> float:
    """
    Calculate a normalized traffic penalty.

    Higher value = less desirable road.

    The score combines:
    - congestion level
    - occupancy
    - speed reduction
    - vehicle load
    """

    congestion_level = str(row["future_congestion_level"])

    congestion_penalty = CONGESTION_PENALTY.get(
        congestion_level,
        1.35,
    )

    speed = max(float(row["average_speed_kmh"]), 1.0)
    speed_limit = max(float(row["speed_limit_kmh"]), 1.0)

    speed_ratio = np.clip(
        speed / speed_limit,
        0.05,
        1.0,
    )

    speed_penalty = 1.0 / speed_ratio

    occupancy = np.clip(
        float(row["occupancy_percent"]) / 100.0,
        0.0,
        1.0,
    )

    occupancy_penalty = 1.0 + occupancy

    return congestion_penalty * speed_penalty * occupancy_penalty


# ---------------------------------------------------------------------
# APPLY TRAFFIC TO OSM
# ---------------------------------------------------------------------


def apply_traffic_weights(
    graph: nx.MultiDiGraph,
    traffic_state: pd.DataFrame,
    mapping: pd.DataFrame,
) -> nx.MultiDiGraph:
    """
    Apply project traffic conditions to mapped OSM edges.

    Each mapped OSM edge receives:
        traffic_penalty
        traffic_weight
        traffic_congestion
        traffic_speed
        traffic_occupancy

    Unmapped OSM edges retain their physical length as weight.
    """

    G = graph.copy()

    for u, v, key, edge_data in G.edges(keys=True, data=True):
        length = float(edge_data.get("length", 1.0))
        edge_data["traffic_weight"] = length
        edge_data["traffic_penalty"] = 1.0

    traffic = traffic_state.copy()

    traffic["traffic_penalty"] = traffic.apply(
        calculate_traffic_penalty,
        axis=1,
    )

    merged = mapping.merge(
        traffic[
            [
                "road_segment_id",
                "future_congestion_level",
                "average_speed_kmh",
                "occupancy_percent",
                "vehicle_count",
                "traffic_flow_vph",
                "traffic_penalty",
            ]
        ],
        on="road_segment_id",
        how="inner",
    )

    for _, row in merged.iterrows():

        u = int(row["osm_u"])
        v = int(row["osm_v"])
        key = int(row["osm_key"])

        if not G.has_edge(u, v, key):
            continue

        edge_data = G[u][v][key]

        length = float(edge_data.get("length", 1.0))

        penalty = float(row["traffic_penalty"])

        edge_data["traffic_penalty"] = penalty
        edge_data["traffic_weight"] = length * penalty

        edge_data["traffic_congestion"] = row["future_congestion_level"]

        edge_data["traffic_speed_kmh"] = float(row["average_speed_kmh"])

        edge_data["traffic_occupancy_percent"] = float(row["occupancy_percent"])

        edge_data["traffic_vehicle_count"] = float(row["vehicle_count"])

        edge_data["traffic_flow_vph"] = float(row["traffic_flow_vph"])

    return G


@lru_cache(maxsize=8)
def get_cached_traffic_graph(
    timestamp: str,
) -> nx.MultiDiGraph:
    """
    Build and cache a traffic-weighted OSM graph by timestamp.

    The underlying traffic-weighting logic is unchanged. This cache
    avoids rebuilding the 357k-edge traffic graph for repeated route
    queries at the same traffic timestamp.
    """

    base_graph = load_osm_graph()
    traffic_state = get_traffic_state(timestamp)
    mapping = load_segment_mapping()

    return apply_traffic_weights(
        base_graph,
        traffic_state,
        mapping,
    )


# ---------------------------------------------------------------------
# ROUTE NODES
# ---------------------------------------------------------------------


def _get_osm_speed_kmh(edge_data: dict) -> float:
    """
    Get a reasonable fallback speed for an OSM edge.

    OSM maxspeed can be:
        - numeric
        - string such as "50"
        - a list
        - missing

    Returns:
        Speed in km/h.
    """

    maxspeed = edge_data.get("maxspeed")

    if maxspeed is not None:

        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0] if maxspeed else None

        if maxspeed is not None:

            text = str(maxspeed)

            # Extract the first numeric value.
            import re

            match = re.search(
                r"\d+(?:\.\d+)?",
                text,
            )

            if match:

                speed = float(match.group())

                if 5 <= speed <= 130:
                    return speed

    # Conservative fallback for an OSM driving edge
    return 40.0


def get_nearest_nodes(
    graph: nx.MultiDiGraph,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
):
    """
    Find nearest OSM nodes.

    Coordinates are supplied as:
        (latitude, longitude)
    """

    origin_lat, origin_lon = origin
    dest_lat, dest_lon = destination

    origin_node = ox.distance.nearest_nodes(
        graph,
        X=origin_lon,
        Y=origin_lat,
    )

    destination_node = ox.distance.nearest_nodes(
        graph,
        X=dest_lon,
        Y=dest_lat,
    )

    return origin_node, destination_node


# ---------------------------------------------------------------------
# ROUTE CALCULATION
# ---------------------------------------------------------------------


def calculate_route(
    graph: nx.MultiDiGraph,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    weight: str,
) -> List[int]:
    """
    Calculate an OSM route using bidirectional Dijkstra.
    """

    origin_node, destination_node = get_nearest_nodes(
        graph,
        origin,
        destination,
    )

    _, route = nx.bidirectional_dijkstra(
        graph,
        origin_node,
        destination_node,
        weight=weight,
    )

    return route

# ---------------------------------------------------------------------
# ROUTE METRICS
# ---------------------------------------------------------------------


def calculate_route_metrics(
    graph: nx.MultiDiGraph,
    route: List[int],
) -> Dict:
    """
    Calculate route-level metrics.

    Traffic-covered OSM edges use actual project traffic observations.

    Unmapped OSM edges use OSM maxspeed where available, otherwise
    a conservative fallback speed.

    This prevents missing traffic observations from being interpreted
    as zero speed.
    """

    if len(route) < 2:
        return {
            "distance_km": 0.0,
            "estimated_time_minutes": 0.0,
            "average_speed_kmh": 0.0,
            "average_occupancy_percent": None,
            "congested_edges": 0,
            "traffic_covered_edges": 0,
            "total_edges": 0,
            "traffic_coverage_percent": 0.0,
        }

    total_distance_m = 0.0

    total_travel_time_hours = 0.0

    weighted_speed_sum = 0.0
    weighted_occupancy_sum = 0.0

    traffic_covered_distance_m = 0.0

    congested_edges = 0
    traffic_covered_edges = 0
    total_edges = 0

    for u, v in zip(
        route[:-1],
        route[1:],
    ):

        edge_dict = graph.get_edge_data(
            u,
            v,
        )

        if not edge_dict:
            continue

        # Select the physically shortest edge between the two nodes.
        key = min(
            edge_dict,
            key=lambda k: edge_dict[k].get(
                "length",
                float("inf"),
            ),
        )

        data = edge_dict[key]

        length_m = float(
            data.get(
                "length",
                0.0,
            )
        )

        if length_m <= 0:
            continue

        total_edges += 1
        total_distance_m += length_m

        # -------------------------------------------------------------
        # Traffic-covered edge
        # -------------------------------------------------------------

        traffic_speed = data.get("traffic_speed_kmh")

        traffic_occupancy = data.get("traffic_occupancy_percent")

        traffic_congestion = data.get("traffic_congestion")

        if traffic_speed is not None:

            speed_kmh = max(
                float(traffic_speed),
                1.0,
            )

            traffic_covered_edges += 1

            traffic_covered_distance_m += length_m

            weighted_speed_sum += speed_kmh * length_m

            if traffic_occupancy is not None:

                weighted_occupancy_sum += float(traffic_occupancy) * length_m

            if traffic_congestion in {
                "Moderate",
                "High",
                "Severe",
            }:

                congested_edges += 1

        # -------------------------------------------------------------
        # Unmapped edge
        # -------------------------------------------------------------

        else:

            speed_kmh = _get_osm_speed_kmh(data)

        # -------------------------------------------------------------
        # Travel time
        # -------------------------------------------------------------

        edge_time_hours = (length_m / 1000.0) / speed_kmh

        total_travel_time_hours += edge_time_hours

    # -----------------------------------------------------------------
    # Final metrics
    # -----------------------------------------------------------------

    distance_km = total_distance_m / 1000.0

    estimated_time_minutes = total_travel_time_hours * 60.0

    if total_travel_time_hours > 0:

        effective_average_speed = distance_km / total_travel_time_hours

    else:

        effective_average_speed = 0.0

    if traffic_covered_distance_m > 0:

        average_traffic_speed = weighted_speed_sum / traffic_covered_distance_m

        average_occupancy = weighted_occupancy_sum / traffic_covered_distance_m

    else:

        average_traffic_speed = None
        average_occupancy = None

    if total_edges > 0:

        traffic_coverage_percent = (traffic_covered_edges / total_edges) * 100.0

    else:

        traffic_coverage_percent = 0.0

    return {
        "distance_km": round(
            distance_km,
            3,
        ),
        "estimated_time_minutes": round(
            estimated_time_minutes,
            2,
        ),
        "average_speed_kmh": round(
            effective_average_speed,
            2,
        ),
        "average_traffic_speed_kmh": (
            round(
                average_traffic_speed,
                2,
            )
            if average_traffic_speed is not None
            else None
        ),
        "average_occupancy_percent": (
            round(
                average_occupancy,
                2,
            )
            if average_occupancy is not None
            else None
        ),
        "congested_edges": congested_edges,
        "traffic_covered_edges": traffic_covered_edges,
        "total_edges": total_edges,
        "traffic_coverage_percent": round(
            traffic_coverage_percent,
            2,
        ),
    }


# ---------------------------------------------------------------------
# MAIN ROUTING FUNCTION
# ---------------------------------------------------------------------


def compare_osm_routes(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    timestamp: str | pd.Timestamp,
) -> Dict:
    """
    Calculate and compare:

        1. Shortest OSM route
        2. Traffic-aware OSM route

    Returns route node lists and route metrics.
    """

    base_graph = load_osm_graph()

    traffic_graph = get_cached_traffic_graph(
        str(pd.Timestamp(timestamp)),
    )

    shortest_route = calculate_route(
        base_graph,
        origin,
        destination,
        weight="length",
    )

    traffic_route = calculate_route(
        traffic_graph,
        origin,
        destination,
        weight="traffic_weight",
    )

    shortest_metrics = calculate_route_metrics(
        traffic_graph,
        shortest_route,
    )

    traffic_metrics = calculate_route_metrics(
        traffic_graph,
        traffic_route,
    )

    return {
        "timestamp_requested": str(
            pd.Timestamp(timestamp)
        ),
        "shortest_route": shortest_route,
        "traffic_aware_route": traffic_route,
        "shortest_metrics": shortest_metrics,
        "traffic_aware_metrics": traffic_metrics,
    }

# ---------------------------------------------------------------------
# SIMPLE CLI TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":

    result = compare_osm_routes(
        origin=(17.4374, 78.4482),
        destination=(17.4401, 78.3489),
        timestamp="2025-09-01 18:00:00",
    )

    print("\n" + "=" * 70)
    print("TRAFFIC-AWARE OSM ROUTING")
    print("=" * 70)

    print("\nTIMESTAMP:")
    print(result["timestamp_requested"])

    print("\nSHORTEST ROUTE:")
    print(result["shortest_metrics"])

    print("\nTRAFFIC-AWARE ROUTE:")
    print(result["traffic_aware_metrics"])

    print("\nROUTE NODE COUNTS:")
    print(
        "Shortest:",
        len(result["shortest_route"]),
    )
    print(
        "Traffic-aware:",
        len(result["traffic_aware_route"]),
    )






