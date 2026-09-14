"""
OSM Traffic Segment Mapper

Maps project road segments to the nearest OpenStreetMap road edges.

Purpose:
    Associate the project's 60 road segments with the real Hyderabad
    OpenStreetMap driving network so traffic information can later be
    incorporated into OSM-based route optimization.

Important:
    This module performs spatial association only.
    It does NOT claim that a project road segment is an official OSM road ID.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import networkx as nx
import osmnx as ox
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OSM_PLACE = "Hyderabad, Telangana, India"
OSM_NETWORK_TYPE = "drive"

# Mapping quality thresholds.
# These are deliberately conservative because we don't want to silently
# accept poor spatial matches.
HIGH_CONFIDENCE_M = 50.0
MEDIUM_CONFIDENCE_M = 100.0


# ---------------------------------------------------------------------------
# OSM graph loading
# ---------------------------------------------------------------------------

_graph_cache: Optional[nx.MultiDiGraph] = None


def load_hyderabad_osm_graph() -> nx.MultiDiGraph:
    """
    Load and cache the Hyderabad driving road network from OpenStreetMap.
    """

    global _graph_cache

    if _graph_cache is None:
        logger.info("Downloading Hyderabad OSM driving network...")

        _graph_cache = ox.graph_from_place(
            OSM_PLACE,
            network_type=OSM_NETWORK_TYPE,
        )

        logger.info(
            "OSM graph loaded: %d nodes, %d edges",
            len(_graph_cache.nodes),
            len(_graph_cache.edges),
        )

    return _graph_cache


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "road_segment_id",
    "latitude",
    "longitude",
    "road_name",
    "road_type",
    "number_of_lanes",
    "speed_limit_kmh",
    "road_length_km",
}


def validate_road_segments(df: pd.DataFrame) -> None:
    """
    Validate that the project road-segment dataset contains the fields
    required for spatial association.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Road-segment dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Road-segment dataset is empty.")

    if df["road_segment_id"].duplicated().any():
        duplicates = df.loc[
            df["road_segment_id"].duplicated(),
            "road_segment_id",
        ].tolist()

        raise ValueError(f"Duplicate road_segment_id values found: {duplicates[:10]}")

    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError(
            "Road-segment dataset contains missing latitude/longitude values."
        )


# ---------------------------------------------------------------------------
# Distance utility
# ---------------------------------------------------------------------------


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance between two coordinates in metres.
    """

    earth_radius_m = 6_371_000.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )

    return earth_radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# Confidence classification
# ---------------------------------------------------------------------------


def classify_mapping_confidence(distance_m: float) -> str:
    """
    Convert spatial distance into a transparent mapping-confidence label.
    """

    if distance_m <= HIGH_CONFIDENCE_M:
        return "High"

    if distance_m <= MEDIUM_CONFIDENCE_M:
        return "Medium"

    return "Low"


# ---------------------------------------------------------------------------
# Single-segment mapping
# ---------------------------------------------------------------------------


def map_segment_to_osm(
    G: nx.MultiDiGraph,
    latitude: float,
    longitude: float,
) -> dict:
    """
    Find the nearest OSM road edge to a project road-segment coordinate.

    Returns:
        Dictionary containing OSM edge identifiers, distance and confidence.
    """

    # Find the nearest OSM edge.
    edge_result = ox.distance.nearest_edges(
        G,
        X=longitude,
        Y=latitude,
        return_dist=True,
    )

    # OSMnx 2.x returns:
    # ((u, v, key), distance)
    edge_tuple, distance = edge_result

    u, v, key = edge_tuple

    distance_m = float(distance)

    edge_data = G.get_edge_data(u, v, key)

    if edge_data is None:
        raise RuntimeError(f"Could not retrieve OSM edge data for ({u}, {v}, {key}).")

    return {
        "osm_u": int(u),
        "osm_v": int(v),
        "osm_key": int(key),
        "osm_distance_m": round(distance_m, 2),
        "mapping_confidence": classify_mapping_confidence(distance_m),
        "osm_highway": edge_data.get("highway"),
        "osm_name": edge_data.get("name"),
        "osm_length_m": round(
            float(edge_data.get("length", 0.0)),
            2,
        ),
    }


# ---------------------------------------------------------------------------
# Full dataset mapping
# ---------------------------------------------------------------------------


def map_road_segments_to_osm(
    road_segments_df: pd.DataFrame,
    G: Optional[nx.MultiDiGraph] = None,
) -> pd.DataFrame:
    """
    Map every project road segment to its nearest OSM road edge.

    The original road-segment columns are preserved.
    OSM association columns are appended.
    """

    validate_road_segments(road_segments_df)

    if G is None:
        G = load_hyderabad_osm_graph()

    mappings = []

    for _, row in road_segments_df.iterrows():

        result = map_segment_to_osm(
            G,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )

        mappings.append(
            {
                "road_segment_id": row["road_segment_id"],
                **result,
            }
        )

    mapping_df = pd.DataFrame(mappings)

    result_df = road_segments_df.merge(
        mapping_df,
        on="road_segment_id",
        how="left",
        validate="one_to_one",
    )

    return result_df


# ---------------------------------------------------------------------------
# Mapping report
# ---------------------------------------------------------------------------


def get_mapping_summary(mapping_df: pd.DataFrame) -> dict:
    """
    Generate a concise quality report for the OSM association.
    """

    required = {
        "road_segment_id",
        "osm_u",
        "osm_v",
        "osm_key",
        "osm_distance_m",
        "mapping_confidence",
    }

    missing = required - set(mapping_df.columns)

    if missing:
        raise ValueError(
            "Mapping dataframe is missing columns: " + ", ".join(sorted(missing))
        )

    confidence_counts = mapping_df["mapping_confidence"].value_counts().to_dict()

    return {
        "total_segments": int(len(mapping_df)),
        "high_confidence": int(confidence_counts.get("High", 0)),
        "medium_confidence": int(confidence_counts.get("Medium", 0)),
        "low_confidence": int(confidence_counts.get("Low", 0)),
        "mean_distance_m": round(
            float(mapping_df["osm_distance_m"].mean()),
            2,
        ),
        "median_distance_m": round(
            float(mapping_df["osm_distance_m"].median()),
            2,
        ),
        "max_distance_m": round(
            float(mapping_df["osm_distance_m"].max()),
            2,
        ),
    }


# ---------------------------------------------------------------------------
# Save mapping
# ---------------------------------------------------------------------------


def save_osm_mapping(
    mapping_df: pd.DataFrame,
    output_path: str = "data/final_dataset/osm_segment_mapping.csv",
) -> str:
    """
    Save the completed project-road → OSM-edge mapping.

    Returns:
        Path to the saved CSV file.
    """

    output_file = output_path

    mapping_df.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "OSM segment mapping saved to: %s",
        output_file,
    )

    return output_file
