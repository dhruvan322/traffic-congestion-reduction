"""Fast regression tests for the saved OSM road-segment mapping."""

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROAD_SEGMENTS_PATH = (
    PROJECT_ROOT / "data" / "final_dataset" / "road_segments.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT / "data" / "final_dataset" / "osm_segment_mapping.csv"
)

REQUIRED_MAPPING_COLUMNS = {
    "road_segment_id",
    "osm_u",
    "osm_v",
    "osm_key",
    "osm_distance_m",
    "mapping_confidence",
}

ALLOWED_CONFIDENCE_LEVELS = {"High", "Medium", "Low"}


@pytest.fixture(scope="module")
def road_segments():
    """Load the project's reference road segments."""
    assert ROAD_SEGMENTS_PATH.is_file(), (
        f"Road-segment dataset not found: {ROAD_SEGMENTS_PATH}"
    )
    return pd.read_csv(ROAD_SEGMENTS_PATH)


@pytest.fixture(scope="module")
def mapping():
    """Load the already-generated OSM mapping without rebuilding it."""
    assert MAPPING_PATH.is_file(), (
        f"Saved OSM mapping not found: {MAPPING_PATH}. "
        "Run tests/generate_osm_segment_mapping.py explicitly to generate it."
    )

    df = pd.read_csv(MAPPING_PATH)

    missing_columns = REQUIRED_MAPPING_COLUMNS - set(df.columns)
    assert not missing_columns, (
        f"Saved OSM mapping is missing columns: {sorted(missing_columns)}"
    )

    return df


def test_saved_mapping_covers_all_road_segments(road_segments, mapping):
    """Every project road segment should have exactly one mapping row."""
    assert len(mapping) == len(road_segments)
    assert mapping["road_segment_id"].is_unique

    assert set(mapping["road_segment_id"]) == set(
        road_segments["road_segment_id"]
    )


def test_saved_mapping_has_valid_osm_edge_ids(mapping):
    """Mapped OSM edge identifiers must be present."""
    for column in ("osm_u", "osm_v", "osm_key"):
        assert mapping[column].notna().all(), (
            f"Missing values found in {column}"
        )


def test_saved_mapping_has_valid_distances(mapping):
    """Mapping distances must be numeric, finite, and non-negative."""
    distances = pd.to_numeric(
        mapping["osm_distance_m"],
        errors="coerce",
    )

    assert distances.notna().all()
    assert distances.map(lambda value: value < float("inf")).all()
    assert (distances >= 0).all()


def test_saved_mapping_has_valid_confidence_levels(mapping):
    """Confidence labels must use the project's supported categories."""
    assert mapping["mapping_confidence"].notna().all()

    assert set(mapping["mapping_confidence"]).issubset(
        ALLOWED_CONFIDENCE_LEVELS
    )
