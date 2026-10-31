"""Standalone demonstration of OSM route generation."""

from pathlib import Path

from engine.osm_route_engine import (
    find_osm_route,
    create_osm_route_map,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "osm_route_engine_test.html"


def main():
    """Calculate a route and save its interactive map."""

    graph, route = find_osm_route(
        17.4374, 78.4482,
        17.4401, 78.3489,
    )

    route_map = create_osm_route_map(
        graph,
        route,
        "Ameerpet",
        "Gachibowli",
    )

    route_map.save(str(OUTPUT_PATH))

    print(f"Map saved successfully: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
