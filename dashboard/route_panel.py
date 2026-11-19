"""
OSM Traffic-Aware Route Optimizer Panel.

Uses the project's final traffic dataset and Hyderabad OSM driving
network to compare shortest-distance and traffic-aware routes.
"""

import math

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from dashboard.ui_theme import render_header
from engine.osm_traffic_router import (
    load_traffic_dataset,
    load_osm_graph,
    compare_osm_routes,
)

ROUTE_TIME_COMPARISON_TOLERANCE_MINUTES = 0.1


def _load_area_coordinates(df: pd.DataFrame) -> dict:
    """
    Build one representative coordinate pair for each project area.
    """

    area_coordinates = {}

    required = {
        "area",
        "latitude",
        "longitude",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Traffic dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    grouped = (
        df.dropna(
            subset=[
                "area",
                "latitude",
                "longitude",
            ]
        )
        .groupby("area", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
    )

    for _, row in grouped.iterrows():

        area = str(row["area"]).strip()

        if not area:
            continue

        area_coordinates[area] = (
            float(row["latitude"]),
            float(row["longitude"]),
        )

    return area_coordinates


def _build_route_map(
    graph,
    shortest_route,
    traffic_route,
    origin_name,
    destination_name,
):
    """
    Create an interactive OSM route comparison map.
    """

    if not shortest_route:
        raise ValueError("Shortest route is empty.")

    origin_node = shortest_route[0]
    destination_node = shortest_route[-1]

    origin_lat = float(graph.nodes[origin_node]["y"])
    origin_lon = float(graph.nodes[origin_node]["x"])

    destination_lat = float(
        graph.nodes[destination_node]["y"]
    )
    destination_lon = float(
        graph.nodes[destination_node]["x"]
    )

    center_lat = (
        origin_lat + destination_lat
    ) / 2.0

    center_lon = (
        origin_lon + destination_lon
    ) / 2.0

    route_map = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    def route_coordinates(route):
        """Build route coordinates from detailed OSM edge geometry."""
        coordinates = []

        if not route:
            return coordinates

        for u, v in zip(route[:-1], route[1:]):
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]

            u_xy = (
                float(u_data["x"]),
                float(u_data["y"]),
            )
            v_xy = (
                float(v_data["x"]),
                float(v_data["y"]),
            )

            edge_options = graph.get_edge_data(u, v) or {}
            edge_data = {}

            if edge_options:
                edge_data = min(
                    edge_options.values(),
                    key=lambda data: float(
                        data.get("length", float("inf"))
                    ),
                )

            geometry = edge_data.get("geometry")
            edge_xy = []

            if geometry is not None and hasattr(geometry, "coords"):
                try:
                    edge_xy = [
                        (float(point[0]), float(point[1]))
                        for point in geometry.coords
                    ]
                except (TypeError, ValueError):
                    edge_xy = []

            # Fall back to endpoint coordinates only when detailed
            # geometry is unavailable for this edge.
            if len(edge_xy) < 2:
                edge_xy = [u_xy, v_xy]
            else:
                # OSM edge geometry can be stored in either direction.
                # Orient it so that it follows the requested route u -> v.
                forward_score = (
                    abs(edge_xy[0][0] - u_xy[0])
                    + abs(edge_xy[0][1] - u_xy[1])
                    + abs(edge_xy[-1][0] - v_xy[0])
                    + abs(edge_xy[-1][1] - v_xy[1])
                )

                reverse_score = (
                    abs(edge_xy[-1][0] - u_xy[0])
                    + abs(edge_xy[-1][1] - u_xy[1])
                    + abs(edge_xy[0][0] - v_xy[0])
                    + abs(edge_xy[0][1] - v_xy[1])
                )

                if reverse_score < forward_score:
                    edge_xy.reverse()

            # Folium expects [latitude, longitude], while OSM geometry
            # stores coordinates as (longitude, latitude).
            edge_latlon = [
                [latitude, longitude]
                for longitude, latitude in edge_xy
            ]

            # Avoid duplicate points where consecutive edges meet.
            if coordinates and edge_latlon:
                last_point = coordinates[-1]
                first_point = edge_latlon[0]

                if (
                    abs(last_point[0] - first_point[0]) < 1e-8
                    and abs(last_point[1] - first_point[1]) < 1e-8
                ):
                    edge_latlon = edge_latlon[1:]

            coordinates.extend(edge_latlon)

        return coordinates

    shortest_coordinates = route_coordinates(
        shortest_route
    )

    traffic_coordinates = route_coordinates(
        traffic_route
    )

    folium.PolyLine(
        shortest_coordinates,
        color="#00F0FF" if traffic_route != shortest_route else "#00E5BC",
        weight=5,
        opacity=0.9,
        tooltip=(
            "Shortest-distance route"
            if traffic_route != shortest_route
            else "Shortest and traffic-aware routes (same path)"
        ),
    ).add_to(route_map)

    if traffic_route != shortest_route:

        folium.PolyLine(
            traffic_coordinates,
            color="#00E5BC",
            weight=5,
            opacity=0.9,
            dash_array="10, 10",
            tooltip="Traffic-aware route",
        ).add_to(route_map)

    folium.Marker(
        [
            origin_lat,
            origin_lon,
        ],
        tooltip=f"Origin: {origin_name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(route_map)

    folium.Marker(
        [
            destination_lat,
            destination_lon,
        ],
        tooltip=f"Destination: {destination_name}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(route_map)

    legend_routes = (
        '<span style="color:#00F0FF">━━━━</span> Shortest-distance route<br>'
        '<span style="color:#00E5BC">┅┅┅┅</span> Traffic-aware route'
        if traffic_route != shortest_route
        else '<span style="color:#00E5BC">━━━━</span> Both routes (same path)'
    )
    legend = f"""
    <div style="position:fixed; bottom:24px; left:24px; z-index:1000;
        background:rgba(10,14,23,.94); border:1px solid rgba(0,240,255,.35);
        border-radius:8px; padding:10px 14px; color:#F8FAFC;
        font:12px/1.8 Inter,Arial,sans-serif; box-shadow:0 4px 16px #0008">
        <strong>ROUTE LEGEND</strong><br>
        {legend_routes}<br>
        <span style="color:#34D399">●</span> Origin&nbsp;&nbsp;
        <span style="color:#F87171">●</span> Destination
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(legend))

    route_map.fit_bounds(
        [
            [
                origin_lat,
                origin_lon,
            ],
            [
                destination_lat,
                destination_lon,
            ],
        ]
    )

    return route_map


def _numeric_metric(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_metric(metrics: dict, key: str, precision: int, unit: str) -> str:
    value = _numeric_metric(metrics, key)
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{unit}"


def _format_traffic_coverage(metrics: dict) -> str:
    coverage = _numeric_metric(metrics, "traffic_coverage_percent")
    covered = _numeric_metric(metrics, "traffic_covered_edges")
    total = _numeric_metric(metrics, "total_edges")

    if coverage is None and covered is not None and total is not None and total > 0:
        coverage = covered / total * 100

    if coverage is None:
        return "N/A"
    if covered is not None and total is not None:
        return f"{coverage:.1f}% · {covered:.0f}/{total:.0f} edges"
    return f"{coverage:.1f}%"


def _render_route_card(title: str, tag: str, metrics: dict) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(tag)
        first_row, second_row = st.columns(2, gap="medium")
        metric_values = (
            (
                first_row,
                "Distance",
                _format_metric(metrics, "distance_km", 2, " km"),
            ),
            (
                first_row,
                "Estimated travel time",
                _format_metric(metrics, "estimated_time_minutes", 2, " min"),
            ),
            (
                second_row,
                "Average speed",
                _format_metric(metrics, "average_speed_kmh", 1, " km/h"),
            ),
            (
                second_row,
                "Historical traffic coverage",
                _format_traffic_coverage(metrics),
            ),
        )
        for column, label, value in metric_values:
            with column:
                st.metric(label, value)


def _render_route_recommendation(
    shortest_route: list,
    traffic_route: list,
    shortest_metrics: dict,
    traffic_metrics: dict,
) -> None:
    st.markdown("### Route recommendation")

    if shortest_route == traffic_route:
        st.info(
            "Both routing algorithms selected the same route for the "
            "selected historical traffic snapshot."
        )
        return

    shortest_time = _numeric_metric(
        shortest_metrics,
        "estimated_time_minutes",
    )
    traffic_time = _numeric_metric(
        traffic_metrics,
        "estimated_time_minutes",
    )
    if shortest_time is None or traffic_time is None:
        st.info(
            "A route preference cannot be determined because one or both "
            "historical estimated travel times are unavailable."
        )
        return

    time_difference = abs(shortest_time - traffic_time)
    if time_difference <= ROUTE_TIME_COMPARISON_TOLERANCE_MINUTES:
        st.info(
            "The routes have effectively similar estimated travel times "
            "for this historical snapshot; no faster option is indicated."
        )
    elif traffic_time < shortest_time:
        st.success(
            f"The traffic-aware route has the lower estimated travel time "
            f"({traffic_time:.2f} min vs {shortest_time:.2f} min) for this "
            "historical snapshot. This is a model estimate, not a live "
            "travel-time observation."
        )
    else:
        st.info(
            f"The shortest route has the lower estimated travel time "
            f"({shortest_time:.2f} min vs {traffic_time:.2f} min) for this "
            "historical snapshot. This is a model estimate, not a live "
            "travel-time observation."
        )


def _render_route_analytics(shortest: dict, traffic_aware: dict) -> None:
    analytics = (
        (
            "Distance",
            _format_metric(shortest, "distance_km", 2, " km"),
            _format_metric(traffic_aware, "distance_km", 2, " km"),
        ),
        (
            "Estimated travel time",
            _format_metric(shortest, "estimated_time_minutes", 2, " min"),
            _format_metric(traffic_aware, "estimated_time_minutes", 2, " min"),
        ),
        (
            "Historical traffic coverage",
            _format_traffic_coverage(shortest),
            _format_traffic_coverage(traffic_aware),
        ),
        (
            "Average speed",
            _format_metric(shortest, "average_speed_kmh", 1, " km/h"),
            _format_metric(traffic_aware, "average_speed_kmh", 1, " km/h"),
        ),
    )

    st.markdown("#### Route analytics")
    for label, shortest_value, traffic_value in analytics:
        st.markdown(f"**{label}**")
        shortest_column, traffic_column = st.columns(2, gap="small")
        with shortest_column:
            st.caption("SHORTEST")
            st.write(shortest_value)
        with traffic_column:
            st.caption("TRAFFIC-AWARE")
            st.write(traffic_value)


def _render_data_provenance(
    timestamp: str,
    shortest_metrics: dict,
    traffic_metrics: dict,
) -> None:
    st.markdown("### Data provenance")
    with st.container(border=True):
        source_col, network_col, snapshot_col = st.columns(3, gap="medium")
        with source_col:
            st.markdown("**Historical observations**")
            st.caption("Project synthetic traffic dataset; no live traffic feed.")
        with network_col:
            st.markdown("**Road network**")
            st.caption("Hyderabad OpenStreetMap driving network.")
        with snapshot_col:
            st.markdown("**Selected traffic snapshot**")
            st.caption(str(timestamp))

        st.caption(
            "Traffic coverage counts route edges matched to project road "
            "segments. Unmapped OSM edges do not have project traffic "
            "observations and use OSM speed data or the router's fallback."
        )
        coverage_columns = st.columns(2, gap="medium")
        with coverage_columns[0]:
            st.markdown(
                f"**Shortest route:** "
                f"{_format_traffic_coverage(shortest_metrics)}"
            )
        with coverage_columns[1]:
            st.markdown(
                f"**Traffic-aware route:** "
                f"{_format_traffic_coverage(traffic_metrics)}"
            )


def render_route_panel():
    """
    Render the OSM traffic-aware route optimizer.
    """

    render_header(
        "Route Intelligence & Optimization",
        "Compare distance-efficient and traffic-aware routes across Hyderabad's road network.",
        badge_text="HISTORICAL DATASET",
        badge_type="cyan",
    )

    # -------------------------------------------------------------
    # Load final traffic dataset
    # -------------------------------------------------------------

    try:

        df = load_traffic_dataset()

    except Exception as exc:

        st.error(
            "Unable to load the final traffic dataset."
        )

        st.exception(exc)

        return

    # -------------------------------------------------------------
    # Build project-area coordinate lookup
    # -------------------------------------------------------------

    try:

        area_coordinates = _load_area_coordinates(df)

    except Exception as exc:

        st.error(
            "Unable to build project-area coordinates."
        )

        st.exception(exc)

        return

    areas = sorted(
        area_coordinates.keys(),
        key=lambda value: value.lower(),
    )

    if len(areas) < 2:

        st.warning(
            "At least two project areas are required "
            "to calculate a route."
        )

        return

    # -------------------------------------------------------------
    # Session state
    # -------------------------------------------------------------

    if "osm_route_result" not in st.session_state:

        st.session_state.osm_route_result = None

    if "osm_route_map" not in st.session_state:

        st.session_state.osm_route_map = None

    # -------------------------------------------------------------
    # Route input
    # -------------------------------------------------------------

    with st.container(border=True):
        st.markdown("#### Plan a route")
        origin_column, destination_column, button_column = st.columns(
            [1, 1, 1.15],
            gap="medium",
            vertical_alignment="bottom",
        )
        with origin_column:
            origin = st.selectbox(
                "Origin Area",
                areas,
                index=0,
                key="osm_route_origin",
            )

        destination_options = [
            area
            for area in areas
            if area != origin
        ]
        default_destination_index = min(
            len(destination_options) - 1,
            1,
        )
        with destination_column:
            destination = st.selectbox(
                "Destination Area",
                destination_options,
                index=default_destination_index,
                key="osm_route_destination",
            )
        with button_column:
            find_routes = st.button(
                "Find Best Routes",
                type="primary",
                width="stretch",
            )
        st.caption(f"Selected route: {origin} → {destination}")
        st.caption(
            "Uses the latest available historical traffic snapshot — "
            "not live navigation."
        )

    # -------------------------------------------------------------
    # Route calculation
    # -------------------------------------------------------------

    if find_routes:

        origin_coords = area_coordinates[origin]

        destination_coords = area_coordinates[
            destination
        ]

        latest_timestamp = pd.to_datetime(
            df["timestamp"]
        ).max()

        try:

            with st.spinner(
                "Calculating shortest and traffic-aware routes..."
            ):

                result = compare_osm_routes(
                    origin=origin_coords,
                    destination=destination_coords,
                    timestamp=latest_timestamp,
                )

                graph = load_osm_graph()

                route_map = _build_route_map(
                    graph=graph,
                    shortest_route=result[
                        "shortest_route"
                    ],
                    traffic_route=result[
                        "traffic_aware_route"
                    ],
                    origin_name=origin,
                    destination_name=destination,
                )

                st.session_state.osm_route_result = result
                st.session_state.osm_route_map = route_map

        except Exception as exc:

            st.session_state.osm_route_result = None
            st.session_state.osm_route_map = None

            st.error(
                "Unable to calculate the OSM traffic-aware routes."
            )

            st.exception(exc)

    result = st.session_state.osm_route_result

    if result is None:
        return

    # -------------------------------------------------------------
    # Route results
    # -------------------------------------------------------------

    shortest = result["shortest_metrics"]

    traffic_aware = result[
        "traffic_aware_metrics"
    ]

    shortest_route = result[
        "shortest_route"
    ]

    traffic_route = result[
        "traffic_aware_route"
    ]
    route_changed = shortest_route != traffic_route

    st.markdown("### Route comparison")
    st.caption(
        f"Historical traffic snapshot: {result['timestamp_requested']}"
    )

    route_columns = st.columns(2, gap="medium")
    with route_columns[0]:
        _render_route_card(
            "Shortest Route",
            "DISTANCE-OPTIMAL",
            shortest,
        )
    with route_columns[1]:
        _render_route_card(
            "Traffic-Aware Route",
            "TRAFFIC-WEIGHTED",
            traffic_aware,
        )

    with st.container(border=True):
        _render_route_recommendation(
            shortest_route,
            traffic_route,
            shortest,
            traffic_aware,
        )
        st.caption(
            "Recommendations compare model-estimated travel times for "
            "historical project data; they do not represent live conditions "
            "or verified real-world travel-time savings."
        )

    st.markdown("### Hyderabad road network")
    map_column, analytics_column = st.columns(
        [1.7, 1],
        gap="medium",
        vertical_alignment="top",
    )
    with map_column:
        if st.session_state.osm_route_map is not None:
            st_folium(
                st.session_state.osm_route_map,
                width=None,
                height=560,
                returned_objects=[],
                key="osm_route_optimizer_map",
            )
    with analytics_column:
        with st.container(border=True):
            _render_route_analytics(shortest, traffic_aware)

    _render_data_provenance(
        result["timestamp_requested"],
        shortest,
        traffic_aware,
    )

    # -------------------------------------------------------------
    # Detailed route information
    # -------------------------------------------------------------

    with st.expander(
        "🔍 Detailed Route Information"
    ):

        st.write(
            {
                "origin": origin,
                "destination": destination,
                "timestamp": result[
                    "timestamp_requested"
                ],
                "shortest_route_nodes": len(
                    shortest_route
                ),
                "traffic_aware_route_nodes": len(
                    traffic_route
                ),
                "route_changed": route_changed,
            }
        )

    # -------------------------------------------------------------
    # Algorithm information
    # -------------------------------------------------------------

    with st.expander(
        "ℹ️ About the Routing Method"
    ):

        st.markdown(
            """
**Shortest Route**

The shortest route uses OSM road length as the routing weight.

**Traffic-Aware Route**

The traffic-aware route uses project traffic penalties
derived from congestion level, speed relative to the speed
limit, and occupancy.

**Routing Algorithm**

Both routes use Bidirectional Dijkstra on the Hyderabad
OSM driving network.

**Traffic Coverage**

Only OSM edges mapped to project road segments receive
project traffic observations. Unmapped OSM edges retain
their physical-length routing weight and use OSM speed
information for route-level metrics.

The output is decision-support based on historical traffic
observations and is not live navigation.
"""
        )
