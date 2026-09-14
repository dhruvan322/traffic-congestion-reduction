"""
OSM Traffic-Aware Route Optimizer Panel.

Uses the project's final traffic dataset and Hyderabad OSM driving
network to compare shortest-distance and traffic-aware routes.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from config import DATASET_PATH
from engine.osm_traffic_router import (
    load_traffic_dataset,
    load_osm_graph,
    compare_osm_routes,
)


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
        coordinates = []

        for node in route:

            node_data = graph.nodes[node]

            coordinates.append(
                [
                    float(node_data["y"]),
                    float(node_data["x"]),
                ]
            )

        return coordinates

    shortest_coordinates = route_coordinates(
        shortest_route
    )

    traffic_coordinates = route_coordinates(
        traffic_route
    )

    folium.PolyLine(
        shortest_coordinates,
        weight=5,
        opacity=0.75,
        tooltip="Shortest-distance route",
    ).add_to(route_map)

    if traffic_route != shortest_route:

        folium.PolyLine(
            traffic_coordinates,
            weight=5,
            opacity=0.8,
            dash_array="10, 10",
            tooltip="Traffic-aware route",
        ).add_to(route_map)

    folium.Marker(
        [
            origin_lat,
            origin_lon,
        ],
        tooltip=f"Origin: {origin_name}",
    ).add_to(route_map)

    folium.Marker(
        [
            destination_lat,
            destination_lon,
        ],
        tooltip=f"Destination: {destination_name}",
    ).add_to(route_map)

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


def render_route_panel():
    """
    Render the OSM traffic-aware route optimizer.
    """

    st.markdown(
        "# 🗺️ OSM Traffic-Aware Route Optimizer"
    )

    st.markdown(
        "Compare the **shortest-distance** route with the "
        "**traffic-aware** route using the Hyderabad OSM driving "
        "network and the project's historical traffic observations."
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

    st.subheader(
        "📍 Select Origin & Destination"
    )

    col1, col2 = st.columns(2)

    with col1:

        origin = st.selectbox(
            "🟢 Origin Area",
            areas,
            index=0,
            key="osm_route_origin",
        )

    with col2:

        destination_options = [
            area
            for area in areas
            if area != origin
        ]

        default_destination_index = min(
            len(destination_options) - 1,
            1,
        )

        destination = st.selectbox(
            "🔴 Destination Area",
            destination_options,
            index=default_destination_index,
            key="osm_route_destination",
        )

    st.caption(
        "Routing uses the latest available historical traffic "
        "timestamp in the project dataset. It is not live navigation."
    )

    st.divider()

    # -------------------------------------------------------------
    # Route calculation
    # -------------------------------------------------------------

    if st.button(
        "🚀 Find Best Routes",
        type="primary",
        use_container_width=True,
    ):

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

    route_changed = (
        shortest_route != traffic_route
    )

    st.markdown("---")

    st.subheader(
        "📊 Route Comparison"
    )

    st.info(
        f"Traffic data timestamp: "
        f"{result['timestamp_requested']}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🚗 Shortest Route"
        )

        st.metric(
            "Distance",
            f"{shortest['distance_km']:.2f} km",
        )

        st.metric(
            "Estimated Time",
            f"{shortest['estimated_time_minutes']:.2f} min",
        )

        st.metric(
            "Average Speed",
            f"{shortest['average_speed_kmh']:.1f} km/h",
        )

        st.metric(
            "Traffic Coverage",
            (
                f"{shortest['traffic_covered_edges']}/"
                f"{shortest['total_edges']}"
            ),
        )

    with col2:

        st.markdown(
            "### 🚦 Traffic-Aware Route"
        )

        st.metric(
            "Distance",
            f"{traffic_aware['distance_km']:.2f} km",
        )

        st.metric(
            "Estimated Time",
            f"{traffic_aware['estimated_time_minutes']:.2f} min",
        )

        st.metric(
            "Average Speed",
            f"{traffic_aware['average_speed_kmh']:.1f} km/h",
        )

        st.metric(
            "Traffic Coverage",
            (
                f"{traffic_aware['traffic_covered_edges']}/"
                f"{traffic_aware['total_edges']}"
            ),
        )

    # -------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🧠 Route Recommendation"
    )

    if route_changed:

        st.success(
            "The traffic-aware route is different from the "
            "shortest-distance route. The traffic-aware route "
            "is recommended for the selected historical conditions."
        )

    else:

        st.info(
            "The shortest and traffic-aware routes are the same "
            "for the selected historical traffic conditions."
        )

    st.caption(
        "This route optimizer provides decision-support using "
        "historical project traffic observations and the Hyderabad "
        "OSM driving network. It does not provide live navigation."
    )

    # -------------------------------------------------------------
    # Route map
    # -------------------------------------------------------------

    if st.session_state.osm_route_map is not None:

        st.markdown("---")

        st.subheader(
            "🗺️ Route Visualization"
        )

        st_folium(
            st.session_state.osm_route_map,
            width=None,
            height=600,
            returned_objects=[],
            key="osm_route_optimizer_map",
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
