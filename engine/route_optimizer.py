"""
Route Optimization Module.
Uses NetworkX with Bidirectional Dijkstra to find:
  1. Shortest path (distance-based)
  2. Least congested path (traffic-aware weighted)
Visualizes both routes on an interactive Folium map.
"""

import logging
import math
import numpy as np
import pandas as pd
import networkx as nx
import folium
from datetime import datetime

from config import (
    ROUTE_SPEED_FACTOR,
    ROUTE_DISTANCE_FACTOR,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
)

logger = logging.getLogger(__name__)


# ──────────────────────────── Distance Utilities ────────────────────────────


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ──────────────────────────── Graph Building ────────────────────────────


def build_junction_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build a weighted graph where:
    - Nodes = junctions with lat/lon attributes
    - Edges = connections between geographically nearby junctions
    - Edge weights: distance_km and traffic_cost (incorporating congestion)
    """
    G = nx.Graph()

    # Get unique junctions with their locations
    junctions = (
        df.groupby("junction_id")
        .agg(
            {
                "latitude": "first",
                "longitude": "first",
                "avg_speed": "mean",
                "vehicle_count": "mean",
                "congestion_level": lambda x: x.mode().iloc[0] if len(x) > 0 else "Low",
            }
        )
        .reset_index()
    )

    # Add all junction nodes
    for _, row in junctions.iterrows():
        G.add_node(
            row["junction_id"],
            lat=row["latitude"],
            lon=row["longitude"],
            avg_speed=row["avg_speed"],
            avg_vehicles=row["vehicle_count"],
            congestion=row["congestion_level"],
        )

    # Connect each junction to its nearest neighbors (build a connected graph)
    junction_list = junctions.to_dict("records")
    n = len(junction_list)

    # Connect to nearest neighbors + ensure connectivity
    for i in range(n):
        distances = []
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(
                junction_list[i]["latitude"],
                junction_list[i]["longitude"],
                junction_list[j]["latitude"],
                junction_list[j]["longitude"],
            )
            distances.append((j, d))

        # Connect to closest 4 neighbors
        distances.sort(key=lambda x: x[1])
        for j, d in distances[:4]:
            ji = junction_list[i]["junction_id"]
            jj = junction_list[j]["junction_id"]

            if G.has_edge(ji, jj):
                continue

            # Traffic cost: penalize low-speed / high-vehicle-count edges
            speed_i = junction_list[i]["avg_speed"]
            speed_j = junction_list[j]["avg_speed"]
            avg_speed = (speed_i + speed_j) / 2
            avg_speed = max(avg_speed, 5)  # Prevent division by zero

            vehicles_i = junction_list[i]["vehicle_count"]
            vehicles_j = junction_list[j]["vehicle_count"]
            avg_vehicles = (vehicles_i + vehicles_j) / 2

            # Distance-only weight
            distance_weight = d * ROUTE_DISTANCE_FACTOR

            # Traffic-aware weight: penalize congestion
            congestion_penalty = 1.0 + (avg_vehicles / 200.0)
            speed_penalty = 60.0 / avg_speed  # Higher when speed is low
            traffic_weight = d * (
                ROUTE_DISTANCE_FACTOR
                + ROUTE_SPEED_FACTOR * speed_penalty * congestion_penalty
            )

            # Estimated travel time (minutes)
            travel_time = (d / avg_speed) * 60

            G.add_edge(
                ji,
                jj,
                distance_km=round(d, 3),
                distance_weight=round(distance_weight, 4),
                traffic_weight=round(traffic_weight, 4),
                travel_time=round(travel_time, 2),
                avg_speed=round(avg_speed, 1),
                avg_vehicles=round(avg_vehicles, 0),
            )

    # Ensure graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            n1 = list(components[k])[0]
            n2 = list(components[k + 1])[0]
            d = haversine_km(
                G.nodes[n1]["lat"],
                G.nodes[n1]["lon"],
                G.nodes[n2]["lat"],
                G.nodes[n2]["lon"],
            )
            G.add_edge(
                n1,
                n2,
                distance_km=d,
                distance_weight=d,
                traffic_weight=d * 2,
                travel_time=(d / 30) * 60,
                avg_speed=30,
                avg_vehicles=100,
            )

    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G


# ──────────────────────────── Bidirectional Dijkstra ────────────────────────────


def find_shortest_path(G: nx.Graph, origin: str, destination: str) -> dict:
    """
    Find shortest path using Bidirectional Dijkstra (distance-based).
    """
    try:
        length, path = nx.bidirectional_dijkstra(
            G, origin, destination, weight="distance_weight"
        )
        # Calculate total distance and time
        total_dist = sum(
            G[path[i]][path[i + 1]]["distance_km"] for i in range(len(path) - 1)
        )
        total_time = sum(
            G[path[i]][path[i + 1]]["travel_time"] for i in range(len(path) - 1)
        )
        return {
            "path": path,
            "total_distance_km": round(total_dist, 2),
            "estimated_minutes": round(total_time, 1),
            "algorithm": "Bidirectional Dijkstra (distance)",
            "weight_used": "distance_weight",
            "path_cost": round(length, 4),
        }
    except nx.NetworkXNoPath:
        return {"error": f"No path between {origin} and {destination}"}
    except nx.NodeNotFound as e:
        return {"error": str(e)}


def find_least_congested_path(G: nx.Graph, origin: str, destination: str) -> dict:
    """
    Find least congested path using Bidirectional Dijkstra (traffic-aware weight).
    """
    try:
        length, path = nx.bidirectional_dijkstra(
            G, origin, destination, weight="traffic_weight"
        )
        total_dist = sum(
            G[path[i]][path[i + 1]]["distance_km"] for i in range(len(path) - 1)
        )
        total_time = sum(
            G[path[i]][path[i + 1]]["travel_time"] for i in range(len(path) - 1)
        )
        return {
            "path": path,
            "total_distance_km": round(total_dist, 2),
            "estimated_minutes": round(total_time, 1),
            "algorithm": "Bidirectional Dijkstra (traffic-aware)",
            "weight_used": "traffic_weight",
            "path_cost": round(length, 4),
        }
    except nx.NetworkXNoPath:
        return {"error": f"No path between {origin} and {destination}"}
    except nx.NodeNotFound as e:
        return {"error": str(e)}


# ──────────────────────────── Route Comparison ────────────────────────────


def compare_routes(G: nx.Graph, origin: str, destination: str) -> dict:
    """
    Find both shortest and least-congested paths, compare them.
    """
    shortest = find_shortest_path(G, origin, destination)
    least_congested = find_least_congested_path(G, origin, destination)

    if "error" in shortest or "error" in least_congested:
        return {"shortest": shortest, "least_congested": least_congested}

    # Calculate time savings
    time_diff = shortest["estimated_minutes"] - least_congested["estimated_minutes"]
    dist_diff = least_congested["total_distance_km"] - shortest["total_distance_km"]

    return {
        "shortest": shortest,
        "least_congested": least_congested,
        "time_saved_minutes": round(time_diff, 1),
        "extra_distance_km": round(dist_diff, 2),
        "recommendation": (
            "Take the least congested route"
            if time_diff > 0
            else "Both routes are similar — shortest path is recommended"
        ),
    }


# ──────────────────────────── Map Visualization ────────────────────────────


def create_route_map(
    G: nx.Graph,
    shortest_result: dict,
    congested_result: dict,
) -> folium.Map:
    """
    Create an interactive Folium map showing both routes.
    - Blue: Shortest path
    - Green: Least congested path
    - Red markers: Start & End
    """
    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=MAP_ZOOM,
        tiles="CartoDB dark_matter",
    )

    # Draw all junction nodes
    for node in G.nodes():
        data = G.nodes[node]
        congestion = data.get("congestion", "Low")
        color = {"Low": "green", "Medium": "orange", "High": "red"}.get(
            congestion, "gray"
        )
        folium.CircleMarker(
            location=[data["lat"], data["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=(
                f"<b>{node}</b><br>"
                f"Avg Speed: {data.get('avg_speed', 0):.1f} km/h<br>"
                f"Avg Vehicles: {data.get('avg_vehicles', 0):.0f}<br>"
                f"Congestion: {congestion}"
            ),
            tooltip=node,
        ).add_to(m)

    # Draw shortest path (blue)
    if "path" in shortest_result:
        path_coords = [
            [G.nodes[n]["lat"], G.nodes[n]["lon"]] for n in shortest_result["path"]
        ]
        folium.PolyLine(
            path_coords,
            color="#4285F4",
            weight=5,
            opacity=0.8,
            popup=(
                f"<b>🔵 Shortest Path</b><br>"
                f"Distance: {shortest_result['total_distance_km']} km<br>"
                f"Time: {shortest_result['estimated_minutes']} min"
            ),
            tooltip="Shortest Path",
        ).add_to(m)

    # Draw least congested path (green)
    if "path" in congested_result:
        path_coords = [
            [G.nodes[n]["lat"], G.nodes[n]["lon"]] for n in congested_result["path"]
        ]
        folium.PolyLine(
            path_coords,
            color="#34A853",
            weight=5,
            opacity=0.8,
            dash_array="10 5",
            popup=(
                f"<b>🟢 Least Congested Path</b><br>"
                f"Distance: {congested_result['total_distance_km']} km<br>"
                f"Time: {congested_result['estimated_minutes']} min"
            ),
            tooltip="Least Congested Path",
        ).add_to(m)

    # Start / End markers
    for result in [shortest_result, congested_result]:
        if "path" in result:
            start = result["path"][0]
            end = result["path"][-1]
            folium.Marker(
                [G.nodes[start]["lat"], G.nodes[start]["lon"]],
                popup=f"<b>🟢 Start: {start}</b>",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
                tooltip="Start",
            ).add_to(m)
            folium.Marker(
                [G.nodes[end]["lat"], G.nodes[end]["lon"]],
                popup=f"<b>🔴 End: {end}</b>",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
                tooltip="Destination",
            ).add_to(m)
            break  # Only add once

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: rgba(0,0,0,0.8); padding: 12px 16px; border-radius: 8px;
         font-family: Arial; font-size: 13px; color: white;">
        <b>Route Legend</b><br>
        <span style="color: #4285F4;">━━━</span> Shortest Path<br>
        <span style="color: #34A853;">╍╍╍</span> Least Congested<br>
        <span style="color: green;">●</span> Low Congestion<br>
        <span style="color: orange;">●</span> Medium Congestion<br>
        <span style="color: red;">●</span> High Congestion
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ──────────────────────────── Public API ────────────────────────────


def optimize_route(df: pd.DataFrame, origin: str, destination: str) -> dict:
    """
    Full route optimization pipeline:
    1. Build graph from traffic data
    2. Find shortest path (Bidirectional Dijkstra)
    3. Find least congested path (Bidirectional Dijkstra with traffic weights)
    4. Generate interactive map
    """
    G = build_junction_graph(df)

    comparison = compare_routes(G, origin, destination)

    if "error" in comparison.get("shortest", {}):
        return comparison

    route_map = create_route_map(
        G, comparison["shortest"], comparison["least_congested"]
    )

    comparison["graph"] = G
    comparison["map"] = route_map
    comparison["timestamp"] = datetime.now().isoformat()

    return comparison
