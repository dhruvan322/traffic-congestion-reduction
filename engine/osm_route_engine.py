import osmnx as ox
import folium

_graph_cache = None


def load_hyderabad_graph():

    global _graph_cache

    if _graph_cache is None:

        print("Loading Hyderabad road network...")

        _graph_cache = ox.graph_from_place(
            "Hyderabad, Telangana, India",
            network_type="drive"
        )

    return _graph_cache

import networkx as nx


def find_osm_route(
    origin_lat,
    origin_lon,
    dest_lat,
    dest_lon
):

    G = load_hyderabad_graph()

    origin_node = ox.distance.nearest_nodes(
        G,
        origin_lon,
        origin_lat
    )

    destination_node = ox.distance.nearest_nodes(
        G,
        dest_lon,
        dest_lat
    )

    route = nx.shortest_path(
        G,
        origin_node,
        destination_node,
        weight="length"
    )

    return G, route


def extract_route_geometry(G, route):

    route_coords = []

    for u, v in zip(route[:-1], route[1:]):

        edge_data = G.get_edge_data(u, v)

        edge = list(edge_data.values())[0]

        if "geometry" in edge:

            xs, ys = edge["geometry"].xy

            coords = list(zip(ys, xs))

            route_coords.extend(coords)

        else:

            route_coords.append(
                (
                    G.nodes[u]["y"],
                    G.nodes[u]["x"]
                )
            )

            route_coords.append(
                (
                    G.nodes[v]["y"],
                    G.nodes[v]["x"]
                )
            )

    return route_coords


def create_osm_route_map(
    G,
    route,
    origin_name,
    destination_name
):

    route_coords = extract_route_geometry(
        G,
        route
    )

    start_node = route[0]
    end_node = route[-1]

    m = folium.Map(
        location=[
            G.nodes[start_node]["y"],
            G.nodes[start_node]["x"]
        ],
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    folium.PolyLine(
        route_coords,
        color="cyan",
        weight=6,
        opacity=0.8
    ).add_to(m)

    folium.Marker(
        [
            G.nodes[start_node]["y"],
            G.nodes[start_node]["x"]
        ],
        popup=origin_name,
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        [
            G.nodes[end_node]["y"],
            G.nodes[end_node]["x"]
        ],
        popup=destination_name,
        icon=folium.Icon(color="red")
    ).add_to(m)

    return m