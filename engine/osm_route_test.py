import osmnx as ox
import networkx as nx
import folium

print("Loading Hyderabad road network...")

G = ox.graph_from_place(
    "Hyderabad, Telangana, India",
    network_type="drive"
)

# Ameerpet
origin_lat = 17.4374
origin_lon = 78.4482

# Gachibowli
dest_lat = 17.4401
dest_lon = 78.3489

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

print("Calculating route...")

route = nx.shortest_path(
    G,
    origin_node,
    destination_node,
    weight="length"
)

print(f"Route contains {len(route)} road nodes")

# Create map centered on origin
m = folium.Map(
    location=[origin_lat, origin_lon],
    zoom_start=12
)

# Extract route coordinates
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

# Draw route
folium.PolyLine(
    route_coords,
    color="blue",
    weight=5,
    opacity=0.8
).add_to(m)

# Start marker
folium.Marker(
    [origin_lat, origin_lon],
    popup="Ameerpet"
).add_to(m)

# End marker
folium.Marker(
    [dest_lat, dest_lon],
    popup="Gachibowli"
).add_to(m)

m.save("osm_route_test.html")

print("Saved: osm_route_test.html")