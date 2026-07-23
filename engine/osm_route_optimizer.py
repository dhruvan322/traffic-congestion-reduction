import osmnx as ox

print("Downloading Hyderabad road network...")

G = ox.graph_from_place(
    "Hyderabad, Telangana, India",
    network_type="drive"
)

print(
    f"Nodes: {len(G.nodes)} | "
    f"Edges: {len(G.edges)}"
)