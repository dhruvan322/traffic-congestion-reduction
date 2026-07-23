from engine.osm_route_engine import (
    find_osm_route,
    create_osm_route_map
)

G, route = find_osm_route(
    17.4374, 78.4482,
    17.4401, 78.3489
)

route_map = create_osm_route_map(
    G,
    route,
    "Ameerpet",
    "Gachibowli"
)

route_map.save(
    "osm_route_engine_test.html"
)

print("Map saved successfully")