from flask import Flask, render_template, request, jsonify
import json
import os
import math

app = Flask(__name__)

NODES_FILE = "stations.json"
EDGES_FILE = "edges.json"


# =========================
# Helpers: persistence
# =========================

def load_nodes():
    """Load list of stations (nodes) from disk."""
    if not os.path.exists(NODES_FILE):
        return []
    with open(NODES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_nodes(nodes):
    with open(NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2)


def load_edges():
    """Load list of edges from disk."""
    if not os.path.exists(EDGES_FILE):
        return []
    with open(EDGES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_edges(edges):
    with open(EDGES_FILE, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2)


# =========================
# Helpers: distance
# =========================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points on Earth in kilometers.
    (Good enough for our use case.)
    """
    R = 6371.0  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def node_index_by_id(nodes, node_id):
    """Return index of node with given id in list, or -1 if not found."""
    for i, n in enumerate(nodes):
        if n["id"] == node_id:
            return i
    return -1


# =========================
# Routes
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/points", methods=["GET"])
def get_points():
    """
    Return all saved stations (nodes).
    Node structure:
      {
        "id": 1,
        "lat": 11.5564,
        "lng": 104.9282
      }
    """
    return jsonify(load_nodes())


@app.route("/edges", methods=["GET"])
def get_edges():
    """
    Return all saved edges.
    Edge structure (undirected):
      {
        "id": 1,
        "from": 1,
        "to": 5,
        "distance_km": 2.34   # this is the edge weight
      }
    """
    return jsonify(load_edges())


@app.route("/add_point", methods=["POST"])
def add_point():
    """
    Add a new station.
    Expected JSON: {"lat": <float>, "lng": <float>}
    """
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng required"}), 400

    nodes = load_nodes()
    point_id = nodes[-1]["id"] + 1 if nodes else 1
    point = {
        "id": point_id,
        "lat": lat,
        "lng": lng,
    }
    nodes.append(point)
    save_nodes(nodes)

    return jsonify(point), 201


@app.route("/add_edge", methods=["POST"])
def add_edge():
    """
    Add an edge between two stations (undirected).
    Expected JSON: {"from": <node_id>, "to": <node_id>}

    Automatically computes the geographic distance in km and stores it
    as "distance_km" (which you can later use as edge weight).
    """
    data = request.get_json()
    source = data.get("from")
    target = data.get("to")

    if source is None or target is None:
        return jsonify({"error": "'from' and 'to' are required"}), 400
    if source == target:
        return jsonify({"error": "from and to must be different nodes"}), 400

    nodes = load_nodes()
    node_ids = {n["id"] for n in nodes}
    if source not in node_ids or target not in node_ids:
        return jsonify({"error": "invalid node id"}), 400

    # remove duplicates (undirected: (a,b) == (b,a))
    edges = load_edges()
    for e in edges:
        if (e["from"] == source and e["to"] == target) or \
           (e["from"] == target and e["to"] == source):
            # already exists, just return it
            return jsonify(e), 200

    # lookup coordinates
    src_index = node_index_by_id(nodes, source)
    tgt_index = node_index_by_id(nodes, target)
    src_node = nodes[src_index]
    tgt_node = nodes[tgt_index]

    distance_km = haversine_km(
        src_node["lat"], src_node["lng"],
        tgt_node["lat"], tgt_node["lng"]
    )

    edge_id = edges[-1]["id"] + 1 if edges else 1
    edge = {
        "id": edge_id,
        "from": source,
        "to": target,
        "distance_km": distance_km
    }
    edges.append(edge)
    save_edges(edges)

    return jsonify(edge), 201


@app.route("/undo", methods=["POST"])
def undo():
    """
    Undo last node:
      - remove the last added station
      - remove all edges incident to that station
    """
    nodes = load_nodes()
    if not nodes:
        return jsonify({"status": "empty"}), 400

    removed_node = nodes.pop()
    save_nodes(nodes)

    edges = load_edges()
    remaining_edges = []
    removed_edges = []
    for e in edges:
        if e["from"] == removed_node["id"] or e["to"] == removed_node["id"]:
            removed_edges.append(e)
        else:
            remaining_edges.append(e)
    save_edges(remaining_edges)

    return jsonify({
        "status": "ok",
        "removed": removed_node,
        "removed_edges": removed_edges,
    })


if __name__ == "__main__":
    app.run(debug=True)
