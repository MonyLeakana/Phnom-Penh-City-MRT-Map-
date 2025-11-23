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
# DSU for Kruskal
# =========================

class DisjointSetUnion:
    """Union-Find data structure for Kruskal's algorithm."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


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

    edges = load_edges()
    # avoid duplicates (undirected)
    for e in edges:
        if (e["from"] == source and e["to"] == target) or \
           (e["from"] == target and e["to"] == source):
            # already exists
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


@app.route("/mst", methods=["POST"])
def mst():
    """
    Compute the Minimum Spanning Tree using Kruskal's algorithm.
    Uses distance_km as the weight.

    Returns:
    {
      "status": "ok",
      "mst_edges": [edge, ...],   # same structure as /edges, subset of them
      "total_distance_km": <float>,
      "num_nodes": <int>,
      "num_edges": <int>,
      "is_spanning": <bool>
    }
    """
    nodes = load_nodes()
    edges = load_edges()

    if not nodes or not edges:
        return jsonify({"error": "Need at least one node and one edge"}), 400

    # Map node IDs to indices [0..n-1] for DSU
    node_ids = sorted(n["id"] for n in nodes)
    id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    # Prepare edge list for Kruskal: (weight, u_idx, v_idx, edge_obj)
    kruskal_edges = []
    for e in edges:
        u_idx = id_to_index[e["from"]]
        v_idx = id_to_index[e["to"]]
        w = e["distance_km"]
        kruskal_edges.append((w, u_idx, v_idx, e))

    # Sort by weight
    kruskal_edges.sort(key=lambda x: x[0])

    dsu = DisjointSetUnion(n)
    mst_edges = []
    total = 0.0

    for w, u_idx, v_idx, e in kruskal_edges:
        if dsu.union(u_idx, v_idx):
            mst_edges.append(e)
            total += w

    is_spanning = (len(mst_edges) == n - 1) if n > 0 else False

    return jsonify({
        "status": "ok",
        "mst_edges": mst_edges,
        "total_distance_km": total,
        "num_nodes": n,
        "num_edges": len(edges),
        "is_spanning": is_spanning
    })


if __name__ == "__main__":
    app.run(debug=True)
