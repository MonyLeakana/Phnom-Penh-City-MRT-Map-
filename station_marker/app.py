from flask import Flask, render_template, request, jsonify
import json
import os
import math

app = Flask(__name__)

# Original design graph
NODES_FILE = "stations.json"
EDGES_FILE = "edges.json"

# Base map graph (MST result)
BASE_NODES_FILE = "basemap_nodes.json"
BASE_EDGES_FILE = "basemap_edges.json"


# =========================
# Helpers: persistence
# =========================

def load_nodes():
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


# === Base-map (MST) files ===

def load_base_nodes():
    if not os.path.exists(BASE_NODES_FILE):
        return []
    with open(BASE_NODES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_base_nodes(nodes):
    with open(BASE_NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2)


def load_base_edges():
    if not os.path.exists(BASE_EDGES_FILE):
        return []
    with open(BASE_EDGES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_base_edges(edges):
    with open(BASE_EDGES_FILE, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2)


# =========================
# Helpers: distance
# =========================

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def node_index_by_id(nodes, node_id):
    for i, n in enumerate(nodes):
        if n["id"] == node_id:
            return i
    return -1


# =========================
# DSU for Kruskal
# =========================

class DisjointSetUnion:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# =========================
# MST helper
# =========================

def compute_mst(nodes, edges):
    """Return (mst_edges_list, total_distance, num_nodes, num_edges, is_spanning)"""
    if not nodes or not edges:
        return None

    node_ids = sorted(n["id"] for n in nodes)
    id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    kruskal_edges = []
    for e in edges:
        u_idx = id_to_index[e["from"]]
        v_idx = id_to_index[e["to"]]
        w = e["distance_km"]
        kruskal_edges.append((w, u_idx, v_idx, e))

    kruskal_edges.sort(key=lambda x: x[0])

    dsu = DisjointSetUnion(n)
    mst_edges = []
    total = 0.0

    for w, u_idx, v_idx, e in kruskal_edges:
        if dsu.union(u_idx, v_idx):
            mst_edges.append(e)
            total += w

    is_spanning = (len(mst_edges) == n - 1) if n > 0 else False
    return mst_edges, total, n, len(edges), is_spanning


# =========================
# Main design page routes
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/points", methods=["GET"])
def get_points():
    return jsonify(load_nodes())


@app.route("/edges", methods=["GET"])
def get_edges():
    return jsonify(load_edges())


@app.route("/add_point", methods=["POST"])
def add_point():
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng required"}), 400

    nodes = load_nodes()
    point_id = nodes[-1]["id"] + 1 if nodes else 1
    point = {"id": point_id, "lat": lat, "lng": lng}
    nodes.append(point)
    save_nodes(nodes)

    return jsonify(point), 201


@app.route("/add_edge", methods=["POST"])
def add_edge():
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
            return jsonify(e), 200

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
    nodes = load_nodes()
    edges = load_edges()

    if not nodes or not edges:
        return jsonify({"error": "Need at least one node and one edge"}), 400

    result = compute_mst(nodes, edges)
    if result is None:
        return jsonify({"error": "MST failure"}), 500

    mst_edges, total, n, num_edges, is_spanning = result

    return jsonify({
        "status": "ok",
        "mst_edges": mst_edges,
        "total_distance_km": total,
        "num_nodes": n,
        "num_edges": num_edges,
        "is_spanning": is_spanning
    })


@app.route("/delete_edge", methods=["POST"])
def delete_edge():
    data = request.get_json()
    edge_id = data.get("edge_id")
    if edge_id is None:
        return jsonify({"error": "edge_id is required"}), 400

    try:
        edge_id = int(edge_id)
    except ValueError:
        return jsonify({"error": "edge_id must be an integer"}), 400

    edges = load_edges()
    new_edges = []
    removed = None
    for e in edges:
        if e["id"] == edge_id:
            removed = e
        else:
            new_edges.append(e)

    if removed is None:
        return jsonify({"error": "edge not found"}), 404

    save_edges(new_edges)
    return jsonify({"status": "ok", "removed": removed})


@app.route("/auto_edges", methods=["POST"])
def auto_edges():
    nodes = load_nodes()
    edges = load_edges()

    if len(nodes) < 2:
        return jsonify({"error": "Need at least two nodes"}), 400

    existing_pairs = set()
    for e in edges:
        a = min(e["from"], e["to"])
        b = max(e["from"], e["to"])
        existing_pairs.add((a, b))

    next_id = edges[-1]["id"] + 1 if edges else 1
    new_edges = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1 = nodes[i]
            n2 = nodes[j]
            a = min(n1["id"], n2["id"])
            b = max(n1["id"], n2["id"])
            pair = (a, b)

            if pair in existing_pairs:
                continue

            d_km = haversine_km(n1["lat"], n1["lng"], n2["lat"], n2["lng"])
            if d_km > 1.0:
                edge = {
                    "id": next_id,
                    "from": n1["id"],
                    "to": n2["id"],
                    "distance_km": d_km
                }
                next_id += 1
                new_edges.append(edge)
                existing_pairs.add(pair)

    edges.extend(new_edges)
    save_edges(edges)

    return jsonify({
        "status": "ok",
        "added": len(new_edges),
        "edges": new_edges
    })


# =========================
# Confirm base map (MST -> basemap_*.json)
# =========================

@app.route("/confirm_base_map", methods=["POST"])
def confirm_base_map():
    """Compute MST, save it as base map, then front-end will redirect to /basemap."""
    nodes = load_nodes()
    edges = load_edges()

    if not nodes or not edges:
        return jsonify({"error": "Need at least one node and one edge"}), 400

    result = compute_mst(nodes, edges)
    if result is None:
        return jsonify({"error": "MST failure"}), 500

    mst_edges, total, n, num_edges, is_spanning = result

    # Save all current nodes and only MST edges as the base map
    save_base_nodes(nodes)
    save_base_edges(mst_edges)

    return jsonify({
        "status": "ok",
        "total_distance_km": total,
        "num_nodes": n,
        "num_mst_edges": len(mst_edges),
        "is_spanning": is_spanning
    })


# =========================
# Base map editor routes
# =========================

@app.route("/basemap")
def basemap_page():
    return render_template("basemap.html")


@app.route("/basemap/points", methods=["GET"])
def basemap_points():
    return jsonify(load_base_nodes())


@app.route("/basemap/edges", methods=["GET"])
def basemap_edges():
    return jsonify(load_base_edges())


@app.route("/basemap/add_point", methods=["POST"])
def basemap_add_point():
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng required"}), 400

    nodes = load_base_nodes()
    point_id = nodes[-1]["id"] + 1 if nodes else 1
    point = {"id": point_id, "lat": lat, "lng": lng}
    nodes.append(point)
    save_base_nodes(nodes)

    return jsonify(point), 201


@app.route("/basemap/add_edge", methods=["POST"])
def basemap_add_edge():
    data = request.get_json()
    source = data.get("from")
    target = data.get("to")

    if source is None or target is None:
        return jsonify({"error": "'from' and 'to' are required"}), 400
    if source == target:
        return jsonify({"error": "from and to must be different nodes"}), 400

    nodes = load_base_nodes()
    node_ids = {n["id"] for n in nodes}
    if source not in node_ids or target not in node_ids:
        return jsonify({"error": "invalid node id"}), 400

    edges = load_base_edges()
    for e in edges:
        if (e["from"] == source and e["to"] == target) or \
           (e["from"] == target and e["to"] == source):
            return jsonify(e), 200

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
    save_base_edges(edges)

    return jsonify(edge), 201


@app.route("/basemap/undo", methods=["POST"])
def basemap_undo():
    nodes = load_base_nodes()
    if not nodes:
        return jsonify({"status": "empty"}), 400

    removed_node = nodes.pop()
    save_base_nodes(nodes)

    edges = load_base_edges()
    remaining_edges = []
    removed_edges = []
    for e in edges:
        if e["from"] == removed_node["id"] or e["to"] == removed_node["id"]:
            removed_edges.append(e)
        else:
            remaining_edges.append(e)
    save_base_edges(remaining_edges)

    return jsonify({
        "status": "ok",
        "removed": removed_node,
        "removed_edges": removed_edges,
    })


@app.route("/basemap/delete_edge", methods=["POST"])
def basemap_delete_edge():
    data = request.get_json()
    edge_id = data.get("edge_id")
    if edge_id is None:
        return jsonify({"error": "edge_id is required"}), 400

    try:
        edge_id = int(edge_id)
    except ValueError:
        return jsonify({"error": "edge_id must be an integer"}), 400

    edges = load_base_edges()
    new_edges = []
    removed = None
    for e in edges:
        if e["id"] == edge_id:
            removed = e
        else:
            new_edges.append(e)

    if removed is None:
        return jsonify({"error": "edge not found"}), 404

    save_base_edges(new_edges)
    return jsonify({"status": "ok", "removed": removed})


@app.route("/basemap/auto_edges", methods=["POST"])
def basemap_auto_edges():
    nodes = load_base_nodes()
    edges = load_base_edges()

    if len(nodes) < 2:
        return jsonify({"error": "Need at least two nodes"}), 400

    existing_pairs = set()
    for e in edges:
        a = min(e["from"], e["to"])
        b = max(e["from"], e["to"])
        existing_pairs.add((a, b))

    next_id = edges[-1]["id"] + 1 if edges else 1
    new_edges = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1 = nodes[i]
            n2 = nodes[j]
            a = min(n1["id"], n2["id"])
            b = max(n1["id"], n2["id"])
            pair = (a, b)

            if pair in existing_pairs:
                continue

            d_km = haversine_km(n1["lat"], n1["lng"], n2["lat"], n2["lng"])
            if d_km > 1.0:
                edge = {
                    "id": next_id,
                    "from": n1["id"],
                    "to": n2["id"],
                    "distance_km": d_km
                }
                next_id += 1
                new_edges.append(edge)
                existing_pairs.add(pair)

    edges.extend(new_edges)
    save_base_edges(edges)

    return jsonify({
        "status": "ok",
        "added": len(new_edges),
        "edges": new_edges
    })


if __name__ == "__main__":
    app.run(debug=True)
