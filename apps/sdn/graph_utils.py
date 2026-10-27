"""Utilitas untuk memuat topologi jaringan (format JSON) ke dalam graf NetworkX."""
import json
import os
import networkx as nx

DEFAULT_TOPOLOGY_PATH = os.path.join(os.path.dirname(__file__), "data", "topology.json")


def load_topology(path=None):
    """Muat file JSON topologi dan kembalikan (networkx.Graph, raw_dict)."""
    path = path or DEFAULT_TOPOLOGY_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    G = nx.Graph()
    G.add_nodes_from(raw["nodes"])
    for e in raw["edges"]:
        G.add_edge(
            e["source"],
            e["target"],
            delay=float(e.get("delay", 1.0)),
            bandwidth=float(e.get("bandwidth", 100.0)),
            loss=float(e.get("loss", 0.001)),
            # 'weight' generik dipakai sebagai bobot default (= delay) oleh algoritma jalur terpendek
            weight=float(e.get("delay", 1.0)),
        )
    return G, raw


def graph_to_dict(G, layout=None):
    """Serialisasi graf + posisi layout (untuk divisualisasikan di frontend)."""
    layout = layout or nx.spring_layout(G, seed=42)
    nodes = [
        {"id": n, "x": round(float(layout[n][0]), 4), "y": round(float(layout[n][1]), 4)}
        for n in G.nodes()
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "delay": d["delay"],
            "bandwidth": d["bandwidth"],
            "loss": d["loss"],
        }
        for u, v, d in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def path_edges(path):
    """Ubah daftar node [S1,S2,S3] menjadi daftar pasangan sisi [(S1,S2),(S2,S3)]."""
    return list(zip(path[:-1], path[1:]))
