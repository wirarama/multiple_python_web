"""Algoritma A* — shortest path dengan heuristik, berbasis NetworkX.

Heuristik menggunakan jarak Euclidean pada layout spring graf (koordinat x,y hasil
nx.spring_layout) sebagai estimasi biaya menuju tujuan. Layout dihitung sekali dan
disimpan sebagai atribut node 'pos' oleh app.py sebelum memanggil modul ini.
"""
import time
import math
import networkx as nx


def _heuristic(G, scale=1.0):
    pos = nx.get_node_attributes(G, "pos")

    def h(u, v):
        if u not in pos or v not in pos:
            return 0
        (x1, y1), (x2, y2) = pos[u], pos[v]
        return scale * math.hypot(x1 - x2, y1 - y2)
    return h


def _admissible_scale(G):
    """Cari skala terkecil (delay / jarak-euclidean) di antara seluruh sisi, agar
    heuristik = scale * euclidean_distance TIDAK PERNAH melebih-lebihkan bobot
    sisi mana pun. Dengan sifat pertidaksamaan segitiga pada jarak Euclidean, ini
    menjamin heuristik tetap admissible untuk keseluruhan jalur, sehingga A* tetap
    menghasilkan solusi optimal (setara Dijkstra), sekaligus mengarahkan pencarian."""
    pos = nx.get_node_attributes(G, "pos")
    ratios = []
    for u, v, d in G.edges(data=True):
        dist = math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1])
        if dist > 1e-9:
            ratios.append(d["delay"] / dist)
    return min(ratios) if ratios else 0.0


def run(G, source, target, **kwargs):
    if not nx.get_node_attributes(G, "pos"):
        pos = nx.spring_layout(G, seed=42)
        nx.set_node_attributes(G, pos, "pos")

    h = _heuristic(G, scale=_admissible_scale(G))

    t0 = time.perf_counter()
    try:
        path = nx.astar_path(G, source, target, heuristic=h, weight="delay")
        cost = nx.astar_path_length(G, source, target, heuristic=h, weight="delay")
    except nx.NetworkXNoPath:
        path, cost = None, None
    runtime_ms = (time.perf_counter() - t0) * 1000
    return {"path": path, "cost": cost, "runtime_ms": round(runtime_ms, 4), "meta": {}}
