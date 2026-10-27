"""Algoritma Dijkstra — shortest path eksak berbasis NetworkX."""
import time
import networkx as nx


def run(G, source, target, **kwargs):
    t0 = time.perf_counter()
    try:
        path = nx.dijkstra_path(G, source, target, weight="delay")
        cost = nx.dijkstra_path_length(G, source, target, weight="delay")
    except nx.NetworkXNoPath:
        path, cost = None, None
    runtime_ms = (time.perf_counter() - t0) * 1000
    return {"path": path, "cost": cost, "runtime_ms": round(runtime_ms, 4), "meta": {}}
