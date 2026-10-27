"""Minimum Spanning Tree — algoritma Kruskal berbasis NetworkX.

Berbeda dari algoritma lain, MST tidak menghasilkan satu jalur source->target,
melainkan pohon yang menghubungkan seluruh node dengan bobot total minimum.
Cocok untuk desain topologi/multicast tree, dievaluasi dengan metrics.compute_mst_metrics.
"""
import time
import networkx as nx


def run(G, source=None, target=None, **kwargs):
    t0 = time.perf_counter()
    T = nx.minimum_spanning_tree(G, weight="delay", algorithm="kruskal")
    edges = list(T.edges())
    total_weight = sum(G[u][v]["delay"] for u, v in edges)
    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "mst_edges": edges,
        "cost": round(total_weight, 3),
        "runtime_ms": round(runtime_ms, 4),
        "meta": {"num_edges": len(edges), "num_nodes": T.number_of_nodes()},
    }
