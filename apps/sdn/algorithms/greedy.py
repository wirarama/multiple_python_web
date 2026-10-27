"""Greedy Routing — setiap langkah memilih tetangga dengan bobot (delay) terkecil
yang belum dikunjungi, tanpa melihat informasi global (tidak menjamin optimal)."""
import time


def run(G, source, target, **kwargs):
    t0 = time.perf_counter()
    current = source
    path = [source]
    visited = {source}
    success = True

    while current != target:
        neighbors = {
            v: G[current][v]["delay"]
            for v in G.neighbors(current)
            if v not in visited
        }
        if not neighbors:
            success = False
            break
        nxt = min(neighbors, key=neighbors.get)
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    runtime_ms = (time.perf_counter() - t0) * 1000
    if not success:
        return {"path": None, "cost": None, "runtime_ms": round(runtime_ms, 4),
                "meta": {"note": "terjebak local minimum / dead-end"}}

    cost = sum(G[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1))
    return {"path": path, "cost": cost, "runtime_ms": round(runtime_ms, 4), "meta": {}}
