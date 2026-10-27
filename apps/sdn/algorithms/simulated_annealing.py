"""Simulated Annealing (SA) untuk optimasi jalur routing."""
import time
import math
import random


def _random_path(G, source, target, rng, max_len=30):
    path, current, visited = [source], source, {source}
    for _ in range(max_len):
        if current == target:
            return path
        neighbors = [v for v in G.neighbors(current) if v not in visited]
        if not neighbors:
            return None
        nxt = rng.choice(neighbors)
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path if path[-1] == target else None


def _cost(G, path):
    return sum(G[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1))


def run(G, source, target, T0=100.0, alpha=0.95, iterations=800, seed=None, **kwargs):
    rng = random.Random(seed)
    t0 = time.perf_counter()

    current = None
    for _ in range(200):
        current = _random_path(G, source, target, rng)
        if current:
            break
    if current is None:
        return {"path": None, "cost": None,
                "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
                "meta": {"note": "tidak ditemukan jalur awal yang valid"}}

    current_cost = _cost(G, current)
    best, best_cost = current, current_cost
    best_found_at = 0
    T = T0

    for it in range(1, iterations + 1):
        candidate = _random_path(G, source, target, rng)
        if candidate is None:
            T *= alpha
            continue
        candidate_cost = _cost(G, candidate)
        delta = candidate_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
            current, current_cost = candidate, candidate_cost
            if current_cost < best_cost:
                best, best_cost = current, current_cost
                best_found_at = it
        T *= alpha

    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "path": best,
        "cost": round(best_cost, 3),
        "runtime_ms": round(runtime_ms, 4),
        "meta": {
            "T0": T0, "alpha": alpha,
            "iterasi_dijalankan": iterations,
            "iterasi_ditemukan_terbaik": best_found_at,
            "suhu_akhir": round(T, 5),
        },
    }
