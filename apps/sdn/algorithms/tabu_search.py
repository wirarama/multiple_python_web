"""Tabu Search untuk optimasi jalur routing dengan memori jangka pendek (tabu list)."""
import time
import random
from collections import deque


def _random_walk(G, source, target, rng, max_len=30, forced_prefix=None):
    if forced_prefix:
        path, visited = list(forced_prefix), set(forced_prefix)
        current = path[-1]
    else:
        path, visited, current = [source], {source}, source
    steps = 0
    while current != target and steps < max_len:
        neighbors = [v for v in G.neighbors(current) if v not in visited]
        if not neighbors:
            return None
        current = rng.choice(neighbors)
        path.append(current)
        visited.add(current)
        steps += 1
    return path if path[-1] == target else None


def _cost(G, path):
    return sum(G[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1))


def _neighbors_of(G, path, source, target, rng, n=10):
    results = []
    for _ in range(n):
        if len(path) > 2:
            idx = rng.randint(1, len(path) - 2)
            prefix = path[:idx + 1]
        else:
            prefix = [source]
        cand = _random_walk(G, source, target, rng, forced_prefix=prefix)
        if cand:
            results.append(cand)
    return results


def run(G, source, target, max_iter=150, tabu_size=15, seed=None, **kwargs):
    rng = random.Random(seed)
    t0 = time.perf_counter()

    current = None
    for _ in range(200):
        current = _random_walk(G, source, target, rng)
        if current:
            break
    if current is None:
        return {"path": None, "cost": None,
                "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
                "meta": {"note": "tidak ditemukan jalur awal yang valid"}}

    best = current
    best_cost = _cost(G, current)
    best_found_at = 0
    tabu = deque(maxlen=tabu_size)

    for it in range(1, max_iter + 1):
        candidates = [c for c in _neighbors_of(G, current, source, target, rng)
                      if tuple(c) not in tabu]
        if not candidates:
            continue
        candidates.sort(key=lambda p: _cost(G, p))
        current = candidates[0]
        tabu.append(tuple(current))
        c_cost = _cost(G, current)
        if c_cost < best_cost:
            best, best_cost = current, c_cost
            best_found_at = it

    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "path": best,
        "cost": round(best_cost, 3),
        "runtime_ms": round(runtime_ms, 4),
        "meta": {
            "tabu_size": tabu_size,
            "iterasi_dijalankan": max_iter,
            "iterasi_ditemukan_terbaik": best_found_at,
        },
    }
