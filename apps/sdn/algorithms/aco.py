"""Ant Colony Optimization (ACO) untuk routing berbasis pheromone trail."""
import time
import random


def run(G, source, target, n_ants=20, n_iter=50, alpha=1.0, beta=2.0,
        rho=0.5, Q=100.0, seed=None, **kwargs):
    rng = random.Random(seed)
    t0 = time.perf_counter()

    pheromone = {(u, v): 1.0 for u in G for v in G[u]}
    best_path, best_cost = None, float("inf")
    best_found_at = 0

    for it in range(1, n_iter + 1):
        all_paths = []
        for _ant in range(n_ants):
            current, path, visited, length = source, [source], {source}, 0.0
            while current != target:
                neighbors = [v for v in G.neighbors(current) if v not in visited]
                if not neighbors:
                    break
                weights = []
                for v in neighbors:
                    tau = pheromone[(current, v)] ** alpha
                    eta = (1.0 / G[current][v]["delay"]) ** beta
                    weights.append(tau * eta)
                total_w = sum(weights) or 1e-9
                probs = [w / total_w for w in weights]
                nxt = rng.choices(neighbors, weights=probs)[0]
                length += G[current][nxt]["delay"]
                path.append(nxt)
                visited.add(nxt)
                current = nxt
            if current == target:
                all_paths.append((path, length))
                if length < best_cost:
                    best_path, best_cost = path, length
                    best_found_at = it

        for edge in pheromone:
            pheromone[edge] *= (1 - rho)
        for path, length in all_paths:
            if length <= 0:
                continue
            for i in range(len(path) - 1):
                pheromone[(path[i], path[i + 1])] += Q / length

    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "path": best_path,
        "cost": round(best_cost, 3) if best_path else None,
        "runtime_ms": round(runtime_ms, 4),
        "meta": {
            "n_ants": n_ants,
            "iterasi_dijalankan": n_iter,
            "iterasi_ditemukan_terbaik": best_found_at,
        },
    }
