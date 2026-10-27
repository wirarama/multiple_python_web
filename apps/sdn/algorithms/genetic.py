"""Genetic Algorithm (GA) untuk optimasi jalur routing."""
import time
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


def _crossover(p1, p2, rng):
    common = set(p1[1:-1]) & set(p2[1:-1])
    if not common:
        return p1
    node = rng.choice(list(common))
    i1, i2 = p1.index(node), p2.index(node)
    return p1[:i1] + p2[i2:]


def run(G, source, target, pop_size=30, generations=100, mutation_rate=0.1,
        seed=None, **kwargs):
    rng = random.Random(seed)
    t0 = time.perf_counter()

    population = [pp for _ in range(pop_size * 3)
                  if (pp := _random_path(G, source, target, rng)) is not None]
    population = population[:pop_size] if population else []

    if not population:
        return {"path": None, "cost": None,
                "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
                "meta": {"note": "tidak ditemukan jalur awal yang valid"}}

    best_cost_so_far = min(_cost(G, p) for p in population)
    best_found_at = 0

    for gen in range(1, generations + 1):
        population.sort(key=lambda p: _cost(G, p))
        parents = population[:max(2, pop_size // 2)]
        offspring = []
        attempts = 0
        while len(offspring) < pop_size // 2 and attempts < pop_size * 5:
            attempts += 1
            p1, p2 = rng.sample(parents, 2)
            child = _crossover(p1, p2, rng)
            if rng.random() < mutation_rate:
                mutated = _random_path(G, source, target, rng)
                if mutated:
                    child = mutated
            if child and child[0] == source and child[-1] == target:
                offspring.append(child)
        population = parents + offspring if offspring else parents

        gen_best_cost = min(_cost(G, p) for p in population)
        if gen_best_cost < best_cost_so_far:
            best_cost_so_far = gen_best_cost
            best_found_at = gen

    population.sort(key=lambda p: _cost(G, p))
    best = population[0]
    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "path": best,
        "cost": round(_cost(G, best), 3),
        "runtime_ms": round(runtime_ms, 4),
        "meta": {
            "pop_size": pop_size,
            "generations_dijalankan": generations,
            "generasi_ditemukan_terbaik": best_found_at,
        },
    }
