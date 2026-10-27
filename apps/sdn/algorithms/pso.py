"""Particle Swarm Optimization (PSO) diadaptasi untuk pemilihan jalur routing.

Setiap partikel merepresentasikan vektor preferensi (skor) untuk setiap sisi graf.
Sebuah jalur didekode dari partikel dengan berjalan dari source, pada tiap node
memilih tetangga (yang belum dikunjungi) dengan skor preferensi tertinggi.
Fitness = biaya (delay total) jalur yang berhasil dibentuk; jalur gagal/tidak
mencapai target diberi penalti besar agar swarm terdorong menuju solusi valid.
"""
import time
import random


def _decode_path(G, source, target, pref, max_len=30):
    current, path, visited = source, [source], {source}
    for _ in range(max_len):
        if current == target:
            return path
        neighbors = [v for v in G.neighbors(current) if v not in visited]
        if not neighbors:
            return None
        nxt = max(neighbors, key=lambda v: pref.get((current, v), 0.0))
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path if path[-1] == target else None


def _fitness(G, path):
    if path is None:
        return 1e6
    return sum(G[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1))


def run(G, source, target, n_particles=25, iterations=60, w=0.7, c1=1.5, c2=1.5,
        seed=None, **kwargs):
    rng = random.Random(seed)
    t0 = time.perf_counter()

    edges = [(u, v) for u in G for v in G[u]]
    dim = len(edges)
    idx = {e: i for i, e in enumerate(edges)}

    X = [[rng.uniform(0, 1) for _ in range(dim)] for _ in range(n_particles)]
    V = [[0.0] * dim for _ in range(n_particles)]

    def pref_dict(vec):
        return {e: vec[idx[e]] for e in edges}

    pbest = [x[:] for x in X]
    pbest_val = [_fitness(G, _decode_path(G, source, target, pref_dict(x))) for x in X]
    g_idx = min(range(n_particles), key=lambda i: pbest_val[i])
    gbest, gbest_val = pbest[g_idx][:], pbest_val[g_idx]
    best_found_at = 0

    for it in range(1, iterations + 1):
        for i in range(n_particles):
            for d in range(dim):
                r1, r2 = rng.random(), rng.random()
                V[i][d] = (w * V[i][d]
                           + c1 * r1 * (pbest[i][d] - X[i][d])
                           + c2 * r2 * (gbest[d] - X[i][d]))
                X[i][d] = min(max(X[i][d] + V[i][d], 0.0), 1.0)
            val = _fitness(G, _decode_path(G, source, target, pref_dict(X[i])))
            if val < pbest_val[i]:
                pbest[i], pbest_val[i] = X[i][:], val
                if val < gbest_val:
                    gbest, gbest_val = X[i][:], val
                    best_found_at = it

    best_path = _decode_path(G, source, target, pref_dict(gbest))
    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "path": best_path,
        "cost": round(gbest_val, 3) if best_path else None,
        "runtime_ms": round(runtime_ms, 4),
        "meta": {
            "n_particles": n_particles,
            "iterasi_dijalankan": iterations,
            "iterasi_ditemukan_terbaik": best_found_at,
        },
    }
