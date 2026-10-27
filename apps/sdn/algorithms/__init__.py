"""Registry algoritma optimasi routing yang tersedia untuk evaluasi.

Setiap modul mengekspos run(G, source, target, **params) -> dict, KECUALI
'mst' yang mengekspos run(G, **params) -> dict berbeda struktur (lihat mst.py).

Setiap entri juga membawa 'params': skema parameter yang dapat diatur pengguna
melalui UI simulator (dipakai untuk membangun formulir dan memvalidasi input).
"""
from . import dijkstra, astar, aco, genetic, pso, simulated_annealing, tabu_search, qlearning, greedy, mst

ALGORITHMS = {
    "dijkstra": {
        "label": "Dijkstra", "module": dijkstra, "type": "path", "params": [],
    },
    "astar": {
        "label": "A*", "module": astar, "type": "path", "params": [],
    },
    "aco": {
        "label": "Ant Colony Optimization", "module": aco, "type": "path",
        "params": [
            {"name": "n_ants", "label": "Jumlah Semut", "type": "int", "default": 20, "min": 5, "max": 100, "step": 1},
            {"name": "n_iter", "label": "Jumlah Iterasi", "type": "int", "default": 50, "min": 10, "max": 300, "step": 10},
            {"name": "alpha", "label": "Alpha (bobot feromon)", "type": "float", "default": 1.0, "min": 0, "max": 5, "step": 0.1},
            {"name": "beta", "label": "Beta (bobot heuristik)", "type": "float", "default": 2.0, "min": 0, "max": 5, "step": 0.1},
            {"name": "rho", "label": "Laju Evaporasi (rho)", "type": "float", "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01},
            {"name": "Q", "label": "Konstanta Q", "type": "float", "default": 100, "min": 1, "max": 1000, "step": 1},
        ],
    },
    "genetic": {
        "label": "Genetic Algorithm", "module": genetic, "type": "path",
        "params": [
            {"name": "pop_size", "label": "Ukuran Populasi", "type": "int", "default": 30, "min": 5, "max": 200, "step": 5},
            {"name": "generations", "label": "Jumlah Generasi", "type": "int", "default": 100, "min": 10, "max": 500, "step": 10},
            {"name": "mutation_rate", "label": "Laju Mutasi", "type": "float", "default": 0.1, "min": 0, "max": 1, "step": 0.05},
        ],
    },
    "pso": {
        "label": "Particle Swarm Optimization", "module": pso, "type": "path",
        "params": [
            {"name": "n_particles", "label": "Jumlah Partikel", "type": "int", "default": 25, "min": 5, "max": 150, "step": 5},
            {"name": "iterations", "label": "Jumlah Iterasi", "type": "int", "default": 60, "min": 10, "max": 300, "step": 10},
            {"name": "w", "label": "Bobot Inersia (w)", "type": "float", "default": 0.7, "min": 0, "max": 1.5, "step": 0.05},
            {"name": "c1", "label": "Koefisien Kognitif (c1)", "type": "float", "default": 1.5, "min": 0, "max": 3, "step": 0.1},
            {"name": "c2", "label": "Koefisien Sosial (c2)", "type": "float", "default": 1.5, "min": 0, "max": 3, "step": 0.1},
        ],
    },
    "simulated_annealing": {
        "label": "Simulated Annealing", "module": simulated_annealing, "type": "path",
        "params": [
            {"name": "T0", "label": "Suhu Awal (T0)", "type": "float", "default": 100, "min": 1, "max": 1000, "step": 1},
            {"name": "alpha", "label": "Laju Pendinginan (alpha)", "type": "float", "default": 0.95, "min": 0.5, "max": 0.999, "step": 0.001},
            {"name": "iterations", "label": "Jumlah Iterasi", "type": "int", "default": 800, "min": 50, "max": 5000, "step": 50},
        ],
    },
    "tabu_search": {
        "label": "Tabu Search", "module": tabu_search, "type": "path",
        "params": [
            {"name": "max_iter", "label": "Jumlah Iterasi", "type": "int", "default": 150, "min": 10, "max": 1000, "step": 10},
            {"name": "tabu_size", "label": "Ukuran Tabu List", "type": "int", "default": 15, "min": 2, "max": 100, "step": 1},
        ],
    },
    "qlearning": {
        "label": "Q-Learning (RL Sederhana)", "module": qlearning, "type": "path",
        "params": [
            {"name": "episodes", "label": "Jumlah Episode", "type": "int", "default": 500, "min": 50, "max": 5000, "step": 50},
            {"name": "alpha", "label": "Learning Rate (alpha)", "type": "float", "default": 0.1, "min": 0.01, "max": 1, "step": 0.01},
            {"name": "gamma", "label": "Discount Factor (gamma)", "type": "float", "default": 0.9, "min": 0, "max": 0.999, "step": 0.01},
            {"name": "epsilon", "label": "Epsilon (eksplorasi)", "type": "float", "default": 0.2, "min": 0, "max": 1, "step": 0.05},
            {"name": "max_steps", "label": "Maks. Langkah/Episode", "type": "int", "default": 40, "min": 5, "max": 200, "step": 5},
        ],
    },
    "greedy": {
        "label": "Greedy Routing", "module": greedy, "type": "path", "params": [],
    },
    "mst": {
        "label": "Minimum Spanning Tree", "module": mst, "type": "tree", "params": [],
    },
}


def clean_params(algo_key, raw_params):
    """Saring & konversi tipe parameter mentah dari request sesuai skema algoritma,
    mengabaikan kunci yang tidak dikenal dan memakai default bila tidak diisi/invalid."""
    spec = ALGORITHMS.get(algo_key)
    if not spec:
        return {}
    cleaned = {}
    for p in spec["params"]:
        name = p["name"]
        val = raw_params.get(name, p["default"]) if raw_params else p["default"]
        try:
            if p["type"] == "int":
                val = int(round(float(val)))
            else:
                val = float(val)
            val = max(p["min"], min(p["max"], val))
        except (TypeError, ValueError):
            val = p["default"]
        cleaned[name] = val
    return cleaned
