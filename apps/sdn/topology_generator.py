"""Generator topologi acak berbasis simulasi penempatan node pada suatu luas
wilayah (area_width x area_height, satuan meter). Dipakai oleh endpoint
/api/topology/generate agar pengguna dapat mengatur jumlah node dan luas
wilayah simulasi, mirip skenario deployment jaringan sensor/SDN nirkabel.

Delay tiap link diturunkan dari jarak Euclidean antar node (semakin jauh,
semakin besar delay), sedangkan bandwidth dan packet loss dasar diambil acak
dalam rentang realistis agar tetap bervariasi antar link.
"""
import math
import random
import networkx as nx

MIN_DELAY_MS = 1.0
MAX_DELAY_MS = 20.0
BANDWIDTH_RANGE = (20.0, 100.0)
LOSS_RANGE = (0.0005, 0.01)


def _distance(pos, a, b):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    return math.hypot(x1 - x2, y1 - y2)


def generate_random_topology(num_nodes, area_width, area_height, k_neighbors=3, seed=None):
    """Bangkitkan graf terhubung dari num_nodes node yang tersebar acak pada
    area_width x area_height meter, dihubungkan ke k_neighbors tetangga terdekat
    (lalu dijamin terhubung penuh bila masih ada komponen terpisah)."""
    num_nodes = max(2, int(num_nodes))
    area_width = max(10.0, float(area_width))
    area_height = max(10.0, float(area_height))
    k_neighbors = max(1, min(int(k_neighbors), num_nodes - 1))
    rng = random.Random(seed)

    nodes = [f"N{i + 1}" for i in range(num_nodes)]
    pos = {n: (rng.uniform(0, area_width), rng.uniform(0, area_height)) for n in nodes}

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for n in nodes:
        nearest = sorted((m for m in nodes if m != n), key=lambda m: _distance(pos, n, m))
        for m in nearest[:k_neighbors]:
            G.add_edge(n, m)

    # jamin graf terhubung penuh: sambungkan komponen terpisah lewat pasangan node terdekat
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        c0 = components[0]
        best = None
        for other in components[1:]:
            for a in c0:
                for b in other:
                    d = _distance(pos, a, b)
                    if best is None or d < best[0]:
                        best = (d, a, b)
        G.add_edge(best[1], best[2])

    max_possible_dist = math.hypot(area_width, area_height)
    for u, v in G.edges():
        d = _distance(pos, u, v)
        ratio = d / max_possible_dist if max_possible_dist else 0
        delay = round(MIN_DELAY_MS + ratio * (MAX_DELAY_MS - MIN_DELAY_MS), 2)
        bandwidth = round(rng.uniform(*BANDWIDTH_RANGE), 1)
        loss = round(rng.uniform(*LOSS_RANGE), 4)
        G[u][v].update(delay=delay, bandwidth=bandwidth, loss=loss, weight=delay,
                        distance_m=round(d, 2))

    return G, pos
