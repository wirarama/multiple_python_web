# =====================================================================
# BAGIAN: FEDERATED REINFORCEMENT LEARNING - Q-ROUTING
# ---------------------------------------------------------------------
# Tiap klien (region) adalah agen Q-Routing (Boyan & Littman, 1994):
#   Q_x(d, y) = estimasi delay total bila paket di node x menuju d
#               diteruskan ke tetangga y.
#   Update  : Q_x(d,y) <- Q_x(d,y) + alpha * [ c(x,y) + min_z Q_y(d,z) - Q_x(d,y) ]
# Agen hanya belajar dari paket yang BERASAL dari regionnya sendiri
# (sumber di region tsb; tujuan sebagian besar lokal -> dest_locality).
# Setiap ronde, server menggabungkan tabel-Q para agen (rata-rata berbobot
# jumlah kunjungan, atau rata-rata biasa) lalu mengirim tabel-Q global ke
# seluruh agen -> pengetahuan rute tiap region dibagikan tanpa berbagi
# data trafik mentah.
# =====================================================================
import numpy as np

from .routing import oracle_routing, evaluate_routing, sample_pairs  # @@internal

DEFAULT_QROUTING = {
    "rounds": 20,
    "episodes_per_round": 200,   # paket (episode) yang disimulasikan tiap klien per ronde
    "alpha": 0.5,                # learning rate Q-learning
    "epsilon": 0.3,              # eksplorasi awal (epsilon-greedy)
    "epsilon_decay": 0.9,        # epsilon dikali faktor ini tiap ronde
    "epsilon_min": 0.02,
    "max_hops_factor": 2.0,      # batas hop per episode = faktor * jumlah node
    "cost_noise_pct": 5.0,       # variasi delay yang dialami tiap paket
    "client_fraction": 1.0,
    "dropout_prob": 0.0,
    "aggregation": "visit_weighted",   # visit_weighted | fedavg
    "train_centralized": True,
}


def _neighbors(net):
    ids = [n["node_id"] for n in net["nodes"]]
    idx = {n: i for i, n in enumerate(ids)}
    nb = [[] for _ in ids]
    for l in net["links"]:
        a, b = idx[l["source"]], idx[l["target"]]
        nb[a].append(b)
        nb[b].append(a)
    nb = [sorted(set(x)) for x in nb]
    return ids, idx, nb


def _cost_lookup(net, idx, true_values):
    cost = {}
    for l, w in zip(net["links"], true_values):
        a, b = idx[l["source"]], idx[l["target"]]
        cost[(a, b)] = float(w)
        cost[(b, a)] = float(w)
    return cost


def new_q(N, nb):
    """Q[x][d] = list nilai untuk tiap tetangga nb[x] (inisialisasi optimis 0)."""
    return [[[0.0] * len(nb[x]) for _ in range(N)] for x in range(N)]


def new_visits(N, nb):
    return [[[0] * len(nb[x]) for _ in range(N)] for x in range(N)]


def copy_q(Q):
    return [[row[:] for row in Qx] for Qx in Q]


def run_episodes(Q, V, nb, cost, sources, dest_pool_local, N, episodes, alpha,
                 epsilon, max_hops, noise, rng, dest_locality):
    """Simulasikan `episodes` paket; update Q dan hitung kunjungan V in-place."""
    total_delay, delivered = 0.0, 0
    all_nodes = list(range(N))
    for _ in range(int(episodes)):
        s = sources[int(rng.integers(len(sources)))]
        if dest_pool_local and len(dest_pool_local) > 1 and rng.random() < dest_locality:
            pool = dest_pool_local
        else:
            pool = all_nodes
        d = s
        while d == s:
            d = pool[int(rng.integers(len(pool)))]
        x, acc = s, 0.0
        for _h in range(max_hops):
            q = Q[x][d]
            nbx = nb[x]
            if rng.random() < epsilon:
                k = int(rng.integers(len(nbx)))
            else:
                k = min(range(len(q)), key=q.__getitem__)
            y = nbx[k]
            c = cost[(x, y)] * float(np.exp(rng.normal(0.0, noise)))
            acc += c
            nxt = 0.0 if y == d else min(Q[y][d])
            q[k] += alpha * (c + nxt - q[k])
            V[x][d][k] += 1
            x = y
            if x == d:
                delivered += 1
                total_delay += acc
                break
    return delivered, total_delay


def greedy_table(Q, Vtot, nb, N):
    """Tabel next-hop greedy: argmin Q, memprioritaskan aksi yang pernah dicoba."""
    table = [[-1] * N for _ in range(N)]
    for d in range(N):
        for x in range(N):
            if x == d:
                table[d][x] = d
                continue
            q, v = Q[x][d], Vtot[x][d]
            cand = [k for k in range(len(q)) if v[k] > 0] or list(range(len(q)))
            k = min(cand, key=q.__getitem__)
            table[d][x] = nb[x][k]
    return table


def aggregate_q(Qs, Vs, prev, nb, N, method="visit_weighted"):
    G = copy_q(prev)
    m = len(Qs)
    for x in range(N):
        for d in range(N):
            for k in range(len(nb[x])):
                if method == "fedavg":
                    G[x][d][k] = sum(Q[x][d][k] for Q in Qs) / m
                else:
                    w = sum(V[x][d][k] for V in Vs)
                    if w > 0:
                        G[x][d][k] = sum(Q[x][d][k] * V[x][d][k] for Q, V in zip(Qs, Vs)) / w
    return G


def add_visits(A, B, nb, N):
    for x in range(N):
        for d in range(N):
            for k in range(len(nb[x])):
                A[x][d][k] += B[x][d][k]


def run_qrouting(net, snapshot, cfg=None, num_pairs=30, seed=42, viz_pair=None, progress=None):
    c = dict(DEFAULT_QROUTING)
    c.update(cfg or {})
    R, EP = int(c["rounds"]), int(c["episodes_per_round"])
    alpha = float(c["alpha"])
    noise = float(c["cost_noise_pct"]) / 100.0
    ids, idx, nb = _neighbors(net)
    N = len(ids)
    max_hops = max(4, int(float(c["max_hops_factor"]) * N))
    true_w = [s["true_delay_ms"] for s in snapshot]
    cost = _cost_lookup(net, idx, true_w)
    clients = [cl["client_id"] for cl in net["clients"]]
    prof = {cl["client_id"]: cl for cl in net["clients"]}
    region_nodes = {cid: [idx[n["node_id"]] for n in net["nodes"] if n["region"] == cid]
                    for cid in clients}
    active = [cid for cid in clients if region_nodes[cid]]

    pairs = sample_pairs(net, num_pairs, seed, first_pair=viz_pair)
    oracle_table, oracle_paths = oracle_routing(net, true_w, pairs)
    entries = sum(len(nb[x]) for x in range(N)) * N
    P = entries

    Qg = new_q(N, nb)
    Vg_tot = new_visits(N, nb)
    rounds, g_tables = [], []
    cum_bytes = 0
    eps = float(c["epsilon"])
    for r in range(1, R + 1):
        rr = np.random.default_rng([seed, 55, r])
        m = max(1, int(round(float(c["client_fraction"]) * len(active))))
        sel = sorted(rr.choice(len(active), size=m, replace=False).tolist())
        selected = [active[i] for i in sel]
        dropped = [cid for cid in selected if rr.random() < float(c["dropout_prob"])]
        part = [cid for cid in selected if cid not in dropped]
        Qs, Vs, deliv, tdelay = [], [], 0, 0.0
        for cid in part:
            ci = clients.index(cid)
            crng = np.random.default_rng([seed, 66, r, ci])
            Ql, Vl = copy_q(Qg), new_visits(N, nb)
            dv, td = run_episodes(Ql, Vl, nb, cost, region_nodes[cid], region_nodes[cid], N, EP,
                                  alpha, eps, max_hops, noise, crng,
                                  float(prof[cid]["dest_locality"]))
            Qs.append(Ql)
            Vs.append(Vl)
            deliv += dv
            tdelay += td
        if Qs:
            Qg = aggregate_q(Qs, Vs, Qg, nb, N, c["aggregation"])
            for V in Vs:
                add_visits(Vg_tot, V, nb, N)
        table = greedy_table(Qg, Vg_tot, nb, N)
        rsum, _ = evaluate_routing(net, table, pairs, true_w, oracle_paths)
        g_tables.append(table)
        b = P * 4
        cum_bytes += b * len(selected) + b * len(part)
        rounds.append({"round": r, "selected": ";".join(selected), "dropped": ";".join(dropped),
                       "participants": len(part), "epsilon": round(eps, 4),
                       "episodes": EP * len(part),
                       "delivered_pct": round(deliv / max(EP * len(part), 1) * 100.0, 2),
                       "avg_episode_delay": round(tdelay / deliv, 3) if deliv else None,
                       "gap_pct": rsum["gap_pct"], "match_pct": rsum["match_pct"],
                       "success_pct": rsum["success_pct"], "avg_delay": rsum["avg_delay"],
                       "bytes_up": b * len(part), "bytes_down": b * len(selected),
                       "cum_bytes": cum_bytes})
        eps = max(float(c["epsilon_min"]), eps * float(c["epsilon_decay"]))
        if progress:
            progress(r, R)

    models = []
    o_sum, o_det = evaluate_routing(net, oracle_table, pairs, true_w, oracle_paths)
    models.append({"key": "oracle", "label": "Oracle (Dijkstra, delay sebenarnya)", "kind": "oracle",
                   "client_id": "", "color": "#111111", "routing": o_sum,
                   "table": oracle_table, "details": o_det})
    t_g = greedy_table(Qg, Vg_tot, nb, N)
    s_g, d_g = evaluate_routing(net, t_g, pairs, true_w, oracle_paths)
    models.append({"key": "global", "label": f"Global FedRL ({c['aggregation']})", "kind": "global",
                   "client_id": "", "color": "#d6006f", "routing": s_g, "table": t_g,
                   "details": d_g})

    def train_alone(sources, local_pool, loc, rseed):
        Q, Vt = new_q(N, nb), new_visits(N, nb)
        e = float(c["epsilon"])
        for r in range(1, R + 1):
            V = new_visits(N, nb)
            run_episodes(Q, V, nb, cost, sources, local_pool, N, EP, alpha, e, max_hops, noise,
                         np.random.default_rng([seed, rseed, r]), loc)
            add_visits(Vt, V, nb, N)
            e = max(float(c["epsilon_min"]), e * float(c["epsilon_decay"]))
        return Q, Vt

    if c.get("train_centralized", True):
        # satu agen terpusat melihat paket dari SEMUA region (episode = total semua klien)
        Qc, Vc = new_q(N, nb), new_visits(N, nb)
        e = float(c["epsilon"])
        for r in range(1, R + 1):
            V = new_visits(N, nb)
            for ci, cid in enumerate(active):
                run_episodes(Qc, V, nb, cost, region_nodes[cid], region_nodes[cid], N, EP, alpha,
                             e, max_hops, noise, np.random.default_rng([seed, 77, r, ci]),
                             float(prof[cid]["dest_locality"]))
            add_visits(Vc, V, nb, N)
            e = max(float(c["epsilon_min"]), e * float(c["epsilon_decay"]))
        t_c = greedy_table(Qc, Vc, nb, N)
        s_c, d_c = evaluate_routing(net, t_c, pairs, true_w, oracle_paths)
        models.append({"key": "centralized", "label": "Terpusat (1 agen, semua trafik)",
                       "kind": "centralized", "client_id": "", "color": "#52514e",
                       "routing": s_c, "table": t_c, "details": d_c})

    for ci, cid in enumerate(clients):
        if not region_nodes[cid]:
            continue
        Ql, Vl = train_alone(region_nodes[cid], region_nodes[cid],
                             float(prof[cid]["dest_locality"]), 88 + ci)
        t_l = greedy_table(Ql, Vl, nb, N)
        s_l, d_l = evaluate_routing(net, t_l, pairs, true_w, oracle_paths)
        # cakupan: proporsi pasangan (x,d) yang pernah dijelajahi agen
        cov = sum(1 for x in range(N) for d in range(N) if x != d and any(Vl[x][d])) / max(N * (N - 1), 1)
        models.append({"key": f"local:{cid}", "label": f"Lokal {cid} ({prof[cid]['name']})",
                       "kind": "local", "client_id": cid, "color": prof[cid]["color"],
                       "routing": s_l, "table": t_l, "details": d_l,
                       "coverage_pct": round(cov * 100.0, 2)})
    cov_g = sum(1 for x in range(N) for d in range(N) if x != d and any(Vg_tot[x][d])) / max(N * (N - 1), 1)
    models[1]["coverage_pct"] = round(cov_g * 100.0, 2)

    return {"mode": "qrouting", "config": c, "seed": seed, "node_ids": ids,
            "link_ids": [l["link_id"] for l in net["links"]],
            "pairs": [list(p) for p in pairs], "n_params": P, "rounds": rounds,
            "models": models, "global_round_tables": g_tables,
            "true_weights": [round(float(x), 4) for x in true_w]}
