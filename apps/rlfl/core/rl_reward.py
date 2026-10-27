# =====================================================================
# BAGIAN: REINFORCEMENT LEARNING ROUTING DENGAN REWARD TERURAI (PLUS/MINUS)
# ---------------------------------------------------------------------
# Satu agen Q-learning belajar merutekan paket dari SOURCE ke TARGET.
#   State  s = node tempat paket berada
#   Aksi   a = tetangga tujuan berikutnya
#   Reward r = jumlah komponen (positif = hadiah, negatif = hukuman):
#     (+) goal        : hadiah besar bila paket tiba di target
#     (+/-) progress  : w_progress * (hop_ke_target(x) - hop_ke_target(y))
#                       -> positif bila mendekat, negatif bila menjauh
#     (-) delay       : -w_delay * delay_link (ms)
#     (-) hop         : -hop_penalty per langkah (mendorong jalur pendek)
#     (-) loop        : -loop_penalty bila mengunjungi node yang sama lagi
#     (-) congestion  : -congestion_penalty * (util - ambang)/(1 - ambang)
#                       bila utilisasi link melewati ambang
#     (-) timeout     : -timeout_penalty bila batas hop habis sebelum tiba
#   Update Q-learning (memaksimalkan total reward / return):
#     Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
# Setiap langkah dari setiap episode dicatat beserta rincian reward,
# nilai Q sebelum/sesudah, dan pilihan (eksplorasi / eksploitasi).
# =====================================================================
import numpy as np
import networkx as nx

from .routing import oracle_routing  # @@internal

DEFAULT_RL = {
    "source": "",
    "target": "",
    "episodes": 300,
    "alpha": 0.3,
    "gamma": 0.95,
    "epsilon": 0.5,
    "epsilon_decay": 0.985,
    "epsilon_min": 0.02,
    "max_hops_factor": 1.5,
    "cost_noise_pct": 5.0,
    "q_init": 0.0,
    "reward_goal": 100.0,
    "w_progress": 5.0,
    "w_delay": 1.0,
    "hop_penalty": 1.0,
    "loop_penalty": 20.0,
    "congestion_threshold": 0.8,
    "congestion_penalty": 30.0,
    "timeout_penalty": 50.0,
}

REWARD_COMPONENTS = [
    # key, label, tanda
    ("goal", "Tiba di target", "+"),
    ("progress", "Kemajuan menuju target", "±"),
    ("delay", "Delay link", "-"),
    ("hop", "Biaya per hop", "-"),
    ("loop", "Mengulang node (loop)", "-"),
    ("congestion", "Link padat (kongesti)", "-"),
    ("timeout", "Batas hop habis", "-"),
]


def _graph_info(net, snapshot):
    ids = [n["node_id"] for n in net["nodes"]]
    idx = {n: i for i, n in enumerate(ids)}
    nb = [[] for _ in ids]
    cost, util = {}, {}
    for l, s in zip(net["links"], snapshot):
        a, b = idx[l["source"]], idx[l["target"]]
        nb[a].append(b)
        nb[b].append(a)
        for k in ((a, b), (b, a)):
            cost[k] = float(s["true_delay_ms"])
            util[k] = float(s["utilization"])
    nb = [sorted(set(x)) for x in nb]
    return ids, idx, nb, cost, util


def hop_distances(nb, target):
    """Jarak hop (BFS) dari setiap node ke target."""
    INF = 10 ** 6
    dist = [INF] * len(nb)
    dist[target] = 0
    queue = [target]
    while queue:
        x = queue.pop(0)
        for y in nb[x]:
            if dist[y] == INF:
                dist[y] = dist[x] + 1
                queue.append(y)
    return dist


def step_reward(x, y, target, visited, delay, util, hopd, c, is_last):
    """Rincian reward satu langkah x -> y. Mengembalikan (dict komponen, total)."""
    thr = float(c["congestion_threshold"])
    comp = {
        "goal": float(c["reward_goal"]) if y == target else 0.0,
        "progress": float(c["w_progress"]) * (hopd[x] - hopd[y]),
        "delay": -float(c["w_delay"]) * delay,
        "hop": -float(c["hop_penalty"]),
        "loop": -float(c["loop_penalty"]) if y in visited else 0.0,
        "congestion": (-float(c["congestion_penalty"]) * (util - thr) / max(1e-9, 1.0 - thr)
                       if util > thr else 0.0),
        "timeout": -float(c["timeout_penalty"]) if (is_last and y != target) else 0.0,
    }
    return comp, sum(comp.values())


def _r(v, d=3):
    return round(float(v), d)


def run_rl(net, snapshot, cfg=None, seed=42, progress=None):
    c = dict(DEFAULT_RL)
    c.update(cfg or {})
    ids, idx, nb, cost, util = _graph_info(net, snapshot)
    N = len(ids)
    src = c["source"] if c["source"] in idx else ids[0]
    dst = c["target"] if c["target"] in idx and c["target"] != src else ids[-1]
    s, t = idx[src], idx[dst]
    hopd = hop_distances(nb, t)
    max_hops = max(3, int(float(c["max_hops_factor"]) * N))
    alpha, gamma = float(c["alpha"]), float(c["gamma"])
    noise = float(c["cost_noise_pct"]) / 100.0
    Q = [[float(c["q_init"])] * len(nb[x]) for x in range(N)]
    visits = [[0] * len(nb[x]) for x in range(N)]
    rng = np.random.default_rng([seed, 909])
    eps = float(c["epsilon"])

    episodes, steps_all = [], []
    first_reach = None
    for ep in range(1, int(c["episodes"]) + 1):
        x, visited, ret = s, {s}, 0.0
        plus = minus = 0.0
        path, steps, explore = [ids[s]], [], 0
        reached, total_delay = False, 0.0
        for k in range(1, max_hops + 1):
            q = Q[x]
            if rng.random() < eps:
                ai = int(rng.integers(len(nb[x])))
                mode = "explore"
                explore += 1
            else:
                best = max(q)
                ai = [i for i, v in enumerate(q) if v == best][0]
                mode = "exploit"
            y = nb[x][ai]
            d = cost[(x, y)] * float(np.exp(rng.normal(0.0, noise)))
            u = util[(x, y)]
            is_last = (k == max_hops)
            comp, r = step_reward(x, y, t, visited, d, u, hopd, c, is_last)
            terminal = (y == t) or is_last
            max_next = 0.0 if terminal else max(Q[y])
            td_target = r + gamma * max_next
            q_before = q[ai]
            q[ai] = q_before + alpha * (td_target - q_before)
            visits[x][ai] += 1
            ret += r
            plus += sum(v for v in comp.values() if v > 0)
            minus += sum(v for v in comp.values() if v < 0)
            total_delay += d
            steps.append({
                "t": k, "x": ids[x], "y": ids[y], "mode": mode,
                "delay": _r(d), "util": _r(u), "comp": {kk: _r(vv) for kk, vv in comp.items()},
                "r": _r(r), "ret": _r(ret), "q_before": _r(q_before), "q_after": _r(q[ai]),
                "max_next": _r(max_next), "td_target": _r(td_target),
                "options": [{"y": ids[nb[x][j]], "q": _r(Q[x][j] if j != ai else q_before)}
                            for j in range(len(nb[x]))],
                "hop_to_target": hopd[y],
            })
            visited.add(y)
            path.append(ids[y])
            x = y
            if y == t:
                reached = True
                break
        if reached and first_reach is None:
            first_reach = ep
        episodes.append({"episode": ep, "epsilon": _r(eps, 4), "total_reward": _r(ret, 2),
                         "plus": _r(plus, 2), "minus": _r(minus, 2), "steps": len(steps),
                         "explore": explore, "reached": reached, "delay": _r(total_delay, 2),
                         "path": path})
        steps_all.append(steps)
        eps = max(float(c["epsilon_min"]), eps * float(c["epsilon_decay"]))
        if progress:
            progress(ep, int(c["episodes"]))

    greedy = evaluate_path_policy(Q, nb, ids, s, t, hopd, cost, util, c, max_hops)
    # jalur optimal berdasarkan delay (Dijkstra) & reward yang akan diperolehnya
    true_w = [float(x["true_delay_ms"]) for x in snapshot]
    _, opaths = oracle_routing(net, true_w, [(src, dst)])
    opt_path = opaths[(src, dst)]["path"]
    opt_steps = replay_path(opt_path, idx, ids, t, hopd, cost, util, c)
    policy = []
    for xi in range(N):
        if xi == t or not any(visits[xi]):
            policy.append(None)
        else:
            best = max(Q[xi])
            policy.append(ids[nb[xi][[i for i, v in enumerate(Q[xi]) if v == best][0]]])
    return {
        "mode": "rl", "config": c, "seed": seed, "source": src, "target": dst,
        "node_ids": ids, "neighbors": [[ids[y] for y in nb[x]] for x in range(N)],
        "hop_dist": hopd, "max_hops": max_hops, "components": REWARD_COMPONENTS,
        "episodes": episodes, "steps": steps_all, "first_reach": first_reach,
        "greedy": greedy,
        "optimal": {"path": opt_path, "delay": _r(sum(s_["delay"] for s_ in opt_steps), 3),
                    "total_reward": _r(sum(s_["r"] for s_ in opt_steps), 2), "steps": opt_steps},
        "q_table": [[_r(v) for v in row] for row in Q], "visits": visits, "policy": policy,
    }


def replay_path(path, idx, ids, t, hopd, cost, util, c):
    """Hitung rincian reward sebuah jalur tertentu (tanpa noise, tanpa update Q)."""
    out, visited, ret = [], set(), 0.0
    if not path:
        return out
    visited.add(idx[path[0]])
    for k, (a, b) in enumerate(zip(path[:-1], path[1:]), start=1):
        x, y = idx[a], idx[b]
        comp, r = step_reward(x, y, t, visited, cost[(x, y)], util[(x, y)], hopd, c, False)
        ret += r
        out.append({"t": k, "x": a, "y": b, "mode": "greedy", "delay": _r(cost[(x, y)]),
                    "util": _r(util[(x, y)]), "comp": {kk: _r(vv) for kk, vv in comp.items()},
                    "r": _r(r), "ret": _r(ret), "hop_to_target": hopd[y]})
        visited.add(y)
    return out


def evaluate_path_policy(Q, nb, ids, s, t, hopd, cost, util, c, max_hops):
    """Ikuti kebijakan greedy (argmax Q) dari source -> target."""
    x, visited, path, steps, ret = s, {s}, [ids[s]], [], 0.0
    for k in range(1, max_hops + 1):
        q = Q[x]
        best = max(q)
        ai = [i for i, v in enumerate(q) if v == best][0]
        y = nb[x][ai]
        is_last = k == max_hops
        comp, r = step_reward(x, y, t, visited, cost[(x, y)], util[(x, y)], hopd, c, is_last)
        ret += r
        steps.append({"t": k, "x": ids[x], "y": ids[y], "mode": "greedy",
                      "delay": _r(cost[(x, y)]), "util": _r(util[(x, y)]),
                      "comp": {kk: _r(vv) for kk, vv in comp.items()}, "r": _r(r), "ret": _r(ret),
                      "q_before": _r(q[ai]), "q_after": _r(q[ai]),
                      "options": [{"y": ids[nb[x][j]], "q": _r(Q[x][j])} for j in range(len(nb[x]))],
                      "hop_to_target": hopd[y]})
        loop = y in visited
        visited.add(y)
        path.append(ids[y])
        x = y
        if y == t or loop:
            break
    reached = path[-1] == ids[t]
    return {"path": path, "steps": steps, "reached": reached,
            "delay": _r(sum(st["delay"] for st in steps), 3), "total_reward": _r(ret, 2)}


# ----------------------------------------------------------- ekspor CSV
def rl_tables(res):
    """Ubah hasil RL menjadi baris-baris CSV (episodes, steps, q_table, greedy)."""
    keys = [k for k, _, _ in REWARD_COMPONENTS]
    ep_rows = [{**{k: v for k, v in e.items() if k != "path"}, "path": ">".join(e["path"])}
               for e in res["episodes"]]
    st_rows = []
    for ei, steps in enumerate(res["steps"], start=1):
        for s_ in steps:
            row = {"episode": ei, "step": s_["t"], "from": s_["x"], "to": s_["y"], "mode": s_["mode"],
                   "delay_ms": s_["delay"], "utilization": s_["util"]}
            for k in keys:
                row[f"r_{k}"] = s_["comp"][k]
            row.update({"reward": s_["r"], "return": s_["ret"], "q_before": s_["q_before"],
                        "q_after": s_["q_after"], "max_q_next": s_["max_next"],
                        "td_target": s_["td_target"]})
            st_rows.append(row)
    q_rows = []
    for x, row in enumerate(res["q_table"]):
        for j, v in enumerate(row):
            q_rows.append({"node": res["node_ids"][x], "action_next": res["neighbors"][x][j],
                           "q_value": v, "visits": res["visits"][x][j],
                           "greedy": int(res["policy"][x] == res["neighbors"][x][j])})
    g_rows = []
    for name in ("greedy", "optimal"):
        for s_ in res[name]["steps"]:
            row = {"policy": name, "step": s_["t"], "from": s_["x"], "to": s_["y"],
                   "delay_ms": s_["delay"], "utilization": s_["util"]}
            for k in keys:
                row[f"r_{k}"] = s_["comp"][k]
            row.update({"reward": s_["r"], "return": s_["ret"]})
            g_rows.append(row)
    return ep_rows, st_rows, q_rows, g_rows
