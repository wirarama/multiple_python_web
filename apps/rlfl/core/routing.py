# =====================================================================
# BAGIAN: ROUTING (tabel next-hop, ekstraksi jalur, metrik jalur)
# ---------------------------------------------------------------------
# Setiap model (lokal / global / terpusat / oracle) pada akhirnya
# direpresentasikan sebagai TABEL NEXT-HOP: table[d][x] = indeks node
# berikutnya dari node x menuju tujuan d. Jalur source->target diperoleh
# dengan mengikuti next-hop tsb. Kualitas jalur selalu diukur memakai
# delay SEBENARNYA (snapshot), lalu dibandingkan dengan jalur optimal
# (oracle = Dijkstra pada delay sebenarnya).
# =====================================================================
import numpy as np
import networkx as nx


def edge_key(u, v):
    return (u, v) if u < v else (v, u)


def weight_map(net, values):
    """values: list bobot dengan urutan sama seperti net['links']."""
    return {edge_key(l["source"], l["target"]): float(w) for l, w in zip(net["links"], values)}


def next_hop_table(net, values):
    """Tabel next-hop (shortest path, Dijkstra) memakai bobot `values` per link."""
    ids = [n["node_id"] for n in net["nodes"]]
    idx = {n: i for i, n in enumerate(ids)}
    G = nx.Graph()
    G.add_nodes_from(ids)
    for l, w in zip(net["links"], values):
        G.add_edge(l["source"], l["target"], weight=max(float(w), 1e-6))
    N = len(ids)
    table = [[-1] * N for _ in range(N)]
    for d in ids:
        pred, _ = nx.dijkstra_predecessor_and_distance(G, d, weight="weight")
        di = idx[d]
        for x in ids:
            if x == d:
                table[di][idx[x]] = di
            elif pred.get(x):
                table[di][idx[x]] = idx[sorted(pred[x])[0]]
    return table


def extract_path(table, ids, source, target, max_hops=None):
    """Ikuti tabel next-hop dari source ke target. None bila gagal/loop."""
    idx = {n: i for i, n in enumerate(ids)}
    s, d = idx[source], idx[target]
    max_hops = max_hops or 2 * len(ids)
    path, seen, x = [source], {s}, s
    for _ in range(max_hops):
        if x == d:
            return path
        y = table[d][x]
        if y is None or y < 0 or y in seen:
            return None
        path.append(ids[y])
        seen.add(y)
        x = y
    return path if x == d else None


def path_metrics(path, net, true_values):
    """Metrik jalur memakai kondisi jaringan sebenarnya."""
    if not path:
        return {"delay": None, "hops": None, "bottleneck_bw": None, "loss_pct": None}
    attr = {edge_key(l["source"], l["target"]): l for l in net["links"]}
    tw = weight_map(net, true_values)
    delay, bw, keep = 0.0, float("inf"), 1.0
    for u, v in zip(path[:-1], path[1:]):
        k = edge_key(u, v)
        delay += tw[k]
        bw = min(bw, float(attr[k]["bandwidth_mbps"]))
        keep *= (1.0 - float(attr[k]["loss"]))
    return {"delay": round(delay, 3), "hops": len(path) - 1,
            "bottleneck_bw": bw if bw != float("inf") else None,
            "loss_pct": round((1.0 - keep) * 100.0, 3)}


def sample_pairs(net, k=30, seed=42, first_pair=None):
    """Pasangan (source, target) untuk evaluasi routing (acak, unik)."""
    ids = [n["node_id"] for n in net["nodes"]]
    rng = np.random.default_rng(seed + 505)
    pairs, seen = [], set()
    if first_pair and first_pair[0] in ids and first_pair[1] in ids and first_pair[0] != first_pair[1]:
        pairs.append(tuple(first_pair))
        seen.add(tuple(first_pair))
    max_pairs = len(ids) * (len(ids) - 1)
    k = min(int(k), max_pairs)
    tries = 0
    while len(pairs) < k and tries < 50 * k:
        tries += 1
        a, b = rng.choice(len(ids), size=2, replace=False)
        p = (ids[int(a)], ids[int(b)])
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    return pairs


def evaluate_routing(net, table, pairs, true_values, oracle_paths):
    """Bandingkan jalur sebuah model dengan jalur optimal untuk tiap pasangan."""
    ids = [n["node_id"] for n in net["nodes"]]
    details, gaps, delays, hops, ok, match = [], [], [], [], 0, 0
    for (s, d) in pairs:
        p = extract_path(table, ids, s, d)
        m = path_metrics(p, net, true_values)
        opt = oracle_paths[(s, d)]
        if p is not None:
            ok += 1
            gap = (m["delay"] - opt["delay"]) / opt["delay"] * 100.0 if opt["delay"] else 0.0
            gaps.append(gap)
            delays.append(m["delay"])
            hops.append(m["hops"])
            same = p == opt["path"]
            match += int(same)
        else:
            gap, same = None, False
        details.append({"source": s, "target": d, "path": p, "delay": m["delay"],
                        "hops": m["hops"], "bottleneck_bw": m["bottleneck_bw"],
                        "loss_pct": m["loss_pct"],
                        "gap_pct": round(gap, 3) if gap is not None else None,
                        "optimal": same, "success": p is not None})
    n = max(len(pairs), 1)
    summary = {
        "success_pct": round(ok / n * 100.0, 2),
        "gap_pct": round(float(np.mean(gaps)), 3) if gaps else None,
        "match_pct": round(match / n * 100.0, 2),
        "avg_delay": round(float(np.mean(delays)), 3) if delays else None,
        "avg_hops": round(float(np.mean(hops)), 2) if hops else None,
    }
    return summary, details


def oracle_routing(net, true_values, pairs):
    table = next_hop_table(net, true_values)
    ids = [n["node_id"] for n in net["nodes"]]
    paths = {}
    for (s, d) in pairs:
        p = extract_path(table, ids, s, d)
        paths[(s, d)] = {"path": p, **path_metrics(p, net, true_values)}
    return table, paths
