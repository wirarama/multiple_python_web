# =====================================================================
# BAGIAN: JARINGAN & DATA TRAFIK
# ---------------------------------------------------------------------
# Membangkitkan topologi jaringan (node, link), membagi node ke dalam
# region/domain yang masing-masing dikelola satu KLIEN federated
# (mis. controller SDN lokal / ISP regional), lalu mensimulasikan
# pengukuran trafik lokal tiap klien (data non-IID) beserta "snapshot"
# kondisi jaringan saat evaluasi routing.
#
# Format data (CSV terpisah):
#   nodes.csv   : node_id, x, y, region
#   links.csv   : link_id, source, target, distance_m, base_delay_ms,
#                 bandwidth_mbps, loss, traffic_weight
#   clients.csv : client_id, name, traffic_intensity, burstiness,
#                 peak_hour, dest_locality, color
#
# Model "kebenaran" (ground truth) delay sebuah link pada utilisasi rho:
#   delay = base_delay + Q * (12 / bandwidth) * rho / (1 - rho) + loss * RTO
# (antrian gaya M/M/1: delay melonjak tajam saat utilisasi mendekati 1).
# Model ML tidak diberi tahu rumus ini -- ia harus mempelajarinya dari
# data pengukuran lokal masing-masing klien.
# Hanya membutuhkan: numpy, networkx (standar library selain itu).
# =====================================================================
import csv
import json
import math
import os

import numpy as np
import networkx as nx

REGION_COLORS = ["#e8743b", "#1baf7a", "#a259d9", "#c98500", "#19a3b5",
                 "#e34948", "#7a8c00", "#8a6d3b", "#6b7b8c", "#b5651d"]
GLOBAL_COLOR = "#d6006f"       # warna jalur model global (berbeda dari warna region)

QUEUE_FACTOR = 8.0          # rata-rata panjang burst antrian (paket)
RTO_MS = 200.0              # retransmission timeout (ms) untuk efek packet loss
MAX_UTIL = 0.97
MIN_UTIL = 0.01

NODE_FIELDS = ["node_id", "x", "y", "region"]
LINK_FIELDS = ["link_id", "source", "target", "distance_m", "base_delay_ms",
               "bandwidth_mbps", "loss", "traffic_weight"]
CLIENT_FIELDS = ["client_id", "name", "traffic_intensity", "burstiness",
                 "peak_hour", "dest_locality", "color"]
MEAS_FIELDS = ["client_id", "t", "hour", "link_id", "source", "target",
               "utilization", "bandwidth_mbps", "base_delay_ms", "loss",
               "delay_ms", "split"]
SNAP_FIELDS = ["link_id", "source", "target", "owner", "utilization",
               "bandwidth_mbps", "base_delay_ms", "loss", "true_delay_ms"]


# ---------------------------------------------------------------- utilitas
def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def base_delay_from_distance(distance_m, max_distance_m):
    """Delay dasar (propagasi + pemrosesan) diturunkan dari jarak: 0.5 - 15 ms."""
    ratio = distance_m / max_distance_m if max_distance_m else 0.0
    return round(0.5 + 14.5 * ratio, 3)


def true_link_delay(utilization, bandwidth_mbps, base_delay_ms, loss):
    """Delay efektif sebenarnya (ms) sebuah link pada utilisasi tertentu.
    Bekerja untuk skalar maupun numpy array."""
    rho = np.clip(utilization, MIN_UTIL, MAX_UTIL)
    service_ms = 12.0 / np.maximum(bandwidth_mbps, 0.1)   # 1500 byte paket
    queue_ms = QUEUE_FACTOR * service_ms * rho / (1.0 - rho)
    return base_delay_ms + queue_ms + loss * RTO_MS


def diurnal(hour, peak_hour):
    """Pola harian trafik: 0.30 (sepi) .. 1.00 (jam puncak)."""
    return 0.65 + 0.35 * np.sin(2.0 * np.pi * (hour - peak_hour + 6.0) / 24.0)


def build_graph(net):
    """networkx.Graph dari dict jaringan (node: pos/region, edge: atribut link)."""
    G = nx.Graph()
    for n in net["nodes"]:
        G.add_node(n["node_id"], pos=(n["x"], n["y"]), region=n["region"])
    for l in net["links"]:
        G.add_edge(l["source"], l["target"], **l)
    return G


# ------------------------------------------------------ partisi region
def kmeans_regions(points, k, seed=42, iters=50):
    """K-means sederhana (numpy) untuk membagi node ke k region (klien)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    k = max(1, min(k, n))
    # inisialisasi k-means++
    centers = [pts[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((pts - c) ** 2, axis=1) for c in centers], axis=0)
        prob = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(pts[rng.choice(n, p=prob)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.stack([np.sum((pts - c) ** 2, axis=1) for c in centers], axis=1)
        new = np.argmin(d, axis=1)
        for j in range(k):                      # jamin tidak ada region kosong
            if not np.any(new == j):
                far = int(np.argmax(np.min(d, axis=1)))
                new[far] = j
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            centers[j] = pts[labels == j].mean(axis=0)
    # urutkan region dari kiri-atas agar penamaan stabil
    order = np.argsort(centers[:, 0] + centers[:, 1] * 0.001)
    remap = {int(old): new for new, old in enumerate(order)}
    return [remap[int(l)] for l in labels]


def default_clients(num_clients, seed=42):
    """Profil trafik bawaan tiap klien: intensitas dibuat berbeda-beda agar
    data lokal tiap klien NON-IID (ada region sepi dan region padat)."""
    rng = np.random.default_rng(seed + 7)
    if num_clients == 1:
        intens = [0.6]
    else:
        intens = list(np.linspace(0.25, 0.85, num_clients))
        rng.shuffle(intens)
    clients = []
    for c in range(num_clients):
        clients.append({
            "client_id": f"C{c + 1}",
            "name": f"Region {c + 1}",
            "traffic_intensity": round(float(intens[c]), 2),
            "burstiness": round(float(rng.uniform(0.03, 0.12)), 3),
            "peak_hour": int(rng.integers(8, 22)),
            "dest_locality": 0.6,
            "color": REGION_COLORS[c % len(REGION_COLORS)],
        })
    return clients


def assign_regions(net, num_clients, seed=42, keep_profiles=True):
    """Bagi ulang node ke num_clients region (k-means posisi)."""
    pts = [(n["x"], n["y"]) for n in net["nodes"]]
    labels = kmeans_regions(pts, num_clients, seed=seed)
    old = {c["client_id"]: c for c in net.get("clients", [])} if keep_profiles else {}
    fresh = default_clients(num_clients, seed=seed)
    clients = []
    for c in fresh:
        clients.append(old.get(c["client_id"], c))
    net["clients"] = clients
    for n, lab in zip(net["nodes"], labels):
        n["region"] = clients[lab]["client_id"]
    return net


# ------------------------------------------------------ topologi acak
def generate_topology(num_nodes=24, area_width=1000.0, area_height=1000.0,
                      k_neighbors=3, num_clients=4, seed=42):
    """Bangkitkan topologi: node acak pada area (meter), dihubungkan ke k
    tetangga terdekat (dijamin terhubung), lalu dibagi ke region klien."""
    num_nodes = int(max(4, min(int(num_nodes), 150)))
    num_clients = int(max(1, min(int(num_clients), 10, num_nodes // 2)))
    k_neighbors = int(max(1, min(int(k_neighbors), num_nodes - 1)))
    area_width, area_height = float(area_width), float(area_height)
    rng = np.random.default_rng(seed)

    nodes = []
    for i in range(num_nodes):
        nodes.append({"node_id": f"N{i + 1}",
                      "x": round(float(rng.uniform(0, area_width)), 1),
                      "y": round(float(rng.uniform(0, area_height)), 1),
                      "region": ""})
    byid = {n["node_id"]: n for n in nodes}
    G = nx.Graph()
    G.add_nodes_from(byid)
    for n in nodes:
        near = sorted((m for m in nodes if m is not n), key=lambda m: _dist(n, m))
        for m in near[:k_neighbors]:
            G.add_edge(n["node_id"], m["node_id"])
    while not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        best = None
        for other in comps[1:]:
            for a in comps[0]:
                for b in other:
                    d = _dist(byid[a], byid[b])
                    if best is None or d < best[0]:
                        best = (d, a, b)
        G.add_edge(best[1], best[2])

    max_d = math.hypot(area_width, area_height)
    links = []
    for i, (u, v) in enumerate(sorted(G.edges(), key=lambda e: (int(e[0][1:]), int(e[1][1:])))):
        d = _dist(byid[u], byid[v])
        links.append({
            "link_id": f"L{i + 1}", "source": u, "target": v,
            "distance_m": round(d, 1),
            "base_delay_ms": base_delay_from_distance(d, max_d),
            "bandwidth_mbps": float(rng.choice([10, 20, 50, 100])),
            "loss": round(float(rng.uniform(0.0005, 0.01)), 4),
            "traffic_weight": round(float(rng.uniform(0.6, 1.4)), 2),
        })
    net = {"nodes": nodes, "links": links, "clients": [],
           "area": {"width": area_width, "height": area_height}}
    assign_regions(net, num_clients, seed=seed, keep_profiles=False)
    return net


def recompute_link_geometry(net, link_ids=None):
    """Hitung ulang jarak & delay dasar link (mis. setelah node digeser)."""
    byid = {n["node_id"]: n for n in net["nodes"]}
    area = net.get("area", {"width": 1000.0, "height": 1000.0})
    max_d = math.hypot(area["width"], area["height"])
    for l in net["links"]:
        if link_ids is not None and l["link_id"] not in link_ids:
            continue
        d = _dist(byid[l["source"]], byid[l["target"]])
        l["distance_m"] = round(d, 1)
        l["base_delay_ms"] = base_delay_from_distance(d, max_d)
    return net


# ------------------------------------------------------ kepemilikan link
def link_observers(net):
    """Klien mana saja yang dapat MENGUKUR tiap link: link di dalam region
    diamati klien region tsb; link lintas-region (border) diamati keduanya."""
    region = {n["node_id"]: n["region"] for n in net["nodes"]}
    obs = {}
    for l in net["links"]:
        a, b = region[l["source"]], region[l["target"]]
        obs[l["link_id"]] = [a] if a == b else [a, b]
    return obs


def link_owner(net):
    """Pemilik utama link (region node sumber) -- dipakai untuk menentukan
    beban trafik link pada snapshot evaluasi."""
    region = {n["node_id"]: n["region"] for n in net["nodes"]}
    return {l["link_id"]: region[l["source"]] for l in net["links"]}


# ------------------------------------------------------ simulasi trafik
def simulate_measurements(net, time_steps=48, seed=42, test_ratio=0.2, noise_pct=5.0):
    """Bangkitkan data pengukuran LOKAL tiap klien: untuk setiap jam t, klien
    mengukur utilisasi & delay pada link yang dapat ia amati. Intensitas,
    burstiness, dan jam puncak berbeda per klien -> distribusi data non-IID.
    Mengembalikan list baris (dict) sesuai MEAS_FIELDS."""
    rng = np.random.default_rng(seed + 101)
    clients = {c["client_id"]: c for c in net["clients"]}
    obs = link_observers(net)
    rows = []
    for c in net["clients"]:
        cid = c["client_id"]
        for t in range(int(time_steps)):
            hour = t % 24
            for l in net["links"]:
                if cid not in obs[l["link_id"]]:
                    continue
                prof = clients[cid]
                rho = (float(prof["traffic_intensity"]) * float(l["traffic_weight"])
                       * diurnal(hour, float(prof["peak_hour"]))
                       + rng.normal(0.0, float(prof["burstiness"])))
                rho = float(np.clip(rho, MIN_UTIL, MAX_UTIL))
                d = true_link_delay(rho, float(l["bandwidth_mbps"]),
                                    float(l["base_delay_ms"]), float(l["loss"]))
                d = float(d * np.exp(rng.normal(0.0, noise_pct / 100.0)))
                rows.append({
                    "client_id": cid, "t": t, "hour": hour,
                    "link_id": l["link_id"], "source": l["source"], "target": l["target"],
                    "utilization": round(rho, 4),
                    "bandwidth_mbps": float(l["bandwidth_mbps"]),
                    "base_delay_ms": float(l["base_delay_ms"]),
                    "loss": float(l["loss"]),
                    "delay_ms": round(d, 4),
                    "split": "test" if rng.random() < test_ratio else "train",
                })
    return rows


def network_snapshot(net, eval_hour=19, seed=42):
    """Kondisi jaringan pada saat evaluasi routing (jam eval_hour): utilisasi
    tiap link ditentukan profil trafik region pemiliknya. true_delay_ms adalah
    bobot 'kebenaran' untuk menghitung jalur optimal (oracle)."""
    rng = np.random.default_rng(seed + 202)
    clients = {c["client_id"]: c for c in net["clients"]}
    owner = link_owner(net)
    rows = []
    for l in net["links"]:
        prof = clients[owner[l["link_id"]]]
        rho = (float(prof["traffic_intensity"]) * float(l["traffic_weight"])
               * diurnal(eval_hour, float(prof["peak_hour"]))
               + rng.normal(0.0, float(prof["burstiness"]) * 0.5))
        rho = float(np.clip(rho, MIN_UTIL, MAX_UTIL))
        d = true_link_delay(rho, float(l["bandwidth_mbps"]),
                            float(l["base_delay_ms"]), float(l["loss"]))
        rows.append({"link_id": l["link_id"], "source": l["source"], "target": l["target"],
                     "owner": owner[l["link_id"]], "utilization": round(rho, 4),
                     "bandwidth_mbps": float(l["bandwidth_mbps"]),
                     "base_delay_ms": float(l["base_delay_ms"]), "loss": float(l["loss"]),
                     "true_delay_ms": round(float(d), 4)})
    return rows


def dataset_stats(net, rows):
    """Ringkasan data lokal per klien (jumlah sampel, sebaran utilisasi)."""
    out = []
    bins = np.linspace(0, 1, 11)
    for c in net["clients"]:
        r = [x for x in rows if x["client_id"] == c["client_id"]]
        u = np.array([x["utilization"] for x in r]) if r else np.zeros(0)
        d = np.array([x["delay_ms"] for x in r]) if r else np.zeros(0)
        hist = np.histogram(u, bins=bins)[0].tolist() if len(u) else [0] * 10
        out.append({
            "client_id": c["client_id"], "name": c["name"], "color": c["color"],
            "samples": len(r),
            "train": sum(1 for x in r if x["split"] == "train"),
            "test": sum(1 for x in r if x["split"] == "test"),
            "links": len({x["link_id"] for x in r}),
            "util_min": round(float(u.min()), 3) if len(u) else 0,
            "util_mean": round(float(u.mean()), 3) if len(u) else 0,
            "util_max": round(float(u.max()), 3) if len(u) else 0,
            "delay_mean": round(float(d.mean()), 2) if len(d) else 0,
            "delay_max": round(float(d.max()), 2) if len(d) else 0,
            "util_hist": hist,
        })
    return out


# ------------------------------------------------------ CSV I/O
def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def rows_to_csv_text(fields, rows):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def parse_nodes(rows):
    return [{"node_id": r["node_id"].strip(), "x": float(r["x"]), "y": float(r["y"]),
             "region": r["region"].strip()} for r in rows]


def parse_links(rows):
    out = []
    for i, r in enumerate(rows):
        out.append({"link_id": (r.get("link_id") or f"L{i + 1}").strip(),
                    "source": r["source"].strip(), "target": r["target"].strip(),
                    "distance_m": float(r.get("distance_m") or 0),
                    "base_delay_ms": float(r["base_delay_ms"]),
                    "bandwidth_mbps": float(r["bandwidth_mbps"]),
                    "loss": float(r["loss"]),
                    "traffic_weight": float(r.get("traffic_weight") or 1.0)})
    return out


def parse_clients(rows):
    out = []
    for i, r in enumerate(rows):
        out.append({"client_id": r["client_id"].strip(),
                    "name": (r.get("name") or r["client_id"]).strip(),
                    "traffic_intensity": float(r["traffic_intensity"]),
                    "burstiness": float(r.get("burstiness") or 0.05),
                    "peak_hour": float(r.get("peak_hour") or 19),
                    "dest_locality": float(r.get("dest_locality") or 0.6),
                    "color": (r.get("color") or REGION_COLORS[i % len(REGION_COLORS)]).strip()})
    return out


def parse_measurements(rows):
    out = []
    for r in rows:
        out.append({"client_id": r["client_id"], "t": int(r["t"]), "hour": int(r["hour"]),
                    "link_id": r["link_id"], "source": r["source"], "target": r["target"],
                    "utilization": float(r["utilization"]),
                    "bandwidth_mbps": float(r["bandwidth_mbps"]),
                    "base_delay_ms": float(r["base_delay_ms"]), "loss": float(r["loss"]),
                    "delay_ms": float(r["delay_ms"]), "split": r["split"]})
    return out


def parse_snapshot(rows):
    out = []
    for r in rows:
        out.append({"link_id": r["link_id"], "source": r["source"], "target": r["target"],
                    "owner": r["owner"], "utilization": float(r["utilization"]),
                    "bandwidth_mbps": float(r["bandwidth_mbps"]),
                    "base_delay_ms": float(r["base_delay_ms"]), "loss": float(r["loss"]),
                    "true_delay_ms": float(r["true_delay_ms"])})
    return out


def save_network_csv(net, folder):
    _write_csv(os.path.join(folder, "nodes.csv"), NODE_FIELDS, net["nodes"])
    _write_csv(os.path.join(folder, "links.csv"), LINK_FIELDS, net["links"])
    _write_csv(os.path.join(folder, "clients.csv"), CLIENT_FIELDS, net["clients"])


def load_network_csv(folder, area=None):
    nodes = parse_nodes(_read_csv(os.path.join(folder, "nodes.csv")))
    links = parse_links(_read_csv(os.path.join(folder, "links.csv")))
    clients = parse_clients(_read_csv(os.path.join(folder, "clients.csv")))
    if area is None:
        area = {"width": max(n["x"] for n in nodes) * 1.02 + 1,
                "height": max(n["y"] for n in nodes) * 1.02 + 1}
    return {"nodes": nodes, "links": links, "clients": clients, "area": area}


def validate_network(net):
    """Periksa konsistensi data jaringan; kembalikan daftar pesan error."""
    errs = []
    ids = [n["node_id"] for n in net["nodes"]]
    if len(set(ids)) != len(ids):
        errs.append("node_id duplikat")
    cids = {c["client_id"] for c in net["clients"]}
    for n in net["nodes"]:
        if n["region"] not in cids:
            errs.append(f"node {n['node_id']}: region '{n['region']}' tidak ada di clients.csv")
    s = set(ids)
    for l in net["links"]:
        if l["source"] not in s or l["target"] not in s:
            errs.append(f"link {l['link_id']}: node tidak dikenal")
        if l["bandwidth_mbps"] <= 0:
            errs.append(f"link {l['link_id']}: bandwidth harus > 0")
    if not errs and net["links"]:
        if not nx.is_connected(build_graph(net)):
            errs.append("graf tidak terhubung penuh (ada node terisolasi)")
    used = {n["region"] for n in net["nodes"]}
    for c in cids - used:
        errs.append(f"klien {c} tidak memiliki node sama sekali")
    return errs
