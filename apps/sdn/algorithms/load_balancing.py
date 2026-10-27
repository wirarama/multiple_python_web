"""Simulator Load Balancing untuk grid / distributed computing (server & client).

Melengkapi paket algorithms/ (yang semula berisi algoritma optimasi ROUTING)
dengan versi yang sama untuk masalah OPTIMASI LOAD BALANCING: menentukan server
mana yang menangani tiap request klien pada sebuah cluster/grid komputasi
terdistribusi, agar beban merata dan waktu respons minimal.

Setiap server disimulasikan sebagai node grid dengan 'capacity' (kapasitas
proses, satuan abstrak, mis. mirip bandwidth pada modul routing) yang tersebar
pada suatu area (mirip topology_generator.py). Klien mengirim sejumlah request,
tiap request punya 'demand' (beban proses, satuan abstrak mirip ukuran paket).
Latensi jaringan klien->server diturunkan dari jarak Euclidean, sama seperti
delay antar-link pada topology_generator.py.

Algoritma yang disediakan mengadaptasi teknik yang SUDAH ADA pada paket ini
untuk masalah routing (aco.py, genetic.py, pso.py, simulated_annealing.py,
tabu_search.py, serta strategi greedy sederhana seperti greedy.py/dijkstra.py)
ke ruang solusi baru: vektor penugasan request->server, bukan lagi jalur pada
graf. qlearning.py dan mst.py tidak diadaptasi karena tidak alami untuk
masalah assignment ini.

Setiap fungsi run(...) mengembalikan {"assignment": [...], "cost": ...,
"runtime_ms": ..., "meta": {...}} -- bentuk yang sama gayanya dengan modul
routing agar mudah dipakai ulang oleh app.py & frontend.
"""
import math
import random
import time

# --------------------------------------------------------------- konstanta --
BASE_LATENCY_MS = 1.0
MAX_LATENCY_MS = 15.0
PROC_MS_PER_UNIT = 2.0          # waktu proses dasar per satuan demand request
QOS_WEIGHTS = {"latency": 0.4, "fairness": 0.35, "dropped": 0.25}
QOS_CLASSES = [
    (85, "Excellent"),
    (70, "Good"),
    (50, "Fair"),
    (0, "Poor"),
]


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


# ----------------------------------------------------------- simulasi grid --
def generate_cluster(num_servers=6, num_clients=8, requests_per_client=4,
                      area_width=1000.0, area_height=1000.0,
                      capacity_range=(60.0, 220.0), demand_range=(1.0, 20.0),
                      seed=None):
    """Bangkitkan cluster/grid komputasi terdistribusi: sejumlah server dengan
    kapasitas & posisi acak pada suatu area, sejumlah klien yang tiap-tiap
    mengirim beberapa request dengan beban (demand) acak. Latensi jaringan
    klien->server diturunkan dari jarak Euclidean (mirip topology_generator.py).
    """
    num_servers = max(1, int(num_servers))
    num_clients = max(1, int(num_clients))
    requests_per_client = max(1, int(requests_per_client))
    area_width = max(10.0, float(area_width))
    area_height = max(10.0, float(area_height))
    rng = random.Random(seed)

    servers = []
    for i in range(num_servers):
        servers.append({
            "id": f"SRV{i + 1}",
            "x": rng.uniform(0, area_width),
            "y": rng.uniform(0, area_height),
            "capacity": round(rng.uniform(*capacity_range), 1),
        })

    clients = []
    for i in range(num_clients):
        clients.append({
            "id": f"CLI{i + 1}",
            "x": rng.uniform(0, area_width),
            "y": rng.uniform(0, area_height),
        })

    area = {"width": area_width, "height": area_height}
    latency = _compute_latency(servers, clients, area)

    requests = []
    rid = 1
    for c in clients:
        for _ in range(requests_per_client):
            requests.append({
                "id": f"R{rid}",
                "client_id": c["id"],
                "demand": round(rng.uniform(*demand_range), 2),
            })
            rid += 1

    return {
        "servers": servers,
        "clients": clients,
        "requests": requests,
        "latency": latency,
        "area": area,
    }


def _compute_latency(servers, clients, area):
    """Turunkan latensi klien->server dari jarak Euclidean posisi (x, y)
    tiap server/klien terhadap luas area aktif -- dipakai baik saat cluster
    dibangkitkan maupun setelah posisi server/klien digeser manual (drag &
    drop) pada visualisasi, lewat recompute_latency()."""
    max_dist = math.hypot(area["width"], area["height"]) or 1.0
    latency = {}
    for c in clients:
        for s in servers:
            d = math.hypot(c["x"] - s["x"], c["y"] - s["y"])
            ratio = min(d / max_dist, 1.0)
            latency[f"{c['id']}|{s['id']}"] = round(
                BASE_LATENCY_MS + ratio * (MAX_LATENCY_MS - BASE_LATENCY_MS), 3)
    return latency


def recompute_latency(cluster):
    """Hitung ulang tabel latensi (in-place pada dict cluster) berdasarkan
    posisi server/klien saat ini. Dipanggil setelah node digeser pada
    visualisasi cluster, sehingga hasil algoritma load balancing (yang
    memakai latensi jaringan sbg salah satu komponen response time) ikut
    menyesuaikan posisi baru."""
    cluster["latency"] = _compute_latency(cluster["servers"], cluster["clients"], cluster["area"])
    return cluster


def _latency_ms(cluster, client_id, server_id):
    return cluster["latency"].get(f"{client_id}|{server_id}", MAX_LATENCY_MS)


# ------------------------------------------------------------------- biaya --
def _congestion_factor(utilization_ratio):
    u = min(max(utilization_ratio, 0.0), 1.0)
    return 1.0 / max(0.05, 1 - u * 0.95)


def evaluate_assignment(cluster, assignment):
    """Hitung metrik lengkap untuk sebuah penugasan request->server (assignment:
    list server_id sepanjang jumlah request, urutan sama dengan cluster['requests']).
    Request diproses berurutan per server; begitu beban kumulatif server melebihi
    kapasitasnya, request berikutnya pada server itu dianggap 'dropped' (mirip
    packet loss karena overload, hanya disimulasikan bukan diukur langsung)."""
    servers = {s["id"]: s for s in cluster["servers"]}
    requests = cluster["requests"]
    if not assignment or len(assignment) != len(requests):
        return {"valid": False}

    cum_load = {sid: 0.0 for sid in servers}
    served_loads = {sid: 0.0 for sid in servers}
    per_request = []          # nilai response_ms saja (None jika dropped) -- dipakai utk agregasi cepat
    request_log = []          # detail tiap request -- dipakai utk log klien/server di UI
    dropped = 0

    for req, sid in zip(requests, assignment):
        srv = servers.get(sid)
        if srv is None:
            dropped += 1
            per_request.append(None)
            request_log.append({"request_id": req["id"], "client_id": req["client_id"],
                                 "server_id": None, "demand": req["demand"],
                                 "response_ms": None, "dropped": True})
            continue
        cum_load[sid] += req["demand"]
        if cum_load[sid] > srv["capacity"]:
            dropped += 1
            per_request.append(None)
            request_log.append({"request_id": req["id"], "client_id": req["client_id"],
                                 "server_id": sid, "demand": req["demand"],
                                 "response_ms": None, "dropped": True})
            continue
        served_loads[sid] += req["demand"]
        util_ratio = cum_load[sid] / srv["capacity"] if srv["capacity"] else 1.0
        latency = _latency_ms(cluster, req["client_id"], sid)
        response_ms = req["demand"] * PROC_MS_PER_UNIT * _congestion_factor(util_ratio) + latency
        response_ms = round(response_ms, 3)
        per_request.append(response_ms)
        request_log.append({"request_id": req["id"], "client_id": req["client_id"],
                             "server_id": sid, "demand": req["demand"],
                             "response_ms": response_ms, "dropped": False})

    response_values = [v for v in per_request if v is not None]
    total = len(requests)
    served = total - dropped

    per_server = []
    util_ratios = []
    for sid, srv in servers.items():
        load = cum_load[sid]
        cap = srv["capacity"] or 1e-6
        util_pct = load / cap * 100
        util_ratios.append(min(load, cap) / cap if cap else 0.0)
        per_server.append({
            "id": sid,
            "capacity": srv["capacity"],
            "load": round(load, 2),
            "requests_count": sum(1 for s in assignment if s == sid),
            "utilization_pct": round(util_pct, 1),
        })

    n = len(util_ratios) or 1
    sum_u = sum(util_ratios)
    sum_u2 = sum(u * u for u in util_ratios) or 1e-9
    fairness_index = round((sum_u ** 2) / (n * sum_u2), 4) if sum_u > 0 else 1.0

    avg_response_ms = round(sum(response_values) / len(response_values), 3) if response_values else None
    max_response_ms = round(max(response_values), 3) if response_values else None
    max_utilization_pct = round(max((p["utilization_pct"] for p in per_server), default=0.0), 1)
    dropped_pct = round(dropped / total * 100, 2) if total else 0.0

    return {
        "valid": True,
        "per_server": per_server,
        "served": served,
        "total": total,
        "dropped": dropped,
        "dropped_pct": dropped_pct,
        "avg_response_ms": avg_response_ms,
        "max_response_ms": max_response_ms,
        "max_utilization_pct": max_utilization_pct,
        "fairness_index": fairness_index,
        "request_log": request_log,
    }


# ------------------------------------------------------ saran auto-scaling --
SCALE_WARN_UTIL_PCT = 80.0     # ambang "sibuk" -- mulai disarankan bersiap scale up
SCALE_CRIT_UTIL_PCT = 100.0    # ambang "overload" -- request mulai di-drop
SCALE_TARGET_UTIL_PCT = 70.0   # target utilisasi setelah scale up (headroom aman)


def scaling_suggestions(cluster, metrics,
                         warn_pct=SCALE_WARN_UTIL_PCT, crit_pct=SCALE_CRIT_UTIL_PCT,
                         target_pct=SCALE_TARGET_UTIL_PCT):
    """Beri saran scale up (horizontal / vertical) untuk server yang beban
    per_server-nya (hasil evaluate_assignment) melebihi ambang aman.

    Heuristik pemilihan jenis scaling:
    - utilization_pct >= crit_pct (100%, sudah ada request yang di-drop):
      cluster kelebihan beban SECARA STRUKTURAL -> disarankan SCALE OUT
      (horizontal): tambah instance server baru berkapasitas serupa agar
      beban terbagi, karena menaikkan kapasitas 1 server saja tidak
      menambah daya tampung keseluruhan cluster saat trafik terus datang.
    - warn_pct <= utilization_pct < crit_pct (sibuk tapi belum drop):
      disarankan SCALE UP (vertical): tambah kapasitas server tsb sampai
      titik aman, langkah pencegahan sebelum terjadi packet/request drop.

    Mengembalikan list terurut (paling kritis dulu) berisi rekomendasi per
    server yang melewati ambang 'warn_pct'."""
    if not metrics or not metrics.get("valid"):
        return []

    servers_by_id = {s["id"]: s for s in cluster["servers"]}
    out = []
    for p in metrics.get("per_server", []):
        util = p["utilization_pct"]
        if util < warn_pct:
            continue
        srv = servers_by_id.get(p["id"], {})
        load = p["load"]
        capacity = p["capacity"] or 1e-6
        target_ratio = target_pct / 100.0

        # Vertical: kapasitas baru agar beban saat ini ~= target_pct dari kapasitas.
        suggested_capacity = round(max(capacity, load / target_ratio), 1)
        capacity_increase_pct = round((suggested_capacity / capacity - 1) * 100, 1) if capacity else None

        # Horizontal: berapa TOTAL instance (server ini + kloningan) yang
        # dibutuhkan agar beban terbagi rata di bawah target_pct per instance.
        total_instances_needed = max(1, math.ceil(load / (capacity * target_ratio))) if capacity else 1
        additional_instances = max(0, total_instances_needed - 1)

        critical = util >= crit_pct
        level = "critical" if critical else "warning"
        recommendation = "horizontal" if critical else "vertical"

        if critical:
            note = (
                f"{p['id']} overload ({util:.1f}% utilisasi, request mulai ditolak). "
                f"Disarankan SCALE OUT (horizontal): tambah {additional_instances} instance "
                f"server baru berkapasitas serupa (~{srv.get('capacity', capacity):.1f}) "
                f"agar beban terbagi dan tidak ada lagi request yang di-drop."
            )
        else:
            note = (
                f"{p['id']} sibuk ({util:.1f}% utilisasi, mendekati kapasitas). "
                f"Disarankan SCALE UP (vertical): naikkan kapasitas dari {capacity:.1f} "
                f"menjadi &ge;{suggested_capacity:.1f} (+{capacity_increase_pct:.1f}%) "
                f"sebagai antisipasi sebelum terjadi request drop."
            )
        note = note.replace("&ge;", ">=")

        out.append({
            "server_id": p["id"],
            "utilization_pct": util,
            "load": load,
            "capacity": capacity,
            "level": level,
            "recommendation": recommendation,
            "suggested_capacity": suggested_capacity,
            "capacity_increase_pct": capacity_increase_pct,
            "additional_instances": additional_instances,
            "note": note,
        })

    out.sort(key=lambda s: s["utilization_pct"], reverse=True)
    return out


def client_log(cluster, request_log):
    """Ringkas request_log (hasil evaluate_assignment) per klien: jumlah request,
    total beban (demand), dan rincian server mana saja yang melayani klien
    tersebut beserta jumlah & beban yang ditangani tiap server. Dipakai UI untuk
    menampilkan log 'klien dilayani server mana'."""
    by_client = {}
    for c in cluster["clients"]:
        by_client[c["id"]] = {"client_id": c["id"], "total_requests": 0, "total_demand": 0.0,
                               "dropped_requests": 0, "servers": {}}

    for entry in request_log or []:
        bucket = by_client.setdefault(entry["client_id"], {
            "client_id": entry["client_id"], "total_requests": 0, "total_demand": 0.0,
            "dropped_requests": 0, "servers": {},
        })
        bucket["total_requests"] += 1
        bucket["total_demand"] += entry["demand"]
        if entry["dropped"] or not entry["server_id"]:
            bucket["dropped_requests"] += 1
            continue
        srv_bucket = bucket["servers"].setdefault(entry["server_id"], {"requests": 0, "demand": 0.0})
        srv_bucket["requests"] += 1
        srv_bucket["demand"] += entry["demand"]

    out = []
    for cid, bucket in by_client.items():
        servers = [
            {"server_id": sid, "requests": v["requests"], "demand": round(v["demand"], 2)}
            for sid, v in sorted(bucket["servers"].items())
        ]
        out.append({
            "client_id": cid,
            "total_requests": bucket["total_requests"],
            "total_demand": round(bucket["total_demand"], 2),
            "dropped_requests": bucket["dropped_requests"],
            "servers": servers,
        })
    out.sort(key=lambda r: r["client_id"])
    return out


def _objective_cost(cluster, assignment):
    """Skalar biaya yang diminimalkan oleh algoritma pencarian/metaheuristik."""
    m = evaluate_assignment(cluster, assignment)
    if not m["valid"] or m["avg_response_ms"] is None:
        return 1e6
    return m["avg_response_ms"] + m["dropped_pct"] * 20 + (1 - m["fairness_index"]) * 200


def reference(cluster):
    """Nilai acuan untuk normalisasi skor QoS load balancing (kondisi terburuk:
    seluruh request ditumpuk pada satu server dengan kapasitas terkecil)."""
    servers = cluster["servers"]
    requests = cluster["requests"]
    min_cap = min((s["capacity"] for s in servers), default=1.0) or 1.0
    total_demand = sum(r["demand"] for r in requests) or 1.0
    worst_util = total_demand / min_cap
    worst_latency_ms = total_demand * PROC_MS_PER_UNIT * _congestion_factor(min(worst_util, 1.0)) + MAX_LATENCY_MS
    return {"max_response_ms": max(worst_latency_ms, 10.0)}


def classify(metrics, ref):
    if not metrics or not metrics.get("valid") or metrics.get("avg_response_ms") is None:
        return {"score": None, "class": None}

    latency_score = _clamp(100 - (metrics["avg_response_ms"] / max(ref["max_response_ms"], 1e-6)) * 100)
    fairness_score = _clamp(metrics["fairness_index"] * 100)
    dropped_score = _clamp(100 - metrics["dropped_pct"] * 5)

    overall = (
        QOS_WEIGHTS["latency"] * latency_score
        + QOS_WEIGHTS["fairness"] * fairness_score
        + QOS_WEIGHTS["dropped"] * dropped_score
    )
    overall = round(overall, 1)
    qos_class = next(label for threshold, label in QOS_CLASSES if overall >= threshold)
    return {
        "score": overall,
        "class": qos_class,
        "components": {
            "latency_score": round(latency_score, 1),
            "fairness_score": round(fairness_score, 1),
            "dropped_score": round(dropped_score, 1),
        },
    }


# ------------------------------------------------------------- algoritma --
def round_robin(cluster, seed=None, **kwargs):
    """Round Robin -- penugasan bergilir tanpa melihat beban server (baseline)."""
    t0 = time.perf_counter()
    server_ids = [s["id"] for s in cluster["servers"]]
    assignment = [server_ids[i % len(server_ids)] for i in range(len(cluster["requests"]))]
    cost = _objective_cost(cluster, assignment)
    return {"assignment": assignment, "cost": round(cost, 3),
            "runtime_ms": round((time.perf_counter() - t0) * 1000, 4), "meta": {}}


def least_connections(cluster, seed=None, **kwargs):
    """Least Connections -- tiap request dikirim ke server dengan rasio
    beban/kapasitas kumulatif terkecil saat itu (greedy, mirip greedy.py)."""
    t0 = time.perf_counter()
    servers = cluster["servers"]
    cum = {s["id"]: 0.0 for s in servers}
    assignment = []
    for req in cluster["requests"]:
        sid = min(servers, key=lambda s: cum[s["id"]] / (s["capacity"] or 1e-6))["id"]
        cum[sid] += req["demand"]
        assignment.append(sid)
    cost = _objective_cost(cluster, assignment)
    return {"assignment": assignment, "cost": round(cost, 3),
            "runtime_ms": round((time.perf_counter() - t0) * 1000, 4), "meta": {}}


def weighted_response_time(cluster, seed=None, **kwargs):
    """Greedy Weighted Response Time -- tiap request dikirim ke server dengan
    estimasi response time (beban + latensi jaringan) terkecil saat itu."""
    t0 = time.perf_counter()
    servers = cluster["servers"]
    cum = {s["id"]: 0.0 for s in servers}
    assignment = []
    for req in cluster["requests"]:
        def est(s):
            util = (cum[s["id"]] + req["demand"]) / (s["capacity"] or 1e-6)
            return req["demand"] * PROC_MS_PER_UNIT * _congestion_factor(util) + _latency_ms(cluster, req["client_id"], s["id"])
        sid = min(servers, key=est)["id"]
        cum[sid] += req["demand"]
        assignment.append(sid)
    cost = _objective_cost(cluster, assignment)
    return {"assignment": assignment, "cost": round(cost, 3),
            "runtime_ms": round((time.perf_counter() - t0) * 1000, 4), "meta": {}}


def _random_assignment(server_ids, n, rng):
    return [rng.choice(server_ids) for _ in range(n)]


def genetic(cluster, pop_size=30, generations=100, mutation_rate=0.1, seed=None, **kwargs):
    """Genetic Algorithm -- populasi vektor penugasan, crossover satu titik,
    mutasi realokasi request acak (adaptasi genetic.py untuk assignment)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    server_ids = [s["id"] for s in cluster["servers"]]
    n = len(cluster["requests"])

    population = [_random_assignment(server_ids, n, rng) for _ in range(pop_size)]
    best_cost_so_far = min(_objective_cost(cluster, p) for p in population)
    best_found_at = 0

    for gen in range(1, generations + 1):
        population.sort(key=lambda p: _objective_cost(cluster, p))
        parents = population[:max(2, pop_size // 2)]
        offspring = []
        while len(offspring) < pop_size // 2:
            p1, p2 = rng.sample(parents, 2)
            cut = rng.randint(1, n - 1) if n > 1 else 0
            child = p1[:cut] + p2[cut:]
            if rng.random() < mutation_rate:
                idx = rng.randrange(n)
                child = child[:idx] + [rng.choice(server_ids)] + child[idx + 1:]
            offspring.append(child)
        population = parents + offspring

        gen_best = min(_objective_cost(cluster, p) for p in population)
        if gen_best < best_cost_so_far:
            best_cost_so_far = gen_best
            best_found_at = gen

    population.sort(key=lambda p: _objective_cost(cluster, p))
    best = population[0]
    return {
        "assignment": best, "cost": round(_objective_cost(cluster, best), 3),
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
        "meta": {"pop_size": pop_size, "generations_dijalankan": generations,
                  "generasi_ditemukan_terbaik": best_found_at},
    }


def pso(cluster, n_particles=25, iterations=60, w=0.7, c1=1.5, c2=1.5, seed=None, **kwargs):
    """Particle Swarm Optimization -- posisi partikel kontinu per request,
    didiskritkan ke indeks server terdekat (adaptasi pso.py untuk assignment)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    server_ids = [s["id"] for s in cluster["servers"]]
    n_servers = len(server_ids)
    n = len(cluster["requests"])

    def decode(position):
        return [server_ids[int(round(min(max(x, 0), n_servers - 1)))] for x in position]

    positions = [[rng.uniform(0, n_servers - 1) for _ in range(n)] for _ in range(n_particles)]
    velocities = [[0.0] * n for _ in range(n_particles)]
    pbest = [list(p) for p in positions]
    pbest_cost = [_objective_cost(cluster, decode(p)) for p in positions]
    gbest_idx = min(range(n_particles), key=lambda i: pbest_cost[i])
    gbest = list(pbest[gbest_idx])
    gbest_cost = pbest_cost[gbest_idx]
    improved_at = 0

    for it in range(1, iterations + 1):
        for i in range(n_particles):
            for d in range(n):
                r1, r2 = rng.random(), rng.random()
                velocities[i][d] = (w * velocities[i][d]
                                     + c1 * r1 * (pbest[i][d] - positions[i][d])
                                     + c2 * r2 * (gbest[d] - positions[i][d]))
                positions[i][d] = min(max(positions[i][d] + velocities[i][d], 0), n_servers - 1)
            cost = _objective_cost(cluster, decode(positions[i]))
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = list(positions[i])
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest = list(positions[i])
                    improved_at = it

    assignment = decode(gbest)
    return {
        "assignment": assignment, "cost": round(_objective_cost(cluster, assignment), 3),
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
        "meta": {"n_particles": n_particles, "iterations_dijalankan": iterations,
                  "iterasi_ditemukan_terbaik": improved_at},
    }


def aco(cluster, n_ants=20, n_iter=50, alpha=1.0, beta=2.0, rho=0.5, Q=100, seed=None, **kwargs):
    """Ant Colony Optimization -- feromon per pasangan (request, server), semut
    membangun penugasan probabilistik berdasar feromon & heuristik estimasi
    response time (adaptasi aco.py untuk assignment)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    servers = cluster["servers"]
    server_ids = [s["id"] for s in servers]
    requests = cluster["requests"]
    n = len(requests)
    n_servers = len(server_ids)

    pheromone = [[1.0] * n_servers for _ in range(n)]
    best_assignment, best_cost = None, float("inf")
    best_found_at = 0

    for it in range(1, n_iter + 1):
        for _ant in range(n_ants):
            assignment = []
            cum = {sid: 0.0 for sid in server_ids}
            for ridx, req in enumerate(requests):
                weights = []
                for sidx, srv in enumerate(servers):
                    util = (cum[srv["id"]] + req["demand"]) / (srv["capacity"] or 1e-6)
                    est_rt = req["demand"] * PROC_MS_PER_UNIT * _congestion_factor(util) + _latency_ms(cluster, req["client_id"], srv["id"])
                    heuristic = 1.0 / max(est_rt, 1e-6)
                    weights.append((pheromone[ridx][sidx] ** alpha) * (heuristic ** beta))
                total = sum(weights) or 1e-9
                probs = [wgt / total for wgt in weights]
                sidx = rng.choices(range(n_servers), weights=probs, k=1)[0]
                sid = server_ids[sidx]
                cum[sid] += req["demand"]
                assignment.append(sid)

            cost = _objective_cost(cluster, assignment)
            if cost < best_cost:
                best_cost, best_assignment, best_found_at = cost, assignment, it
            for ridx in range(n):
                sidx = server_ids.index(assignment[ridx])
                pheromone[ridx][sidx] += Q / max(cost, 1e-6)

        for ridx in range(n):
            for sidx in range(n_servers):
                pheromone[ridx][sidx] *= (1 - rho)

    return {
        "assignment": best_assignment, "cost": round(best_cost, 3),
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
        "meta": {"n_ants": n_ants, "n_iter_dijalankan": n_iter, "iterasi_ditemukan_terbaik": best_found_at},
    }


def simulated_annealing(cluster, T0=100, alpha=0.95, iterations=800, seed=None, **kwargs):
    """Simulated Annealing -- perturbasi satu request ke server lain tiap
    iterasi, diterima probabilistik sesuai jadwal pendinginan (adaptasi
    simulated_annealing.py untuk assignment)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    server_ids = [s["id"] for s in cluster["servers"]]
    n = len(cluster["requests"])

    current = _random_assignment(server_ids, n, rng)
    current_cost = _objective_cost(cluster, current)
    best, best_cost = list(current), current_cost
    best_found_at = 0
    T = T0

    for it in range(1, iterations + 1):
        idx = rng.randrange(n)
        neighbor = list(current)
        neighbor[idx] = rng.choice(server_ids)
        neighbor_cost = _objective_cost(cluster, neighbor)
        delta = neighbor_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost, best_found_at = list(current), current_cost, it
        T *= alpha

    return {
        "assignment": best, "cost": round(best_cost, 3),
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
        "meta": {"iterations_dijalankan": iterations, "iterasi_ditemukan_terbaik": best_found_at, "T_akhir": round(T, 4)},
    }


def tabu_search(cluster, max_iter=150, tabu_size=15, seed=None, **kwargs):
    """Tabu Search -- pencarian lokal memindahkan satu request ke server lain,
    menghindari langkah dalam tabu list (adaptasi tabu_search.py untuk assignment)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    server_ids = [s["id"] for s in cluster["servers"]]
    n = len(cluster["requests"])

    current = least_connections(cluster)["assignment"] if n else []
    current_cost = _objective_cost(cluster, current)
    best, best_cost = list(current), current_cost
    best_found_at = 0
    tabu = []

    for it in range(1, max_iter + 1):
        candidates = []
        for _ in range(min(20, n * len(server_ids)) or 1):
            idx = rng.randrange(n)
            new_sid = rng.choice(server_ids)
            move = (idx, new_sid)
            if move in tabu:
                continue
            neighbor = list(current)
            neighbor[idx] = new_sid
            candidates.append((move, neighbor, _objective_cost(cluster, neighbor)))
        if not candidates:
            continue
        move, neighbor, cost = min(candidates, key=lambda c: c[2])
        current, current_cost = neighbor, cost
        tabu.append(move)
        if len(tabu) > tabu_size:
            tabu.pop(0)
        if current_cost < best_cost:
            best, best_cost, best_found_at = list(current), current_cost, it

    return {
        "assignment": best, "cost": round(best_cost, 3),
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 4),
        "meta": {"max_iter_dijalankan": max_iter, "iterasi_ditemukan_terbaik": best_found_at},
    }


LB_ALGORITHMS = {
    "round_robin": {"label": "Round Robin", "run": round_robin, "params": []},
    "least_connections": {"label": "Least Connections", "run": least_connections, "params": []},
    "weighted_response_time": {"label": "Weighted Response Time", "run": weighted_response_time, "params": []},
    "genetic": {
        "label": "Genetic Algorithm", "run": genetic,
        "params": [
            {"name": "pop_size", "label": "Ukuran Populasi", "type": "int", "default": 30, "min": 5, "max": 200, "step": 5},
            {"name": "generations", "label": "Jumlah Generasi", "type": "int", "default": 100, "min": 10, "max": 500, "step": 10},
            {"name": "mutation_rate", "label": "Laju Mutasi", "type": "float", "default": 0.1, "min": 0, "max": 1, "step": 0.05},
        ],
    },
    "pso": {
        "label": "Particle Swarm Optimization", "run": pso,
        "params": [
            {"name": "n_particles", "label": "Jumlah Partikel", "type": "int", "default": 25, "min": 5, "max": 150, "step": 5},
            {"name": "iterations", "label": "Jumlah Iterasi", "type": "int", "default": 60, "min": 10, "max": 300, "step": 10},
            {"name": "w", "label": "Bobot Inersia (w)", "type": "float", "default": 0.7, "min": 0, "max": 1.5, "step": 0.05},
            {"name": "c1", "label": "Koefisien Kognitif (c1)", "type": "float", "default": 1.5, "min": 0, "max": 3, "step": 0.1},
            {"name": "c2", "label": "Koefisien Sosial (c2)", "type": "float", "default": 1.5, "min": 0, "max": 3, "step": 0.1},
        ],
    },
    "aco": {
        "label": "Ant Colony Optimization", "run": aco,
        "params": [
            {"name": "n_ants", "label": "Jumlah Semut", "type": "int", "default": 20, "min": 5, "max": 100, "step": 1},
            {"name": "n_iter", "label": "Jumlah Iterasi", "type": "int", "default": 50, "min": 10, "max": 300, "step": 10},
            {"name": "alpha", "label": "Alpha (bobot feromon)", "type": "float", "default": 1.0, "min": 0, "max": 5, "step": 0.1},
            {"name": "beta", "label": "Beta (bobot heuristik)", "type": "float", "default": 2.0, "min": 0, "max": 5, "step": 0.1},
            {"name": "rho", "label": "Laju Evaporasi (rho)", "type": "float", "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01},
            {"name": "Q", "label": "Konstanta Q", "type": "float", "default": 100, "min": 1, "max": 1000, "step": 1},
        ],
    },
    "simulated_annealing": {
        "label": "Simulated Annealing", "run": simulated_annealing,
        "params": [
            {"name": "T0", "label": "Suhu Awal (T0)", "type": "float", "default": 100, "min": 1, "max": 1000, "step": 1},
            {"name": "alpha", "label": "Laju Pendinginan (alpha)", "type": "float", "default": 0.95, "min": 0.5, "max": 0.999, "step": 0.001},
            {"name": "iterations", "label": "Jumlah Iterasi", "type": "int", "default": 800, "min": 50, "max": 5000, "step": 50},
        ],
    },
    "tabu_search": {
        "label": "Tabu Search", "run": tabu_search,
        "params": [
            {"name": "max_iter", "label": "Jumlah Iterasi", "type": "int", "default": 150, "min": 10, "max": 1000, "step": 10},
            {"name": "tabu_size", "label": "Ukuran Tabu List", "type": "int", "default": 15, "min": 2, "max": 100, "step": 1},
        ],
    },
}


def clean_params(algo_key, raw_params):
    spec = LB_ALGORITHMS.get(algo_key)
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
