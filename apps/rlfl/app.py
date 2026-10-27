"""
FedNet Edu - Simulator Federated Learning untuk Optimasi Jaringan.

Flask + NumPy + NetworkX (+ matplotlib untuk gambar unduhan). Menyediakan:
- Simulator topologi jaringan & region klien federated (generate, edit node/link,
  drag & drop, profil trafik tiap klien, import/export CSV)
- Simulasi data trafik lokal tiap klien (non-IID) + snapshot kondisi jaringan
- Mode 1: Federated Learning (supervised) prediksi delay link -> routing Dijkstra
- Mode 2: Federated Reinforcement Learning (Q-Routing)
- Visualisasi jalur teroptimasi tiap model LOKAL, GLOBAL, TERPUSAT, ORACLE
- Unduh script Python standalone (01 data, 02 FL, 03 visualisasi), CSV jaringan,
  konfigurasi FL (JSON), hasil (CSV) dan gambar (PNG)

Jalankan:
    pip install -r requirements.txt
    python app.py
Lalu buka http://127.0.0.1:5000
"""
import csv
import io
import json
import os
import tempfile
import threading
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_file, Response

from core import network as netmod
from core.federated import run_supervised
from core.qrouting import run_qrouting
from core.rl_reward import run_rl, rl_tables, DEFAULT_RL
from core import rl_viz
from core.routing import extract_path, path_metrics
from core.results_io import results_to_tables
from core import viz
from core.exporter import build_export_files, build_zip, default_config, SCRIPT_SPECS

app = Flask(__name__)
LOCK = threading.Lock()

STATE = {}


def _reset_state(net=None, cfg=None):
    net = net or netmod.generate_topology(24, 1000, 1000, 3, 4, 42)
    STATE["net"] = net
    STATE["cfg"] = cfg or default_config(net)
    _ensure_rl_cfg()
    STATE["measurements"] = None
    STATE["snapshot"] = None
    STATE["data_sim"] = None
    STATE["results"] = {"supervised": None, "qrouting": None, "rl": None}




# --------------------------------------------------------------- helpers
def _ensure_rl_cfg():
    ids = [n["node_id"] for n in STATE["net"]["nodes"]]
    rl = STATE["cfg"].setdefault("rl", {})
    for k, v in DEFAULT_RL.items():
        rl.setdefault(k, v)
    if rl.get("source") not in ids:
        rl["source"] = ids[0]
    if rl.get("target") not in ids or rl.get("target") == rl.get("source"):
        rl["target"] = ids[-1] if ids[-1] != rl["source"] else ids[0]


def _invalidate():
    """Topologi/klien berubah -> data trafik & hasil lama tidak berlaku lagi."""
    STATE["measurements"] = None
    STATE["snapshot"] = None
    STATE["data_sim"] = None
    STATE["results"] = {"supervised": None, "qrouting": None, "rl": None}
    ids = [n["node_id"] for n in STATE["net"]["nodes"]]
    ev = STATE["cfg"].setdefault("evaluation", {})
    if ev.get("viz_source") not in ids:
        ev["viz_source"] = ids[0]
    if ev.get("viz_target") not in ids or ev.get("viz_target") == ev.get("viz_source"):
        ev["viz_target"] = ids[-1]
    _ensure_rl_cfg()


def _sync_topology_cfg():
    net = STATE["net"]
    t = STATE["cfg"].setdefault("topology", {})
    t["num_nodes"] = len(net["nodes"])
    t["num_clients"] = len(net["clients"])
    t["area_width"] = net["area"]["width"]
    t["area_height"] = net["area"]["height"]


def _ensure_data(force=False):
    sim = STATE["cfg"]["simulation"]
    if force or STATE["measurements"] is None or STATE["data_sim"] != sim:
        errs = netmod.validate_network(STATE["net"])
        if errs:
            raise ValueError("; ".join(errs))
        STATE["measurements"] = netmod.simulate_measurements(
            STATE["net"], int(sim["time_steps"]), int(sim["seed"]), float(sim["test_ratio"]),
            float(sim["noise_pct"]))
        STATE["snapshot"] = netmod.network_snapshot(STATE["net"], float(sim["eval_hour"]), int(sim["seed"]))
        STATE["data_sim"] = dict(sim)
        STATE["results"] = {"supervised": None, "qrouting": None, "rl": None}


def _data_payload():
    if STATE["measurements"] is None:
        return None
    return {"stats": netmod.dataset_stats(STATE["net"], STATE["measurements"]),
            "snapshot": STATE["snapshot"], "sim": STATE["data_sim"],
            "total": len(STATE["measurements"])}


def _state_payload():
    _sync_topology_cfg()
    return {"net": STATE["net"], "cfg": STATE["cfg"], "data": _data_payload(),
            "results": {k: (v is not None) for k, v in STATE["results"].items()},
            "errors": netmod.validate_network(STATE["net"])}


def _ok(extra=None):
    p = _state_payload()
    if extra:
        p.update(extra)
    return jsonify(p)


def _err(msg, code=400):
    return jsonify({"error": msg}), code


def _next_node_id():
    used = {n["node_id"] for n in STATE["net"]["nodes"]}
    i = len(used) + 1
    while f"N{i}" in used:
        i += 1
    return f"N{i}"


def _next_link_id():
    used = {l["link_id"] for l in STATE["net"]["links"]}
    i = len(used) + 1
    while f"L{i}" in used:
        i += 1
    return f"L{i}"


def _ui_results(res):
    """Ringankan hasil untuk dikirim ke browser (tanpa detail berlebih)."""
    out = {k: v for k, v in res.items() if k not in ("global_params",)}
    return out


_reset_state()


# --------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")


# --------------------------------------------------------------- state & config
@app.get("/api/state")
def api_state():
    with LOCK:
        return _ok()


@app.post("/api/config")
def api_config():
    body = request.get_json(force=True) or {}
    with LOCK:
        cfg = STATE["cfg"]
        for sec in ("simulation", "supervised", "qrouting", "evaluation", "rl"):
            if isinstance(body.get(sec), dict):
                cfg.setdefault(sec, {}).update(body[sec])
        if body.get("mode"):
            cfg["mode"] = body["mode"]
        return _ok()


@app.get("/api/config/export")
def api_config_export():
    with LOCK:
        _sync_topology_cfg()
        payload = {"app": "FedNet Edu", "exported": datetime.now().isoformat(timespec="seconds"),
                   "net": STATE["net"], "cfg": STATE["cfg"]}
    return Response(json.dumps(payload, indent=2, ensure_ascii=False), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=fednet_config.json"})


@app.post("/api/config/import")
def api_config_import():
    body = request.get_json(force=True) or {}
    net, cfg = body.get("net"), body.get("cfg")
    if not net or not cfg:
        return _err("File konfigurasi harus berisi 'net' dan 'cfg'.")
    errs = netmod.validate_network(net)
    if errs:
        return _err("Konfigurasi tidak valid: " + "; ".join(errs[:5]))
    base = default_config(net)
    for sec in base:
        if isinstance(base[sec], dict):
            base[sec].update(cfg.get(sec, {}))
        elif sec in cfg:
            base[sec] = cfg[sec]
    with LOCK:
        _reset_state(net, base)
        return _ok()


# --------------------------------------------------------------- topology editing
@app.post("/api/topology/generate")
def api_generate():
    b = request.get_json(force=True) or {}
    with LOCK:
        net = netmod.generate_topology(int(b.get("num_nodes", 24)), float(b.get("area_width", 1000)),
                                       float(b.get("area_height", 1000)), int(b.get("k_neighbors", 3)),
                                       int(b.get("num_clients", 4)), int(b.get("seed", 42)))
        STATE["net"] = net
        STATE["cfg"]["topology"].update({"k_neighbors": int(b.get("k_neighbors", 3)),
                                         "seed": int(b.get("seed", 42))})
        _invalidate()
        return _ok()


@app.post("/api/topology/partition")
def api_partition():
    b = request.get_json(force=True) or {}
    with LOCK:
        netmod.assign_regions(STATE["net"], int(b.get("num_clients", 4)), int(b.get("seed", 42)),
                              keep_profiles=True)
        _invalidate()
        return _ok()


@app.post("/api/node/move")
def api_node_move():
    b = request.get_json(force=True) or {}
    with LOCK:
        net = STATE["net"]
        n = next((n for n in net["nodes"] if n["node_id"] == b.get("node_id")), None)
        if not n:
            return _err("Node tidak ditemukan")
        n["x"] = round(min(max(float(b["x"]), 0), net["area"]["width"]), 1)
        n["y"] = round(min(max(float(b["y"]), 0), net["area"]["height"]), 1)
        inc = {l["link_id"] for l in net["links"] if n["node_id"] in (l["source"], l["target"])}
        netmod.recompute_link_geometry(net, inc)
        _invalidate()
        return _ok()


@app.post("/api/node/region")
def api_node_region():
    b = request.get_json(force=True) or {}
    with LOCK:
        n = next((n for n in STATE["net"]["nodes"] if n["node_id"] == b.get("node_id")), None)
        if not n:
            return _err("Node tidak ditemukan")
        if b.get("region") not in {c["client_id"] for c in STATE["net"]["clients"]}:
            return _err("Region tidak dikenal")
        n["region"] = b["region"]
        _invalidate()
        return _ok()


@app.post("/api/node/add")
def api_node_add():
    b = request.get_json(force=True) or {}
    import random
    with LOCK:
        net = STATE["net"]
        rng = random.Random()
        nid = (b.get("node_id") or "").strip() or _next_node_id()
        if any(n["node_id"] == nid for n in net["nodes"]):
            return _err(f"Node {nid} sudah ada")
        x = float(b.get("x", rng.uniform(0, net["area"]["width"])))
        y = float(b.get("y", rng.uniform(0, net["area"]["height"])))
        node = {"node_id": nid, "x": round(x, 1), "y": round(y, 1), "region": ""}
        others = sorted(net["nodes"], key=lambda m: (m["x"] - x) ** 2 + (m["y"] - y) ** 2)
        node["region"] = b.get("region") or (others[0]["region"] if others else net["clients"][0]["client_id"])
        net["nodes"].append(node)
        for m in others[:max(1, int(b.get("connect", 2)))]:
            net["links"].append({"link_id": _next_link_id(), "source": m["node_id"], "target": nid,
                                 "distance_m": 0, "base_delay_ms": 1, "bandwidth_mbps": 50.0,
                                 "loss": 0.002, "traffic_weight": 1.0})
        netmod.recompute_link_geometry(net)
        _invalidate()
        return _ok()


@app.delete("/api/node/<node_id>")
def api_node_delete(node_id):
    with LOCK:
        net = STATE["net"]
        if len(net["nodes"]) <= 3:
            return _err("Minimal 3 node")
        net["nodes"] = [n for n in net["nodes"] if n["node_id"] != node_id]
        net["links"] = [l for l in net["links"] if node_id not in (l["source"], l["target"])]
        _invalidate()
        return _ok()


@app.post("/api/link")
def api_link_save():
    b = request.get_json(force=True) or {}
    with LOCK:
        net = STATE["net"]
        s, t = b.get("source"), b.get("target")
        ids = {n["node_id"] for n in net["nodes"]}
        if s not in ids or t not in ids or s == t:
            return _err("Pilih dua node berbeda")
        link = next((l for l in net["links"] if {l["source"], l["target"]} == {s, t}), None)
        if link is None:
            link = {"link_id": _next_link_id(), "source": s, "target": t, "distance_m": 0,
                    "base_delay_ms": 1, "bandwidth_mbps": 50.0, "loss": 0.002, "traffic_weight": 1.0}
            net["links"].append(link)
            netmod.recompute_link_geometry(net, {link["link_id"]})
        for k in ("bandwidth_mbps", "loss", "traffic_weight", "base_delay_ms"):
            if b.get(k) not in (None, ""):
                link[k] = float(b[k])
        _invalidate()
        return _ok()


@app.delete("/api/link/<link_id>")
def api_link_delete(link_id):
    with LOCK:
        STATE["net"]["links"] = [l for l in STATE["net"]["links"] if l["link_id"] != link_id]
        _invalidate()
        return _ok()


@app.post("/api/clients")
def api_clients():
    b = request.get_json(force=True) or {}
    with LOCK:
        byid = {c["client_id"]: c for c in STATE["net"]["clients"]}
        for upd in b.get("clients", []):
            c = byid.get(upd.get("client_id"))
            if not c:
                continue
            for k in ("name", "color"):
                if upd.get(k):
                    c[k] = str(upd[k])
            for k in ("traffic_intensity", "burstiness", "peak_hour", "dest_locality"):
                if upd.get(k) not in (None, ""):
                    c[k] = float(upd[k])
        _invalidate()
        return _ok()


@app.post("/api/import/csv")
def api_import_csv():
    files = request.files
    if not all(k in files for k in ("nodes", "links", "clients")):
        return _err("Unggah ketiga file: nodes.csv, links.csv, clients.csv")
    try:
        rd = lambda f: list(csv.DictReader(io.StringIO(f.read().decode("utf-8-sig"))))  # noqa: E731
        nodes = netmod.parse_nodes(rd(files["nodes"]))
        links = netmod.parse_links(rd(files["links"]))
        clients = netmod.parse_clients(rd(files["clients"]))
    except Exception as e:  # noqa: BLE001
        return _err(f"Gagal membaca CSV: {e}")
    area = {"width": max(1000.0, max(n["x"] for n in nodes)), "height": max(1000.0, max(n["y"] for n in nodes))}
    net = {"nodes": nodes, "links": links, "clients": clients, "area": area}
    errs = netmod.validate_network(net)
    if errs:
        return _err("Data tidak valid: " + "; ".join(errs[:6]))
    with LOCK:
        STATE["net"] = net
        _invalidate()
        return _ok()


# --------------------------------------------------------------- data & training
@app.post("/api/data/generate")
def api_data_generate():
    b = request.get_json(force=True) or {}
    with LOCK:
        if isinstance(b.get("simulation"), dict):
            STATE["cfg"]["simulation"].update(b["simulation"])
        try:
            _ensure_data(force=True)
        except ValueError as e:
            return _err(str(e))
        return _ok()


@app.post("/api/run/<mode>")
def api_run(mode):
    if mode not in ("supervised", "qrouting"):
        return _err("mode tidak dikenal")
    b = request.get_json(force=True) or {}
    with LOCK:
        cfg = STATE["cfg"]
        for sec in ("simulation", mode, "evaluation"):
            if isinstance(b.get(sec), dict):
                cfg[sec].update(b[sec])
        try:
            _ensure_data()
        except ValueError as e:
            return _err(str(e))
        ev, sim = cfg["evaluation"], cfg["simulation"]
        pair = [ev.get("viz_source"), ev.get("viz_target")]
        if mode == "supervised":
            res = run_supervised(STATE["net"], STATE["measurements"], STATE["snapshot"], cfg["supervised"],
                                 int(ev.get("num_pairs", 30)), int(sim["seed"]), pair)
        else:
            res = run_qrouting(STATE["net"], STATE["snapshot"], cfg["qrouting"],
                               int(ev.get("num_pairs", 30)), int(sim["seed"]), pair)
        STATE["results"][mode] = res
        return _ok({"result": _ui_results(res)})


@app.post("/api/rl/run")
def api_rl_run():
    b = request.get_json(force=True) or {}
    with LOCK:
        cfg = STATE["cfg"]
        if isinstance(b.get("simulation"), dict):
            cfg["simulation"].update(b["simulation"])
        if isinstance(b.get("rl"), dict):
            cfg["rl"].update(b["rl"])
        _ensure_rl_cfg()
        try:
            _ensure_data()
        except ValueError as e:
            return _err(str(e))
        res = run_rl(STATE["net"], STATE["snapshot"], cfg["rl"], int(cfg["simulation"]["seed"]))
        STATE["results"]["rl"] = res
        return _ok({"result": res})


@app.get("/api/result/<mode>")
def api_result(mode):
    with LOCK:
        res = STATE["results"].get(mode)
        if not res:
            return _err("Belum ada hasil", 404)
        return jsonify(_ui_results(res))


# --------------------------------------------------------------- figures (matplotlib)
def _csv_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def _figure_bytes(mode, kind, source=None, target=None, episode=None):
    res = STATE["results"].get(mode)
    net = STATE["net"]
    if mode == "rl":
        if not res:
            return None
        s_, t_ = res["source"], res["target"]
        if kind == "reward_curve":
            ep_rows = rl_tables(res)[0]
            fn = lambda out: rl_viz.fig_rl_rewards(ep_rows, out, f"Reward per episode {s_} \u2192 {t_}")  # noqa: E731
        elif kind == "steps_greedy":
            fn = lambda out: rl_viz.fig_rl_steps(net, res["greedy"]["steps"], out,  # noqa: E731
                                                 f"Reward per langkah \u2014 kebijakan akhir agen {s_} \u2192 {t_}")
        elif kind == "steps_optimal":
            fn = lambda out: rl_viz.fig_rl_steps(net, res["optimal"]["steps"], out,  # noqa: E731
                                                 f"Reward per langkah \u2014 jalur delay minimum {s_} \u2192 {t_}")
        elif kind == "steps_episode":
            try:
                k = int(episode or 1)
            except ValueError:
                k = 1
            k = min(max(k, 1), len(res["steps"]))
            if not res["steps"][k - 1]:
                return None
            fn = lambda out: rl_viz.fig_rl_steps(net, res["steps"][k - 1], out,  # noqa: E731
                                                 f"Reward per langkah \u2014 episode {k} ({s_} \u2192 {t_})")
        elif kind == "policy":
            pol = list(zip(res["node_ids"], res["policy"]))
            fn = lambda out: rl_viz.fig_rl_policy(net, pol, s_, t_, res["greedy"]["path"],  # noqa: E731
                                                  res["optimal"]["path"], out)
        else:
            return None
    elif kind == "data_distribution":
        if STATE["measurements"] is None:
            return None
        rows = [{k: str(v) for k, v in r.items()} for r in STATE["measurements"]]
        fn = lambda out: viz.fig_data_distribution(net, rows, out)  # noqa: E731
    else:
        if not res:
            return None
        tables = results_to_tables(res, net)
        model_rows = _csv_rows(tables["model_summary.csv"])
        round_rows = _csv_rows(tables["round_metrics.csv"])
        ids = res["node_ids"]
        if source not in ids or target not in ids or source == target:
            source, target = res["pairs"][0]
        true_w = res["true_weights"]
        tbl = {m["key"]: m["table"] for m in res["models"]}
        opt = extract_path(tbl["oracle"], ids, source, target)
        opt_d = path_metrics(opt, net, true_w)["delay"]
        if kind == "paths":
            models, paths = [], {}
            for m in res["models"]:
                models.append({"key": m["key"], "label": m["label"], "color": m["color"], "kind": m["kind"]})
                p = extract_path(m["table"], ids, source, target)
                d = path_metrics(p, net, true_w)["delay"]
                paths[m["key"]] = {"path": p, "delay": d,
                                   "gap": (d - opt_d) / opt_d * 100 if (p and opt_d) else None}
            fn = lambda out: viz.fig_paths(net, models, paths, source, target, out, mode)  # noqa: E731
        elif kind == "learning_curves":
            fn = lambda out: viz.fig_learning_curves(mode, round_rows, model_rows, out)  # noqa: E731
        elif kind == "model_comparison":
            fn = lambda out: viz.fig_model_comparison(mode, model_rows, out)  # noqa: E731
        elif kind == "global_evolution":
            rp = [extract_path(t, ids, source, target) for t in res["global_round_tables"]]
            fn = lambda out: viz.fig_global_evolution(net, rp, opt, source, target, out)  # noqa: E731
        elif kind == "link_weights" and mode == "supervised":
            lw = _csv_rows(tables["link_weights.csv"])
            fn = lambda out: viz.fig_link_weights(net, lw, model_rows, out)  # noqa: E731
        else:
            return None
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        fn(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


@app.get("/api/figure/<mode>/<kind>.png")
def api_figure(mode, kind):
    with LOCK:
        data = _figure_bytes(mode, kind, request.args.get("source"), request.args.get("target"),
                             request.args.get("episode"))
    if data is None:
        return _err("Gambar belum tersedia (jalankan simulasi terlebih dahulu)", 404)
    name = f"fig_{mode}_{kind}.png"
    return send_file(io.BytesIO(data), mimetype="image/png", as_attachment=bool(request.args.get("dl")),
                     download_name=name)


# --------------------------------------------------------------- downloads
FILE_DESC = {
    "data/nodes.csv": "Daftar node: posisi (meter) & region (klien federated)",
    "data/links.csv": "Daftar link: jarak, delay dasar, bandwidth, packet loss, bobot trafik",
    "data/clients.csv": "Profil klien federated: intensitas, burstiness, jam puncak, lokalitas tujuan",
    "data/measurements.csv": "Data pengukuran trafik lokal tiap klien (non-IID) - keluaran langkah 1",
    "data/snapshot.csv": "Kondisi jaringan saat evaluasi (delay sebenarnya untuk oracle)",
    "config/fl_config.json": "Konfigurasi simulasi, FL supervised, Q-routing, RL reward, evaluasi",
    "01_data_jaringan.py": SCRIPT_SPECS["01_data_jaringan.py"]["title"],
    "02_federated_learning.py": SCRIPT_SPECS["02_federated_learning.py"]["title"],
    "03_visualisasi.py": SCRIPT_SPECS["03_visualisasi.py"]["title"],
    "04_reinforcement_learning.py": SCRIPT_SPECS["04_reinforcement_learning.py"]["title"],
    "run_all.py": "Menjalankan langkah 1 -> 2 -> 3 -> 4 berurutan",
    "results/rl/episodes.csv": "RL: total reward, reward +, reward -, langkah, tiba/tidak per episode",
    "results/rl/steps.csv": "RL: rincian komponen reward setiap langkah setiap episode + update Q",
    "results/rl/q_table.csv": "RL: tabel Q akhir (node, aksi, nilai Q, kunjungan, greedy)",
    "results/rl/greedy_vs_optimal.csv": "RL: rincian reward jalur agen vs jalur delay minimum",
    "requirements.txt": "Dependensi script standalone (numpy, networkx, matplotlib)",
    "README.md": "Petunjuk paket script",
}

FIG_KINDS = {"supervised": ["paths", "global_evolution", "learning_curves", "model_comparison",
                            "link_weights", "data_distribution"],
             "qrouting": ["paths", "global_evolution", "learning_curves", "model_comparison"],
             "rl": ["reward_curve", "steps_greedy", "steps_optimal", "policy"]}


RL_FIG_NAMES = {"reward_curve": "fig_rl_reward_per_episode.png", "steps_greedy": "fig_rl_steps_greedy.png",
                "steps_optimal": "fig_rl_steps_optimal.png", "policy": "fig_rl_policy.png"}


def _export_cfg():
    _sync_topology_cfg()
    cfg = json.loads(json.dumps(STATE["cfg"]))
    cfg["mode"] = "both"
    return cfg


def _export_files(with_figures=False):
    figs = {}
    if with_figures:
        ev = STATE["cfg"]["evaluation"]
        s, t = ev.get("viz_source"), ev.get("viz_target")
        for mode, kinds in FIG_KINDS.items():
            if not STATE["results"].get(mode):
                continue
            for k in kinds:
                data = _figure_bytes(mode, k, s, t)
                if data:
                    if mode == "rl":
                        name = RL_FIG_NAMES[k]
                    else:
                        name = f"fig_{k}_{s}_{t}.png" if k in ("paths", "global_evolution") else f"fig_{k}.png"
                    figs[f"results/{mode}/{name}"] = data
    return build_export_files(STATE["net"], _export_cfg(), STATE["measurements"], STATE["snapshot"],
                              STATE["results"], figs)


@app.get("/api/files")
def api_files():
    with LOCK:
        files = _export_files()
    out = []
    for path in sorted(files, key=lambda p: (p.startswith("results"), p.count("/") == 0, p)):
        data = files[path]
        out.append({"path": path, "size": len(data), "desc": FILE_DESC.get(path, ""),
                    "group": ("script" if path.endswith(".py") or path in ("requirements.txt", "README.md")
                              else "data" if path.startswith("data/")
                              else "config" if path.startswith("config/") else "result")})
    for mode, kinds in FIG_KINDS.items():
        if STATE["results"].get(mode) or (mode == "supervised" and STATE["measurements"] is not None):
            for k in kinds:
                if STATE["results"].get(mode) or k == "data_distribution":
                    out.append({"path": f"results/{mode}/fig_{k}.png", "size": None, "group": "figure",
                                "desc": f"Gambar matplotlib: {k.replace('_', ' ')}",
                                "figure": f"/api/figure/{mode}/{k}.png"})
    return jsonify(out)


@app.get("/api/download/<path:fname>")
def api_download(fname):
    with LOCK:
        files = _export_files()
    if fname not in files:
        return _err("File tidak tersedia", 404)
    data = files[fname]
    data = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")
    mt = "text/csv" if fname.endswith(".csv") else "application/json" if fname.endswith(".json") \
        else "text/x-python" if fname.endswith(".py") else "text/plain"
    if request.args.get("preview"):
        return Response(data.decode("utf-8", "replace"), mimetype="text/plain; charset=utf-8")
    return send_file(io.BytesIO(data), mimetype=mt, as_attachment=True,
                     download_name=os.path.basename(fname))


@app.get("/api/download.zip")
def api_download_zip():
    with LOCK:
        files = _export_files(with_figures=request.args.get("figures", "1") == "1")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return send_file(build_zip(files), mimetype="application/zip", as_attachment=True,
                     download_name=f"fednet_scripts_{stamp}.zip")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
