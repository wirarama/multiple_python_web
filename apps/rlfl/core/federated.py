# =====================================================================
# BAGIAN: FEDERATED LEARNING (SUPERVISED) - PREDIKSI DELAY LINK
# ---------------------------------------------------------------------
# Setiap klien (region) melatih model prediksi delay link HANYA dari data
# pengukuran lokalnya. Server menggabungkan bobot model (FedAvg / FedProx
# / FedMedian / Trimmed-Mean) tanpa pernah melihat data mentah klien.
# Model hasil (lokal maupun global) dipakai untuk memprediksi delay seluruh
# link pada snapshot jaringan, lalu jalur dioptimasi dengan Dijkstra
# memakai delay prediksi tsb.
#
# Model: MLP 1 hidden layer (ReLU) atau regresi linear, diimplementasikan
# murni dengan numpy (forward, backprop, mini-batch SGD).
# Fitur  : utilization, bandwidth_mbps, base_delay_ms, loss
# Target : log(1 + delay_ms)  (distandarisasi secara federated)
# =====================================================================
import numpy as np

from .routing import (next_hop_table, oracle_routing, evaluate_routing,  # @@internal
                      sample_pairs)  # @@internal

FEATURES = ["utilization", "bandwidth_mbps", "base_delay_ms", "loss"]

DEFAULT_SUPERVISED = {
    "model": "mlp",              # "mlp" atau "linear"
    "hidden_units": 16,
    "rounds": 30,
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.05,
    "client_fraction": 1.0,      # proporsi klien yang dipilih tiap ronde
    "dropout_prob": 0.0,         # peluang klien terpilih gagal mengirim (straggler)
    "aggregation": "fedavg",     # fedavg | fedprox | fedmedian | trimmed_mean
    "prox_mu": 0.01,             # koefisien proximal (FedProx)
    "trim_ratio": 0.2,           # proporsi dipangkas tiap sisi (trimmed_mean)
    "weighted": True,            # bobot agregasi = jumlah sampel klien
    "dp_clip": 0.0,              # >0: clipping norma update (Differential Privacy)
    "dp_noise": 0.0,             # sigma noise Gaussian relatif terhadap dp_clip
    "accuracy_tolerance_pct": 15.0,
    "train_centralized": True,   # baseline terpusat (semua data digabung)
}


# ----------------------------------------------------------- data
def to_xy(rows):
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=float).reshape(-1, len(FEATURES))
    y = np.array([float(r.get("delay_ms", r.get("true_delay_ms", 0.0))) for r in rows], dtype=float)
    return X, y


def split_client_data(rows, client_ids):
    data = {}
    for cid in client_ids:
        tr = [r for r in rows if r["client_id"] == cid and r["split"] == "train"]
        te = [r for r in rows if r["client_id"] == cid and r["split"] == "test"]
        Xtr, ytr = to_xy(tr)
        Xte, yte = to_xy(te)
        data[cid] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
    return data


def local_stats(X, y):
    """Statistik yang DIKIRIM klien untuk standarisasi federated (bukan data)."""
    ly = np.log1p(y)
    return {"n": len(y), "sx": X.sum(axis=0), "sxx": (X ** 2).sum(axis=0),
            "sy": float(ly.sum()), "syy": float((ly ** 2).sum())}


def merge_stats(stats):
    n = sum(s["n"] for s in stats)
    sx = sum(s["sx"] for s in stats)
    sxx = sum(s["sxx"] for s in stats)
    sy = sum(s["sy"] for s in stats)
    syy = sum(s["syy"] for s in stats)
    mx = sx / n
    vx = np.maximum(sxx / n - mx ** 2, 1e-12)
    my = sy / n
    vy = max(syy / n - my ** 2, 1e-12)
    return {"mx": mx, "sx": np.sqrt(vx), "my": my, "sy": float(np.sqrt(vy))}


def tx(X, sc):
    return (X - sc["mx"]) / sc["sx"]


def ty(y, sc):
    return (np.log1p(y) - sc["my"]) / sc["sy"]


def inv_y(z, sc):
    return np.expm1(z * sc["sy"] + sc["my"])


# ----------------------------------------------------------- model numpy
def init_model(d, hidden, rng):
    if hidden and hidden > 0:
        return {"W1": rng.normal(0.0, np.sqrt(2.0 / d), (d, hidden)),
                "b1": np.zeros(hidden),
                "W2": rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, 1)),
                "b2": np.zeros(1)}
    return {"W": rng.normal(0.0, 0.1, (d, 1)), "b": np.zeros(1)}


def n_params(p):
    return int(sum(v.size for v in p.values()))


def forward(p, X):
    if "W1" in p:
        H = np.maximum(X @ p["W1"] + p["b1"], 0.0)
        return (H @ p["W2"] + p["b2"]).ravel(), H
    return (X @ p["W"] + p["b"]).ravel(), None


def gradients(p, X, y):
    out, H = forward(p, X)
    err = (out - y) / len(y)
    if "W1" in p:
        g = {"W2": H.T @ err[:, None], "b2": np.array([err.sum()])}
        dH = err[:, None] @ p["W2"].T
        dH[H <= 0] = 0.0
        g["W1"] = X.T @ dH
        g["b1"] = dH.sum(axis=0)
        return g
    return {"W": X.T @ err[:, None], "b": np.array([err.sum()])}


def mse(p, X, y):
    if len(y) == 0:
        return 0.0
    out, _ = forward(p, X)
    return float(np.mean((out - y) ** 2))


def local_train(p, X, y, epochs, lr, batch, rng, mu=0.0, anchor=None):
    """Mini-batch SGD lokal. mu>0 menambahkan term proximal FedProx:
    (mu/2)*||w - w_global||^2 agar model lokal tidak menyimpang jauh."""
    p = {k: v.copy() for k, v in p.items()}
    n = len(y)
    if n == 0:
        return p, 0.0
    batch = max(1, min(int(batch), n))
    for _ in range(int(epochs)):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            g = gradients(p, X[b], y[b])
            if mu > 0 and anchor is not None:
                for k in g:
                    g[k] = g[k] + mu * (p[k] - anchor[k])
            norm = np.sqrt(sum(float(np.sum(v ** 2)) for v in g.values()))
            scale = 5.0 / norm if norm > 5.0 else 1.0
            for k in g:
                p[k] -= lr * scale * g[k]
    return p, mse(p, X, y)


# ----------------------------------------------------------- agregasi
def aggregate(models, weights, method="fedavg", trim_ratio=0.2, weighted=True):
    keys = list(models[0].keys())
    w = np.asarray(weights, dtype=float) if weighted else np.ones(len(models))
    w = w / w.sum()
    out = {}
    for k in keys:
        S = np.stack([m[k] for m in models], axis=0)
        if method == "fedmedian":
            out[k] = np.median(S, axis=0)
        elif method == "trimmed_mean":
            m = len(models)
            cut = int(np.floor(trim_ratio * m))
            if m - 2 * cut <= 0:
                cut = 0
            Ss = np.sort(S, axis=0)
            out[k] = Ss[cut:m - cut].mean(axis=0)
        else:  # fedavg & fedprox: rata-rata berbobot
            out[k] = np.tensordot(w, S, axes=1)
    return out


def privatize(local, glob, clip, noise, rng):
    """Local Differential Privacy sederhana: clip norma update lalu tambah
    noise Gaussian (sigma = noise * clip)."""
    if clip <= 0:
        return local
    delta = {k: local[k] - glob[k] for k in local}
    norm = np.sqrt(sum(float(np.sum(v ** 2)) for v in delta.values()))
    s = min(1.0, clip / (norm + 1e-12))
    return {k: glob[k] + delta[k] * s + rng.normal(0.0, noise * clip, size=delta[k].shape)
            for k in local}


# ----------------------------------------------------------- evaluasi
def regression_metrics(y_true, y_pred, tol_pct=15.0):
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None, "acc": None, "mape": None, "n": 0}
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-12
    return {"mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "r2": round(1.0 - ss_res / ss_tot, 4),
            "acc": round(float(np.mean(np.abs(err) <= tol_pct / 100.0 * y_true)) * 100.0, 2),
            "mape": round(float(np.mean(np.abs(err) / np.maximum(y_true, 1e-9))) * 100.0, 2),
            "n": int(len(y_true))}


def mae_by_load(X, y_true, y_pred):
    """MAE per kelompok utilisasi: rendah (<0.5), sedang (0.5-0.8), tinggi (>0.8)."""
    u = X[:, 0]
    out = {}
    for name, lo, hi in (("rendah", 0.0, 0.5), ("sedang", 0.5, 0.8), ("tinggi", 0.8, 1.01)):
        m = (u >= lo) & (u < hi)
        out[name] = round(float(np.mean(np.abs(y_pred[m] - y_true[m]))), 3) if m.any() else None
    return out


def predict_ms(p, X, sc):
    return inv_y(forward(p, tx(X, sc))[0], sc)


# ----------------------------------------------------------- pelatihan FL
def run_supervised(net, rows, snapshot, cfg=None, num_pairs=30, seed=42,
                   viz_pair=None, progress=None):
    """Jalankan seluruh eksperimen: Federated (global), Lokal-saja tiap
    klien, Terpusat (opsional), dan Oracle. Mengembalikan dict hasil."""
    c = dict(DEFAULT_SUPERVISED)
    c.update(cfg or {})
    R, E = int(c["rounds"]), int(c["local_epochs"])
    hidden = int(c["hidden_units"]) if c["model"] == "mlp" else 0
    lr, bs = float(c["learning_rate"]), int(c["batch_size"])
    tol = float(c["accuracy_tolerance_pct"])
    agg = c["aggregation"]
    mu = float(c["prox_mu"]) if agg == "fedprox" else 0.0

    clients = [cl["client_id"] for cl in net["clients"]]
    colors = {cl["client_id"]: cl["color"] for cl in net["clients"]}
    names = {cl["client_id"]: cl["name"] for cl in net["clients"]}
    ids = [n["node_id"] for n in net["nodes"]]
    data = split_client_data(rows, clients)
    active_clients = [cid for cid in clients if len(data[cid]["ytr"]) > 0]

    # 1) standarisasi federated: klien hanya mengirim jumlah & jumlah kuadrat
    sc = merge_stats([local_stats(data[cid]["Xtr"], data[cid]["ytr"]) for cid in active_clients])
    for cid in clients:
        d = data[cid]
        d["Xtr_s"], d["ytr_s"] = tx(d["Xtr"], sc), ty(d["ytr"], sc) if len(d["ytr"]) else d["ytr"]
        d["Xte_s"] = tx(d["Xte"], sc)
    Xte_all = np.vstack([data[cid]["Xte"] for cid in clients])
    yte_all = np.concatenate([data[cid]["yte"] for cid in clients])

    # 2) snapshot & oracle
    Xsnap, _ = to_xy(snapshot)
    true_w = [s["true_delay_ms"] for s in snapshot]
    pairs = sample_pairs(net, num_pairs, seed, first_pair=viz_pair)
    oracle_table, oracle_paths = oracle_routing(net, true_w, pairs)

    def routing_of(p):
        w = np.maximum(predict_ms(p, Xsnap, sc), 1e-3)
        table = next_hop_table(net, w)
        summ, det = evaluate_routing(net, table, pairs, true_w, oracle_paths)
        return w, table, summ, det

    rng0 = np.random.default_rng(seed + 303)
    init = init_model(len(FEATURES), hidden, rng0)
    P = n_params(init)
    glob = {k: v.copy() for k, v in init.items()}
    rounds, g_tables, g_weights = [], [], []
    cum_bytes = 0
    for r in range(1, R + 1):
        rr = np.random.default_rng([seed, 11, r])
        m = max(1, int(round(float(c["client_fraction"]) * len(active_clients))))
        sel = sorted(rr.choice(len(active_clients), size=m, replace=False).tolist())
        selected = [active_clients[i] for i in sel]
        dropped = [cid for cid in selected if rr.random() < float(c["dropout_prob"])]
        part = [cid for cid in selected if cid not in dropped]
        locals_, ns, losses = [], [], []
        for cid in part:
            ci = clients.index(cid)
            crng = np.random.default_rng([seed, 22, r, ci])
            d = data[cid]
            p_loc, loss = local_train(glob, d["Xtr_s"], d["ytr_s"], E, lr, bs, crng,
                                      mu=mu, anchor=glob)
            p_loc = privatize(p_loc, glob, float(c["dp_clip"]), float(c["dp_noise"]), crng)
            locals_.append(p_loc)
            ns.append(len(d["ytr"]))
            losses.append(loss)
        if locals_:
            glob = aggregate(locals_, ns, agg, float(c["trim_ratio"]), bool(c["weighted"]))
        b_down = P * 4 * len(selected)
        b_up = P * 4 * len(part)
        cum_bytes += b_up + b_down
        met = regression_metrics(yte_all, predict_ms(glob, Xte_all, sc), tol)
        w, table, rsum, _ = routing_of(glob)
        g_tables.append(table)
        g_weights.append([round(float(x), 4) for x in w])
        rounds.append({"round": r, "selected": ";".join(selected), "dropped": ";".join(dropped),
                       "participants": len(part),
                       "train_loss": round(float(np.average(losses, weights=ns)), 5) if losses else None,
                       "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"], "acc": met["acc"],
                       "gap_pct": rsum["gap_pct"], "match_pct": rsum["match_pct"],
                       "success_pct": rsum["success_pct"], "avg_delay": rsum["avg_delay"],
                       "bytes_up": b_up, "bytes_down": b_down, "cum_bytes": cum_bytes})
        if progress:
            progress(r, R)

    # 3) baseline: model lokal-saja (tanpa kolaborasi) & terpusat
    models = []

    def pack(key, label, kind, cid, color, p):
        mg = regression_metrics(yte_all, predict_ms(p, Xte_all, sc), tol)
        mbl = mae_by_load(Xte_all, yte_all, predict_ms(p, Xte_all, sc))
        own = None
        if cid:
            d = data[cid]
            own = regression_metrics(d["yte"], predict_ms(p, d["Xte"], sc), tol) if len(d["yte"]) else None
        w, table, rsum, det = routing_of(p)
        return {"key": key, "label": label, "kind": kind, "client_id": cid or "", "color": color,
                "metrics": mg, "metrics_own": own, "mae_by_load": mbl, "routing": rsum,
                "weights": [round(float(x), 4) for x in w], "table": table, "details": det}

    o_sum, o_det = evaluate_routing(net, oracle_table, pairs, true_w, oracle_paths)
    models.append({"key": "oracle", "label": "Oracle (delay sebenarnya)", "kind": "oracle",
                   "client_id": "", "color": "#111111",
                   "metrics": regression_metrics(yte_all, yte_all, tol), "metrics_own": None,
                   "mae_by_load": None, "routing": o_sum,
                   "weights": [round(float(x), 4) for x in true_w], "table": oracle_table,
                   "details": o_det})
    models.append(pack("global", f"Global FL ({agg})", "global", None, "#d6006f", glob))

    if c.get("train_centralized", True):
        Xc = np.vstack([data[cid]["Xtr_s"] for cid in active_clients])
        yc = np.concatenate([data[cid]["ytr_s"] for cid in active_clients])
        pc, _ = local_train(init, Xc, yc, R * E, lr, bs, np.random.default_rng([seed, 33]))
        models.append(pack("centralized", "Terpusat (data digabung)", "centralized", None,
                           "#52514e", pc))
    for ci, cid in enumerate(clients):
        d = data[cid]
        if len(d["ytr"]) == 0:
            continue
        pl, _ = local_train(init, d["Xtr_s"], d["ytr_s"], R * E, lr, bs,
                            np.random.default_rng([seed, 44, ci]))
        models.append(pack(f"local:{cid}", f"Lokal {cid} ({names[cid]})", "local", cid,
                           colors[cid], pl))

    return {
        "mode": "supervised", "config": c, "seed": seed,
        "node_ids": ids, "link_ids": [l["link_id"] for l in net["links"]],
        "pairs": [list(p) for p in pairs], "n_params": P,
        "scaler": {"mx": sc["mx"].tolist(), "sx": sc["sx"].tolist(), "my": sc["my"], "sy": sc["sy"]},
        "global_params": {k: v.tolist() for k, v in glob.items()},
        "rounds": rounds, "models": models,
        "global_round_tables": g_tables, "global_round_weights": g_weights,
        "true_weights": [round(float(x), 4) for x in true_w],
    }
