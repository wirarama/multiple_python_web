# =====================================================================
# BAGIAN: PENYIMPANAN HASIL (CSV) - dipakai script 02 (tulis) & 03 (baca)
# ---------------------------------------------------------------------
# results/round_metrics.csv   : metrik model global tiap ronde
# results/model_summary.csv   : ringkasan akhir tiap model (oracle, global,
#                               terpusat, lokal tiap klien)
# results/paths.csv           : jalur hasil optimasi tiap model per pasangan
# results/global_round_paths.csv : jalur model global tiap ronde
# results/routing_tables.csv  : tabel next-hop tiap model (dest, node, next)
# results/link_weights.csv    : delay link sebenarnya vs prediksi (supervised)
# results/summary.json        : ringkasan + konfigurasi yang dipakai
# =====================================================================
import csv
import io
import json
import os

from .routing import extract_path  # @@internal


def _csv_text(rows, fields=None):
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return buf.getvalue()


def _flat(prefix, d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, str)) or v is None:
            out[f"{prefix}{k}"] = v
    return out


def results_to_tables(res, net):
    """Ubah dict hasil menjadi {nama_file: teks_csv/json}."""
    ids = res["node_ids"]
    files = {}
    files["round_metrics.csv"] = _csv_text(res["rounds"])

    summ = []
    for m in res["models"]:
        row = {"model": m["key"], "label": m["label"], "kind": m["kind"],
               "client_id": m.get("client_id", ""), "color": m.get("color", "")}
        row.update(_flat("route_", m.get("routing")))
        row.update(_flat("test_", m.get("metrics")))
        row.update(_flat("own_", m.get("metrics_own")))
        row.update(_flat("mae_load_", m.get("mae_by_load")))
        if "coverage_pct" in m:
            row["coverage_pct"] = m["coverage_pct"]
        summ.append(row)
    files["model_summary.csv"] = _csv_text(summ)

    prow = []
    for m in res["models"]:
        for i, d in enumerate(m["details"]):
            prow.append({"pair_id": i + 1, "source": d["source"], "target": d["target"],
                         "model": m["key"], "path": ">".join(d["path"]) if d["path"] else "",
                         "true_delay_ms": d["delay"], "hops": d["hops"],
                         "bottleneck_bw_mbps": d["bottleneck_bw"], "path_loss_pct": d["loss_pct"],
                         "gap_pct": d["gap_pct"], "optimal": int(bool(d["optimal"])),
                         "success": int(bool(d["success"]))})
    files["paths.csv"] = _csv_text(prow)

    grow = []
    for r, table in enumerate(res["global_round_tables"], start=1):
        for i, (s, t) in enumerate(res["pairs"]):
            p = extract_path(table, ids, s, t)
            grow.append({"round": r, "pair_id": i + 1, "source": s, "target": t,
                         "path": ">".join(p) if p else ""})
    files["global_round_paths.csv"] = _csv_text(grow)

    trow = []
    for m in res["models"]:
        for di, d in enumerate(ids):
            for xi, x in enumerate(ids):
                nh = m["table"][di][xi]
                trow.append({"model": m["key"], "dest": d, "node": x,
                             "next_hop": ids[nh] if nh is not None and nh >= 0 else ""})
    files["routing_tables.csv"] = _csv_text(trow, ["model", "dest", "node", "next_hop"])

    if res["mode"] == "supervised":
        wrow = []
        for li, l in enumerate(net["links"]):
            row = {"link_id": l["link_id"], "source": l["source"], "target": l["target"]}
            for m in res["models"]:
                row["true" if m["key"] == "oracle" else m["key"]] = m["weights"][li]
            wrow.append(row)
        files["link_weights.csv"] = _csv_text(wrow)

    summary = {"mode": res["mode"], "seed": res["seed"], "n_params_per_model": res["n_params"],
               "config": res["config"], "pairs": res["pairs"],
               "models": [{k: m.get(k) for k in ("key", "label", "kind", "client_id", "color",
                                                  "routing", "metrics", "metrics_own",
                                                  "mae_by_load", "coverage_pct") if k in m}
                          for m in res["models"]]}
    files["summary.json"] = json.dumps(summary, indent=2, ensure_ascii=False)
    return files


def save_results(res, net, folder):
    os.makedirs(folder, exist_ok=True)
    for name, text in results_to_tables(res, net).items():
        with open(os.path.join(folder, name), "w", encoding="utf-8", newline="") as f:
            f.write(text)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
