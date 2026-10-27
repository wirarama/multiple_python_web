# =====================================================================
# PROGRAM UTAMA 03 - VISUALISASI JALUR LOKAL & GLOBAL + METRIK
# =====================================================================
import argparse
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tables(rows, ids):
    idx = {n: i for i, n in enumerate(ids)}
    tables = {}
    for r in rows:
        t = tables.setdefault(r["model"], [[-1] * len(ids) for _ in ids])
        t[idx[r["dest"]]][idx[r["node"]]] = idx[r["next_hop"]] if r["next_hop"] else -1
    return tables


def visualize_mode(mode, net, data_dir, res_dir, source, target, meas_rows):
    if not os.path.exists(os.path.join(res_dir, "model_summary.csv")):
        print(f"    (lewati {mode}: hasil belum ada di {res_dir})")
        return
    ids = [n["node_id"] for n in net["nodes"]]
    snapshot = parse_snapshot(_read_csv(os.path.join(data_dir, "snapshot.csv")))
    true_w = [s["true_delay_ms"] for s in snapshot]
    model_rows = read_csv_rows(os.path.join(res_dir, "model_summary.csv"))
    round_rows = read_csv_rows(os.path.join(res_dir, "round_metrics.csv"))
    tables = load_tables(read_csv_rows(os.path.join(res_dir, "routing_tables.csv")), ids)
    with open(os.path.join(res_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if not source or not target or source not in ids or target not in ids:
        source, target = summary["pairs"][0]

    # jalur tiap model untuk pasangan terpilih (diikuti dari tabel next-hop)
    opt = extract_path(tables["oracle"], ids, source, target)
    opt_d = path_metrics(opt, net, true_w)["delay"]
    models, paths = [], {}
    for m in model_rows:
        models.append({"key": m["model"], "label": m["label"], "color": m["color"], "kind": m["kind"]})
        p = extract_path(tables[m["model"]], ids, source, target)
        d = path_metrics(p, net, true_w)["delay"]
        paths[m["model"]] = {"path": p, "delay": d,
                             "gap": (d - opt_d) / opt_d * 100 if (p and opt_d) else None}
        print(f"    {m['label'][:38]:<40} {' > '.join(p) if p else 'GAGAL'}"
              + (f"  ({d:.2f} ms)" if d is not None else ""))

    tag = f"{mode}"
    out = lambda name: os.path.join(res_dir, name)  # noqa: E731
    fig_paths(net, models, paths, source, target, out(f"fig_paths_{source}_{target}.png"), mode)
    fig_learning_curves(mode, round_rows, model_rows, out("fig_learning_curves.png"))
    fig_model_comparison(mode, model_rows, out("fig_model_comparison.png"))

    grp = read_csv_rows(os.path.join(res_dir, "global_round_paths.csv"))
    pair_rows = [r for r in grp if r["source"] == source and r["target"] == target]
    if pair_rows:
        rp = [r["path"].split(">") if r["path"] else None for r in pair_rows]
        fig_global_evolution(net, rp, opt, source, target, out(f"fig_global_evolution_{source}_{target}.png"))
    if mode == "supervised" and os.path.exists(out("link_weights.csv")):
        fig_link_weights(net, read_csv_rows(out("link_weights.csv")), model_rows, out("fig_link_weights.png"))
        if meas_rows:
            fig_data_distribution(net, meas_rows, out("fig_data_distribution.png"))
    print(f"    -> gambar {tag} disimpan di {res_dir}")


def main():
    ap = argparse.ArgumentParser(description="Langkah 3: visualisasi jalur teroptimasi (lokal vs global).")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--config", default=os.path.join(HERE, "config", "fl_config.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--mode", choices=["supervised", "qrouting", "both"], default=None)
    ap.add_argument("--source", default=None, help="node sumber, mis. N1")
    ap.add_argument("--target", default=None, help="node tujuan, mis. N10")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    ev = cfg.get("evaluation", {})
    topo = cfg.get("topology", {})
    area = {"width": float(topo.get("area_width", 1000)), "height": float(topo.get("area_height", 1000))}
    net = load_network_csv(args.data, area=area)
    src = args.source or ev.get("viz_source")
    dst = args.target or ev.get("viz_target")
    mpath = os.path.join(args.data, "measurements.csv")
    meas = _read_csv(mpath) if os.path.exists(mpath) else []
    mode = args.mode or cfg.get("mode", "both")
    for md in (["supervised", "qrouting"] if mode == "both" else [mode]):
        print(f"[3] Visualisasi {md}: {src} -> {dst}")
        visualize_mode(md, net, args.data, os.path.join(args.results, md), src, dst, meas)


if __name__ == "__main__":
    main()
