# =====================================================================
# PROGRAM UTAMA 02 - PELATIHAN FEDERATED LEARNING
# =====================================================================
import argparse
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _progress(r, R):
    bar = "#" * int(30 * r / R)
    print(f"\r    ronde {r:>3}/{R} [{bar:<30}]", end="", flush=True)
    if r == R:
        print()


def _print_summary(res):
    print()
    if res["mode"] == "supervised":
        print(f"{'Model':<34}{'MAE':>8}{'R2':>8}{'Akurasi':>9}{'Gap%':>8}{'Optimal%':>10}")
        for m in res["models"]:
            mt, rt = m["metrics"], m["routing"]
            gap = rt["gap_pct"] if rt["gap_pct"] is not None else float("nan")
            print(f"{m['label'][:33]:<34}{mt['mae']:>8.3f}{mt['r2']:>8.3f}{mt['acc']:>8.1f}%"
                  f"{gap:>8.2f}{rt['match_pct']:>10.1f}")
    else:
        print(f"{'Model':<40}{'Berhasil%':>10}{'Gap%':>8}{'Optimal%':>10}{'Cakupan%':>10}")
        for m in res["models"]:
            rt = m["routing"]
            gap = rt["gap_pct"] if rt["gap_pct"] is not None else float("nan")
            cov = m.get("coverage_pct")
            print(f"{m['label'][:39]:<40}{rt['success_pct']:>10.1f}{gap:>8.2f}{rt['match_pct']:>10.1f}"
                  f"{(cov if cov is not None else float('nan')):>10.1f}")


def main():
    ap = argparse.ArgumentParser(description="Langkah 2: latih federated learning untuk optimasi jalur.")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--config", default=os.path.join(HERE, "config", "fl_config.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--mode", choices=["supervised", "qrouting", "both"], default=None,
                    help="menimpa 'mode' pada config")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    sim, ev = cfg["simulation"], cfg.get("evaluation", {})
    topo = cfg.get("topology", {})
    area = {"width": float(topo.get("area_width", 1000)), "height": float(topo.get("area_height", 1000))}
    mode = args.mode or cfg.get("mode", "both")
    net = load_network_csv(args.data, area=area)
    snap_path = os.path.join(args.data, "snapshot.csv")
    if not os.path.exists(snap_path):
        print("snapshot.csv belum ada. Jalankan dulu: python 01_data_jaringan.py")
        sys.exit(1)
    snapshot = parse_snapshot(_read_csv(snap_path))
    seed = int(sim["seed"])
    pair = [ev.get("viz_source"), ev.get("viz_target")]
    npairs = int(ev.get("num_pairs", 30))

    if mode in ("supervised", "both"):
        rows = parse_measurements(_read_csv(os.path.join(args.data, "measurements.csv")))
        print(f"[2] Federated Learning (supervised) - agregasi {cfg['supervised'].get('aggregation')}")
        res = run_supervised(net, rows, snapshot, cfg["supervised"], npairs, seed, pair, _progress)
        out = os.path.join(args.results, "supervised")
        save_results(res, net, out)
        _print_summary(res)
        print(f"    -> hasil disimpan di {out}")
    if mode in ("qrouting", "both"):
        print(f"[2] Federated Reinforcement Learning (Q-Routing) - agregasi {cfg['qrouting'].get('aggregation')}")
        res = run_qrouting(net, snapshot, cfg["qrouting"], npairs, seed, pair, _progress)
        out = os.path.join(args.results, "qrouting")
        save_results(res, net, out)
        _print_summary(res)
        print(f"    -> hasil disimpan di {out}")


if __name__ == "__main__":
    main()
