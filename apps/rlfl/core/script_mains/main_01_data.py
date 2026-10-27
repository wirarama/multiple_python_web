# =====================================================================
# PROGRAM UTAMA 01 - DATA KONFIGURASI JARINGAN & DATA TRAFIK LOKAL
# =====================================================================
import argparse
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(
        description="Langkah 1: baca/bangkitkan data jaringan (CSV) lalu simulasikan "
                    "data pengukuran trafik lokal tiap klien federated.")
    ap.add_argument("--data", default=os.path.join(HERE, "data"), help="folder CSV jaringan")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "fl_config.json"))
    ap.add_argument("--generate", action="store_true",
                    help="bangkitkan ULANG topologi acak dari bagian 'topology' pada config "
                         "(menimpa nodes.csv, links.csv, clients.csv)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    sim, topo = cfg["simulation"], cfg.get("topology", {})
    area = {"width": float(topo.get("area_width", 1000)), "height": float(topo.get("area_height", 1000))}

    if args.generate or not os.path.exists(os.path.join(args.data, "nodes.csv")):
        print("[1] Membangkitkan topologi acak dari config ...")
        net = generate_topology(topo.get("num_nodes", 24), area["width"], area["height"],
                                topo.get("k_neighbors", 3), topo.get("num_clients", 4),
                                seed=int(topo.get("seed", sim.get("seed", 42))))
        save_network_csv(net, args.data)
    net = load_network_csv(args.data, area=area)
    errs = validate_network(net)
    if errs:
        print("Data jaringan TIDAK valid:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"[1] Jaringan: {len(net['nodes'])} node, {len(net['links'])} link, "
          f"{len(net['clients'])} klien federated")

    rows = simulate_measurements(net, int(sim["time_steps"]), int(sim["seed"]),
                                 float(sim["test_ratio"]), float(sim["noise_pct"]))
    snap = network_snapshot(net, float(sim["eval_hour"]), int(sim["seed"]))
    _write_csv(os.path.join(args.data, "measurements.csv"), MEAS_FIELDS, rows)
    _write_csv(os.path.join(args.data, "snapshot.csv"), SNAP_FIELDS, snap)

    print(f"[1] measurements.csv : {len(rows)} baris pengukuran lokal")
    print(f"[1] snapshot.csv     : kondisi jaringan jam {sim['eval_hour']} (bobot oracle)")
    print()
    print(f"{'Klien':<6}{'Nama':<14}{'Sampel':>8}{'Link':>6}{'Util min':>10}{'rata2':>8}{'maks':>8}{'Delay maks':>12}")
    for s in dataset_stats(net, rows):
        print(f"{s['client_id']:<6}{s['name'][:13]:<14}{s['samples']:>8}{s['links']:>6}"
              f"{s['util_min']:>10.2f}{s['util_mean']:>8.2f}{s['util_max']:>8.2f}{s['delay_max']:>12.1f}")
    print("\nPerhatikan: rentang utilisasi tiap klien berbeda -> data NON-IID.")


if __name__ == "__main__":
    main()
