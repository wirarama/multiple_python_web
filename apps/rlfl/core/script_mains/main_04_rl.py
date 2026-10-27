# =====================================================================
# PROGRAM UTAMA 04 - REINFORCEMENT LEARNING ROUTING: REWARD PER LANGKAH
# =====================================================================
import argparse
import csv as _csv
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _print_steps(title, steps):
    keys = [k for k, _, _ in REWARD_COMPONENTS]
    print(f"\n{title}")
    head = f"{'#':>3} {'langkah':<12}" + "".join(f"{k[:8]:>10}" for k in keys) + f"{'reward':>10}{'return':>10}"
    print(head)
    print("-" * len(head))
    for s in steps:
        print(f"{s['t']:>3} {s['x'] + '>' + s['y']:<12}"
              + "".join(f"{s['comp'][k]:>10.2f}" for k in keys) + f"{s['r']:>10.2f}{s['ret']:>10.2f}")


def main():
    ap = argparse.ArgumentParser(description="Langkah 4: RL routing dengan reward plus/minus per langkah.")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--config", default=os.path.join(HERE, "config", "fl_config.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--source", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--episode", type=int, default=None, help="gambar rincian langkah episode ke-n")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    sim, topo = cfg["simulation"], cfg.get("topology", {})
    area = {"width": float(topo.get("area_width", 1000)), "height": float(topo.get("area_height", 1000))}
    net = load_network_csv(args.data, area=area)
    snap_path = os.path.join(args.data, "snapshot.csv")
    if os.path.exists(snap_path):
        snapshot = parse_snapshot(_read_csv(snap_path))
    else:
        snapshot = network_snapshot(net, float(sim["eval_hour"]), int(sim["seed"]))
    rl = dict(cfg.get("rl", {}))
    if args.source:
        rl["source"] = args.source
    if args.target:
        rl["target"] = args.target

    print(f"[4] Melatih agen Q-learning {rl.get('source')} -> {rl.get('target')} "
          f"({rl.get('episodes', DEFAULT_RL['episodes'])} episode)")
    res = run_rl(net, snapshot, rl, int(sim["seed"]))
    out = os.path.join(args.results, "rl")
    os.makedirs(out, exist_ok=True)
    ep_rows, st_rows, q_rows, g_rows = rl_tables(res)
    _save(os.path.join(out, "episodes.csv"), ep_rows)
    _save(os.path.join(out, "steps.csv"), st_rows)
    _save(os.path.join(out, "q_table.csv"), q_rows)
    _save(os.path.join(out, "greedy_vs_optimal.csv"), g_rows)
    summ = {k: res[k] for k in ("source", "target", "first_reach", "max_hops", "config")}
    summ["greedy"] = {k: res["greedy"][k] for k in ("path", "reached", "delay", "total_reward")}
    summ["optimal"] = {k: res["optimal"][k] for k in ("path", "delay", "total_reward")}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    g, o = res["greedy"], res["optimal"]
    print(f"    episode pertama tiba di target : {res['first_reach']}")
    print(f"    jalur agen (greedy) : {' > '.join(g['path'])}  delay {g['delay']:.2f} ms, reward {g['total_reward']:+.2f}"
          + ("" if g["reached"] else "  [TIDAK sampai]"))
    print(f"    jalur delay minimum : {' > '.join(o['path'])}  delay {o['delay']:.2f} ms, reward {o['total_reward']:+.2f}")
    _print_steps("Rincian reward jalur agen (greedy):", g["steps"])

    s, t = res["source"], res["target"]
    fig_rl_rewards(ep_rows, os.path.join(out, "fig_rl_reward_per_episode.png"),
                   f"Reward per episode {s} → {t}")
    fig_rl_steps(net, g["steps"], os.path.join(out, "fig_rl_steps_greedy.png"),
                 f"Reward per langkah — kebijakan akhir agen {s} → {t}")
    fig_rl_steps(net, o["steps"], os.path.join(out, "fig_rl_steps_optimal.png"),
                 f"Reward per langkah — jalur delay minimum {s} → {t}")
    ep_n = args.episode or 1
    if 1 <= ep_n <= len(res["steps"]) and res["steps"][ep_n - 1]:
        fig_rl_steps(net, res["steps"][ep_n - 1], os.path.join(out, f"fig_rl_steps_episode_{ep_n}.png"),
                     f"Reward per langkah — episode {ep_n} (masih banyak eksplorasi)")
    pol = list(zip(res["node_ids"], res["policy"]))
    fig_rl_policy(net, pol, s, t, g["path"], o["path"], os.path.join(out, "fig_rl_policy.png"))
    print(f"\n    -> CSV & gambar disimpan di {out}")


if __name__ == "__main__":
    main()
