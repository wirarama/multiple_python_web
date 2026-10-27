# =====================================================================
# BAGIAN: VISUALISASI REINFORCEMENT LEARNING (reward per episode & langkah)
# =====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .viz import draw_network  # @@internal

COMP_COLORS = {"goal": "#1baf7a", "progress": "#2a78d6", "delay": "#e8743b", "hop": "#c98500",
               "loop": "#a259d9", "congestion": "#e34948", "timeout": "#52514e"}
COMP_LABELS = {"goal": "+ tiba di target", "progress": "± kemajuan", "delay": "- delay",
               "hop": "- biaya hop", "loop": "- loop", "congestion": "- kongesti",
               "timeout": "- timeout"}


def fig_rl_rewards(ep_rows, out_png, title="Reward per episode"):
    ep = [int(e["episode"]) for e in ep_rows]
    tot = [float(e["total_reward"]) for e in ep_rows]
    plus = [float(e["plus"]) for e in ep_rows]
    minus = [float(e["minus"]) for e in ep_rows]
    steps = [int(e["steps"]) for e in ep_rows]
    reached = [str(e["reached"]) in ("True", "1", "true") for e in ep_rows]
    eps = [float(e["epsilon"]) for e in ep_rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.bar(ep, plus, color="#1baf7a", alpha=0.45, width=1.0, label="jumlah reward positif (+)")
    a1.bar(ep, minus, color="#e34948", alpha=0.45, width=1.0, label="jumlah reward negatif (-)")
    a1.plot(ep, tot, color="#111111", lw=1.2, label="total reward (return)")
    w = max(1, len(tot) // 20)
    if len(tot) >= w:
        ma = [sum(tot[max(0, i - w + 1):i + 1]) / len(tot[max(0, i - w + 1):i + 1]) for i in range(len(tot))]
        a1.plot(ep, ma, color="#d6006f", lw=2.4, label=f"rata-rata bergerak ({w} ep)")
    a1.axhline(0, color="#888888", lw=0.8)
    a1.set_ylabel("Reward")
    a1.set_title(title, fontsize=11)
    a1.legend(fontsize=8, loc="lower right")
    a1.grid(alpha=0.3)
    cols = ["#1baf7a" if r else "#e34948" for r in reached]
    a2.scatter(ep, steps, c=cols, s=10)
    a2.set_ylabel("Jumlah langkah (hijau = tiba)")
    a2.set_xlabel("Episode")
    a2.grid(alpha=0.3)
    b2 = a2.twinx()
    b2.plot(ep, eps, color="#2a78d6", lw=1.5)
    b2.set_ylabel("ε eksplorasi", color="#2a78d6")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_rl_steps(net, steps, out_png, title="Reward per langkah"):
    """Kiri: jalur dengan warna segmen = tanda reward. Kanan: komponen reward
    tiap langkah (bertumpuk positif/negatif) + return kumulatif."""
    keys = list(COMP_COLORS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    path = [steps[0]["x"]] + [s["y"] for s in steps] if steps else None
    draw_network(a1, net, path=None, show_labels=True, node_size=55)
    pos = {n["node_id"]: (float(n["x"]), float(n["y"])) for n in net["nodes"]}
    for s in steps:
        (x1, y1), (x2, y2) = pos[s["x"]], pos[s["y"]]
        r = float(s["r"])
        a1.plot([x1, x2], [y1, y2], color="#1baf7a" if r >= 0 else "#e34948", lw=4, zorder=3,
                solid_capstyle="round")
        a1.text((x1 + x2) / 2, (y1 + y2) / 2, str(s["t"]), fontsize=7, ha="center", va="center",
                zorder=6, bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#888888", lw=0.5))
    if path:
        a1.scatter(*pos[path[0]], s=160, marker="s", facecolors="none", edgecolors="#111111", lw=1.6, zorder=7)
        a1.scatter(*pos[path[-1]], s=200, marker="*", c="#111111", zorder=7)
    a1.set_title("Jalur: segmen hijau = reward langkah positif, merah = negatif", fontsize=9)

    xs = list(range(1, len(steps) + 1))
    pos_base = [0.0] * len(steps)
    neg_base = [0.0] * len(steps)
    for k in keys:
        vals = [float(s["comp"][k]) for s in steps]
        if not any(vals):
            continue
        pv = [v if v > 0 else 0 for v in vals]
        nv = [v if v < 0 else 0 for v in vals]
        a2.bar(xs, pv, bottom=pos_base, color=COMP_COLORS[k], label=COMP_LABELS[k])
        a2.bar(xs, nv, bottom=neg_base, color=COMP_COLORS[k])
        pos_base = [b + v for b, v in zip(pos_base, pv)]
        neg_base = [b + v for b, v in zip(neg_base, nv)]
    a2.plot(xs, [float(s["ret"]) for s in steps], color="#111111", marker="o", lw=1.8,
            label="return kumulatif")
    for x_, s in zip(xs, steps):
        a2.text(x_, float(s["r"]), f"{float(s['r']):+.1f}", fontsize=7, ha="center",
                va="bottom" if float(s["r"]) >= 0 else "top")
    a2.axhline(0, color="#888888", lw=0.8)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{s['t']}\n{s['x']}→{s['y']}" for s in steps], fontsize=7)
    a2.set_ylabel("Reward")
    a2.set_title("Komponen reward tiap langkah (+ di atas, - di bawah)", fontsize=9)
    a2.legend(fontsize=7, loc="best")
    a2.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_rl_policy(net, policy_rows, source, target, greedy_path, opt_path, out_png):
    """Panah kebijakan greedy tiap node (aksi dengan Q tertinggi) + jalur."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    pos = {n["node_id"]: (float(n["x"]), float(n["y"])) for n in net["nodes"]}
    draw_network(axes[0], net, show_labels=True, node_size=55,
                 title="Kebijakan agen: panah = aksi dengan Q tertinggi di tiap node")
    for node, nxt in policy_rows:
        if not nxt:
            continue
        (x1, y1), (x2, y2) = pos[node], pos[nxt]
        axes[0].annotate("", xy=(x1 + (x2 - x1) * 0.62, y1 + (y2 - y1) * 0.62), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="-|>", color="#d6006f", lw=1.6), zorder=5)
    axes[0].scatter(*pos[target], s=260, marker="*", c="#111111", zorder=7)
    draw_network(axes[1], net, path=opt_path, path_color="#111111", show_labels=True, node_size=55,
                 title="Jalur agen (magenta) vs jalur delay minimum (hitam)")
    if greedy_path:
        for u, v in zip(greedy_path[:-1], greedy_path[1:]):
            (x1, y1), (x2, y2) = pos[u], pos[v]
            axes[1].plot([x1, x2], [y1, y2], color="#d6006f", lw=2.2, ls="--", zorder=4)
    fig.suptitle(f"Kebijakan RL {source} → {target}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
