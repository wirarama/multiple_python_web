# =====================================================================
# BAGIAN: VISUALISASI (matplotlib)
# ---------------------------------------------------------------------
# Semua fungsi menerima data "polos" (list of dict hasil baca CSV) agar
# bisa dipakai oleh aplikasi web maupun script 03_visualisasi.py.
# =====================================================================
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _f(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _region_colors(net):
    return {c["client_id"]: c["color"] for c in net["clients"]}


def draw_network(ax, net, path=None, path_color="#2a78d6", edge_values=None, cmap="viridis",
                 vmin=None, vmax=None, title=None, show_labels=True, node_size=70):
    pos = {n["node_id"]: (float(n["x"]), float(n["y"])) for n in net["nodes"]}
    rc = _region_colors(net)
    sm = None
    if edge_values is not None:
        vals = [v for v in edge_values if v is not None]
        vmin = min(vals) if vmin is None else vmin
        vmax = max(vals) if vmax is None else vmax
        norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin + 1e-3))
        cm = plt.get_cmap(cmap)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    for i, l in enumerate(net["links"]):
        (x1, y1), (x2, y2) = pos[l["source"]], pos[l["target"]]
        if sm is not None:
            ax.plot([x1, x2], [y1, y2], color=sm.to_rgba(max(edge_values[i], 1e-3)), lw=2.2, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color="#c9c9c4", lw=1.0, zorder=1)
    if path:
        for u, v in zip(path[:-1], path[1:]):
            (x1, y1), (x2, y2) = pos[u], pos[v]
            ax.plot([x1, x2], [y1, y2], color=path_color, lw=3.6, zorder=2, solid_capstyle="round")
            ax.annotate("", xy=((x1 + 2 * x2) / 3, (y1 + 2 * y2) / 3), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=path_color, lw=0, mutation_scale=11),
                        zorder=3)
    xs = [pos[n["node_id"]][0] for n in net["nodes"]]
    ys = [pos[n["node_id"]][1] for n in net["nodes"]]
    cols = [rc.get(n["region"], "#888888") for n in net["nodes"]]
    ax.scatter(xs, ys, s=node_size, c=cols, edgecolors="white", linewidths=0.8, zorder=4)
    if path:
        s, t = pos[path[0]], pos[path[-1]]
        ax.scatter([s[0]], [s[1]], s=node_size * 2.4, marker="s", facecolors="none",
                   edgecolors="#111111", linewidths=1.6, zorder=5)
        ax.scatter([t[0]], [t[1]], s=node_size * 3.2, marker="*", c="#111111", zorder=5)
    if show_labels:
        for n in net["nodes"]:
            x, y = pos[n["node_id"]]
            ax.text(x, y + (max(ys) - min(ys) + 1) * 0.025, n["node_id"], fontsize=6.5,
                    ha="center", va="bottom", color="#333333", zorder=6)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dddddd")
    if title:
        ax.set_title(title, fontsize=9)
    return sm


def fig_paths(net, models, paths_for_pair, source, target, out_png, mode="supervised"):
    """Small-multiples: satu peta per model (oracle, global, terpusat, lokal)
    dengan jalur hasil optimasi untuk pasangan source->target."""
    n = len(models)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.3 * rows))
    axes = [axes] if n == 1 else list(getattr(axes, "flat", axes))
    for ax, m in zip(axes, models):
        info = paths_for_pair.get(m["key"], {})
        path = info.get("path")
        if path:
            d = _f(info.get("delay"))
            g = _f(info.get("gap"))
            sub = f"delay {d:.2f} ms, {len(path) - 1} hop" + (f", gap {g:+.1f}%" if g is not None and m["key"] != "oracle" else "")
        else:
            sub = "GAGAL menemukan jalur (loop / buntu)"
        color = "#111111" if m["key"] == "oracle" else m.get("color", "#2a78d6")
        draw_network(ax, net, path=path, path_color=color, title=f"{m['label']}\n{sub}")
    for ax in axes[n:]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c["color"], markersize=8,
                      label=f"{c['client_id']} {c['name']}") for c in net["clients"]]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=8,
               frameon=False, title="Region / klien federated", title_fontsize=8)
    title = "Federated Learning (prediksi delay)" if mode == "supervised" else "Federated Q-Routing"
    fig.suptitle(f"Jalur teroptimasi {source} → {target} — {title}", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_learning_curves(mode, round_rows, model_rows, out_png):
    """Kurva metrik model global per ronde + garis acuan model lokal & terpusat."""
    r = [int(x["round"]) for x in round_rows]
    locals_ = [m for m in model_rows if m["kind"] == "local"]
    cent = [m for m in model_rows if m["kind"] == "centralized"]
    if mode == "supervised":
        panels = [("acc", "test_acc", "Akurasi (±toleransi, %)"), ("mae", "test_mae", "MAE delay (ms)"),
                  ("r2", "test_r2", "R²"), ("gap_pct", "route_gap_pct", "Gap jalur vs optimal (%)")]
    else:
        panels = [("success_pct", "route_success_pct", "Jalur berhasil (%)"),
                  ("gap_pct", "route_gap_pct", "Gap jalur vs optimal (%)"),
                  ("match_pct", "route_match_pct", "Jalur = optimal (%)"),
                  ("delivered_pct", None, "Paket terkirim saat training (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, (rk, mk, lab) in zip(axes.flat, panels):
        y = [_f(x.get(rk)) for x in round_rows]
        ax.plot(r, [v if v is not None else float("nan") for v in y], color="#d6006f", lw=2.2,
                marker="o", ms=3, label="Global (federated)")
        if mk:
            for m in locals_:
                v = _f(m.get(mk))
                if v is not None:
                    ax.axhline(v, color=m["color"], ls="--", lw=1.1, label=f"Lokal {m['client_id']}")
            for m in cent:
                v = _f(m.get(mk))
                if v is not None:
                    ax.axhline(v, color="#52514e", ls=":", lw=1.6, label="Terpusat")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Ronde komunikasi")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle("Kurva pembelajaran model global vs model lokal-saja", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_model_comparison(mode, model_rows, out_png):
    rows = [m for m in model_rows if m["kind"] != "oracle"]
    labels = [m["model"].replace("local:", "Lokal ").replace("global", "Global")
              .replace("centralized", "Terpusat") for m in rows]
    colors = [m["color"] or "#2a78d6" for m in rows]
    if mode == "supervised":
        panels = [("test_acc", "Akurasi (%) ↑"), ("test_mae", "MAE (ms) ↓"),
                  ("route_gap_pct", "Gap jalur (%) ↓"), ("route_match_pct", "Jalur optimal (%) ↑")]
    else:
        panels = [("route_success_pct", "Jalur berhasil (%) ↑"), ("route_gap_pct", "Gap jalur (%) ↓"),
                  ("route_match_pct", "Jalur optimal (%) ↑"), ("coverage_pct", "Cakupan state (%) ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, (k, lab) in zip(axes, panels):
        vals = [_f(m.get(k), 0.0) for m in rows]
        bars = ax.bar(range(len(rows)), vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Perbandingan model: global federated vs terpusat vs lokal-saja", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_link_weights(net, link_rows, model_rows, out_png):
    """Peta delay link: sebenarnya vs prediksi global vs prediksi lokal terburuk."""
    locals_ = [m for m in model_rows if m["kind"] == "local"]
    worst = max(locals_, key=lambda m: _f(m.get("test_mae"), 0.0)) if locals_ else None
    keys = [("true", "Delay sebenarnya (snapshot)"), ("global", "Prediksi model GLOBAL")]
    if worst:
        keys.append((worst["model"], f"Prediksi {worst['label']} (MAE terburuk)"))
    allv = [_f(r.get(k)) for r in link_rows for k, _ in keys if _f(r.get(k)) is not None]
    vmin, vmax = min(allv), max(allv)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5.2))
    sm = None
    for ax, (k, lab) in zip(axes, keys):
        vals = [_f(r.get(k), 0.0) for r in link_rows]
        sm = draw_network(ax, net, edge_values=vals, cmap="plasma", vmin=vmin, vmax=vmax,
                          title=lab, show_labels=False, node_size=40)
    cb = fig.colorbar(sm, ax=list(axes), shrink=0.75, pad=0.02)
    cb.set_label("Delay link (ms, skala log)")
    fig.suptitle("Bobot link yang dipakai routing: sebenarnya vs prediksi", fontsize=12)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_data_distribution(net, meas_rows, out_png):
    """Histogram utilisasi link yang diamati tiap klien -> memperlihatkan
    data non-IID (klien di region sepi tidak pernah melihat kongesti)."""
    import numpy as np
    clients = net["clients"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 1, 21)
    for c in clients:
        u = [float(r["utilization"]) for r in meas_rows if r["client_id"] == c["client_id"]]
        d = [float(r["delay_ms"]) for r in meas_rows if r["client_id"] == c["client_id"]]
        axes[0].hist(u, bins=bins, histtype="step", lw=2, color=c["color"],
                     label=f"{c['client_id']} (n={len(u)})")
        axes[1].scatter(u, d, s=5, alpha=0.45, color=c["color"], label=c["client_id"])
    axes[0].set_xlabel("Utilisasi link")
    axes[0].set_ylabel("Jumlah sampel")
    axes[0].set_title("Sebaran utilisasi per klien (non-IID)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Utilisasi link")
    axes[1].set_ylabel("Delay terukur (ms, log)")
    axes[1].set_title("Utilisasi vs delay: data lokal tiap klien", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_global_evolution(net, round_paths, oracle_path, source, target, out_png, n_panels=4):
    """Jalur model global pada beberapa ronde (konvergensi menuju optimal)."""
    R = len(round_paths)
    picks = sorted({max(1, int(round(R * f))) for f in [1 / R if R else 1, 0.25, 0.5, 1.0][:n_panels]})
    picks = [p for p in picks if 1 <= p <= R]
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(4.2 * (len(picks) + 1), 4.4))
    for ax, rr in zip(axes, picks):
        p = round_paths[rr - 1]
        same = p == oracle_path
        draw_network(ax, net, path=p, path_color="#1baf7a" if same else "#d6006f",
                     title=f"Global ronde {rr}" + (" (= optimal)" if same else ""), show_labels=False,
                     node_size=45)
    draw_network(axes[-1], net, path=oracle_path, path_color="#111111", title="Oracle (optimal)",
                 show_labels=False, node_size=45)
    fig.suptitle(f"Evolusi jalur model global {source} → {target}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
