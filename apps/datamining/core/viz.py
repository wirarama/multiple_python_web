"""Pustaka visualisasi matplotlib bertema untuk seluruh proses."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .i18n import L
from .theme import CATEGORICAL, CMAP_DIV, CMAP_SEQ, GRID, INK, INK_MUTED, apply_mpl_style, color

apply_mpl_style()

BAR_R = 0.0  # radius diatur via path effects sederhana (dijaga tetap tipis)


def _finish(ax, title=None, xlabel=None, ylabel=None, source=None):
    if title:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if source:
        ax.figure.text(0.01, -0.02, source, fontsize=8, color=INK_MUTED, ha="left")
    return ax


def _bar(ax, x, y, c=None, horizontal=False, labels=True, fmt="{:.2f}"):
    c = c or color(0)
    if horizontal:
        bars = ax.barh(x, y, color=c, height=0.62)
        ax.grid(axis="y", visible=False)
        if labels:
            span = (max(y) - min(min(y), 0)) or 1
            for b, v in zip(bars, y):
                ax.text(v + span * 0.015, b.get_y() + b.get_height() / 2, fmt.format(v),
                        va="center", fontsize=8.5, color=INK_MUTED)
    else:
        bars = ax.bar(x, y, color=c, width=0.62)
        ax.grid(axis="x", visible=False)
        if labels:
            span = (max(y) - min(min(y), 0)) or 1
            for b, v in zip(bars, y):
                ax.text(b.get_x() + b.get_width() / 2, v + span * 0.015, fmt.format(v),
                        ha="center", fontsize=8.5, color=INK_MUTED)
    return bars


# =============================================================== EKSPLORASI
def fig_missing(df: pd.DataFrame):
    m = (df.isna().mean() * 100).sort_values(ascending=True)
    m = m[m > 0]
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.34 * max(len(m), 1) + 1.2)))
    if m.empty:
        ax.text(0.5, 0.5, L("Tidak ada nilai kosong", "No missing values"),
                ha="center", va="center", fontsize=12, color=INK_MUTED)
        ax.axis("off")
    else:
        _bar(ax, m.index.tolist(), m.values.tolist(), horizontal=True, fmt="{:.1f}%")
        ax.set_xlim(0, max(m.max() * 1.18, 1))
    _finish(ax, L("Persentase Nilai Kosong per Kolom", "Missing Values per Column (%)"),
            L("% kosong", "% missing"), "")
    fig.tight_layout()
    return fig


def fig_hist_grid(df: pd.DataFrame, cols=None, bins=20):
    num = df.select_dtypes(include=[np.number])
    cols = cols or list(num.columns)[:12]
    cols = [c for c in cols if c in num.columns]
    if not cols:
        return None
    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(cols):
        ax = axes[i]
        s = num[c].dropna()
        ax.hist(s, bins=bins, color=color(0), edgecolor="#fcfcfb", linewidth=0.8)
        ax.axvline(s.mean(), color=color(1), linewidth=2, label=L("Rata-rata", "Mean"))
        ax.axvline(s.median(), color=color(2), linewidth=2, linestyle="--", label=L("Median", "Median"))
        ax.set_title(c, fontsize=10)
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(L("Distribusi Variabel Numerik", "Distribution of Numeric Variables"),
                 x=0.01, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def fig_box_grid(df: pd.DataFrame, cols=None):
    num = df.select_dtypes(include=[np.number])
    cols = cols or list(num.columns)[:12]
    cols = [c for c in cols if c in num.columns]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(cols) + 2), 4.4))
    data = [num[c].dropna().values for c in cols]
    # standardisasi agar skala berbeda tetap terbaca dalam satu sumbu
    data = [(d - d.mean()) / (d.std() if d.std() else 1) for d in data]
    bp = ax.boxplot(
        data,
        tick_labels=cols,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color=INK, linewidth=1.6),
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor=color(7),
            markeredgecolor="#fcfcfb",
            markeredgewidth=0.8,
            alpha=0.85
        )
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color(0))
        patch.set_alpha(0.75)
        patch.set_edgecolor("#fcfcfb")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(GRID)
        w.set_linewidth(1.4)
    ax.axhline(0, color=GRID, linewidth=1)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    _finish(ax, L("Boxplot Terstandardisasi (deteksi outlier)", "Standardized Boxplot (outlier detection)"),
            "", L("skor-z", "z-score"))
    fig.tight_layout()
    return fig


def fig_corr_heatmap(corr: pd.DataFrame, title=None):
    if corr is None or corr.empty:
        return None
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.72 * n + 2.6), max(4.6, 0.62 * n + 2.2)))
    im = ax.imshow(corr.values, cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=42, ha="right", fontsize=9)
    ax.set_yticks(range(n), corr.index, fontsize=9)
    ax.grid(False)
    thr = 0.55
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > thr else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=GRID)
    _finish(ax, title or L("Matriks Korelasi", "Correlation Matrix"), "", "")
    fig.tight_layout()
    return fig


def fig_category_bar(freq: pd.DataFrame, label_col, value_col, title, pct_col=None):
    d = freq.head(15)
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.36 * len(d) + 1.4)))
    _bar(ax, d[label_col].astype(str).tolist()[::-1], d[value_col].tolist()[::-1],
         horizontal=True, fmt="{:.0f}")
    ax.set_xlim(0, max(d[value_col]) * 1.16)
    _finish(ax, title, L("Frekuensi", "Frequency"), "")
    fig.tight_layout()
    return fig


# ============================================================= KLASIFIKASI
def fig_class_balance(y: pd.Series):
    vc = pd.Series(y).value_counts()
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.42 * len(vc) + 1.6)))
    colors = [color(i) for i in range(len(vc))]
    bars = ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], height=0.6,
                   color=colors[::-1])
    tot = vc.sum()
    for b, v in zip(bars, vc.values[::-1]):
        ax.text(v + tot * 0.008, b.get_y() + b.get_height() / 2,
                f"{v} ({v / tot * 100:.1f}%)", va="center", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0, vc.max() * 1.24)
    ax.grid(axis="y", visible=False)
    _finish(ax, L("Distribusi Kelas Target", "Target Class Distribution"),
            L("Jumlah sampel", "Sample count"), "")
    fig.tight_layout()
    return fig


def fig_confusion(cm, labels, normalize=False):
    m = np.array(cm, dtype=float)
    if normalize:
        m = m / m.sum(axis=1, keepdims=True).clip(min=1e-9)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4.6, 0.85 * n + 3.0), max(4.0, 0.75 * n + 2.6)))
    im = ax.imshow(m, cmap=CMAP_SEQ, vmin=0, vmax=m.max() if m.max() else 1)
    ax.set_xticks(range(n), labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n), labels, fontsize=9)
    ax.grid(False)
    thr = m.max() * 0.55 if m.max() else 0.5
    for i in range(n):
        for j in range(n):
            txt = f"{m[i, j]:.2f}" if normalize else f"{int(m[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="#ffffff" if m[i, j] > thr else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.outline.set_visible(False)
    _finish(ax,
            L("Matriks Konfusi (ternormalisasi)", "Confusion Matrix (normalized)") if normalize
            else L("Matriks Konfusi", "Confusion Matrix"),
            L("Prediksi", "Predicted"), L("Aktual", "Actual"))
    fig.tight_layout()
    return fig


def fig_roc(curves: dict, auc_map: dict):
    """curves: {label: (fpr, tpr)}"""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for i, (lab, (fpr, tpr)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=color(i), linewidth=2,
                label=f"{lab} (AUC={auc_map.get(lab, float('nan')):.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--",
            label=L("Tebakan acak", "Random guess"))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    _finish(ax, L("Kurva ROC", "ROC Curve"),
            L("Laju Positif Palsu (FPR)", "False Positive Rate"),
            L("Laju Positif Benar (TPR)", "True Positive Rate"))
    fig.tight_layout()
    return fig


def fig_pr_curve(curves: dict, ap_map: dict):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for i, (lab, (rec, prec)) in enumerate(curves.items()):
        ax.plot(rec, prec, color=color(i), linewidth=2,
                label=f"{lab} (AP={ap_map.get(lab, float('nan')):.3f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    _finish(ax, L("Kurva Precision-Recall", "Precision-Recall Curve"), "Recall", "Precision")
    fig.tight_layout()
    return fig


def fig_importance(names, values, title=None, top=20):
    s = pd.Series(np.asarray(values, dtype=float), index=list(names))
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.36 * len(s) + 1.4)))
    cols = [color(7) if v < 0 else color(0) for v in s.values]
    bars = ax.barh(s.index.astype(str), s.values, color=cols, height=0.62)
    span = (s.values.max() - min(s.values.min(), 0)) or 1
    for b, v in zip(bars, s.values):
        ax.text(v + np.sign(v if v else 1) * span * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.4g}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=INK_MUTED)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(s.values.min() * 1.3, 0), max(s.values.max() * 1.3, 0.0001))
    _finish(ax, title or L("Kepentingan Fitur", "Feature Importance"), "", "")
    fig.tight_layout()
    return fig


def fig_model_comparison(df: pd.DataFrame, name_col, metric_col, title, higher_better=True):
    d = df.sort_values(metric_col, ascending=higher_better)
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.42 * len(d) + 1.4)))
    best = d[metric_col].max() if higher_better else d[metric_col].min()
    cols = [color(0) if v == best else "#9ec5f4" for v in d[metric_col]]
    bars = ax.barh(d[name_col].astype(str), d[metric_col].values, color=cols, height=0.6)
    span = d[metric_col].max() or 1
    for b, v in zip(bars, d[metric_col].values):
        ax.text(v + span * 0.012, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0, span * 1.2)
    ax.grid(axis="y", visible=False)
    _finish(ax, title, metric_col, "")
    fig.tight_layout()
    return fig


def fig_learning_curve(train_sizes, train_scores, test_scores, metric_name="Skor"):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    tm, ts = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    vm, vs = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    ax.plot(train_sizes, tm, color=color(0), linewidth=2, marker="o", label=L("Data latih", "Training"))
    ax.fill_between(train_sizes, tm - ts, tm + ts, color=color(0), alpha=0.14, linewidth=0)
    ax.plot(train_sizes, vm, color=color(1), linewidth=2, marker="s", label=L("Validasi silang", "Cross-validation"))
    ax.fill_between(train_sizes, vm - vs, vm + vs, color=color(1), alpha=0.14, linewidth=0)
    ax.legend(loc="best", fontsize=9)
    _finish(ax, L("Kurva Pembelajaran", "Learning Curve"),
            L("Jumlah sampel latih", "Training samples"), metric_name)
    fig.tight_layout()
    return fig


# ================================================================= REGRESI
def fig_actual_vs_pred(y_true, y_pred, r2=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(y_true, y_pred, s=34, color=color(0), alpha=0.72,
               edgecolor="#fcfcfb", linewidth=0.9, label=L("Observasi", "Observations"))
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = (hi - lo) * 0.05 or 1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=color(1), linewidth=2,
            linestyle="--", label=L("Prediksi sempurna", "Perfect prediction"))
    ax.legend(loc="upper left", fontsize=9)
    ttl = L("Nilai Aktual vs Prediksi", "Actual vs Predicted")
    if r2 is not None:
        ttl += f"  (R² = {r2:.4f})"
    _finish(ax, ttl, L("Aktual", "Actual"), L("Prediksi", "Predicted"))
    fig.tight_layout()
    return fig


def fig_residual_plot(y_pred, resid):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(y_pred, resid, s=32, color=color(0), alpha=0.72,
               edgecolor="#fcfcfb", linewidth=0.9)
    ax.axhline(0, color=color(1), linewidth=2, linestyle="--")
    _finish(ax, L("Plot Residual (homoskedastisitas)", "Residual Plot (homoscedasticity)"),
            L("Nilai prediksi", "Predicted value"), L("Residual", "Residual"))
    fig.tight_layout()
    return fig


def fig_residual_hist(resid):
    from scipy import stats as sps

    r = np.asarray(pd.Series(resid).dropna())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(r, bins=22, color=color(0), edgecolor="#fcfcfb", linewidth=0.8, density=True)
    xs = np.linspace(r.min(), r.max(), 200)
    axes[0].plot(xs, sps.norm.pdf(xs, r.mean(), r.std() if r.std() else 1),
                 color=color(1), linewidth=2, label=L("Kurva normal", "Normal curve"))
    axes[0].legend(fontsize=9)
    axes[0].set_title(L("Histogram Residual", "Residual Histogram"), fontsize=11)
    axes[0].grid(axis="x", visible=False)
    sps.probplot(r, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set(marker="o", markersize=4, color=color(0),
                               markerfacecolor=color(0), markeredgecolor="#fcfcfb", linestyle="none")
    axes[1].get_lines()[1].set(color=color(1), linewidth=2)
    axes[1].set_title(L("Q-Q Plot Residual", "Residual Q-Q Plot"), fontsize=11)
    axes[1].set_xlabel(L("Kuantil teoretis", "Theoretical quantiles"))
    axes[1].set_ylabel(L("Kuantil sampel", "Sample quantiles"))
    fig.tight_layout()
    return fig


def fig_pred_error_dist(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    idx = np.arange(len(err))
    ax.vlines(idx, 0, err, color=color(0), linewidth=1.4, alpha=0.85)
    ax.axhline(0, color=GRID, linewidth=1.2)
    _finish(ax, L("Galat Prediksi per Observasi", "Prediction Error per Observation"),
            L("Indeks observasi (data uji)", "Observation index (test set)"),
            L("Aktual - Prediksi", "Actual - Predicted"))
    fig.tight_layout()
    return fig


# =============================================================== KLASTERING
def fig_elbow(ks, inertias, silhouettes=None):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(ks, inertias, color=color(0), linewidth=2, marker="o", markersize=7,
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, label="Inertia (WCSS)")
    for k, v in zip(ks, inertias):
        ax.annotate(f"{v:,.0f}", (k, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=INK_MUTED)
    ax.legend(fontsize=9)
    _finish(ax, L("Metode Elbow", "Elbow Method"),
            L("Jumlah klaster (k)", "Number of clusters (k)"),
            L("Inertia / WCSS", "Inertia / WCSS"))
    fig.tight_layout()
    return fig


def fig_metric_by_k(ks, values, metric_name, higher_better=True):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    best_i = int(np.argmax(values) if higher_better else np.argmin(values))
    ax.plot(ks, values, color=color(0), linewidth=2, marker="o", markersize=7,
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=metric_name)
    ax.scatter([ks[best_i]], [values[best_i]], s=150, facecolor="none",
               edgecolor=color(1), linewidth=2.4, zorder=5,
               label=L("Nilai terbaik", "Best value"))
    ax.annotate(f"k={ks[best_i]}\n{values[best_i]:.4f}", (ks[best_i], values[best_i]),
                textcoords="offset points", xytext=(12, 10), fontsize=9, color=INK)
    ax.legend(fontsize=9)
    _finish(ax, f"{metric_name} " + L("per jumlah klaster", "by number of clusters"),
            L("Jumlah klaster (k)", "Number of clusters (k)"), metric_name)
    fig.tight_layout()
    return fig


def fig_silhouette(X, labels, sil_values, sil_avg):
    uniq = np.unique(labels)
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.55 * len(uniq) + 3)))
    y_lower = 10
    for i, k in enumerate(uniq):
        vals = np.sort(sil_values[labels == k])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=color(i), edgecolor="#fcfcfb", linewidth=0.6, alpha=0.9)
        ax.text(-0.05, y_lower + 0.5 * len(vals), f"K{k}", fontsize=9, color=INK_MUTED, ha="right")
        y_lower = y_upper + 10
    ax.axvline(sil_avg, color=color(1), linestyle="--", linewidth=2,
               label=L(f"Rata-rata = {sil_avg:.3f}", f"Average = {sil_avg:.3f}"))
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    _finish(ax, L("Plot Siluet per Klaster", "Silhouette Plot by Cluster"),
            L("Koefisien siluet", "Silhouette coefficient"), "")
    fig.tight_layout()
    return fig


def fig_cluster_scatter(coords, labels, xlabel, ylabel, centers=None, title=None):
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    uniq = sorted(pd.unique(labels))
    # Bentuk "semua pasangan": >3 klaster diberi penanda berbeda (encoding ganda)
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    for i, k in enumerate(uniq):
        mask = np.asarray(labels) == k
        lab = L("Noise", "Noise") if k == -1 else f"{L('Klaster', 'Cluster')} {k}"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=46,
                   color="#9a9a95" if k == -1 else color(i),
                   marker=markers[i % len(markers)],
                   alpha=0.82, edgecolor="#fcfcfb", linewidth=1.0, label=lab)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], s=260, marker="X", color=INK,
                   edgecolor="#fcfcfb", linewidth=2, zorder=6,
                   label=L("Pusat klaster", "Centroids"))
    ax.legend(fontsize=9, loc="best")
    _finish(ax, title or L("Sebaran Klaster", "Cluster Scatter"), xlabel, ylabel)
    fig.tight_layout()
    return fig


def fig_cluster_profile(profile: pd.DataFrame):
    """profile: index = klaster, kolom = fitur (nilai terstandardisasi)."""
    fig, ax = plt.subplots(figsize=(max(6.5, 0.95 * profile.shape[1] + 3), max(3.4, 0.6 * len(profile) + 2.4)))
    vmax = np.abs(profile.values).max() or 1
    im = ax.imshow(profile.values, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(profile.shape[1]), profile.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(profile)), [f"{L('Klaster', 'Cluster')} {i}" for i in profile.index], fontsize=9)
    ax.grid(False)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            v = profile.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(v) > vmax * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)
    _finish(ax, L("Profil Klaster (rata-rata skor-z per fitur)",
                  "Cluster Profile (mean z-score per feature)"), "", "")
    fig.tight_layout()
    return fig


def fig_cluster_sizes(labels):
    vc = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.42 * len(vc) + 1.6)))
    names = [L("Noise", "Noise") if k == -1 else f"{L('Klaster', 'Cluster')} {k}" for k in vc.index]
    cols = ["#9a9a95" if k == -1 else color(i) for i, k in enumerate(vc.index)]
    bars = ax.barh(names[::-1], vc.values[::-1], color=cols[::-1], height=0.6)
    tot = vc.sum()
    for b, v in zip(bars, vc.values[::-1]):
        ax.text(v + tot * 0.008, b.get_y() + b.get_height() / 2,
                f"{v} ({v / tot * 100:.1f}%)", va="center", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0, vc.max() * 1.25)
    ax.grid(axis="y", visible=False)
    _finish(ax, L("Ukuran Tiap Klaster", "Cluster Sizes"), L("Jumlah anggota", "Members"), "")
    fig.tight_layout()
    return fig


def fig_dendrogram(X, method="ward", max_leaves=60):
    from scipy.cluster.hierarchy import dendrogram, linkage

    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=min(max_leaves, len(X)),
               color_threshold=0.7 * max(Z[:, 2]), above_threshold_color=GRID,
               leaf_rotation=90, leaf_font_size=8)
    ax.grid(False)
    _finish(ax, L("Dendrogram (Agglomerative)", "Dendrogram (Agglomerative)"),
            L("Sampel / kelompok", "Samples / groups"), L("Jarak gabung", "Merge distance"))
    fig.tight_layout()
    return fig


# ============================================================== ASOSIASI
def fig_item_frequency(counts: pd.Series, top=20):
    d = counts.sort_values(ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, max(3, 0.36 * len(d) + 1.4)))
    _bar(ax, d.index.astype(str).tolist(), d.values.tolist(), horizontal=True, fmt="{:.0f}")
    ax.set_xlim(0, d.values.max() * 1.16)
    _finish(ax, L("Frekuensi Item Terbanyak", "Most Frequent Items"),
            L("Jumlah transaksi", "Transaction count"), "")
    fig.tight_layout()
    return fig


def fig_rules_scatter(rules: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    sizes = 40 + 320 * (rules["lift"] - rules["lift"].min()) / (np.ptp(rules["lift"]) or 1)
    sc = ax.scatter(rules["support"], rules["confidence"], s=sizes,
                    c=rules["lift"], cmap=CMAP_SEQ, alpha=0.85,
                    edgecolor="#fcfcfb", linewidth=1.0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Lift", fontsize=9, color=INK_MUTED)
    cb.outline.set_visible(False)
    top = rules.nlargest(min(5, len(rules)), "lift")
    for _, r in top.iterrows():
        ax.annotate(f"{r['antecedents_str']} → {r['consequents_str']}"[:34],
                    (r["support"], r["confidence"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=7.6, color=INK_MUTED)
    _finish(ax, L("Support vs Confidence (ukuran & warna = lift)",
                  "Support vs Confidence (size & color = lift)"),
            "Support", "Confidence")
    fig.tight_layout()
    return fig


def fig_rules_bar(rules: pd.DataFrame, metric="lift", top=15):
    d = rules.nlargest(min(top, len(rules)), metric).iloc[::-1]
    labels = [f"{a} → {c}"[:52] for a, c in zip(d["antecedents_str"], d["consequents_str"])]
    fig, ax = plt.subplots(figsize=(9.6, max(3, 0.38 * len(d) + 1.5)))
    _bar(ax, labels, d[metric].tolist(), horizontal=True, fmt="{:.3f}")
    if metric == "lift":
        ax.axvline(1.0, color=color(1), linewidth=1.8, linestyle="--",
                   label=L("Lift = 1 (independen)", "Lift = 1 (independent)"))
        ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, d[metric].max() * 1.18)
    _finish(ax, L(f"Aturan Teratas berdasarkan {metric}", f"Top Rules by {metric}"), metric, "")
    fig.tight_layout()
    return fig


def fig_rules_network(rules: pd.DataFrame, top=18):
    import networkx as nx

    d = rules.nlargest(min(top, len(rules)), "lift")
    G = nx.DiGraph()
    for _, r in d.iterrows():
        a, c = r["antecedents_str"], r["consequents_str"]
        G.add_edge(a, c, weight=float(r["lift"]))
    if len(G) == 0:
        return None
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    wmin, wmax = min(weights), max(weights)
    widths = [1.2 + 2.6 * (w - wmin) / ((wmax - wmin) or 1) for w in weights]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=CATEGORICAL[0],
                           alpha=0.55, arrowsize=14, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08", node_size=1400)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1400, node_color="#cde2fb",
                           edgecolors=CATEGORICAL[0], linewidths=1.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.6, font_color=INK)
    ax.axis("off")
    ax.set_title(L("Jaringan Aturan Asosiasi (tebal garis = lift)",
                   "Association Rule Network (edge width = lift)"))
    fig.tight_layout()
    return fig


def fig_rules_matrix(rules: pd.DataFrame, top=12):
    d = rules.nlargest(min(top * 3, len(rules)), "lift")
    piv = d.pivot_table(index="antecedents_str", columns="consequents_str",
                        values="lift", aggfunc="max")
    piv = piv.iloc[:top, :top]
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 0.95 * piv.shape[1] + 4), max(4, 0.55 * piv.shape[0] + 3)))
    im = ax.imshow(piv.values, cmap=CMAP_SEQ, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [c[:22] for c in piv.columns], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), [c[:26] for c in piv.index], fontsize=8)
    ax.grid(False)
    vmax = np.nanmax(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                        color="#ffffff" if v > vmax * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Lift", fontsize=9)
    cb.outline.set_visible(False)
    _finish(ax, L("Matriks Antecedent x Consequent (lift)", "Antecedent x Consequent Matrix (lift)"),
            L("Consequent", "Consequent"), L("Antecedent", "Antecedent"))
    fig.tight_layout()
    return fig


def fig_basket_size(sizes: pd.Series):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    vc = sizes.value_counts().sort_index()
    _bar(ax, vc.index.astype(str).tolist(), vc.values.tolist(), fmt="{:.0f}")
    ax.axvline(np.argmin(np.abs(vc.index.values - sizes.mean())), color=color(1),
               linewidth=2, linestyle="--", label=L(f"Rata-rata = {sizes.mean():.2f}", f"Mean = {sizes.mean():.2f}"))
    ax.legend(fontsize=9)
    _finish(ax, L("Distribusi Ukuran Keranjang", "Basket Size Distribution"),
            L("Jumlah item per transaksi", "Items per transaction"),
            L("Jumlah transaksi", "Transactions"))
    fig.tight_layout()
    return fig


# ============================================================== TIME SERIES
def fig_ts_line(s: pd.Series, title=None, rolling=None):
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.plot(s.index, s.values, color=color(0), linewidth=2, label=L("Nilai aktual", "Actual value"))
    if rolling and len(s) > rolling:
        ax.plot(s.index, s.rolling(rolling).mean(), color=color(1), linewidth=2,
                label=L(f"Rata-rata bergerak ({rolling})", f"Moving average ({rolling})"))
    ax.legend(fontsize=9)
    _finish(ax, title or L("Plot Deret Waktu", "Time Series Plot"),
            L("Waktu", "Time"), L("Nilai", "Value"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_decompose(result, model_name=""):
    fig, axes = plt.subplots(4, 1, figsize=(10.4, 9.0), sharex=True)
    parts = [
        (result.observed, L("Observasi", "Observed"), color(0)),
        (result.trend, L("Tren", "Trend"), color(1)),
        (result.seasonal, L("Musiman", "Seasonal"), color(2)),
        (result.resid, L("Residual", "Residual"), color(6)),
    ]
    for ax, (data, name, c) in zip(axes, parts):
        if name in (L("Residual", "Residual"),):
            ax.plot(data.index, data.values, color=c, linewidth=0, marker="o", markersize=3.4,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.6)
            ax.axhline(0, color=GRID, linewidth=1.2)
        else:
            ax.plot(data.index, data.values, color=c, linewidth=2)
        ax.set_ylabel(name, fontsize=9)
    axes[0].set_title(L(f"Dekomposisi Deret Waktu {model_name}", f"Time Series Decomposition {model_name}"))
    axes[-1].set_xlabel(L("Waktu", "Time"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_acf_pacf(s: pd.Series, lags=24):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    lags = int(min(lags, max(2, len(s) // 2 - 1)))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    plot_acf(s.dropna(), lags=lags, ax=axes[0], color=color(0), vlines_kwargs={"colors": color(0)})
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], method="ywm", color=color(1),
              vlines_kwargs={"colors": color(1)})
    axes[0].set_title("ACF (Autocorrelation Function)", fontsize=11)
    axes[1].set_title("PACF (Partial Autocorrelation Function)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Lag")
        for coll in ax.collections:
            coll.set_facecolor("#cde2fb")
            coll.set_alpha(0.55)
    fig.tight_layout()
    return fig


def fig_ts_fit_forecast(train, test, fitted, forecast, conf_int=None, title=None):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(train.index, train.values, color=color(0), linewidth=2, label=L("Data latih", "Training data"))
    if fitted is not None and len(fitted):
        ax.plot(fitted.index, fitted.values, color=color(2), linewidth=1.8, linestyle="--",
                label=L("Nilai fit (in-sample)", "Fitted (in-sample)"))
    if test is not None and len(test):
        ax.plot(test.index, test.values, color=INK, linewidth=2, label=L("Data uji (aktual)", "Test data (actual)"))
    if forecast is not None and len(forecast):
        ax.plot(forecast.index, forecast.values, color=color(1), linewidth=2.4,
                label=L("Ramalan", "Forecast"))
        if conf_int is not None and len(conf_int):
            ax.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                            color=color(1), alpha=0.16, linewidth=0,
                            label=L("Interval keyakinan 95%", "95% confidence interval"))
    ax.legend(fontsize=9, loc="best", ncol=2)
    _finish(ax, title or L("Pemodelan & Peramalan", "Model Fit & Forecast"),
            L("Waktu", "Time"), L("Nilai", "Value"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_forecast_table_plot(future: pd.DataFrame, hist: pd.Series):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(hist.index, hist.values, color=color(0), linewidth=2, label=L("Historis", "Historical"))
    ax.plot(future.index, future["forecast"], color=color(1), linewidth=2.4, marker="o",
            markersize=5, markeredgecolor="#fcfcfb", markeredgewidth=1.0,
            label=L("Ramalan ke depan", "Future forecast"))
    if "lower" in future.columns:
        ax.fill_between(future.index, future["lower"], future["upper"], color=color(1),
                        alpha=0.16, linewidth=0, label=L("IK 95%", "95% CI"))
    ax.legend(fontsize=9, loc="best")
    _finish(ax, L("Peramalan Periode Mendatang", "Future Period Forecast"),
            L("Waktu", "Time"), L("Nilai", "Value"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_ts_residual_diag(resid):
    from scipy import stats as sps
    from statsmodels.graphics.tsaplots import plot_acf

    r = pd.Series(resid).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    axes[0, 0].plot(r.index, r.values, color=color(0), linewidth=1.6)
    axes[0, 0].axhline(0, color=GRID, linewidth=1.2)
    axes[0, 0].set_title(L("Residual terhadap waktu", "Residuals over time"), fontsize=11)
    axes[0, 1].hist(r, bins=20, color=color(0), edgecolor="#fcfcfb", linewidth=0.8, density=True)
    xs = np.linspace(r.min(), r.max(), 200)
    axes[0, 1].plot(xs, sps.norm.pdf(xs, r.mean(), r.std() if r.std() else 1),
                    color=color(1), linewidth=2, label=L("Kurva normal", "Normal curve"))
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title(L("Histogram residual", "Residual histogram"), fontsize=11)
    axes[0, 1].grid(axis="x", visible=False)
    sps.probplot(r.values, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set(marker="o", markersize=4, color=color(0), linestyle="none",
                                  markeredgecolor="#fcfcfb")
    axes[1, 0].get_lines()[1].set(color=color(1), linewidth=2)
    axes[1, 0].set_title(L("Q-Q plot residual", "Residual Q-Q plot"), fontsize=11)
    lags = int(min(20, max(2, len(r) // 2 - 1)))
    plot_acf(r, lags=lags, ax=axes[1, 1], color=color(0), vlines_kwargs={"colors": color(0)})
    axes[1, 1].set_title(L("ACF residual", "Residual ACF"), fontsize=11)
    for coll in axes[1, 1].collections:
        coll.set_facecolor("#cde2fb")
        coll.set_alpha(0.55)
    fig.suptitle(L("Diagnostik Residual Model", "Model Residual Diagnostics"),
                 x=0.01, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def fig_seasonal_boxplot(s: pd.Series, period_label="Bulan"):
    df = pd.DataFrame({"v": s.values}, index=s.index)
    if period_label.lower().startswith("bulan") or period_label.lower().startswith("month"):
        df["p"] = df.index.month
    elif period_label.lower().startswith("kuartal") or period_label.lower().startswith("quart"):
        df["p"] = df.index.quarter
    else:
        df["p"] = df.index.dayofweek + 1
    groups = [g["v"].values for _, g in df.groupby("p")]
    names = [str(k) for k, _ in df.groupby("p")]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    bp = ax.boxplot(groups, labels=names, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.6),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor=color(7),
                                    markeredgecolor="#fcfcfb", alpha=0.85))
    for patch in bp["boxes"]:
        patch.set_facecolor(color(0))
        patch.set_alpha(0.75)
        patch.set_edgecolor("#fcfcfb")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(GRID)
        w.set_linewidth(1.4)
    ax.grid(axis="x", visible=False)
    _finish(ax, L(f"Pola Musiman per {period_label}", f"Seasonal Pattern by {period_label}"),
            period_label, L("Nilai", "Value"))
    fig.tight_layout()
    return fig


# ========================================================= FINE TUNING
def _is_numeric_labels(labels) -> bool:
    try:
        [float(str(v)) for v in labels]
        return True
    except (TypeError, ValueError):
        return False


def fig_tune_top_configs(labels, scores, stds, metric_name, best_pos=0, top=12):
    """Bar konfigurasi terbaik dengan galat baku antar lipatan."""
    n = min(top, len(labels))
    lab = [str(l)[:64] for l in labels[:n]][::-1]
    sc = np.asarray(scores[:n], dtype=float)[::-1]
    sd = np.asarray(stds[:n], dtype=float)[::-1]
    best_at = n - 1 - best_pos if 0 <= best_pos < n else None
    fig, ax = plt.subplots(figsize=(10.4, max(3.0, 0.42 * n + 1.6)))
    cols = ["#9ec5f4"] * n
    if best_at is not None:
        cols[best_at] = color(0)
    ax.barh(lab, sc, color=cols, height=0.62,
            xerr=sd, error_kw=dict(ecolor=INK_MUTED, elinewidth=1.2, capsize=3, alpha=0.75))
    span = float(np.nanmax(sc) - np.nanmin(sc)) or 1.0
    for i, v in enumerate(sc):
        ax.text(v + sd[i] + span * 0.03, i, f"{v:.4f}", va="center", fontsize=8.5, color=INK_MUTED)
    lo = float(np.nanmin(sc - sd))
    hi = float(np.nanmax(sc + sd))
    pad = (hi - lo) * 0.22 or abs(hi) * 0.1 or 1
    ax.set_xlim(lo - pad * 0.3, hi + pad)
    ax.grid(axis="y", visible=False)
    _finish(ax, L(f"Konfigurasi Parameter Teratas ({metric_name})",
                  f"Top Parameter Configurations ({metric_name})"),
            metric_name + L("  (batang gelap = terbaik, garis = ±1 std antar lipatan)",
                            "  (dark bar = best, whisker = ±1 std across folds)"), "")
    fig.tight_layout()
    return fig


def fig_param_importance(names, eta2, ranges=None):
    """Seberapa besar tiap parameter menjelaskan variasi skor."""
    s = pd.Series(np.asarray(eta2, dtype=float), index=list(names)).sort_values()
    fig, ax = plt.subplots(figsize=(8.6, max(2.6, 0.44 * len(s) + 1.6)))
    cols = [color(0) if v >= 0.14 else color(3) if v >= 0.06 else "#c9c8c2" for v in s.values]
    ax.barh(s.index.astype(str), s.values, color=cols, height=0.6)
    for i, v in enumerate(s.values):
        extra = f"  ({ranges[s.index[i]]:.4f})" if ranges and s.index[i] in ranges else ""
        ax.text(v + max(s.values.max(), 0.01) * 0.02, i, f"{v:.3f}{extra}",
                va="center", fontsize=9, color=INK_MUTED)
    ax.axvline(0.06, color=GRID, linewidth=1.4, linestyle="--")
    ax.axvline(0.14, color=GRID, linewidth=1.4, linestyle="--")
    ax.set_ylim(-0.62, len(s) - 0.5 + 0.28)
    ax.set_xlim(0, max(float(s.values.max()) * 1.28, 0.2))
    ax.grid(axis="y", visible=False)
    _finish(ax, L("Kepentingan Parameter (Eta² terhadap skor CV)",
                  "Parameter Importance (Eta² on the CV score)"),
            L("Eta² — proporsi variasi skor yang dijelaskan parameter\n"
              "garis putus-putus = ambang 0,06 (sedang) dan 0,14 (besar)",
              "Eta² — share of score variance explained by the parameter\n"
              "dashed lines = thresholds 0.06 (medium) and 0.14 (large)"), "")
    fig.tight_layout()
    return fig


def fig_param_effect_grid(effects: dict, metric_name: str, col_value, col_cv, col_train, col_std,
                          negated: bool = False):
    """Kurva validasi tiap parameter: skor latih vs skor validasi silang."""
    items = [(k, v) for k, v in effects.items() if v is not None and len(v)]
    if not items:
        return None
    ncol = min(3, len(items))
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, (pname, e) in enumerate(items):
        ax = axes[i]
        labels = [str(v) for v in e[col_value]]
        cv = e[col_cv].to_numpy(dtype=float)
        tr = e[col_train].to_numpy(dtype=float)
        sd = e[col_std].to_numpy(dtype=float)
        x = np.arange(len(labels))
        if _is_numeric_labels(labels):
            ax.plot(x, tr, color=color(1), linewidth=2, marker="s", markersize=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.0,
                    label=L("Skor latih", "Train score"))
            ax.plot(x, cv, color=color(0), linewidth=2, marker="o", markersize=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.0,
                    label=L("Skor validasi silang", "Cross-validation score"))
            ax.fill_between(x, cv - sd, cv + sd, color=color(0), alpha=0.14, linewidth=0)
        else:
            # nilai kategorik: penanda titik, bukan batang, supaya sumbu boleh
            # dizoom tanpa memotong batang (yang akan menyesatkan)
            ax.scatter(x, tr, s=70, marker="s", color=color(1), edgecolor="#fcfcfb",
                       linewidth=1.0, zorder=4, label=L("Skor latih", "Train score"))
            ax.errorbar(x, cv, yerr=sd, fmt="o", markersize=9, color=color(0),
                        markeredgecolor="#fcfcfb", markeredgewidth=1.0, ecolor=INK_MUTED,
                        elinewidth=1.2, capsize=4, zorder=5,
                        label=L("Skor validasi silang", "Cross-validation score"))
            for xi in x:
                ax.axvline(xi, color=GRID, linewidth=0.9, alpha=0.7, zorder=1)
            ax.set_xlim(-0.5, len(x) - 0.5)
            lo = float(min((cv - sd).min(), tr.min()))
            hi = float(max((cv + sd).max(), tr.max()))
            pad = (hi - lo) * 0.22 or abs(hi) * 0.05 or 0.01
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(axis="x", visible=False)
        best = int(np.argmin(cv)) if negated else int(np.argmax(cv))
        ax.scatter([x[best]], [cv[best]], s=210, facecolor="none", edgecolor=color(2),
                   linewidth=2.4, zorder=7)
        ax.annotate(L("terbaik", "best"), (x[best], cv[best]), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=8, color=color(2), fontweight="600")
        ax.set_xticks(x, labels, fontsize=9, rotation=25 if max(len(l) for l in labels) > 6 else 0,
                      ha="right" if max(len(l) for l in labels) > 6 else "center")
        ax.set_title(pname, fontsize=11)
        ax.set_xlabel("")
        if i == 0:
            ax.legend(loc="best", fontsize=8.5)
    for j in range(len(items), len(axes)):
        axes[j].axis("off")
    fig.suptitle(L(f"Kurva Validasi per Parameter ({metric_name})",
                   f"Validation Curve per Parameter ({metric_name})"),
                 x=0.01, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.supylabel(metric_name, fontsize=10, color=INK_MUTED)
    fig.tight_layout(rect=[0.01, 0, 1, 0.955])
    return fig


def fig_param_heatmap(pivot: pd.DataFrame, p1: str, p2: str, metric_name: str,
                      negated: bool = False):
    """Interaksi dua parameter terpenting."""
    if pivot is None or pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(5.6, 1.0 * pivot.shape[1] + 3.4),
                                    max(4.0, 0.62 * pivot.shape[0] + 2.6)))
    vals = pivot.values.astype(float)
    im = ax.imshow(-vals if negated else vals, cmap=CMAP_SEQ, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index], fontsize=9)
    ax.grid(False)
    finite = vals[np.isfinite(vals)]
    vmax = finite.max() if finite.size else 1.0
    vmin = finite.min() if finite.size else 0.0
    thr = (vmin + (vmax - vmin) * 0.45) if negated else (vmin + (vmax - vmin) * 0.55)
    best = None
    if finite.size:
        best = np.unravel_index(np.nanargmin(vals) if negated else np.nanargmax(vals), vals.shape)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            dark = (v < thr) if negated else (v > thr)
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK)
    if best is not None:
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                                   edgecolor=CATEGORICAL[1], linewidth=3, zorder=5))
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(metric_name, fontsize=9, color=INK_MUTED)
    cb.outline.set_visible(False)
    _finish(ax, L(f"Interaksi {p1} × {p2} (kotak oranye = terbaik)",
                  f"Interaction {p1} × {p2} (orange box = best)"), p2, p1)
    fig.tight_layout()
    return fig


def fig_train_cv_scatter(train, cv, best_pos, metric_name):
    """Deteksi overfitting: skor latih vs skor validasi tiap kombinasi."""
    tr = np.asarray(train, dtype=float)
    va = np.asarray(cv, dtype=float)
    m = np.isfinite(tr) & np.isfinite(va)
    tr, va = tr[m], va[m]
    if len(tr) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(tr, va, s=44, color=color(0), alpha=0.7, edgecolor="#fcfcfb", linewidth=0.9,
               label=L("Kombinasi parameter", "Parameter combinations"))
    lo = float(min(tr.min(), va.min()))
    hi = float(max(tr.max(), va.max()))
    pad = (hi - lo) * 0.06 or 0.01
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID, linewidth=1.6, linestyle="--",
            label=L("Latih = validasi (tidak overfitting)", "Train = validation (no overfitting)"))
    if 0 <= best_pos < len(tr):
        ax.scatter([tr[best_pos]], [va[best_pos]], s=210, facecolor="none",
                   edgecolor=color(1), linewidth=2.6, zorder=6,
                   label=L("Konfigurasi terbaik", "Best configuration"))
    ax.legend(loc="upper left", fontsize=9)
    _finish(ax, L("Skor Latih vs Skor Validasi Silang", "Train Score vs Cross-Validation Score"),
            L(f"Skor latih ({metric_name})", f"Train score ({metric_name})"),
            L(f"Skor validasi silang ({metric_name})", f"Cross-validation score ({metric_name})"))
    fig.tight_layout()
    return fig


def fig_fold_box(fold_matrix, labels, metric_name):
    """Sebaran skor antar lipatan untuk konfigurasi teratas — memperlihatkan
    konfigurasi mana yang benar-benar berbeda dan mana yang hanya berimpit."""
    data = [np.asarray(row, dtype=float) for row in fold_matrix]
    data = [d[np.isfinite(d)] for d in data]
    keep = [i for i, d in enumerate(data) if len(d) > 0]
    if not keep:
        return None
    data = [data[i] for i in keep]
    labs = [str(labels[i])[:46] for i in keep]
    fig, ax = plt.subplots(figsize=(10.4, max(3.2, 0.5 * len(data) + 2.0)))
    bp = ax.boxplot(
        data[::-1],
        tick_labels=labs[::-1],
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color=INK, linewidth=1.6),
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor=color(7),
            markeredgecolor="#fcfcfb",
            alpha=0.85
        )
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(color(0) if i == len(bp["boxes"]) - 1 else "#9ec5f4")
        patch.set_alpha(0.85)
        patch.set_edgecolor("#fcfcfb")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(GRID)
        w.set_linewidth(1.4)
    ax.grid(axis="y", visible=False)
    _finish(ax, L("Sebaran Skor Antar Lipatan (konfigurasi teratas)",
                  "Score Spread Across Folds (top configurations)"),
            metric_name, "")
    fig.tight_layout()
    return fig


def fig_search_progress(scores, metric_name, negated=False):
    """Skor terbaik kumulatif sepanjang pencarian."""
    s = np.asarray(scores, dtype=float)
    s = np.where(np.isfinite(s), s, np.nan)
    if np.all(np.isnan(s)):
        return None
    running = pd.Series(s).cummin() if negated else pd.Series(s).cummax()
    x = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.plot(x, s, color="#c9c8c2", linewidth=0, marker="o", markersize=4,
            markeredgecolor="#fcfcfb", markeredgewidth=0.6,
            label=L("Skor tiap kombinasi", "Score of each combination"))
    ax.plot(x, running, color=color(0), linewidth=2.4,
            label=L("Skor terbaik sejauh ini", "Best score so far"))
    best_i = int(np.nanargmin(s)) if negated else int(np.nanargmax(s))
    ax.scatter([best_i + 1], [s[best_i]], s=190, facecolor="none", edgecolor=color(1),
               linewidth=2.4, zorder=6, label=L("Kombinasi terbaik", "Best combination"))
    ax.legend(loc="lower right", fontsize=9)
    _finish(ax, L("Perkembangan Pencarian Parameter", "Parameter Search Progress"),
            L("Kombinasi ke-", "Combination #"), metric_name)
    fig.tight_layout()
    return fig


def fig_paired_folds(base_folds, tuned_folds, metric_name):
    """Perbandingan berpasangan per lipatan: parameter bawaan vs hasil tuning."""
    b = np.asarray(base_folds, dtype=float)
    t = np.asarray(tuned_folds, dtype=float)
    m = np.isfinite(b) & np.isfinite(t)
    b, t = b[m], t[m]
    if len(b) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i in range(len(b)):
        up = t[i] >= b[i]
        ax.plot([0, 1], [b[i], t[i]], color=color(2) if up else color(7),
                linewidth=1.8, alpha=0.85, zorder=2)
        ax.annotate(f"F{i + 1}", (1, t[i]), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=INK_MUTED)
    ax.scatter(np.zeros(len(b)), b, s=70, color=color(1), edgecolor="#fcfcfb", linewidth=1.2,
               zorder=4, label=L("Parameter bawaan", "Default parameters"))
    ax.scatter(np.ones(len(t)), t, s=70, color=color(0), edgecolor="#fcfcfb", linewidth=1.2,
               zorder=4, label=L("Setelah tuning", "After tuning"))
    ax.plot([0, 1], [b.mean(), t.mean()], color=INK, linewidth=2.6, linestyle="--", zorder=5,
            label=L("Rata-rata", "Mean"))
    ax.set_xticks([0, 1], [L("Bawaan", "Default"), L("Tuning", "Tuned")], fontsize=10)
    ax.set_xlim(-0.25, 1.35)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best", fontsize=9)
    _finish(ax, L("Perbandingan Berpasangan per Lipatan CV",
                  "Paired Comparison per CV Fold"), "", metric_name)
    fig.tight_layout()
    return fig
