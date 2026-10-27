"""Visualisasi khusus per algoritma.

Berbeda dengan `core.viz` yang berisi grafik umum (matriks konfusi, ROC, residual),
modul ini menggambarkan *cara kerja* algoritma itu sendiri: struktur pohon,
hyperplane dan margin SVM, batas keputusan k-NN, jalur regularisasi Lasso/Ridge,
sel Voronoi K-Means, elips kovarians GMM, kisi itemset Apriori, dan seterusnya.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402

from . import theme as TH  # noqa: E402
from .i18n import L  # noqa: E402

TH.apply_mpl_style()

GRID_STEPS = 260
MAX_SCATTER = 3000


# =============================================================== utilitas
def _fin(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def _subsample(X, y=None, cap=MAX_SCATTER, seed=0):
    X = np.asarray(X)
    if len(X) <= cap:
        return (X, y) if y is not None else (X, None)
    idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
    return X[idx], (np.asarray(y)[idx] if y is not None else None)


def project_2d(X, mode: str = "pca", cols: tuple[int, int] | None = None,
               names: list[str] | None = None):
    """Kembalikan (koordinat 2D, label sumbu, objek proyektor/None).

    mode='pca'  : dua komponen utama (semua fitur terwakili)
    mode='pair' : dua kolom asli yang dipilih pengguna
    """
    X = np.asarray(X, dtype=float)
    if mode == "pair" and cols is not None and X.shape[1] >= 2:
        i, j = cols
        lab = (names[i] if names else f"x{i}", names[j] if names else f"x{j}")
        return X[:, [i, j]], lab, None
    if X.shape[1] == 1:
        return np.c_[X[:, 0], np.zeros(len(X))], ((names or ["x0"])[0], ""), None
    from sklearn.decomposition import PCA

    p = PCA(n_components=2).fit(X)
    ev = p.explained_variance_ratio_ * 100
    lab = (f"PC1 ({ev[0]:.1f}% {L('varians', 'variance')})",
           f"PC2 ({ev[1]:.1f}% {L('varians', 'variance')})")
    return p.transform(X), lab, p


def fit_2d(estimator, X2, y):
    """Latih ulang salinan model pada ruang 2D agar batas keputusan tepat."""
    from sklearn.base import clone

    m = clone(estimator)
    m.fit(np.asarray(X2, dtype=float), y)
    return m


def _mesh(X2, pad=0.08, steps=GRID_STEPS):
    X2 = np.asarray(X2, dtype=float)
    x0, x1 = np.nanmin(X2[:, 0]), np.nanmax(X2[:, 0])
    y0, y1 = np.nanmin(X2[:, 1]), np.nanmax(X2[:, 1])
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    x0, x1 = x0 - pad * dx, x1 + pad * dx
    y0, y1 = y0 - pad * dy, y1 + pad * dy
    xx, yy = np.meshgrid(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _class_colors(classes):
    return {c: TH.color(i) for i, c in enumerate(classes)}


def _scatter_classes(ax, X2, y, classes, colors, size=26, alpha=0.9, edge="#ffffff"):
    for c in classes:
        m = np.asarray(y) == c
        if not m.any():
            continue
        ax.scatter(X2[m, 0], X2[m, 1], s=size, c=colors[c], alpha=alpha,
                   edgecolors=edge, linewidths=0.6, label=str(c), zorder=3)


# ================================================================== POHON
def fig_tree(estimator, feature_names, class_names=None, max_depth=3, figsize=None):
    """Struktur pohon keputusan (dipangkas sampai kedalaman tertentu agar terbaca)."""
    from sklearn.tree import plot_tree

    depth = int(min(max_depth, getattr(estimator, "get_depth", lambda: max_depth)() or max_depth))
    depth = max(depth, 1)
    width = min(26, 4 + 2.6 * (2 ** min(depth, 4)) ** 0.55)
    fig, ax = plt.subplots(figsize=figsize or (width, 3.0 + 2.1 * depth))
    plot_tree(estimator, feature_names=list(feature_names),
              class_names=[str(c) for c in class_names] if class_names is not None else None,
              filled=True, rounded=True, proportion=False, impurity=True, precision=3,
              max_depth=depth, fontsize=9, ax=ax)
    ax.set_title(L(f"Struktur Pohon Keputusan (kedalaman ditampilkan: {depth})",
                   f"Decision Tree Structure (depth shown: {depth})"))
    ax.grid(False)
    fig.tight_layout()
    return fig


def tree_rules_table(estimator, feature_names, class_names=None, max_depth=4) -> pd.DataFrame:
    """Aturan jika-maka dari pohon, satu baris per daun."""
    t = estimator.tree_
    feat = list(feature_names)
    rows = []

    def walk(node, conds, depth):
        if t.children_left[node] == -1 or depth > max_depth:
            val = t.value[node][0]
            n = int(t.n_node_samples[node])
            if class_names is not None and len(val) == len(class_names):
                out = str(class_names[int(np.argmax(val))])
                conf = float(val.max() / max(val.sum(), 1))
                extra = f"{conf:.2%}"
            else:
                out = f"{float(val[0]):,.4f}"
                extra = "-"
            rows.append({
                L("Aturan (JIKA)", "Rule (IF)"): (f" {L('DAN', 'AND')} ".join(conds)
                                                  if conds else L("(akar)", "(root)")),
                L("MAKA", "THEN"): out,
                L("Jumlah sampel", "Samples"): n,
                L("Kemurnian/keyakinan", "Purity/confidence"): extra,
                L("Kedalaman", "Depth"): depth,
            })
            return
        f = feat[t.feature[node]] if t.feature[node] < len(feat) else f"f{t.feature[node]}"
        thr = t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4f}"], depth + 1)
        walk(t.children_right[node], conds + [f"{f} > {thr:.4f}"], depth + 1)

    walk(0, [], 0)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(L("Jumlah sampel", "Samples"), ascending=False).reset_index(drop=True)
    return df


def fig_tree_importance_path(estimator, feature_names, top=15):
    """Kontribusi tiap fitur terhadap penurunan ketidakmurnian total pohon."""
    imp = np.asarray(getattr(estimator, "feature_importances_", []), dtype=float)
    if imp.size == 0:
        return None
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(order) + 1.9))
    ax.barh(range(len(order)), imp[order], color=TH.color(0), height=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(feature_names[i]) for i in order])
    for i, v in enumerate(imp[order]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=TH.INK_MUTED)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    return _fin(ax, L("Penurunan Ketidakmurnian per Fitur",
                      "Impurity Decrease per Feature"),
                L("Total penurunan ketidakmurnian (ternormalisasi)",
                  "Total impurity decrease (normalized)")).figure


def fig_ccp_path(estimator_factory, X, y, alphas=None):
    """Jalur pemangkasan cost-complexity: jumlah daun & ketidakmurnian vs alpha."""
    from sklearn.base import clone

    base = estimator_factory()
    try:
        path = base.cost_complexity_pruning_path(np.asarray(X, dtype=float), y)
    except Exception:
        return None
    a, imp = path.ccp_alphas, path.impurities
    if len(a) < 3:
        return None
    a, imp = a[:-1], imp[:-1]
    keep = np.linspace(0, len(a) - 1, min(len(a), 24)).astype(int)
    leaves, depths = [], []
    for al in a[keep]:
        m = clone(base)
        m.set_params(ccp_alpha=float(al))
        m.fit(np.asarray(X, dtype=float), y)
        leaves.append(m.get_n_leaves())
        depths.append(m.get_depth())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].plot(a, imp, marker="o", ms=3, color=TH.color(0))
    _fin(axes[0], L("Ketidakmurnian vs alpha", "Impurity vs alpha"), "ccp_alpha",
         L("Total ketidakmurnian daun", "Total leaf impurity"))
    axes[1].plot(a[keep], leaves, marker="o", ms=4, color=TH.color(1),
                 label=L("Jumlah daun", "Number of leaves"))
    axes[1].plot(a[keep], depths, marker="s", ms=4, color=TH.color(2),
                 label=L("Kedalaman", "Depth"))
    axes[1].legend()
    _fin(axes[1], L("Ukuran pohon vs alpha", "Tree size vs alpha"), "ccp_alpha",
         L("Ukuran", "Size"))
    fig.tight_layout()
    return fig


# ============================================================ HUTAN/ANSAMBEL
def fig_forest_grid(forest, feature_names, class_names=None, n_trees=3, max_depth=2):
    """Beberapa pohon contoh dari hutan, memperlihatkan keragaman antar pohon."""
    from sklearn.tree import plot_tree

    trees = list(getattr(forest, "estimators_", []))
    if not trees:
        return None
    trees = trees[:max(1, int(n_trees))]
    fig, axes = plt.subplots(1, len(trees), figsize=(6.0 * len(trees), 4.4))
    axes = np.atleast_1d(axes)
    for i, (ax, t) in enumerate(zip(axes, trees)):
        est = t[0] if isinstance(t, (list, np.ndarray)) else t
        plot_tree(est, feature_names=list(feature_names),
                  class_names=[str(c) for c in class_names] if class_names is not None else None,
                  filled=True, rounded=True, max_depth=int(max_depth), fontsize=7,
                  precision=2, impurity=False, ax=ax)
        ax.set_title(L(f"Pohon #{i + 1}", f"Tree #{i + 1}"))
        ax.grid(False)
    fig.suptitle(L("Contoh Pohon dalam Ansambel (dipangkas agar terbaca)",
                   "Sample Trees Inside the Ensemble (pruned for legibility)"),
                 x=0.01, ha="left", fontsize=12, fontweight="600")
    fig.tight_layout()
    return fig


def fig_forest_structure(forest):
    """Sebaran kedalaman & jumlah daun seluruh pohon dalam hutan."""
    trees = list(getattr(forest, "estimators_", []))
    if not trees:
        return None
    ests = [t[0] if isinstance(t, (list, np.ndarray)) else t for t in trees]
    depths = np.array([e.get_depth() for e in ests], dtype=float)
    leaves = np.array([e.get_n_leaves() for e in ests], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    axes[0].hist(depths, bins=min(20, max(4, len(set(depths)))), color=TH.color(0))
    axes[0].axvline(depths.mean(), color=TH.STATUS["critical"], ls="--", lw=1.4,
                    label=L(f"rata-rata {depths.mean():.1f}", f"mean {depths.mean():.1f}"))
    axes[0].legend()
    _fin(axes[0], L("Sebaran Kedalaman Pohon", "Tree Depth Distribution"),
         L("Kedalaman", "Depth"), L("Jumlah pohon", "Trees"))
    axes[1].hist(leaves, bins=min(20, max(4, len(set(leaves)))), color=TH.color(2))
    axes[1].axvline(leaves.mean(), color=TH.STATUS["critical"], ls="--", lw=1.4,
                    label=L(f"rata-rata {leaves.mean():.0f}", f"mean {leaves.mean():.0f}"))
    axes[1].legend()
    _fin(axes[1], L("Sebaran Jumlah Daun", "Leaf Count Distribution"),
         L("Jumlah daun", "Leaves"), L("Jumlah pohon", "Trees"))
    fig.tight_layout()
    return fig


def fig_forest_convergence(forest, X_te, y_te, task="classification", steps=25):
    """Skor uji sebagai fungsi jumlah pohon — memperlihatkan titik jenuh ansambel."""
    ests = list(getattr(forest, "estimators_", []))
    if len(ests) < 3:
        return None
    X_te = np.asarray(X_te, dtype=float)
    ns = np.unique(np.linspace(1, len(ests), min(steps, len(ests))).astype(int))
    scores = []
    if task == "classification":
        classes = list(forest.classes_)
        probs = np.zeros((len(X_te), len(classes)))
        y_true = np.asarray(y_te)
        j = 0
        for n in ns:
            while j < n:
                p = ests[j].predict_proba(X_te)
                if p.shape[1] != len(classes):
                    return None
                probs += p
                j += 1
            pred = np.array([classes[i] for i in probs.argmax(axis=1)])
            scores.append(float((pred == y_true).mean()))
        ylab = L("Akurasi data uji", "Test accuracy")
    else:
        from sklearn.metrics import r2_score

        acc = np.zeros(len(X_te))
        j = 0
        for n in ns:
            while j < n:
                e = ests[j]
                acc += (e[0] if isinstance(e, (list, np.ndarray)) else e).predict(X_te)
                j += 1
            scores.append(float(r2_score(np.asarray(y_te, dtype=float), acc / n)))
        ylab = L("R² data uji", "Test R²")
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.plot(ns, scores, marker="o", ms=3.5, color=TH.color(0))
    best = int(np.argmax(scores))
    ax.scatter([ns[best]], [scores[best]], s=90, facecolors="none",
               edgecolors=TH.STATUS["critical"], linewidths=1.8, zorder=4)
    ax.annotate(L(f"terbaik: {ns[best]} pohon ({scores[best]:.4f})",
                  f"best: {ns[best]} trees ({scores[best]:.4f})"),
                (ns[best], scores[best]), textcoords="offset points", xytext=(8, -14),
                fontsize=9, color=TH.INK_MUTED)
    return _fin(ax, L("Konvergensi Ansambel: Skor vs Jumlah Pohon",
                      "Ensemble Convergence: Score vs Number of Trees"),
                L("Jumlah pohon yang digabung", "Trees combined"), ylab).figure


def fig_importance_spread(forest, feature_names, top=15):
    """Sebaran kepentingan fitur antar pohon (boxplot) — menunjukkan kestabilannya."""
    ests = [t[0] if isinstance(t, (list, np.ndarray)) else t
            for t in getattr(forest, "estimators_", [])]
    mats = [e.feature_importances_ for e in ests if hasattr(e, "feature_importances_")]
    if len(mats) < 2:
        return None
    M = np.vstack(mats)
    mean = M.mean(axis=0)
    order = np.argsort(mean)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(order) + 2.0))
    bp = ax.boxplot([M[:, i] for i in order], vert=False, widths=0.62, patch_artist=True,
                    medianprops=dict(color=TH.INK, lw=1.4),
                    flierprops=dict(marker=".", ms=3, mfc=TH.INK_MUTED, mec="none"))
    for p in bp["boxes"]:
        p.set_facecolor(TH.color(0))
        p.set_alpha(0.55)
    ax.set_yticklabels([str(feature_names[i]) for i in order])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    return _fin(ax, L("Kestabilan Kepentingan Fitur antar Pohon",
                      "Feature Importance Stability Across Trees"),
                L("Kepentingan (impurity decrease)", "Importance (impurity decrease)")).figure


def fig_oob_curve(forest_factory, X, y, max_trees=300, steps=12):
    """Kurva galat out-of-bag terhadap jumlah pohon."""
    from sklearn.base import clone

    base = forest_factory()
    if not hasattr(base, "set_params"):
        return None
    ns = np.unique(np.linspace(10, max_trees, steps).astype(int))
    errs = []
    for n in ns:
        m = clone(base)
        try:
            m.set_params(n_estimators=int(n), bootstrap=True, oob_score=True, warm_start=False)
            m.fit(np.asarray(X, dtype=float), y)
            errs.append(1.0 - float(m.oob_score_))
        except Exception:
            return None
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.plot(ns, errs, marker="o", ms=4, color=TH.color(1))
    lo = int(np.argmin(errs))
    ax.scatter([ns[lo]], [errs[lo]], s=90, facecolors="none",
               edgecolors=TH.STATUS["critical"], linewidths=1.8, zorder=4)
    return _fin(ax, L("Galat Out-of-Bag vs Jumlah Pohon", "Out-of-Bag Error vs Number of Trees"),
                L("Jumlah pohon", "Trees"), L("Galat OOB (1 - skor)", "OOB error (1 - score)")).figure


# =============================================== BATAS KEPUTUSAN & HYPERPLANE
def fig_decision_regions(model2d, X2, y, xlabel, ylabel, title=None, proba=False):
    """Peta wilayah keputusan pada bidang 2D + sebaran data sebenarnya."""
    X2 = np.asarray(X2, dtype=float)
    y = np.asarray(y)
    classes = list(getattr(model2d, "classes_", sorted(pd.unique(y))))
    colors = _class_colors(classes)
    xx, yy, grid = _mesh(X2)
    try:
        if proba and hasattr(model2d, "predict_proba") and len(classes) == 2:
            Z = model2d.predict_proba(grid)[:, 1].reshape(xx.shape)
        else:
            pred = model2d.predict(grid)
            Z = np.array([classes.index(v) for v in pred], dtype=float).reshape(xx.shape)
            proba = False
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    if proba:
        cf = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap=TH.CMAP_DIV, alpha=0.85)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[TH.INK], linewidths=1.6)
        cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(L(f"P(kelas = {classes[1]})", f"P(class = {classes[1]})"))
    else:
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap([colors[c] for c in classes])
        ax.contourf(xx, yy, Z, levels=np.arange(-0.5, len(classes), 1), cmap=cmap, alpha=0.20)
        ax.contour(xx, yy, Z, levels=np.arange(0.5, len(classes) - 0.5, 1),
                   colors=[TH.INK], linewidths=1.2)
    Xs, ys = _subsample(X2, y)
    _scatter_classes(ax, Xs, ys, classes, colors)
    ax.legend(title=L("Kelas", "Class"), loc="best")
    ax.grid(False)
    return _fin(ax, title or L("Wilayah Keputusan Model", "Model Decision Regions"),
                xlabel, ylabel).figure


def fig_svm_hyperplane(svc2d, X2, y, xlabel, ylabel, title=None):
    """Hyperplane pemisah, margin, dan support vector pada bidang 2D."""
    X2 = np.asarray(X2, dtype=float)
    y = np.asarray(y)
    classes = list(getattr(svc2d, "classes_", sorted(pd.unique(y))))
    colors = _class_colors(classes)
    xx, yy, grid = _mesh(X2)
    binary = len(classes) == 2
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    try:
        if binary:
            Z = svc2d.decision_function(grid).reshape(xx.shape)
            lim = float(np.nanmax(np.abs(Z))) or 1.0
            ax.contourf(xx, yy, Z, levels=np.linspace(-lim, lim, 25), cmap=TH.CMAP_DIV, alpha=0.55)
            cs = ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[TH.INK_MUTED, TH.INK, TH.INK_MUTED],
                            linestyles=["--", "-", "--"], linewidths=[1.2, 2.0, 1.2])
            ax.clabel(cs, fmt={-1: L("margin -1", "margin -1"), 0: L("hyperplane", "hyperplane"),
                               1: L("margin +1", "margin +1")}, fontsize=8)
        else:
            pred = svc2d.predict(grid)
            Z = np.array([classes.index(v) for v in pred], dtype=float).reshape(xx.shape)
            from matplotlib.colors import ListedColormap

            ax.contourf(xx, yy, Z, levels=np.arange(-0.5, len(classes), 1),
                        cmap=ListedColormap([colors[c] for c in classes]), alpha=0.20)
            ax.contour(xx, yy, Z, levels=np.arange(0.5, len(classes) - 0.5, 1),
                       colors=[TH.INK], linewidths=1.4)
    except Exception:
        return None
    Xs, ys = _subsample(X2, y)
    _scatter_classes(ax, Xs, ys, classes, colors, size=24, alpha=0.85)
    sv = getattr(svc2d, "support_vectors_", None)
    if sv is not None and len(sv):
        ax.scatter(sv[:, 0], sv[:, 1], s=140, facecolors="none", edgecolors=TH.INK,
                   linewidths=1.3, zorder=5,
                   label=L(f"Support vector ({len(sv)})", f"Support vectors ({len(sv)})"))
    ax.legend(loc="best")
    ax.grid(False)
    sub = L("garis tengah = hyperplane, garis putus-putus = batas margin",
            "solid line = hyperplane, dashed lines = margin boundaries") if binary else \
        L("batas antar wilayah kelas (kernel non-linear tampak melengkung)",
          "boundaries between class regions (a non-linear kernel appears curved)")
    return _fin(ax, (title or L("Hyperplane & Margin SVM", "SVM Hyperplane & Margin")) + f"\n{sub}",
                xlabel, ylabel).figure


def fig_svm_margin_dist(svc, X, y):
    """Sebaran nilai fungsi keputusan: seberapa jauh tiap titik dari hyperplane."""
    try:
        d = svc.decision_function(np.asarray(X, dtype=float))
    except Exception:
        return None
    if d.ndim > 1:
        d = d.max(axis=1) - np.sort(d, axis=1)[:, -2]
        xlab = L("Selisih skor kelas terbaik vs kedua (margin efektif)",
                 "Gap between best and runner-up class score (effective margin)")
    else:
        xlab = L("Nilai fungsi keputusan (jarak berskala ke hyperplane)",
                 "Decision function value (scaled distance to the hyperplane)")
    y = np.asarray(y)
    classes = sorted(pd.unique(y))
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for i, c in enumerate(classes[:8]):
        ax.hist(d[y == c], bins=30, alpha=0.6, color=TH.color(i), label=str(c))
    if d.ndim == 1:
        for v, st_ in [(-1, "--"), (0, "-"), (1, "--")]:
            ax.axvline(v, color=TH.INK, ls=st_, lw=1.2)
    ax.legend(title=L("Kelas", "Class"))
    return _fin(ax, L("Sebaran Jarak ke Hyperplane", "Distance-to-Hyperplane Distribution"),
                xlab, L("Jumlah sampel", "Samples")).figure


def fig_svr_tube(model1d, x, y, epsilon, xlabel, title=None):
    """Pita epsilon SVR: galat di dalam pita tidak dikenai penalti."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    grid = np.linspace(xs.min(), xs.max(), 400).reshape(-1, 1)
    try:
        pred = model1d.predict(grid).ravel()
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.fill_between(grid.ravel(), pred - epsilon, pred + epsilon, color=TH.color(0), alpha=0.18,
                    label=L(f"pita ε = {epsilon:g}", f"ε-tube = {epsilon:g}"))
    ax.plot(grid.ravel(), pred, color=TH.color(0), lw=2.2, label=L("prediksi SVR", "SVR prediction"))
    Xs, ys = _subsample(np.c_[x, y])
    ax.scatter(Xs[:, 0], Xs[:, 1], s=20, color=TH.INK_MUTED, alpha=0.55,
               label=L("data aktual", "actual data"))
    sv = getattr(model1d, "support_", None)
    if sv is not None and len(sv):
        ax.scatter(x[sv], y[sv], s=70, facecolors="none", edgecolors=TH.STATUS["critical"],
                   linewidths=1.2, label=L(f"support vector ({len(sv)})",
                                           f"support vectors ({len(sv)})"))
    ax.legend()
    return _fin(ax, title or L("Pita Epsilon & Support Vector (SVR)",
                               "Epsilon Tube & Support Vectors (SVR)"),
                xlabel, L("Nilai target", "Target value")).figure


# ======================================================== K-NEAREST NEIGHBORS
def fig_knn_neighbors(X2, y, k, xlabel, ylabel, n_points=3, seed=0):
    """Beberapa titik contoh beserta lingkaran yang memuat k tetangga terdekatnya."""
    from sklearn.neighbors import NearestNeighbors

    X2 = np.asarray(X2, dtype=float)
    y = np.asarray(y)
    if len(X2) <= k + 1:
        return None
    classes = sorted(pd.unique(y))
    colors = _class_colors(classes)
    nn = NearestNeighbors(n_neighbors=int(k) + 1).fit(X2)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(X2), size=min(n_points, len(X2)), replace=False)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    Xs, ys = _subsample(X2, y)
    _scatter_classes(ax, Xs, ys, classes, colors, size=20, alpha=0.45)
    for j, i in enumerate(picks):
        dist, idx = nn.kneighbors(X2[i].reshape(1, -1))
        r = float(dist[0][-1])
        ax.add_patch(plt.Circle((X2[i, 0], X2[i, 1]), r, fill=False, ls="--", lw=1.3,
                                color=TH.INK, alpha=0.8))
        nb = idx[0][1:]
        ax.scatter(X2[nb, 0], X2[nb, 1], s=90, facecolors="none", edgecolors=TH.STATUS["critical"],
                   linewidths=1.3, zorder=5)
        ax.scatter([X2[i, 0]], [X2[i, 1]], marker="*", s=260, color=TH.INK, zorder=6)
        vals, cnt = np.unique(y[nb], return_counts=True)
        ax.annotate(L(f"→ {vals[cnt.argmax()]}", f"→ {vals[cnt.argmax()]}"),
                    (X2[i, 0], X2[i, 1]), textcoords="offset points", xytext=(10, 10),
                    fontsize=9, fontweight="600", color=TH.INK)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(title=L("Kelas", "Class"), loc="best")
    ax.grid(False)
    return _fin(ax, L(f"Cara k-NN Memutuskan (k = {k})", f"How k-NN Decides (k = {k})"),
                xlabel, ylabel).figure


def fig_k_sensitivity(factory, X_tr, y_tr, X_te, y_te, ks=None, metric_name="Akurasi"):
    """Skor latih & uji untuk berbagai nilai k — memperlihatkan titik keseimbangan."""
    from sklearn.base import clone

    ks = list(ks or range(1, min(31, max(3, len(X_tr) // 3))))
    base = factory()
    tr, te = [], []
    for k in ks:
        m = clone(base)
        try:
            m.set_params(n_neighbors=int(k))
            m.fit(X_tr, y_tr)
            tr.append(float(m.score(X_tr, y_tr)))
            te.append(float(m.score(X_te, y_te)))
        except Exception:
            return None
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(ks, tr, marker="o", ms=3.5, color=TH.color(1), label=L("Data latih", "Train"))
    ax.plot(ks, te, marker="s", ms=3.5, color=TH.color(0), label=L("Data uji", "Test"))
    b = int(np.argmax(te))
    ax.axvline(ks[b], color=TH.INK, ls=":", lw=1.2)
    ax.annotate(L(f"k terbaik = {ks[b]}", f"best k = {ks[b]}"), (ks[b], te[b]),
                textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.legend()
    return _fin(ax, L("Sensitivitas terhadap Jumlah Tetangga (k)",
                      "Sensitivity to the Number of Neighbours (k)"),
                "k", metric_name).figure


# ================================================================= BOOSTING
def fig_boosting_stages(model, X_tr, y_tr, X_te, y_te, task="classification"):
    """Skor per tahap boosting: kapan model berhenti membaik dan mulai overfitting."""
    try:
        if task == "classification":
            from sklearn.metrics import accuracy_score

            tr = [accuracy_score(y_tr, p) for p in model.staged_predict(X_tr)]
            te = [accuracy_score(y_te, p) for p in model.staged_predict(X_te)]
            ylab = L("Akurasi", "Accuracy")
            best = int(np.argmax(te))
        else:
            from sklearn.metrics import r2_score

            tr = [r2_score(y_tr, p) for p in model.staged_predict(X_tr)]
            te = [r2_score(y_te, p) for p in model.staged_predict(X_te)]
            ylab = "R²"
            best = int(np.argmax(te))
    except Exception:
        return None
    n = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    ax.plot(n, tr, color=TH.color(1), lw=1.8, label=L("Data latih", "Train"))
    ax.plot(n, te, color=TH.color(0), lw=1.8, label=L("Data uji", "Test"))
    ax.axvline(best + 1, color=TH.INK, ls=":", lw=1.3)
    ax.annotate(L(f"puncak uji: tahap {best + 1}", f"test peak: stage {best + 1}"),
                (best + 1, te[best]), textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax.fill_between(n, te, tr, where=np.array(tr) > np.array(te), color=TH.STATUS["warning"],
                    alpha=0.14, label=L("selisih latih-uji (overfitting)",
                                        "train-test gap (overfitting)"))
    ax.legend()
    return _fin(ax, L("Perkembangan Skor per Tahap Boosting",
                      "Score Progression per Boosting Stage"),
                L("Tahap boosting (jumlah pohon)", "Boosting stage (number of trees)"), ylab).figure


def fig_boosting_loss(model):
    """Penurunan kerugian pada data latih di setiap tahap."""
    loss = getattr(model, "train_score_", None)
    if loss is None or len(np.atleast_1d(loss)) < 2:
        return None
    loss = np.asarray(loss, dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.plot(np.arange(1, len(loss) + 1), loss, color=TH.color(3), lw=1.9)
    return _fin(ax, L("Penurunan Kerugian (Loss) per Tahap", "Loss Reduction per Stage"),
                L("Tahap boosting", "Boosting stage"),
                L("Kerugian data latih", "Training loss")).figure


# ================================================== MODEL LINEAR & KOEFISIEN
def fig_coefficients(names, coefs, title=None, top=20, intercept=None):
    """Koefisien model linear, diurutkan berdasarkan besarnya pengaruh."""
    coefs = np.asarray(coefs, dtype=float).ravel()
    names = list(names)[:len(coefs)]
    order = np.argsort(np.abs(coefs))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(order) + 2.0))
    cols = [TH.color(0) if coefs[i] >= 0 else TH.color(7) for i in order]
    ax.barh(range(len(order)), coefs[order], color=cols, height=0.7)
    ax.axvline(0, color=TH.INK, lw=1.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(names[i]) for i in order])
    for i, v in enumerate(coefs[order]):
        ax.text(v, i, f" {v:+.4f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=TH.INK_MUTED)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(handles=[Patch(color=TH.color(0), label=L("positif", "positive")),
                       Patch(color=TH.color(7), label=L("negatif", "negative"))], loc="lower right")
    sub = "" if intercept is None else L(f"  (intersep = {intercept:,.4f})",
                                         f"  (intercept = {intercept:,.4f})")
    return _fin(ax, (title or L("Koefisien Model Linear", "Linear Model Coefficients")) + sub,
                L("Nilai koefisien", "Coefficient value")).figure


def fig_reg_path(kind, X, y, names, alphas=None, l1_ratio=0.5, chosen=None, top=12):
    """Jalur regularisasi: bagaimana tiap koefisien menyusut seiring alpha membesar."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alphas = np.logspace(-4, 2, 60) if alphas is None else np.asarray(alphas, dtype=float)
    try:
        if kind == "lasso":
            from sklearn.linear_model import lasso_path

            a, coefs, _ = lasso_path(X, y, alphas=alphas)
        elif kind == "elasticnet":
            from sklearn.linear_model import enet_path

            a, coefs, _ = enet_path(X, y, alphas=alphas, l1_ratio=float(l1_ratio))
        else:
            from sklearn.linear_model import Ridge

            coefs = []
            for al in alphas:
                coefs.append(Ridge(alpha=float(al)).fit(X, y).coef_.ravel())
            coefs = np.asarray(coefs).T
            a = alphas
    except Exception:
        return None
    imp = np.abs(coefs).max(axis=1)
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for j, i in enumerate(order):
        ax.plot(a, coefs[i], lw=1.8, color=TH.color(j), label=str(names[i])[:28])
    ax.set_xscale("log")
    ax.axhline(0, color=TH.INK, lw=1.0)
    if chosen:
        ax.axvline(float(chosen), color=TH.STATUS["critical"], ls="--", lw=1.5,
                   label=L(f"alpha dipakai = {chosen:g}", f"alpha in use = {chosen:g}"))
    ax.legend(ncol=2, fontsize=8)
    nm = {"lasso": "Lasso (L1)", "elasticnet": "Elastic Net", "ridge": "Ridge (L2)"}.get(kind, kind)
    return _fin(ax, L(f"Jalur Regularisasi {nm}", f"{nm} Regularization Path"),
                L("alpha (kekuatan regularisasi, skala log)",
                  "alpha (regularization strength, log scale)"),
                L("Nilai koefisien", "Coefficient value")).figure


def fig_logistic_curve(model, X, y, title=None):
    """Kurva logistik: skor linear -> probabilitas, dengan sebaran data sebenarnya."""
    try:
        d = model.decision_function(X)
        p = model.predict_proba(X)
    except Exception:
        return None
    if d.ndim > 1:
        return None
    y = np.asarray(y)
    classes = list(model.classes_)
    pos = p[:, 1]
    grid = np.linspace(np.nanmin(d) - 1, np.nanmax(d) + 1, 300)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(grid, 1 / (1 + np.exp(-grid)), color=TH.INK, lw=2.0,
            label=L("fungsi logistik", "logistic function"))
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(d[m], pos[m], s=22, color=TH.color(i), alpha=0.7, label=f"{L('kelas','class')} {c}")
    ax.axhline(0.5, color=TH.STATUS["warning"], ls="--", lw=1.2,
               label=L("ambang 0,5", "0.5 threshold"))
    ax.axvline(0, color=TH.INK_MUTED, ls=":", lw=1.1)
    ax.legend(fontsize=8)
    return _fin(ax, title or L("Kurva Logistik & Ambang Keputusan",
                               "Logistic Curve & Decision Threshold"),
                L("Skor linear (log-odds)", "Linear score (log-odds)"),
                L(f"P(kelas = {classes[-1]})", f"P(class = {classes[-1]})")).figure


def fig_odds_ratio(names, coefs, top=15):
    """Rasio odds (exp koefisien) untuk regresi logistik."""
    coefs = np.asarray(coefs, dtype=float).ravel()
    names = list(names)[:len(coefs)]
    order = np.argsort(np.abs(coefs))[::-1][:top][::-1]
    odds = np.exp(coefs[order])
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(order) + 2.0))
    ax.barh(range(len(order)), odds, color=[TH.color(0) if v >= 1 else TH.color(7) for v in odds],
            height=0.7)
    ax.axvline(1.0, color=TH.INK, lw=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(names[i]) for i in order])
    for i, v in enumerate(odds):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=TH.INK_MUTED)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    return _fin(ax, L("Rasio Odds per Fitur (exp koefisien)",
                      "Odds Ratio per Feature (exp of coefficient)"),
                L("Rasio odds — >1 menaikkan peluang, <1 menurunkan",
                  "Odds ratio — >1 raises the odds, <1 lowers them")).figure


# =============================================================== NAIVE BAYES
def fig_nb_densities(model, X, y, names, top=4):
    """Kurva Gauss yang dipelajari Naive Bayes untuk tiap fitur pada tiap kelas."""
    theta = getattr(model, "theta_", None)
    var = getattr(model, "var_", getattr(model, "sigma_", None))
    if theta is None or var is None:
        return None
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = list(model.classes_)
    spread = np.abs(theta - theta.mean(axis=0)).sum(axis=0) / (np.sqrt(var).mean(axis=0) + 1e-12)
    order = np.argsort(spread)[::-1][:min(top, X.shape[1])]
    n = len(order)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.4 * rows), squeeze=False)
    for ax, j in zip(axes.ravel(), order):
        lo, hi = np.nanmin(X[:, j]), np.nanmax(X[:, j])
        pad = 0.15 * ((hi - lo) or 1.0)
        grid = np.linspace(lo - pad, hi + pad, 300)
        for i, c in enumerate(classes):
            mu, sd = theta[i, j], np.sqrt(var[i, j])
            dens = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(grid, dens, color=TH.color(i), lw=2.0, label=str(c))
            ax.fill_between(grid, dens, color=TH.color(i), alpha=0.12)
            ax.axvline(mu, color=TH.color(i), ls=":", lw=1.1)
        ax.set_title(str(names[j]), fontsize=10)
        ax.set_xlabel(L("Nilai fitur", "Feature value"))
        ax.set_ylabel(L("Kepadatan", "Density"))
    axes.ravel()[0].legend(title=L("Kelas", "Class"), fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(L("Distribusi Gauss yang Diasumsikan Naive Bayes",
                   "Gaussian Distributions Assumed by Naive Bayes"),
                 x=0.01, ha="left", fontsize=12, fontweight="600")
    fig.tight_layout()
    return fig


def fig_nb_loglik(model, X, y):
    """Kontribusi log-likelihood tiap kelas untuk sampel uji."""
    try:
        jll = model._joint_log_likelihood(np.asarray(X, dtype=float))
    except Exception:
        return None
    classes = list(model.classes_)
    top = np.sort(jll, axis=1)
    margin = top[:, -1] - top[:, -2] if jll.shape[1] > 1 else top[:, -1]
    y = np.asarray(y)
    pred = np.array([classes[i] for i in jll.argmax(axis=1)])
    ok = pred == y
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.hist(margin[ok], bins=30, color=TH.STATUS["good"], alpha=0.7, label=L("benar", "correct"))
    if (~ok).any():
        ax.hist(margin[~ok], bins=30, color=TH.STATUS["critical"], alpha=0.7,
                label=L("salah", "wrong"))
    ax.legend()
    return _fin(ax, L("Keyakinan Naive Bayes (selisih log-likelihood)",
                      "Naive Bayes Confidence (log-likelihood gap)"),
                L("Selisih log-likelihood kelas terbaik vs kedua",
                  "Log-likelihood gap, best vs runner-up class"),
                L("Jumlah sampel", "Samples")).figure


# ================================================================ KLASTERING
def fig_kmeans_voronoi(centers2d, X2, labels, xlabel, ylabel, title=None):
    """Sel Voronoi: wilayah yang 'dimiliki' tiap pusat klaster."""
    centers2d = np.asarray(centers2d, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if centers2d.ndim != 2 or centers2d.shape[0] < 1:
        return None
    xx, yy, grid = _mesh(np.vstack([X2, centers2d]))
    d = ((grid[:, None, :] - centers2d[None, :, :]) ** 2).sum(axis=2)
    Z = d.argmin(axis=1).reshape(xx.shape).astype(float)
    k = centers2d.shape[0]
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.contourf(xx, yy, Z, levels=np.arange(-0.5, k, 1),
                cmap=ListedColormap([TH.color(i) for i in range(k)]), alpha=0.18)
    ax.contour(xx, yy, Z, levels=np.arange(0.5, k - 0.5, 1), colors=[TH.INK], linewidths=1.1)
    labels = np.asarray(labels)
    Xs, ls = _subsample(X2, labels)
    for i in range(k):
        m = ls == i
        if m.any():
            ax.scatter(Xs[m, 0], Xs[m, 1], s=22, color=TH.color(i), alpha=0.85,
                       edgecolors="#ffffff", linewidths=0.5, label=f"{L('Klaster','Cluster')} {i}")
    ax.scatter(centers2d[:, 0], centers2d[:, 1], marker="X", s=230, c=TH.INK, zorder=6,
               label=L("Pusat klaster", "Centroids"))
    for i, (cx, cy) in enumerate(centers2d):
        ax.annotate(str(i), (cx, cy), textcoords="offset points", xytext=(10, 8),
                    fontsize=10, fontweight="700", color=TH.INK)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(False)
    return _fin(ax, title or L("Sel Voronoi K-Means (wilayah tiap pusat)",
                               "K-Means Voronoi Cells (region owned by each centroid)"),
                xlabel, ylabel).figure


def fig_center_distance(X, labels, centers):
    """Sebaran jarak tiap titik ke pusat klasternya — klaster rapat vs longgar."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=float)
    ids = [i for i in sorted(set(labels)) if i != -1 and i < len(centers)]
    if not ids:
        return None
    data = [np.linalg.norm(X[labels == i] - centers[i], axis=1) for i in ids]
    fig, ax = plt.subplots(figsize=(9.0, 0.5 * len(ids) + 2.4))
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=TH.INK, lw=1.4),
                    flierprops=dict(marker=".", ms=3, mfc=TH.INK_MUTED, mec="none"))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(TH.color(ids[i]))
        p.set_alpha(0.6)
    ax.set_yticklabels([f"{L('Klaster','Cluster')} {i} (n={int((labels == i).sum())})" for i in ids])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    return _fin(ax, L("Kerapatan Klaster: Jarak Anggota ke Pusat",
                      "Cluster Compactness: Member Distance to Centroid"),
                L("Jarak Euclid (ruang berskala)", "Euclidean distance (scaled space)")).figure


def fig_kdist(X, min_samples, eps=None):
    """Grafik k-distance untuk memilih eps DBSCAN (cari titik siku)."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    k = int(min_samples)
    if len(X) <= k:
        return None
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    kd = np.sort(d[:, -1])
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(np.arange(len(kd)), kd, color=TH.color(0), lw=2.0)
    if eps is not None:
        ax.axhline(float(eps), color=TH.STATUS["critical"], ls="--", lw=1.5,
                   label=L(f"eps dipakai = {eps:g}", f"eps in use = {eps:g}"))
        below = int((kd <= float(eps)).sum())
        ax.axvline(below, color=TH.INK_MUTED, ls=":", lw=1.2)
        ax.annotate(L(f"{below} titik ({below / len(kd):.0%}) menjadi titik inti",
                      f"{below} points ({below / len(kd):.0%}) become core points"),
                    (below, float(eps)), textcoords="offset points", xytext=(10, 10), fontsize=9)
        ax.legend()
    return _fin(ax, L(f"Grafik k-distance (k = {k}) untuk Memilih eps",
                      f"k-distance Plot (k = {k}) for Choosing eps"),
                L("Titik data diurutkan", "Points sorted by distance"),
                L(f"Jarak ke tetangga ke-{k}", f"Distance to the {k}-th neighbour")).figure


def fig_dbscan_points(X2, labels, core_mask, xlabel, ylabel):
    """Pemisahan titik inti, titik batas, dan noise pada DBSCAN."""
    X2 = np.asarray(X2, dtype=float)
    labels = np.asarray(labels)
    core = np.zeros(len(labels), dtype=bool)
    core[np.asarray(core_mask, dtype=int)] = True
    noise = labels == -1
    border = (~core) & (~noise)
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for i in sorted(set(labels[labels != -1])):
        m = (labels == i) & core
        if m.any():
            ax.scatter(X2[m, 0], X2[m, 1], s=42, color=TH.color(i), alpha=0.95,
                       edgecolors="#ffffff", linewidths=0.5,
                       label=f"{L('Inti klaster','Core cluster')} {i}")
    if border.any():
        ax.scatter(X2[border, 0], X2[border, 1], s=26, facecolors="none",
                   edgecolors=TH.INK_MUTED, linewidths=1.0, label=L("Titik batas", "Border points"))
    if noise.any():
        ax.scatter(X2[noise, 0], X2[noise, 1], s=34, marker="x", color=TH.STATUS["critical"],
                   linewidths=1.4, label=L(f"Noise ({int(noise.sum())})",
                                           f"Noise ({int(noise.sum())})"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(False)
    return _fin(ax, L("DBSCAN: Titik Inti, Titik Batas, dan Noise",
                      "DBSCAN: Core, Border, and Noise Points"), xlabel, ylabel).figure


def _cov_2d(gmm, i):
    ct = gmm.covariance_type
    c = gmm.covariances_
    if ct == "full":
        return c[i][:2, :2]
    if ct == "tied":
        return c[:2, :2]
    if ct == "diag":
        return np.diag(c[i][:2])
    return np.eye(2) * float(c[i])


def fig_gmm_ellipses(gmm2d, X2, labels, xlabel, ylabel, title=None):
    """Elips kovarians tiap komponen Gauss (1, 2, dan 3 simpangan baku)."""
    X2 = np.asarray(X2, dtype=float)
    means = np.asarray(gmm2d.means_, dtype=float)[:, :2]
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    labels = np.asarray(labels)
    Xs, ls = _subsample(X2, labels)
    for i in sorted(set(ls)):
        m = ls == i
        ax.scatter(Xs[m, 0], Xs[m, 1], s=20, color=TH.color(int(i)), alpha=0.65,
                   edgecolors="#ffffff", linewidths=0.4,
                   label=f"{L('Komponen','Component')} {int(i)}")
    for i in range(len(means)):
        cov = _cov_2d(gmm2d, i)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for nsd, alpha in ((1, 0.30), (2, 0.18), (3, 0.10)):
            w, h = 2 * nsd * np.sqrt(np.maximum(vals, 1e-12))
            ax.add_patch(Ellipse(means[i], w, h, angle=ang, facecolor=TH.color(i),
                                 alpha=alpha, edgecolor=TH.color(i), lw=1.2))
        ax.scatter([means[i, 0]], [means[i, 1]], marker="X", s=160, c=TH.INK, zorder=6)
    w = getattr(gmm2d, "weights_", None)
    if w is not None:
        for i, mu in enumerate(means):
            ax.annotate(f"{w[i]:.0%}", mu, textcoords="offset points", xytext=(10, 8),
                        fontsize=9, fontweight="600")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(False)
    return _fin(ax, title or L("Elips Kovarians Gaussian Mixture (1σ, 2σ, 3σ)",
                               "Gaussian Mixture Covariance Ellipses (1σ, 2σ, 3σ)"),
                xlabel, ylabel).figure


def fig_gmm_information_criteria(X, ks, cov_type="full", rs=42):
    """AIC & BIC untuk berbagai jumlah komponen — dasar memilih k pada GMM."""
    from sklearn.mixture import GaussianMixture

    X = np.asarray(X, dtype=float)
    ks = list(ks)
    aic, bic = [], []
    for k in ks:
        try:
            g = GaussianMixture(n_components=int(k), covariance_type=cov_type,
                                random_state=rs).fit(X)
            aic.append(float(g.aic(X)))
            bic.append(float(g.bic(X)))
        except Exception:
            return None
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(ks, aic, marker="o", ms=4, color=TH.color(0), label="AIC")
    ax.plot(ks, bic, marker="s", ms=4, color=TH.color(1), label="BIC")
    for arr, c, nm in ((aic, TH.color(0), "AIC"), (bic, TH.color(1), "BIC")):
        j = int(np.argmin(arr))
        ax.scatter([ks[j]], [arr[j]], s=110, facecolors="none", edgecolors=c, linewidths=2.0)
        ax.annotate(L(f"{nm} min: k={ks[j]}", f"{nm} min: k={ks[j]}"), (ks[j], arr[j]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, color=c)
    ax.legend()
    return _fin(ax, L("Kriteria Informasi GMM (semakin kecil semakin baik)",
                      "GMM Information Criteria (lower is better)"),
                L("Jumlah komponen (k)", "Number of components (k)"),
                L("Nilai kriteria", "Criterion value")).figure


def fig_gmm_uncertainty(proba):
    """Ketidakpastian keanggotaan: probabilitas tertinggi tiap titik."""
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        return None
    top = p.max(axis=1)
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.hist(top, bins=30, color=TH.color(2))
    ax.axvline(0.9, color=TH.STATUS["good"], ls="--", lw=1.3, label=L("sangat yakin (>0,9)",
                                                                      "confident (>0.9)"))
    ax.axvline(0.6, color=TH.STATUS["warning"], ls="--", lw=1.3, label=L("ragu (<0,6)",
                                                                         "ambiguous (<0.6)"))
    ax.legend()
    amb = float((top < 0.6).mean())
    return _fin(ax, L(f"Keyakinan Keanggotaan Klaster — {amb:.1%} titik masih ragu",
                      f"Cluster Membership Confidence — {amb:.1%} of points are ambiguous"),
                L("Probabilitas keanggotaan tertinggi", "Highest membership probability"),
                L("Jumlah titik", "Points")).figure


def fig_linkage_merge(Z, top=25):
    """Jarak penggabungan pada tiap langkah aglomeratif (siku = jumlah klaster wajar)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or len(Z) < 3:
        return None
    d = Z[:, 2][::-1][:top]
    ks = np.arange(2, len(d) + 2)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(ks, d, marker="o", ms=4, color=TH.color(0))
    gap = np.diff(d)
    if len(gap):
        j = int(np.argmin(gap))
        ax.axvline(ks[j + 1], color=TH.STATUS["critical"], ls="--", lw=1.4,
                   label=L(f"lompatan terbesar setelah k = {ks[j + 1]}",
                           f"largest jump after k = {ks[j + 1]}"))
        ax.legend()
    return _fin(ax, L("Jarak Penggabungan Hierarkis per Jumlah Klaster",
                      "Hierarchical Merge Distance by Cluster Count"),
                L("Jumlah klaster bila dendrogram dipotong di sini",
                  "Clusters if the dendrogram is cut here"),
                L("Jarak penggabungan", "Merge distance")).figure


# =============================================================== ASSOCIATION
def _sets(freq: pd.DataFrame):
    col = "itemsets" if "itemsets" in freq.columns else freq.columns[-1]
    return [frozenset(s) for s in freq[col]], freq["support"].astype(float).values


def fig_itemset_lattice(freq: pd.DataFrame, top_per_level: int = 8):
    """Kisi (lattice) itemset: bagaimana itemset panjang dibentuk dari yang pendek."""
    if freq is None or freq.empty:
        return None
    sets, sup = _sets(freq)
    lens = np.array([len(s) for s in sets])
    levels = sorted(set(lens))
    if not levels:
        return None
    nodes = {}
    for lv in levels:
        idx = np.where(lens == lv)[0]
        idx = idx[np.argsort(sup[idx])[::-1][:top_per_level]]
        n = len(idx)
        for j, i in enumerate(idx):
            nodes[i] = ((j + 0.5) / n, lv)
    fig, ax = plt.subplots(figsize=(11.0, 1.9 * len(levels) + 2.2))
    for i, (xi, yi) in nodes.items():
        for j, (xj, yj) in nodes.items():
            if yj == yi + 1 and sets[i] < sets[j]:
                ax.plot([xi, xj], [yi, yj], color=TH.GRID, lw=1.0, zorder=1)
    smax = float(sup.max()) or 1.0
    for i, (xi, yi) in nodes.items():
        s = 120 + 900 * (sup[i] / smax)
        ax.scatter([xi], [yi], s=s, color=TH.color(int(yi) - 1), alpha=0.85,
                   edgecolors=TH.INK, linewidths=0.7, zorder=3)
        ax.annotate(", ".join(sorted(map(str, sets[i])))[:34], (xi, yi),
                    textcoords="offset points", xytext=(0, 13), ha="center", fontsize=7.5,
                    color=TH.INK)
        ax.annotate(f"{sup[i]:.3f}", (xi, yi), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=7, color=TH.INK_MUTED)
    ax.set_yticks(levels)
    ax.set_yticklabels([L(f"{lv}-itemset", f"{lv}-itemset") for lv in levels])
    ax.set_xticks([])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(min(levels) - 0.5, max(levels) + 0.6)
    ax.grid(False)
    return _fin(ax, L("Kisi Itemset (ukuran lingkaran = support)",
                      "Itemset Lattice (circle size = support)"),
                L("Itemset teratas pada tiap tingkat", "Top itemsets at each level"),
                L("Panjang itemset", "Itemset length")).figure


def fig_itemset_levels(freq: pd.DataFrame, min_support=None):
    """Jumlah dan support itemset per tingkat — jejak pemangkasan Apriori."""
    if freq is None or freq.empty:
        return None
    sets, sup = _sets(freq)
    lens = np.array([len(s) for s in sets])
    levels = sorted(set(lens))
    counts = [int((lens == lv).sum()) for lv in levels]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].bar([str(lv) for lv in levels], counts, color=TH.color(0), width=0.62)
    for i, v in enumerate(counts):
        axes[0].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9, color=TH.INK_MUTED)
    _fin(axes[0], L("Jumlah Itemset Sering per Tingkat", "Frequent Itemsets per Level"),
         L("Panjang itemset", "Itemset length"), L("Jumlah itemset", "Itemsets"))
    data = [sup[lens == lv] for lv in levels]
    bp = axes[1].boxplot(data, widths=0.6, patch_artist=True,
                         medianprops=dict(color=TH.INK, lw=1.4),
                         flierprops=dict(marker=".", ms=3, mfc=TH.INK_MUTED, mec="none"))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(TH.color(i))
        p.set_alpha(0.6)
    axes[1].set_xticklabels([str(lv) for lv in levels])
    if min_support is not None:
        axes[1].axhline(float(min_support), color=TH.STATUS["critical"], ls="--", lw=1.4,
                        label=L(f"min. support = {min_support:g}",
                                f"min. support = {min_support:g}"))
        axes[1].legend(fontsize=8)
    _fin(axes[1], L("Sebaran Support per Tingkat", "Support Distribution per Level"),
         L("Panjang itemset", "Itemset length"), "Support")
    fig.tight_layout()
    return fig


def fig_rule_quadrant(rules: pd.DataFrame, min_conf=None, min_lift=None, top=200):
    """Peta support-confidence dengan kuadran mutu aturan."""
    if rules is None or rules.empty:
        return None
    r = rules.head(top)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sc = ax.scatter(r["support"], r["confidence"], c=r["lift"], cmap=TH.CMAP_SEQ,
                    s=40 + 160 * (r["lift"] / max(float(r["lift"].max()), 1e-9)),
                    alpha=0.85, edgecolors=TH.INK, linewidths=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Lift")
    if min_conf is not None:
        ax.axhline(float(min_conf), color=TH.STATUS["warning"], ls="--", lw=1.3,
                   label=L(f"min. confidence = {min_conf:g}", f"min. confidence = {min_conf:g}"))
    med_s = float(r["support"].median())
    ax.axvline(med_s, color=TH.INK_MUTED, ls=":", lw=1.1)
    ax.annotate(L("sering & andal\n(prioritas tindakan)", "frequent & reliable\n(act on these)"),
                (r["support"].max(), r["confidence"].max()), ha="right", va="top",
                fontsize=9, color=TH.INK_MUTED)
    ax.annotate(L("jarang tapi andal\n(peluang ceruk)", "rare but reliable\n(niche opportunity)"),
                (r["support"].min(), r["confidence"].max()), ha="left", va="top",
                fontsize=9, color=TH.INK_MUTED)
    if min_conf is not None:
        ax.legend(fontsize=8)
    return _fin(ax, L("Kuadran Mutu Aturan Asosiasi", "Association Rule Quality Quadrants"),
                "Support", "Confidence").figure


# =============================================================== TIME SERIES
def fig_arima_roots(res, title=None):
    """Akar terbalik AR & MA pada lingkaran satuan: di dalam = stabil."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    for ax, kind in zip(axes, ("ar", "ma")):
        roots = getattr(res, f"{kind}roots", None)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=TH.INK_MUTED, lw=1.3, ls="--"))
        ax.axhline(0, color=TH.GRID, lw=1.0)
        ax.axvline(0, color=TH.GRID, lw=1.0)
        if roots is not None and len(np.atleast_1d(roots)):
            inv = 1.0 / np.asarray(roots, dtype=complex)
            inside = np.abs(inv) < 1
            ax.scatter(inv.real[inside], inv.imag[inside], s=90, color=TH.STATUS["good"],
                       edgecolors=TH.INK, linewidths=0.6, zorder=3,
                       label=L("di dalam (stabil)", "inside (stable)"))
            if (~inside).any():
                ax.scatter(inv.real[~inside], inv.imag[~inside], s=90, marker="X",
                           color=TH.STATUS["critical"], zorder=3,
                           label=L("di luar (tidak stabil)", "outside (unstable)"))
            ax.legend(fontsize=8, loc="lower left")
        else:
            ax.text(0, 0, L("tidak ada akar", "no roots"), ha="center", va="center",
                    fontsize=10, color=TH.INK_MUTED)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.grid(False)
        _fin(ax, L(f"Akar terbalik {kind.upper()}", f"Inverse {kind.upper()} roots"),
             L("Bagian riil", "Real part"), L("Bagian imajiner", "Imaginary part"))
    fig.suptitle(title or L("Kestabilan & Invertibilitas Model (akar terbalik)",
                            "Model Stability & Invertibility (inverse roots)"),
                 x=0.01, ha="left", fontsize=12, fontweight="600")
    fig.tight_layout()
    return fig


def fig_order_heatmap(results, ic="aic", best=None):
    """Peta panas kriteria informasi untuk tiap kombinasi ordo (p, q)."""
    if not results:
        return None
    col = 2 if ic == "aic" else 3
    rec = {}
    for o, so, aic, bic in results:
        v = aic if col == 2 else bic
        key = (o[0], o[2])
        if key not in rec or v < rec[key]:
            rec[key] = v
    ps = sorted({k[0] for k in rec})
    qs = sorted({k[1] for k in rec})
    M = np.full((len(ps), len(qs)), np.nan)
    for (p, q), v in rec.items():
        M[ps.index(p), qs.index(q)] = v
    fig, ax = plt.subplots(figsize=(1.1 * len(qs) + 4.2, 0.9 * len(ps) + 3.2))
    im = ax.imshow(M, cmap=TH.CMAP_SEQ.reversed(), aspect="auto")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"q={q}" for q in qs])
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels([f"p={p}" for p in ps])
    fin = np.isfinite(M)
    vmid = (np.nanmin(M) + np.nanmax(M)) / 2 if fin.any() else 0
    for i in range(len(ps)):
        for j in range(len(qs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if M[i, j] < vmid else TH.INK)
    if best is not None:
        try:
            bi, bj = ps.index(best[0]), qs.index(best[2])
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                       edgecolor=TH.STATUS["critical"], lw=2.6))
        except (ValueError, IndexError):
            pass
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(ic.upper())
    ax.grid(False)
    return _fin(ax, L(f"Peta {ic.upper()} untuk Kombinasi Ordo (semakin kecil semakin baik)",
                      f"{ic.upper()} Map Across Orders (lower is better)"),
                L("Ordo MA (q)", "MA order (q)"), L("Ordo AR (p)", "AR order (p)")).figure


def fig_arima_coefficients(res, alpha=0.05):
    """Koefisien model beserta selang kepercayaan dan penanda signifikansi."""
    try:
        params = res.params
        conf = res.conf_int(alpha=alpha)
        pvals = res.pvalues
    except Exception:
        return None
    names = list(getattr(params, "index", range(len(params))))
    vals = np.asarray(params, dtype=float)
    conf = np.asarray(conf, dtype=float)
    p = np.asarray(pvals, dtype=float)
    keep = [i for i, n in enumerate(names) if str(n) != "sigma2"]
    if not keep:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 0.5 * len(keep) + 2.2))
    for row, i in enumerate(keep):
        sig = p[i] < 0.05
        c = TH.color(0) if sig else TH.INK_MUTED
        ax.plot([conf[i, 0], conf[i, 1]], [row, row], color=c, lw=2.0, solid_capstyle="round")
        ax.scatter([vals[i]], [row], s=70, color=c, zorder=4)
        ax.text(conf[i, 1], row, f"  p={p[i]:.3g}" + ("*" if sig else ""), va="center",
                fontsize=8, color=TH.INK_MUTED)
    ax.axvline(0, color=TH.INK, lw=1.2)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([str(names[i]) for i in keep])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(handles=[Line2D([0], [0], color=TH.color(0), lw=2.4,
                              label=L("signifikan (p < 0,05)", "significant (p < 0.05)")),
                       Line2D([0], [0], color=TH.INK_MUTED, lw=2.4,
                              label=L("tidak signifikan", "not significant"))], fontsize=8)
    return _fin(ax, L(f"Koefisien Model & Selang Kepercayaan {(1 - alpha):.0%}",
                      f"Model Coefficients & {(1 - alpha):.0%} Confidence Intervals"),
                L("Nilai koefisien", "Coefficient value")).figure


def fig_impulse_response(res, steps=24):
    """Respons impuls: berapa lama guncangan satu periode terasa pada deret."""
    try:
        irf = np.asarray(res.impulse_responses(int(steps)), dtype=float).ravel()
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ax.bar(np.arange(len(irf)), irf, color=TH.color(0), width=0.65)
    ax.axhline(0, color=TH.INK, lw=1.0)
    cum = np.cumsum(irf)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(irf)), cum, color=TH.color(1), lw=1.8,
             label=L("dampak kumulatif", "cumulative impact"))
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="upper right")
    return _fin(ax, L("Respons Impuls: Dampak Satu Guncangan",
                      "Impulse Response: Effect of a One-Off Shock"),
                L("Periode setelah guncangan", "Periods after the shock"),
                L("Perubahan nilai deret", "Change in the series")).figure
