"""Modul Klastering."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core import algviz as AV
from core import params as PR
from core import paramui as PUI
from core import settings as SET
from core import stats as S
from core import ui as U
from core import viz as V
from core.i18n import L

ALGOS = PR.algos("clustering")


def config(df: pd.DataFrame) -> dict | None:
    SET.consume_pending("clustering")
    st.markdown("### " + L("Konfigurasi Klastering", "Clustering Setup"))
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        U.warn(L("Butuh minimal 2 kolom numerik.", "At least 2 numeric columns are required."))
        return None
    c1, c2 = st.columns([2, 1])
    features = SET.multiselect(L("Kolom fitur numerik", "Numeric feature columns"), num_cols,
                               key="clu_feats", default=num_cols, container=c1)
    algo = SET.selectbox(L("Algoritma", "Algorithm"), ALGOS, key="clu_algo", container=c2)

    c3, c4, c5 = st.columns(3)
    scaling = SET.selectbox(L("Penskalaan", "Scaling"), ["standard", "minmax", "none"],
                            key="clu_scale", container=c3)
    rs = SET.number_input(L("Random state", "Random state"), 0, 9999, 42, key="clu_rs", container=c4)
    kmax = SET.slider(L("Rentang k untuk evaluasi", "k range for evaluation"), 3, 12, 8,
                      key="clu_kmax", container=c5)

    if len(features) < 2:
        U.warn(L("Pilih minimal 2 fitur.", "Select at least 2 features."))
        return None

    params = PUI.editor("clustering", [algo], "clu", expanded=True,
                        title=L(f"Parameter algoritma — {algo}", f"Algorithm parameters — {algo}"))
    rp = PR.resolve("clustering", algo, params.get(algo, {}))
    cfg = dict(features=features, algo=algo, scaling=scaling, rs=int(rs), kmax=int(kmax),
               params=params)
    # kunci ringkas agar sisa modul (dan pengujian lama) tetap sederhana
    if algo == "K-Means":
        cfg.update(k=int(rp["n_clusters"]), init=rp["init"])
    elif algo == "Agglomerative (Hierarchical)":
        cfg.update(k=int(rp["n_clusters"]), linkage=rp["linkage"])
    elif algo == "Gaussian Mixture":
        cfg.update(k=int(rp["n_components"]), cov=rp["covariance_type"])
    else:
        cfg.update(eps=float(rp["eps"]), min_samples=int(rp["min_samples"]))
    SET.panel("clustering", cfg)
    return cfg


def _scale(X, how):
    if how == "standard":
        return StandardScaler().fit_transform(X)
    if how == "minmax":
        return MinMaxScaler().fit_transform(X)
    return np.asarray(X, dtype=float)


def _algo_visuals(algo, model, Xs, labels, centers, features, cfg, tables, figs):
    """Bagian 5: visualisasi yang menjelaskan cara kerja algoritma klastering."""
    U.section(L(f"5. Visualisasi Khusus Algoritma — {algo}",
                f"5. Algorithm-Specific Visualization — {algo}"),
              L("Grafik berikut menunjukkan bagaimana algoritma membentuk klaster: wilayah milik "
                "tiap pusat, elips kovarians, titik inti vs noise, atau jarak penggabungan.",
                "The charts below show how the algorithm forms clusters: the region owned by each "
                "centroid, covariance ellipses, core versus noise points, or merge distances."))
    mode, cols = PUI.projection_controls("clu", features)
    X2, lab, proj = AV.project_2d(Xs, mode, cols, features)

    def add(title, fig):
        if fig is None:
            return
        figs[title] = fig
        st.markdown(f"**{title}**")
        st.pyplot(fig, width="stretch")

    if algo == "K-Means":
        U.note(L(
            "K-Means membagi ruang menjadi sel Voronoi: setiap titik menjadi anggota pusat "
            "terdekat, sehingga batas antar klaster selalu berupa garis lurus. Klaster berbentuk "
            "memanjang atau melengkung sulit ditangkap dengan baik oleh K-Means.",
            "K-Means partitions the space into Voronoi cells: every point joins its nearest "
            "centroid, so cluster boundaries are always straight lines. Elongated or curved "
            "clusters are poorly captured by K-Means."))
        km2 = PR.build("clustering", algo, cfg.get("params", {}).get(algo), cfg["rs"]).fit(X2)
        add(L("Sel Voronoi K-Means", "K-Means Voronoi cells"),
            AV.fig_kmeans_voronoi(km2.cluster_centers_, X2, km2.labels_, *lab))
        if centers is not None:
            add(L("Kerapatan klaster (jarak ke pusat)", "Cluster compactness (distance to centroid)"),
                AV.fig_center_distance(Xs, labels, centers))

    elif algo == "DBSCAN":
        rp = PR.resolve("clustering", algo, cfg.get("params", {}).get(algo))
        U.note(L(
            "DBSCAN menumbuhkan klaster dari <b>titik inti</b> — titik yang punya minimal "
            "<code>min_samples</code> tetangga dalam radius <code>eps</code>. Titik yang tidak "
            "memenuhi syarat itu dan tidak bertetangga dengan titik inti ditandai sebagai noise. "
            "Grafik k-distance membantu memilih eps: pilih nilai pada titik siku kurva.",
            "DBSCAN grows clusters from <b>core points</b> — points with at least "
            "<code>min_samples</code> neighbours within radius <code>eps</code>. Points that fail "
            "that test and touch no core point are marked noise. The k-distance plot helps choose "
            "eps: pick the value at the elbow of the curve."))
        add(L("Grafik k-distance untuk memilih eps", "k-distance plot for choosing eps"),
            AV.fig_kdist(Xs, int(rp["min_samples"]), float(rp["eps"])))
        add(L("Titik inti, batas, dan noise", "Core, border, and noise points"),
            AV.fig_dbscan_points(X2, labels, getattr(model, "core_sample_indices_", []), *lab))
        core = np.zeros(len(labels), dtype=bool)
        core[np.asarray(getattr(model, "core_sample_indices_", []), dtype=int)] = True
        tables[L("Komposisi Titik DBSCAN", "DBSCAN Point Composition")] = pd.DataFrame({
            L("Jenis titik", "Point type"): [L("Titik inti", "Core"), L("Titik batas", "Border"),
                                             L("Noise", "Noise")],
            L("Jumlah", "Count"): [int(core.sum()),
                                   int(((~core) & (labels != -1)).sum()),
                                   int((labels == -1).sum())]})
        U.show_table(tables[L("Komposisi Titik DBSCAN", "DBSCAN Point Composition")])

    elif algo == "Gaussian Mixture":
        rp = PR.resolve("clustering", algo, cfg.get("params", {}).get(algo))
        U.note(L(
            "GMM memodelkan data sebagai campuran beberapa distribusi normal. Berbeda dari "
            "K-Means, klaster boleh berbentuk elips dengan orientasi dan ukuran berbeda, dan "
            "setiap titik memiliki <b>probabilitas</b> keanggotaan, bukan keputusan tegas.",
            "GMM models the data as a mixture of normal distributions. Unlike K-Means, clusters may "
            "be ellipses with different sizes and orientations, and every point carries a membership "
            "<b>probability</b> rather than a hard assignment."))
        gm2 = PR.build("clustering", algo, cfg.get("params", {}).get(algo), cfg["rs"]).fit(X2)
        add(L("Elips kovarians komponen", "Component covariance ellipses"),
            AV.fig_gmm_ellipses(gm2, X2, gm2.predict(X2), *lab))
        try:
            add(L("Keyakinan keanggotaan", "Membership confidence"),
                AV.fig_gmm_uncertainty(model.predict_proba(Xs)))
        except Exception:
            pass
        add(L("Kriteria informasi AIC/BIC", "AIC/BIC information criteria"),
            AV.fig_gmm_information_criteria(Xs, range(2, int(cfg["kmax"]) + 1),
                                            str(rp["covariance_type"]), cfg["rs"]))

    else:  # Agglomerative
        rp = PR.resolve("clustering", algo, cfg.get("params", {}).get(algo))
        U.note(L(
            "Klastering hierarkis menggabungkan dua kelompok terdekat berulang kali. Lompatan "
            "besar pada jarak penggabungan menandakan bahwa dua kelompok yang digabung sebenarnya "
            "sudah jauh berbeda — jumlah klaster tepat sebelum lompatan itu biasanya wajar.",
            "Hierarchical clustering repeatedly merges the two closest groups. A large jump in merge "
            "distance means the two groups joined were in fact far apart — the number of clusters "
            "just before that jump is usually a sensible choice."))
        if len(Xs) <= 2000:
            try:
                from scipy.cluster.hierarchy import linkage

                method = str(rp["linkage"])
                Z = linkage(Xs, method=method if method != "single" else "single")
                add(L("Jarak penggabungan per jumlah klaster",
                      "Merge distance by cluster count"), AV.fig_linkage_merge(Z))
            except Exception:
                pass
        if centers is not None:
            add(L("Kerapatan klaster (jarak ke pusat)", "Cluster compactness (distance to centroid)"),
                AV.fig_center_distance(Xs, labels, centers))


def run(df: pd.DataFrame, cfg: dict):
    features = cfg["features"]
    data = df[features].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    keep_idx = data.dropna(how="all").index
    data = data.loc[keep_idx]
    imputed = SimpleImputer(strategy="median").fit_transform(data)
    data = pd.DataFrame(imputed, columns=features, index=data.index)
    Xs = _scale(data.values, cfg["scaling"])

    tables, figs = {}, {}

    U.section(L("1. Analisis Statistik Data", "1. Data Statistical Analysis"))
    t0, f0 = U.descriptive_block(data, "clu")
    tables.update(t0)
    figs.update(f0)

    # ------------------------------------------- penentuan jumlah klaster
    U.section(L("2. Penentuan Jumlah Klaster Optimal", "2. Determining the Optimal Number of Clusters"))
    ks = list(range(2, min(cfg["kmax"], max(3, len(data) - 1)) + 1))
    rows, inertias, sils, dbs, chs = [], [], [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg["rs"]).fit(Xs)
        lab = km.labels_
        sil = silhouette_score(Xs, lab) if len(set(lab)) > 1 else np.nan
        db = davies_bouldin_score(Xs, lab) if len(set(lab)) > 1 else np.nan
        ch = calinski_harabasz_score(Xs, lab) if len(set(lab)) > 1 else np.nan
        inertias.append(float(km.inertia_))
        sils.append(float(sil))
        dbs.append(float(db))
        chs.append(float(ch))
        rows.append({"k": k, "Inertia (WCSS)": round(float(km.inertia_), 4),
                     L("Silhouette", "Silhouette"): round(float(sil), 4),
                     "Davies-Bouldin": round(float(db), 4),
                     "Calinski-Harabasz": round(float(ch), 4)})
    eval_k = pd.DataFrame(rows)
    tables[L("Evaluasi Jumlah Klaster", "Cluster Count Evaluation")] = eval_k
    U.note(L(
        "Silhouette tinggi (mendekati 1) = klaster rapat & terpisah. Davies-Bouldin rendah = lebih baik. "
        "Calinski-Harabasz tinggi = lebih baik. Elbow: cari titik siku pada kurva inertia.",
        "High silhouette (near 1) = compact, well-separated clusters. Lower Davies-Bouldin = better. "
        "Higher Calinski-Harabasz = better. Elbow: look for the bend in the inertia curve.",
    ))
    U.show_table(eval_k)
    k_sil = ks[int(np.nanargmax(sils))]
    k_db = ks[int(np.nanargmin(dbs))]
    k_ch = ks[int(np.nanargmax(chs))]
    U.metric_row([
        (L("k terbaik (Silhouette)", "Best k (Silhouette)"), str(k_sil)),
        (L("k terbaik (Davies-Bouldin)", "Best k (Davies-Bouldin)"), str(k_db)),
        (L("k terbaik (Calinski-H.)", "Best k (Calinski-H.)"), str(k_ch)),
        (L("k yang dipakai", "k in use"), str(cfg.get("k", "-"))),
    ])
    figs[L("Metode elbow", "Elbow method")] = V.fig_elbow(ks, inertias)
    figs[L("Silhouette per k", "Silhouette by k")] = V.fig_metric_by_k(ks, sils, L("Silhouette", "Silhouette"))
    figs[L("Davies-Bouldin per k", "Davies-Bouldin by k")] = V.fig_metric_by_k(
        ks, dbs, "Davies-Bouldin", higher_better=False)
    e1, e2 = st.columns(2)
    e1.pyplot(figs[L("Metode elbow", "Elbow method")], width="stretch")
    e2.pyplot(figs[L("Silhouette per k", "Silhouette by k")], width="stretch")
    st.pyplot(figs[L("Davies-Bouldin per k", "Davies-Bouldin by k")], width="stretch")

    # ------------------------------------------------------- klastering
    U.section(L("3. Hasil Klastering", "3. Clustering Result"))
    algo = cfg["algo"]
    aparams = cfg.get("params", {}).get(algo)
    model = PR.build("clustering", algo, aparams, cfg["rs"])
    if algo == "K-Means":
        model.fit(Xs)
        labels = model.labels_
        centers_scaled = model.cluster_centers_
    elif algo == "Agglomerative (Hierarchical)":
        labels = model.fit_predict(Xs)
        centers_scaled = np.vstack([Xs[labels == k].mean(axis=0) for k in sorted(set(labels))])
    elif algo == "Gaussian Mixture":
        model.fit(Xs)
        labels = model.predict(Xs)
        centers_scaled = model.means_
    else:
        labels = model.fit_predict(Xs)
        valid = sorted([k for k in set(labels) if k != -1])
        centers_scaled = np.vstack([Xs[labels == k].mean(axis=0) for k in valid]) if valid else None

    ptab = PUI.combined_params_table("clustering", [algo], cfg.get("params", {}))
    if not ptab.empty:
        tables[L("Parameter Algoritma", "Algorithm Parameters")] = ptab

    n_clusters = len([k for k in set(labels) if k != -1])
    n_noise = int((labels == -1).sum())
    if n_clusters < 1:
        st.error(L("Algoritma tidak menemukan klaster. Sesuaikan parameter (mis. eps/min_samples).",
                   "The algorithm found no clusters. Adjust the parameters (e.g. eps/min_samples)."))
        return

    mask = labels != -1
    sil = silhouette_score(Xs[mask], labels[mask]) if n_clusters > 1 and mask.sum() > n_clusters else np.nan
    db = davies_bouldin_score(Xs[mask], labels[mask]) if n_clusters > 1 and mask.sum() > n_clusters else np.nan
    ch = calinski_harabasz_score(Xs[mask], labels[mask]) if n_clusters > 1 and mask.sum() > n_clusters else np.nan

    U.metric_row([
        (L("Jumlah klaster", "Clusters found"), str(n_clusters)),
        (L("Silhouette", "Silhouette"), f"{sil:.4f}" if np.isfinite(sil) else "-"),
        ("Davies-Bouldin", f"{db:.4f}" if np.isfinite(db) else "-"),
        ("Calinski-Harabasz", f"{ch:.2f}" if np.isfinite(ch) else "-"),
    ])
    if n_noise:
        st.caption(L(f"Titik noise (label -1): {n_noise} ({n_noise / len(labels) * 100:.1f}%)",
                     f"Noise points (label -1): {n_noise} ({n_noise / len(labels) * 100:.1f}%)"))

    metrik = pd.DataFrame([
        (L("Algoritma", "Algorithm"), algo),
        (L("Jumlah klaster", "Number of clusters"), n_clusters),
        (L("Titik noise", "Noise points"), n_noise),
        (L("Silhouette Score", "Silhouette Score"), round(float(sil), 4) if np.isfinite(sil) else "-"),
        ("Davies-Bouldin Index", round(float(db), 4) if np.isfinite(db) else "-"),
        ("Calinski-Harabasz Index", round(float(ch), 4) if np.isfinite(ch) else "-"),
    ], columns=[L("Metrik", "Metric"), L("Nilai", "Value")])
    tables[L("Evaluasi Klastering", "Clustering Evaluation")] = metrik
    U.show_table(metrik)

    # profil klaster
    labeled = data.copy()
    labeled[L("Klaster", "Cluster")] = labels
    prof_mean = labeled.groupby(L("Klaster", "Cluster"))[features].mean().round(4)
    prof_std = labeled.groupby(L("Klaster", "Cluster"))[features].std().round(4)
    sizes = labeled[L("Klaster", "Cluster")].value_counts().sort_index()
    prof = prof_mean.copy()
    prof.insert(0, L("Jumlah Anggota", "Members"), sizes)
    prof.insert(1, L("Persentase (%)", "Percentage (%)"), (sizes / len(labeled) * 100).round(2))
    tables[L("Profil Klaster (rata-rata)", "Cluster Profile (mean)")] = prof.reset_index()
    tables[L("Profil Klaster (std)", "Cluster Profile (std)")] = prof_std.reset_index()
    st.markdown("**" + L("Profil rata-rata tiap klaster", "Mean profile per cluster") + "**")
    U.show_table(prof.reset_index())

    zprof = ((prof_mean - data[features].mean()) / data[features].std().replace(0, 1)).round(4)
    tables[L("Profil Klaster (skor-z)", "Cluster Profile (z-score)")] = zprof.reset_index()

    p1, p2 = st.columns(2)
    figs[L("Ukuran klaster", "Cluster sizes")] = V.fig_cluster_sizes(labels)
    figs[L("Profil klaster", "Cluster profile")] = V.fig_cluster_profile(zprof)
    p1.pyplot(figs[L("Ukuran klaster", "Cluster sizes")], width="stretch")
    p2.pyplot(figs[L("Profil klaster", "Cluster profile")], width="stretch")

    # visualisasi PCA
    U.section(L("4. Visualisasi Sebaran Klaster", "4. Cluster Scatter Visualization"))
    pca = PCA(n_components=min(3, Xs.shape[1]))
    coords = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_
    centers_2d = pca.transform(centers_scaled)[:, :2] if centers_scaled is not None else None
    figs[L("Sebaran klaster (PCA)", "Cluster scatter (PCA)")] = V.fig_cluster_scatter(
        coords[:, :2], labels,
        f"PC1 ({ev[0] * 100:.1f}% {L('varians', 'variance')})",
        f"PC2 ({ev[1] * 100:.1f}% {L('varians', 'variance')})" if len(ev) > 1 else "PC2",
        centers_2d,
        L(f"Sebaran Klaster - {algo} (PCA 2D)", f"Cluster Scatter - {algo} (PCA 2D)"))
    st.pyplot(figs[L("Sebaran klaster (PCA)", "Cluster scatter (PCA)")], width="stretch")
    tables[L("Varians PCA", "PCA Variance")] = pd.DataFrame({
        L("Komponen", "Component"): [f"PC{i + 1}" for i in range(len(ev))],
        L("Proporsi Varians (%)", "Variance Explained (%)"): (ev * 100).round(2),
        L("Kumulatif (%)", "Cumulative (%)"): (np.cumsum(ev) * 100).round(2)})
    U.show_table(tables[L("Varians PCA", "PCA Variance")])

    # scatter dua fitur asli
    fa, fb = st.columns(2)
    x_f = fa.selectbox(L("Sumbu X (fitur asli)", "X axis (original feature)"), features, index=0, key="clu_x")
    y_f = fb.selectbox(L("Sumbu Y (fitur asli)", "Y axis (original feature)"), features,
                       index=min(1, len(features) - 1), key="clu_y")
    raw_coords = data[[x_f, y_f]].values
    figs[L(f"Sebaran {x_f} vs {y_f}", f"Scatter {x_f} vs {y_f}")] = V.fig_cluster_scatter(
        raw_coords, labels, x_f, y_f, None,
        L(f"Sebaran Klaster: {x_f} vs {y_f}", f"Cluster Scatter: {x_f} vs {y_f}"))
    st.pyplot(figs[L(f"Sebaran {x_f} vs {y_f}", f"Scatter {x_f} vs {y_f}")], width="stretch")

    if algo == "Agglomerative (Hierarchical)" and len(Xs) <= 2000:
        try:
            figs[L("Dendrogram", "Dendrogram")] = V.fig_dendrogram(Xs, cfg.get("linkage", "ward"))
            st.pyplot(figs[L("Dendrogram", "Dendrogram")], width="stretch")
        except Exception:
            pass

    if n_clusters > 1 and mask.sum() > n_clusters:
        sv = silhouette_samples(Xs[mask], labels[mask])
        figs[L("Plot siluet", "Silhouette plot")] = V.fig_silhouette(Xs[mask], labels[mask], sv, float(sil))
        st.pyplot(figs[L("Plot siluet", "Silhouette plot")], width="stretch")
        sil_per = pd.DataFrame({L("Klaster", "Cluster"): labels[mask], L("Siluet", "Silhouette"): sv}) \
            .groupby(L("Klaster", "Cluster")).agg(["mean", "min", "max", "count"]).round(4)
        sil_per.columns = [f"{a} | {b}" for a, b in sil_per.columns]
        tables[L("Siluet per Klaster", "Silhouette per Cluster")] = sil_per.reset_index()
        U.show_table(sil_per.reset_index())

    # uji beda antar klaster
    if n_clusters > 1:
        lab_df = labeled[labeled[L("Klaster", "Cluster")] != -1].copy()
        lab_df[L("Klaster", "Cluster")] = lab_df[L("Klaster", "Cluster")].astype(str)
        an = S.anova_feature_target(lab_df, L("Klaster", "Cluster"), features)
        if not an.empty:
            tables[L("Uji Beda Antar Klaster", "Between-Cluster Difference Test")] = an
            st.markdown("**" + L("Uji beda rerata fitur antar klaster (ANOVA)",
                                 "Between-cluster mean difference test (ANOVA)") + "**")
            U.show_table(an)

    # data berlabel
    out = df.loc[data.index].copy()
    out[L("Klaster", "Cluster")] = labels
    tables[L("Data Berlabel Klaster", "Cluster-Labeled Data")] = out.reset_index(drop=True)
    with st.expander(L("Lihat data dengan label klaster", "View data with cluster labels")):
        U.show_table(out.reset_index(drop=True), height=380)

    # -------------------------------------------------------- interpretasi
    _algo_visuals(algo, model, Xs, labels, centers_scaled, features, cfg, tables, figs)

    U.section(L("6. Interpretasi Otomatis", "6. Automatic Interpretation"))
    mutu = (L("sangat baik", "very good") if sil >= 0.7 else
            L("baik", "good") if sil >= 0.5 else
            L("cukup", "reasonable") if sil >= 0.25 else
            L("lemah", "weak")) if np.isfinite(sil) else "-"
    interp = [
        L(f"Algoritma **{algo}** membentuk **{n_clusters} klaster** dengan Silhouette Score "
          f"**{sil:.4f}** ({mutu}), Davies-Bouldin {db:.4f}, dan Calinski-Harabasz {ch:.2f}.",
          f"**{algo}** produced **{n_clusters} clusters** with a Silhouette Score of "
          f"**{sil:.4f}** ({mutu}), Davies-Bouldin {db:.4f}, and Calinski-Harabasz {ch:.2f}."),
        L(f"Evaluasi lintas-k menyarankan k = {k_sil} berdasarkan Silhouette, k = {k_db} "
          f"berdasarkan Davies-Bouldin, dan k = {k_ch} berdasarkan Calinski-Harabasz.",
          f"Cross-k evaluation suggests k = {k_sil} by Silhouette, k = {k_db} "
          f"by Davies-Bouldin, and k = {k_ch} by Calinski-Harabasz."),
        L(f"Dua komponen utama PCA menjelaskan {ev[:2].sum() * 100:.1f}% keragaman data, "
          "sehingga sebaran 2D di atas cukup representatif untuk melihat pemisahan klaster."
          if len(ev) > 1 else "",
          f"The first two PCA components explain {ev[:2].sum() * 100:.1f}% of the variance, "
          "so the 2D scatter above represents the separation reasonably well." if len(ev) > 1 else ""),
    ]
    for c in sorted(zprof.index):
        row = zprof.loc[c]
        hi = row.sort_values(ascending=False).head(2)
        lo = row.sort_values().head(2)
        n_mem = int(sizes.get(c, 0))
        interp.append(L(
            f"**Klaster {c}** ({n_mem} anggota, {n_mem / len(labeled) * 100:.1f}%): menonjol tinggi pada "
            f"{', '.join(f'{k} ({v:+.2f} SD)' for k, v in hi.items())}; rendah pada "
            f"{', '.join(f'{k} ({v:+.2f} SD)' for k, v in lo.items())}.",
            f"**Cluster {c}** ({n_mem} members, {n_mem / len(labeled) * 100:.1f}%): high on "
            f"{', '.join(f'{k} ({v:+.2f} SD)' for k, v in hi.items())}; low on "
            f"{', '.join(f'{k} ({v:+.2f} SD)' for k, v in lo.items())}."))
    if n_noise:
        interp.append(L(f"Terdapat {n_noise} titik yang ditandai noise oleh DBSCAN "
                        f"({n_noise / len(labels) * 100:.1f}% dari data).",
                        f"DBSCAN flagged {n_noise} points as noise "
                        f"({n_noise / len(labels) * 100:.1f}% of the data)."))
    interp = [i for i in interp if i]
    for line in interp:
        st.markdown("- " + line)
    tables[L("Interpretasi", "Interpretation")] = pd.DataFrame(
        {L("No", "No"): range(1, len(interp) + 1),
         L("Interpretasi", "Interpretation"): [t.replace("**", "") for t in interp]})

    meta = {
        L("Proses", "Process"): L("Klastering", "Clustering"),
        L("Algoritma", "Algorithm"): algo,
        L("Fitur", "Features"): ", ".join(features),
        L("Jumlah data", "Row count"): len(data),
        L("Penskalaan", "Scaling"): cfg["scaling"],
        L("Jumlah klaster", "Clusters"): n_clusters,
        L("Silhouette", "Silhouette"): f"{sil:.4f}" if np.isfinite(sil) else "-",
    }
    U.export_bar(tables, figs, "klastering", meta)
