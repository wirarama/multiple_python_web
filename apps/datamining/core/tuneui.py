"""Panel Streamlit untuk fine tuning hiperparameter klasifikasi & regresi."""

from __future__ import annotations

import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from . import tuning as T  # noqa: E402
from . import ui as U  # noqa: E402
from . import viz as V  # noqa: E402
from .i18n import L  # noqa: E402

MAX_SAFE_FITS = 4000


# ------------------------------------------------------------ konfigurasi
def config_panel(task: str, models_available: list[str], prefix: str,
                 cv_default: int = 5, seed_default: int = 42) -> dict:
    sp = T.space(task)
    tunable = [m for m in models_available if m in sp] or list(sp)
    with st.expander("🎛 " + L("Fine tuning hiperparameter", "Hyperparameter fine tuning"),
                     expanded=False):
        U.note(L(
            "Cari kombinasi parameter terbaik untuk satu algoritma, lengkap dengan bukti "
            "<b>mengapa</b> kombinasi itu menang: kepentingan tiap parameter, kurva validasi, "
            "uji signifikansi antar konfigurasi, dan perbandingan terhadap parameter bawaan.",
            "Search for the best parameter combination of one algorithm, with evidence for "
            "<b>why</b> it wins: per-parameter importance, validation curves, significance tests "
            "between configurations, and a comparison against the default parameters."))
        enable = st.checkbox(L("Aktifkan fine tuning", "Enable fine tuning"), False,
                             key=f"{prefix}_tune_on")
        if not enable:
            return {"enable": False}

        c1, c2, c3 = st.columns([1.4, 1, 1])
        algo = c1.selectbox(L("Algoritma yang di-tuning", "Algorithm to tune"), tunable,
                            key=f"{prefix}_tune_algo")
        sc = T.scorings(task)
        scoring = c2.selectbox(L("Metrik yang dioptimalkan", "Metric to optimize"), list(sc),
                               format_func=lambda k: L(*sc[k]), key=f"{prefix}_tune_metric")
        folds = c3.slider(L("Lipatan validasi silang", "CV folds"), 3, 10, int(cv_default),
                          key=f"{prefix}_tune_cv")

        c4, c5 = st.columns([1, 1])
        method = c4.radio(L("Metode pencarian", "Search method"), ["grid", "random"],
                          horizontal=True, key=f"{prefix}_tune_method",
                          format_func=lambda k: L("Grid Search (semua kombinasi)",
                                                  "Grid Search (all combinations)") if k == "grid"
                          else L("Random Search (sampel acak)", "Random Search (random sample)"))
        n_iter = c5.number_input(L("Jumlah sampel (Random Search)", "Samples (Random Search)"),
                                 5, 500, 40, key=f"{prefix}_tune_niter",
                                 disabled=(method == "grid"))

        st.markdown("**" + L(f"Nilai parameter yang diuji — {algo}",
                             f"Parameter values to test — {algo}") + "**")
        st.caption(L(
            "Nonaktifkan parameter yang tidak ingin diuji, atau ubah daftar nilainya "
            "(pisahkan dengan koma). Tulis `None`, `True`, `False` apa adanya.",
            "Untick parameters you do not want to search, or edit the value list "
            "(comma-separated). Write `None`, `True`, `False` literally."))
        frame = T.default_grid_frame(task, algo)
        edited = st.data_editor(
            frame, width="stretch", hide_index=True, num_rows="fixed",
            key=f"{prefix}_tune_grid_{algo}",
            column_config={
                "parameter": st.column_config.TextColumn(L("Parameter", "Parameter"),
                                                         disabled=True, width="small"),
                "aktif": st.column_config.CheckboxColumn(L("Diuji", "Search"), width="small"),
                "nilai_diuji": st.column_config.TextColumn(L("Nilai yang diuji", "Values to test"),
                                                           width="medium"),
                "keterangan": st.column_config.TextColumn(L("Arti parameter", "What it does"),
                                                          disabled=True, width="large"),
            })
        grid = T.parse_grid(edited)
        combos = T.count_combos(grid) if grid else 0
        n_fit = (min(combos, int(n_iter)) if method == "random" else combos) * int(folds)
        m1, m2, m3 = st.columns(3)
        m1.metric(L("Parameter diuji", "Parameters searched"), f"{len(grid)}")
        m2.metric(L("Kombinasi", "Combinations"), f"{combos:,}")
        m3.metric(L("Total pelatihan model", "Total model fits"), f"{n_fit:,}")
        if not grid:
            U.warn(L("Tidak ada parameter aktif — centang minimal satu baris.",
                     "No active parameter — tick at least one row."))
            return {"enable": False}
        if n_fit > MAX_SAFE_FITS:
            U.warn(L(
                f"{n_fit:,} pelatihan model akan memakan waktu lama. Kurangi nilai yang diuji "
                "atau gunakan Random Search.",
                f"{n_fit:,} model fits will take a long time. Trim the value lists "
                "or switch to Random Search."))
        elif method == "grid" and combos > 200:
            U.note(L(
                f"{combos:,} kombinasi cukup banyak untuk Grid Search. Random Search biasanya "
                "menemukan konfigurasi setara dengan pencarian jauh lebih sedikit.",
                f"{combos:,} combinations is a lot for Grid Search. Random Search usually finds "
                "an equivalent configuration with far fewer trials."))

        return {"enable": True, "algo": algo, "scoring": scoring, "folds": int(folds),
                "method": method, "n_iter": int(n_iter), "grid": grid,
                "grid_frame": edited.to_dict("records")}


# ------------------------------------------------------------- evaluasi uji
def _eval_test(task, pipe, X_te, y_te) -> dict:
    yp = pipe.predict(X_te)
    if task == "classification":
        return {
            L("Akurasi", "Accuracy"): accuracy_score(y_te, yp),
            L("Akurasi seimbang", "Balanced accuracy"): balanced_accuracy_score(y_te, yp),
            "Precision (macro)": precision_score(y_te, yp, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_te, yp, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_te, yp, average="macro", zero_division=0),
            "Cohen Kappa": cohen_kappa_score(y_te, yp),
            "MCC": matthews_corrcoef(y_te, yp),
        }
    y = np.asarray(y_te, dtype=float)
    p = np.asarray(yp, dtype=float)
    mask = np.abs(y) > 1e-9
    mape = float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100) if mask.any() else np.nan
    return {
        "R²": r2_score(y_te, yp),
        "MAE": mean_absolute_error(y_te, yp),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, yp))),
        "MAPE (%)": mape,
    }


def _signature(task, t, X, y, scaling, seed) -> str:
    try:
        fp = int(pd.util.hash_pandas_object(X, index=False).sum()) ^ int(
            pd.util.hash_pandas_object(pd.Series(np.asarray(y).astype(str)), index=False).sum())
    except Exception:
        fp = len(X) * 131 + X.shape[1]
    payload = {
        "task": task, "algo": t["algo"], "scoring": t["scoring"], "folds": t["folds"],
        "method": t["method"], "n_iter": t["n_iter"],
        "grid": {k: [T.format_scalar(x) for x in v] for k, v in sorted(t["grid"].items())},
        "scaling": scaling, "seed": seed, "shape": list(X.shape), "fp": fp,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ==================================================================== render
def render(task: str, t: dict, model_factory, X_tr, y_tr, X_te, y_te,
           scaling: str, seed: int, tables: dict, figs: dict, prefix: str = "tune"):
    """Jalankan pencarian & tampilkan seluruh analisis perbandingan parameter."""
    grid = t["grid"]
    algo, scoring = t["algo"], t["scoring"]
    mlab = T.metric_label(scoring)
    neg = T.is_negated(scoring)
    disp_metric = T.positive_name(scoring) if neg else mlab

    U.section(L(f"7. Fine Tuning Hiperparameter — {algo}",
                f"7. Hyperparameter Fine Tuning — {algo}"),
              L("Pencarian kombinasi parameter terbaik beserta bukti mengapa kombinasi itu menang.",
                "Search for the best parameter combination, with evidence for why it wins."))

    combos = T.count_combos(grid)
    n_fit = (min(combos, t["n_iter"]) if t["method"] == "random" else combos) * t["folds"]
    run_key = f"do_{prefix}_{task}"
    st.session_state.setdefault(run_key, False)
    b1, b2 = st.columns([1, 3])
    if b1.button("▶ " + L("Jalankan Fine Tuning", "Run Fine Tuning"), type="primary",
                 width="stretch", key=f"btn_{prefix}_{task}"):
        st.session_state[run_key] = True
    b2.caption(L(f"Akan melatih {n_fit:,} model ({combos:,} kombinasi × {t['folds']} lipatan). "
                 "Hasil disimpan sementara sehingga tidak dihitung ulang setiap interaksi.",
                 f"Will fit {n_fit:,} models ({combos:,} combinations × {t['folds']} folds). "
                 "Results are cached so they are not recomputed on every interaction."))
    if not st.session_state[run_key]:
        st.info(L("Tekan tombol di atas untuk memulai pencarian parameter.",
                  "Press the button above to start the parameter search."))
        return

    sig = _signature(task, t, X_tr, y_tr, scaling, seed)
    cache_key = f"{prefix}_cache_{task}_{sig}"
    if cache_key not in st.session_state:
        for k in [k for k in list(st.session_state) if k.startswith(f"{prefix}_cache_{task}_")]:
            del st.session_state[k]
        try:
            with st.spinner(L(f"Menguji {combos:,} kombinasi parameter...",
                              f"Evaluating {combos:,} parameter combinations...")):
                t0 = time.time()
                search, cv, _ = T.run_search(
                    model_factory, X_tr, y_tr, grid, task, method=t["method"],
                    n_iter=t["n_iter"], folds=t["folds"], scoring=scoring,
                    scaling=scaling, seed=seed)
                elapsed = time.time() - t0
                cmp = T.baseline_comparison(model_factory, X_tr, y_tr, cv, scoring,
                                            scaling, seed, search)
            st.session_state[cache_key] = (search, cv, cmp, elapsed)
        except Exception as e:
            st.error(L(f"Pencarian gagal: {e}", f"Search failed: {e}"))
            return
    st.session_state[f"{prefix}_last_{task}"] = cache_key
    search, cv, cmp, elapsed = st.session_state[cache_key]

    r = search.cv_results_
    sign = -1.0 if neg else 1.0
    n_cand = len(r["mean_test_score"])
    best_i = int(np.nanargmax(np.asarray(r["mean_test_score"], dtype=float)))

    res = T.results_table(search, grid, scoring)
    imp = T.param_importance(search, grid, scoring)
    effects = T.param_effects(search, grid, scoring)
    sig_tab = T.significance_vs_best(search, grid, scoring)
    ose = T.one_se_rule(search, grid, scoring)

    tables[L("Hasil Semua Kombinasi", "All Combinations Result")] = res
    tables[L("Kepentingan Parameter", "Parameter Importance")] = imp
    tables[L("Signifikansi Antar Konfigurasi", "Between-Config Significance")] = sig_tab
    tables[L("Perbandingan Bawaan vs Tuning", "Default vs Tuned")] = T.baseline_table(cmp, scoring)
    for p, e in effects.items():
        tables[f"Efek_{p}"[:31]] = e

    U.metric_row([
        (L("Kombinasi diuji", "Combinations tested"), f"{n_cand:,}"),
        (L("Model dilatih", "Models fitted"), f"{n_cand * t['folds']:,}"),
        (L(f"Skor CV terbaik ({disp_metric})", f"Best CV score ({disp_metric})"),
         f"{ose['best_score']:.4f}"),
        (L("Waktu pencarian", "Search time"), f"{elapsed:.1f} s"),
    ])

    best_params_tab = pd.DataFrame(
        [(p, T.format_scalar(v)) for p, v in ose["best_params"].items()],
        columns=[L("Parameter", "Parameter"), L("Nilai terbaik", "Best value")])
    tables[L("Parameter Terbaik", "Best Parameters")] = best_params_tab
    st.success(L(
        "Konfigurasi terbaik: " + " · ".join(f"**{p} = {T.format_scalar(v)}**"
                                             for p, v in ose["best_params"].items()),
        "Best configuration: " + " · ".join(f"**{p} = {T.format_scalar(v)}**"
                                            for p, v in ose["best_params"].items())))

    tabs = st.tabs([
        L("Peringkat konfigurasi", "Configuration ranking"),
        L("Kepentingan parameter", "Parameter importance"),
        L("Kurva validasi", "Validation curves"),
        L("Interaksi parameter", "Parameter interaction"),
        L("Uji signifikansi", "Significance test"),
        L("Sebelum vs sesudah", "Before vs after"),
    ])

    # ---------------------------------------------------- 1. peringkat
    with tabs[0]:
        U.note(L(
            "Kolom <b>Selisih latih-CV</b> adalah indikator overfitting: makin besar, makin "
            "model hanya hafal data latih. Konfigurasi ideal punya skor CV tinggi "
            "<i>dan</i> selisih kecil.",
            "The <b>Train-CV gap</b> column indicates overfitting: the larger it is, the more the "
            "model merely memorizes the training data. An ideal configuration has a high CV score "
            "<i>and</i> a small gap."))
        U.show_table(res.head(60), height=380)
        labels = [", ".join(f"{p}={res.iloc[i][p]}" for p in grid) for i in range(len(res))]
        figs[L("Konfigurasi teratas", "Top configurations")] = V.fig_tune_top_configs(
            labels, res[L("Skor CV", "CV score")].to_numpy(dtype=float),
            res[L("Std CV", "CV std")].to_numpy(dtype=float), disp_metric, best_pos=0)
        st.pyplot(figs[L("Konfigurasi teratas", "Top configurations")], width="stretch")

        order = np.argsort(np.asarray(r["rank_test_score"]))
        prog = sign * np.asarray(r["mean_test_score"], dtype=float)
        figs[L("Perkembangan pencarian", "Search progress")] = V.fig_search_progress(
            prog, disp_metric, negated=neg)
        st.pyplot(figs[L("Perkembangan pencarian", "Search progress")], width="stretch")

    # ------------------------------------------------- 2. kepentingan
    with tabs[1]:
        if imp.empty:
            st.caption(L("Butuh minimal 2 nilai per parameter.", "Needs at least 2 values per parameter."))
        else:
            U.note(L(
                "<b>Eta²</b> menyatakan berapa persen variasi skor antar kombinasi yang dijelaskan "
                "oleh satu parameter: &lt;0,06 kecil, 0,06–0,14 sedang, &gt;0,14 besar. "
                "Parameter ber-Eta² besar itulah yang benar-benar menentukan hasil; "
                "parameter ber-Eta² kecil boleh dikunci pada nilai bawaan.",
                "<b>Eta²</b> is the share of score variance across combinations explained by one "
                "parameter: &lt;0.06 small, 0.06–0.14 medium, &gt;0.14 large. Parameters with a "
                "large Eta² are what really drive the result; those with a small Eta² can be "
                "fixed at their defaults."))
            U.show_table(imp)
            ranges = dict(zip(imp[L("Parameter", "Parameter")],
                              imp[L("Rentang pengaruh", "Effect range")]))
            figs[L("Kepentingan parameter", "Parameter importance")] = V.fig_param_importance(
                imp[L("Parameter", "Parameter")].tolist(), imp["Eta²"].tolist(), ranges)
            st.pyplot(figs[L("Kepentingan parameter", "Parameter importance")], width="stretch")

    # -------------------------------------------------- 3. kurva validasi
    with tabs[2]:
        U.note(L(
            "Tiap panel menunjukkan rata-rata skor untuk setiap nilai parameter. Jarak antara "
            "garis <b>skor latih</b> dan <b>skor validasi silang</b> adalah besarnya overfitting. "
            "Nilai terbaik adalah titik saat skor validasi tertinggi — di luar itu model menjadi "
            "terlalu sederhana (underfitting) atau terlalu kompleks (overfitting).",
            "Each panel shows the mean score for every parameter value. The distance between the "
            "<b>train score</b> and the <b>cross-validation score</b> lines is the amount of "
            "overfitting. The best value is where the validation score peaks — beyond it the model "
            "becomes too simple (underfitting) or too complex (overfitting)."))
        figs[L("Kurva validasi parameter", "Parameter validation curves")] = V.fig_param_effect_grid(
            effects, disp_metric,
            L("Nilai", "Value"), L("Rata-rata skor CV", "Mean CV score"),
            L("Rata-rata skor latih", "Mean train score"),
            L("Std antar kombinasi", "Std across combos"), negated=neg)
        if figs[L("Kurva validasi parameter", "Parameter validation curves")] is not None:
            st.pyplot(figs[L("Kurva validasi parameter", "Parameter validation curves")],
                      width="stretch")
        pick = st.selectbox(L("Tabel efek untuk parameter", "Effect table for parameter"),
                            list(effects), key=f"{prefix}_{task}_eff")
        U.show_table(effects[pick])

    # ------------------------------------------------- 4. interaksi
    with tabs[3]:
        if len(grid) < 2:
            st.caption(L("Butuh minimal 2 parameter aktif.", "Needs at least 2 active parameters."))
        else:
            top2 = (imp[L("Parameter", "Parameter")].head(2).tolist()
                    if len(imp) >= 2 else list(grid)[:2])
            i1, i2 = st.columns(2)
            p1 = i1.selectbox(L("Parameter baris", "Row parameter"), list(grid),
                              index=list(grid).index(top2[0]), key=f"{prefix}_{task}_p1")
            rest = [p for p in grid if p != p1]
            p2 = i2.selectbox(L("Parameter kolom", "Column parameter"), rest,
                              index=rest.index(top2[1]) if top2[1] in rest else 0,
                              key=f"{prefix}_{task}_p2")
            piv = T.interaction_pivot(search, grid, p1, p2, scoring)
            tables[f"Interaksi_{p1}_{p2}"[:31]] = piv.reset_index()
            figs[L(f"Interaksi {p1} x {p2}", f"Interaction {p1} x {p2}")] = V.fig_param_heatmap(
                piv, p1, p2, disp_metric, negated=neg)
            fg = figs[L(f"Interaksi {p1} x {p2}", f"Interaction {p1} x {p2}")]
            if fg is not None:
                st.pyplot(fg, width="stretch")
            U.show_table(piv.reset_index())
            U.note(L(
                "Baca baris/kolom terbaik untuk melihat apakah nilai optimal satu parameter "
                "bergantung pada parameter lain. Bila pola tiap baris mirip, kedua parameter "
                "bekerja mandiri; bila berbeda, ada interaksi dan keduanya harus disetel bersama.",
                "Read the best row/column to see whether one parameter's optimum depends on the "
                "other. If every row shows a similar pattern the two parameters act independently; "
                "if they differ there is an interaction and both must be tuned together."))

    # ------------------------------------------------- 5. signifikansi
    with tabs[4]:
        U.note(L(
            "Uji-t berpasangan membandingkan skor <i>lipatan demi lipatan</i> antara konfigurasi "
            "terbaik dan pesaingnya. p ≥ 0,05 berarti keduanya <b>setara secara statistik</b> — "
            "selisih skornya hanya efek keacakan pembagian data, sehingga memilih yang lebih "
            "sederhana justru lebih aman.",
            "The paired t-test compares scores <i>fold by fold</i> between the best configuration "
            "and its rivals. p ≥ 0.05 means they are <b>statistically tied</b> — the score "
            "difference is only an artefact of the data split, so picking the simpler one is "
            "safer."))
        U.show_table(sig_tab)

        folds_mat = T.fold_scores(search)
        top_idx = np.argsort(np.asarray(r["rank_test_score"]))[:8]
        labs = [", ".join(f"{p}={T.format_scalar(r[f'param_model__{p}'][i])}" for p in grid)
                for i in top_idx]
        figs[L("Sebaran skor antar lipatan", "Score spread across folds")] = V.fig_fold_box(
            [sign * folds_mat[i] for i in top_idx], labs, disp_metric)
        fg = figs[L("Sebaran skor antar lipatan", "Score spread across folds")]
        if fg is not None:
            st.pyplot(fg, width="stretch")

        rank_order = np.argsort(np.asarray(r["rank_test_score"]))
        figs[L("Latih vs validasi", "Train vs validation")] = V.fig_train_cv_scatter(
            sign * np.asarray(r["mean_train_score"], dtype=float)[rank_order],
            sign * np.asarray(r["mean_test_score"], dtype=float)[rank_order],
            0, disp_metric)
        fg = figs[L("Latih vs validasi", "Train vs validation")]
        if fg is not None:
            st.pyplot(fg, width="stretch")

        ose_tab = pd.DataFrame([
            (L("Skor terbaik", "Best score"), round(ose["best_score"], 6)),
            (L("Galat baku skor terbaik", "Standard error of best"), round(ose["se_best"], 6)),
            (L("Ambang 1 galat baku", "One-standard-error threshold"), round(ose["threshold"], 6)),
            (L("Konfigurasi dalam ambang", "Configurations within threshold"), ose["n_within"]),
            (L("Konfigurasi paling sederhana dalam ambang", "Simplest within threshold"),
             ", ".join(f"{p}={T.format_scalar(v)}" for p, v in ose["pick_params"].items())),
            (L("Skornya", "Its score"), round(ose["pick_score"], 6)),
            (L("Waktu latih terbaik (detik)", "Best fit time (s)"), round(ose["best_time"], 4)),
            (L("Waktu latih pilihan sederhana (detik)", "Simple pick fit time (s)"),
             round(ose["pick_time"], 4)),
        ], columns=[L("Aspek", "Aspect"), L("Nilai", "Value")])
        tables[L("Kaidah One-Standard-Error", "One-Standard-Error Rule")] = ose_tab
        st.markdown("**" + L("Kaidah one-standard-error", "One-standard-error rule") + "**")
        U.show_table(ose_tab)

    # ---------------------------------------------- 6. sebelum vs sesudah
    with tabs[5]:
        U.show_table(T.baseline_table(cmp, scoring))
        figs[L("Bawaan vs tuning per lipatan", "Default vs tuned per fold")] = V.fig_paired_folds(
            sign * cmp["base_folds"], sign * cmp["tuned_folds"], disp_metric)
        fg = figs[L("Bawaan vs tuning per lipatan", "Default vs tuned per fold")]
        if fg is not None:
            st.pyplot(fg, width="stretch")

        try:
            base_pipe, _, _ = T.build_pipeline(model_factory, X_tr, scaling, seed)
            base_pipe.fit(X_tr, y_tr)
            m_base = _eval_test(task, base_pipe, X_te, y_te)
            m_tuned = _eval_test(task, search.best_estimator_, X_te, y_te)
            rows = []
            for k in m_base:
                b, t2 = float(m_base[k]), float(m_tuned[k])
                rows.append({
                    L("Metrik", "Metric"): k,
                    L("Parameter bawaan", "Default parameters"): round(b, 6),
                    L("Setelah tuning", "After tuning"): round(t2, 6),
                    L("Selisih", "Difference"): round(t2 - b, 6),
                    L("Perubahan (%)", "Change (%)"): round((t2 - b) / abs(b) * 100, 3)
                    if abs(b) > 1e-12 else "-",
                })
            test_tab = pd.DataFrame(rows)
            tables[L("Evaluasi Data Uji: Bawaan vs Tuning",
                     "Test Set: Default vs Tuned")] = test_tab
            st.markdown("**" + L("Evaluasi pada data uji (data yang belum pernah dilihat model)",
                                 "Evaluation on the test set (data the model has never seen)") + "**")
            U.show_table(test_tab)
            U.note(L(
                "Perbaikan pada validasi silang belum tentu terbawa ke data uji. Bila kolom "
                "<b>Selisih</b> di sini kecil atau negatif sementara skor CV naik, itu tanda "
                "tuning mulai menyesuaikan diri pada lipatan validasi (overfitting terhadap "
                "prosedur pencarian).",
                "A cross-validation gain does not always carry over to the test set. If the "
                "<b>Difference</b> column here is small or negative while the CV score rose, "
                "the tuning is starting to fit the validation folds themselves (overfitting the "
                "search procedure)."))
        except Exception as e:
            st.info(L(f"Evaluasi data uji dilewati: {e}", f"Test set evaluation skipped: {e}"))

    # ------------------------------------------------------- interpretasi
    st.markdown("#### " + L("Mengapa konfigurasi ini yang terbaik",
                            "Why this configuration is the best"))
    lines = T.narrative(search, grid, scoring, imp, sig_tab, ose, cmp, effects, task, algo)
    for line in lines:
        st.markdown("- " + line)
    tables[L("Interpretasi Tuning", "Tuning Interpretation")] = pd.DataFrame({
        L("No", "No"): range(1, len(lines) + 1),
        L("Interpretasi", "Interpretation"): [x.replace("**", "").replace("*", "")
                                              .replace("<b>", "").replace("</b>", "")
                                              .replace("&lt;", "<").replace("&gt;", ">")
                                              for x in lines]})
    return {"search": search, "best_params": ose["best_params"], "ose": ose, "cmp": cmp}
