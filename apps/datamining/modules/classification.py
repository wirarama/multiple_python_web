"""Modul Klasifikasi."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline

from core import algviz as AV
from core import params as PR
from core import paramui as PUI
from core import settings as SET
from core import stats as S
from core import tuneui as TUI
from core import ui as U
from core import viz as V
from core.i18n import L
from core.mlutils import build_preprocessor, feature_names_from, importance_from

ALGOS = PR.algos("classification")


def make_model(name: str, params: dict | None, rs: int):
    """Bangun estimator sesuai parameter yang diatur pengguna."""
    return PR.build("classification", name, params, rs)


# Dipertahankan agar kode/uji lama tetap berjalan (parameter bawaan).
MODELS = {name: (lambda rs, _n=name: make_model(_n, None, rs)) for name in ALGOS}


def config(df: pd.DataFrame) -> dict | None:
    SET.consume_pending("classification")
    st.markdown("### " + L("Konfigurasi Klasifikasi", "Classification Setup"))
    cand_target = [c for c in df.columns if df[c].nunique(dropna=True) <= 20 and df[c].nunique(dropna=True) >= 2]
    if not cand_target:
        U.warn(L("Tidak ada kolom yang cocok sebagai target (butuh 2-20 kategori unik).",
                 "No suitable target column (needs 2-20 unique categories)."))
        return None

    c1, c2 = st.columns([1, 1])
    default_t = cand_target[-1]
    target = SET.selectbox(L("Kolom target (kelas)", "Target column (class)"), cand_target,
                           key="cls_target", index=cand_target.index(default_t), container=c1)
    feats_all = [c for c in df.columns if c != target]
    auto_drop = [c for c in feats_all if df[c].nunique(dropna=True) >= max(len(df) * 0.9, 20)
                 and not pd.api.types.is_numeric_dtype(df[c])]
    features = SET.multiselect(L("Kolom fitur", "Feature columns"), feats_all, key="cls_feats",
                               default=[c for c in feats_all if c not in auto_drop], container=c2)
    if auto_drop:
        st.caption(L(f"Kolom mirip-ID dikecualikan otomatis: {', '.join(auto_drop)}",
                     f"ID-like columns auto-excluded: {', '.join(auto_drop)}"))

    c3, c4, c5, c6 = st.columns(4)
    test_size = SET.slider(L("Proporsi data uji", "Test size"), 0.1, 0.5, 0.25, 0.05,
                           key="cls_ts", container=c3)
    scaling = SET.selectbox(L("Penskalaan fitur", "Feature scaling"),
                            ["standard", "minmax", "none"], key="cls_scale", container=c4)
    cv = SET.slider(L("Lipatan validasi silang", "CV folds"), 2, 10, 5, key="cls_cv", container=c5)
    rs = SET.number_input(L("Random state", "Random state"), 0, 9999, 42, key="cls_rs", container=c6)

    chosen = SET.multiselect(L("Algoritma yang diuji", "Algorithms to evaluate"), ALGOS,
                             key="cls_models",
                             default=["Logistic Regression", "Decision Tree", "Random Forest",
                                      "Gradient Boosting", "K-Nearest Neighbors"])
    stratify = SET.checkbox(L("Pembagian data terstratifikasi", "Stratified split"), True,
                            key="cls_strat")

    if not features or not chosen:
        U.warn(L("Pilih minimal satu fitur dan satu algoritma.", "Select at least one feature and one algorithm."))
        return None

    params = PUI.editor("classification", chosen, "cls")
    tune = TUI.config_panel("classification", chosen, "cls", cv_default=int(cv), seed_default=int(rs))
    cfg = dict(target=target, features=features, test_size=test_size, scaling=scaling,
               cv=cv, rs=int(rs), models=chosen, stratify=stratify, params=params, tune=tune)
    SET.panel("classification", {k: v for k, v in cfg.items() if k != "tune"})
    return cfg


def _algo_visuals(name, pipe, X_tr, y_tr, X_te, y_te, classes, fnames, cfg, tables, figs):
    """Bagian 5: visualisasi yang menjelaskan cara kerja algoritma terpilih."""
    U.section(L(f"5. Visualisasi Khusus Algoritma — {name}",
                f"5. Algorithm-Specific Visualization — {name}"),
              L("Grafik pada bagian ini menggambarkan mekanisme internal algoritma: "
                "struktur pohon, hyperplane pemisah, batas keputusan, koefisien, dan sejenisnya.",
                "The charts here depict the algorithm's internal mechanism: tree structure, "
                "separating hyperplane, decision boundary, coefficients, and the like."))
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    params = cfg.get("params", {}).get(name, {})
    try:
        Ztr = np.asarray(pre.transform(X_tr), dtype=float)
        Zte = np.asarray(pre.transform(X_te), dtype=float)
    except Exception as e:
        U.warn(L(f"Data tidak dapat ditransformasi untuk visualisasi: {e}",
                 f"The data could not be transformed for visualization: {e}"))
        return

    def add(title, fig, wide=True):
        if fig is None:
            return
        figs[title] = fig
        st.markdown(f"**{title}**")
        st.pyplot(fig, width="stretch" if wide else "content")

    # ---------------------------------------------------------------- pohon
    if name == "Decision Tree":
        d = SET.slider(L("Kedalaman pohon yang ditampilkan", "Tree depth to display"),
                       1, 6, 3, key="cls_tree_depth")
        U.note(L(
            "Setiap kotak adalah simpul: baris pertama adalah aturan pemecahan, "
            "<b>samples</b> adalah jumlah data yang sampai di simpul itu, <b>value</b> adalah "
            "sebarannya per kelas, dan warna menunjukkan kelas yang dominan.",
            "Each box is a node: the first line is the split rule, <b>samples</b> is how many rows "
            "reach that node, <b>value</b> is their per-class distribution, and the colour shows "
            "the dominant class."))
        add(L("Struktur pohon keputusan", "Decision tree structure"),
            AV.fig_tree(model, fnames, classes, max_depth=int(d)))
        rules = AV.tree_rules_table(model, fnames, classes)
        if not rules.empty:
            tables[L("Aturan Pohon Keputusan", "Decision Tree Rules")] = rules
            st.markdown("**" + L("Aturan jika-maka yang dipelajari pohon",
                                 "If-then rules learned by the tree") + "**")
            U.show_table(rules.head(30))
        add(L("Penurunan ketidakmurnian per fitur", "Impurity decrease per feature"),
            AV.fig_tree_importance_path(model, fnames))
        if len(Ztr) <= 3000:
            add(L("Jalur pemangkasan cost-complexity", "Cost-complexity pruning path"),
                AV.fig_ccp_path(lambda: make_model(name, params, cfg["rs"]), Ztr, y_tr))

    # ------------------------------------------------------- hutan/ansambel
    elif name in ("Random Forest", "Extra Trees"):
        n_show = SET.slider(L("Jumlah pohon contoh yang ditampilkan", "Sample trees to display"),
                            1, 4, 3, key="cls_forest_n")
        U.note(L(
            "Hutan acak menggabungkan banyak pohon yang masing-masing dilatih pada sampel dan "
            "subset fitur berbeda. Perhatikan bahwa pohon-pohon contoh di bawah ini berbeda satu "
            "sama lain — keragaman itulah yang membuat gabungannya lebih stabil daripada satu pohon.",
            "A random forest combines many trees, each trained on a different sample and feature "
            "subset. Notice how the sample trees below differ from one another — that diversity is "
            "what makes their combination more stable than any single tree."))
        add(L("Contoh pohon dalam ansambel", "Sample trees in the ensemble"),
            AV.fig_forest_grid(model, fnames, classes, n_trees=int(n_show), max_depth=2))
        add(L("Struktur pohon dalam hutan", "Tree structure across the forest"),
            AV.fig_forest_structure(model))
        add(L("Konvergensi ansambel", "Ensemble convergence"),
            AV.fig_forest_convergence(model, Zte, y_te, "classification"))
        add(L("Kestabilan kepentingan fitur", "Feature importance stability"),
            AV.fig_importance_spread(model, fnames))

    # -------------------------------------------------------------- boosting
    elif name == "Gradient Boosting":
        U.note(L(
            "Boosting membangun pohon secara berurutan; tiap pohon memperbaiki sisa kesalahan "
            "pohon sebelumnya. Titik puncak kurva data uji menandai jumlah tahap yang optimal — "
            "setelah itu model mulai menghafal data latih.",
            "Boosting builds trees sequentially; each tree corrects the previous ones' residual "
            "error. The peak of the test curve marks the optimal number of stages — beyond it the "
            "model starts memorizing the training data."))
        add(L("Skor per tahap boosting", "Score per boosting stage"),
            AV.fig_boosting_stages(model, Ztr, y_tr, Zte, y_te, "classification"))
        add(L("Penurunan kerugian per tahap", "Loss reduction per stage"),
            AV.fig_boosting_loss(model))
        ests = getattr(model, "estimators_", None)
        if ests is not None and len(ests):
            add(L("Pohon lemah pertama", "First weak learner"),
                AV.fig_tree(ests[0][0], fnames, None, max_depth=2))

    # ------------------------------------------------------------------ SVM
    elif name == "SVM (RBF)":
        mode, cols = PUI.projection_controls("cls", fnames)
        X2, lab, _ = AV.project_2d(Ztr, mode, cols, fnames)
        U.note(L(
            "SVM mencari bidang pemisah (hyperplane) dengan margin selebar mungkin. "
            "Titik berlingkar adalah <b>support vector</b> — hanya titik-titik inilah yang "
            "menentukan letak batas; sisanya boleh digeser tanpa mengubah model.",
            "An SVM looks for a separating hyperplane with the widest possible margin. Circled "
            "points are <b>support vectors</b> — only they determine where the boundary sits; the "
            "rest could be moved without changing the model."))
        with st.spinner(L("Melatih ulang SVM pada bidang 2D...", "Refitting the SVM on the 2-D plane...")):
            svc2d = AV.fit_2d(model, X2, y_tr)
        add(L("Hyperplane & margin SVM", "SVM hyperplane & margin"),
            AV.fig_svm_hyperplane(svc2d, X2, y_tr, *lab))
        add(L("Sebaran jarak ke hyperplane", "Distance-to-hyperplane distribution"),
            AV.fig_svm_margin_dist(model, Zte, y_te))
        nsv = getattr(model, "n_support_", None)
        if nsv is not None:
            tables[L("Support Vector per Kelas", "Support Vectors per Class")] = pd.DataFrame({
                L("Kelas", "Class"): [str(c) for c in model.classes_],
                L("Jumlah support vector", "Support vectors"): list(map(int, nsv)),
                L("Persentase data latih (%)", "Share of training data (%)"):
                    [round(float(v) / max(len(Ztr), 1) * 100, 2) for v in nsv]})
            U.show_table(tables[L("Support Vector per Kelas", "Support Vectors per Class")])

    # ------------------------------------------------------------------ kNN
    elif name == "K-Nearest Neighbors":
        mode, cols = PUI.projection_controls("cls", fnames)
        X2, lab, _ = AV.project_2d(Ztr, mode, cols, fnames)
        k = int(PR.resolve("classification", name, params).get("n_neighbors", 5))
        U.note(L(
            "k-NN tidak membangun model; ia menyimpan seluruh data latih lalu memungut suara "
            "dari k tetangga terdekat. Lingkaran putus-putus di bawah memuat tepat k tetangga "
            "dari titik bertanda bintang.",
            "k-NN builds no model; it stores the training data and takes a vote among the k nearest "
            "neighbours. Each dashed circle below contains exactly k neighbours of the starred point."))
        add(L("Cara k-NN memutuskan", "How k-NN decides"),
            AV.fig_knn_neighbors(X2, y_tr, k, *lab))
        add(L("Batas keputusan k-NN", "k-NN decision boundary"),
            AV.fig_decision_regions(AV.fit_2d(model, X2, y_tr), X2, y_tr, *lab))
        add(L("Sensitivitas terhadap k", "Sensitivity to k"),
            AV.fig_k_sensitivity(lambda: make_model(name, params, cfg["rs"]),
                                 Ztr, y_tr, Zte, y_te, metric_name=L("Akurasi", "Accuracy")))

    # ------------------------------------------------------ regresi logistik
    elif name == "Logistic Regression":
        U.note(L(
            "Regresi logistik menjumlahkan fitur berbobot menjadi satu skor linear, lalu "
            "mengubahnya menjadi probabilitas lewat fungsi logistik. Rasio odds &gt; 1 berarti "
            "fitur menaikkan peluang kelas positif.",
            "Logistic regression sums weighted features into one linear score, then maps it to a "
            "probability through the logistic function. An odds ratio &gt; 1 means the feature "
            "raises the odds of the positive class."))
        coef = np.asarray(model.coef_, dtype=float)
        add(L("Koefisien model", "Model coefficients"),
            AV.fig_coefficients(fnames, coef[0] if coef.ndim > 1 else coef,
                                intercept=float(np.ravel(model.intercept_)[0])))
        if len(classes) == 2:
            add(L("Rasio odds per fitur", "Odds ratio per feature"),
                AV.fig_odds_ratio(fnames, coef[0] if coef.ndim > 1 else coef))
            add(L("Kurva logistik & ambang", "Logistic curve & threshold"),
                AV.fig_logistic_curve(model, Zte, y_te))
        mode, cols = PUI.projection_controls("cls", fnames)
        X2, lab, _ = AV.project_2d(Ztr, mode, cols, fnames)
        add(L("Wilayah keputusan", "Decision regions"),
            AV.fig_decision_regions(AV.fit_2d(model, X2, y_tr), X2, y_tr, *lab,
                                    proba=(len(classes) == 2)))

    # ---------------------------------------------------------- naive bayes
    elif name == "Naive Bayes":
        U.note(L(
            "Naive Bayes mengasumsikan tiap fitur berdistribusi normal dalam setiap kelas dan "
            "saling bebas. Semakin jauh puncak kurva antar kelas, semakin kuat fitur itu "
            "memisahkan kelas.",
            "Naive Bayes assumes each feature is normally distributed within every class and "
            "independent of the others. The further apart the class peaks, the stronger that "
            "feature separates the classes."))
        add(L("Distribusi Gauss per kelas", "Per-class Gaussian distributions"),
            AV.fig_nb_densities(model, Ztr, y_tr, fnames))
        add(L("Keyakinan Naive Bayes", "Naive Bayes confidence"), AV.fig_nb_loglik(model, Zte, y_te))
        mode, cols = PUI.projection_controls("cls", fnames)
        X2, lab, _ = AV.project_2d(Ztr, mode, cols, fnames)
        add(L("Wilayah keputusan", "Decision regions"),
            AV.fig_decision_regions(AV.fit_2d(model, X2, y_tr), X2, y_tr, *lab))

    else:
        mode, cols = PUI.projection_controls("cls", fnames)
        X2, lab, _ = AV.project_2d(Ztr, mode, cols, fnames)
        add(L("Wilayah keputusan", "Decision regions"),
            AV.fig_decision_regions(AV.fit_2d(model, X2, y_tr), X2, y_tr, *lab))


def run(df: pd.DataFrame, cfg: dict):
    target, features = cfg["target"], cfg["features"]
    data = df[features + [target]].copy()
    data = data[data[target].notna()]
    y = data[target].astype(str)
    X = data[features]

    vc = y.value_counts()
    if (vc < 2).any():
        drop = vc[vc < 2].index.tolist()
        data = data[~y.isin(drop)]
        y = data[target].astype(str)
        X = data[features]
        st.warning(L(f"Kelas dengan <2 sampel dihapus: {', '.join(drop)}",
                     f"Classes with <2 samples removed: {', '.join(drop)}"))

    classes = sorted(y.unique())
    strat = y if cfg["stratify"] and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["rs"], stratify=strat)

    tables, figs = {}, {}

    # ---------------------------------------------------- statistik dasar
    U.section(L("1. Analisis Statistik Data", "1. Data Statistical Analysis"),
              L("Profil data sebelum pemodelan: sebaran, korelasi, normalitas, outlier.",
                "Data profile before modeling: distribution, correlation, normality, outliers."))
    t0, f0 = U.descriptive_block(data, "cls")
    tables.update(t0)
    figs.update(f0)

    # ------------------------------------------- statistik khusus klasifikasi
    U.section(L("2. Statistik Target & Hubungan Fitur-Target",
                "2. Target Statistics & Feature-Target Relationship"))
    bal = S.frequency_table(data, target)
    tables[L("Distribusi Kelas", "Class Distribution")] = bal
    imb = float(vc.max() / max(vc.min(), 1))
    U.metric_row([
        (L("Jumlah kelas", "Number of classes"), f"{len(classes)}"),
        (L("Kelas mayoritas", "Majority class"), f"{vc.index[0]} ({vc.iloc[0]})"),
        (L("Rasio ketidakseimbangan", "Imbalance ratio"), f"{imb:.2f} : 1"),
        (L("Data latih / uji", "Train / test"), f"{len(X_tr)} / {len(X_te)}"),
    ])
    if imb > 3:
        U.warn(L(
            f"Data tidak seimbang (rasio {imb:.2f}:1). Gunakan metrik F1-macro / balanced accuracy "
            "dan bukan akurasi mentah sebagai acuan utama.",
            f"Imbalanced data (ratio {imb:.2f}:1). Use F1-macro / balanced accuracy "
            "rather than raw accuracy as the primary metric.",
        ))
    ct1, ct2 = st.columns([1, 1])
    with ct1:
        U.show_table(bal)
    with ct2:
        figs[L("Distribusi kelas", "Class distribution")] = V.fig_class_balance(y)
        st.pyplot(figs[L("Distribusi kelas", "Class distribution")], width="stretch")

    an = S.anova_feature_target(data, target, features)
    if not an.empty:
        tables[L("Uji Beda Fitur Numerik", "Numeric Feature Difference Test")] = an
        st.markdown("**" + L("Uji beda rerata fitur numerik antar kelas (ANOVA & Kruskal-Wallis)",
                             "Mean difference of numeric features across classes (ANOVA & Kruskal-Wallis)") + "**")
        U.note(L("Eta² adalah ukuran efek: &lt;0,06 kecil, 0,06-0,14 sedang, &gt;0,14 besar.",
                 "Eta² is the effect size: &lt;0.06 small, 0.06-0.14 medium, &gt;0.14 large."))
        U.show_table(an)
    ch = S.chi2_feature_target(data, target, features)
    if not ch.empty:
        tables[L("Uji Chi-Square Fitur Kategorik", "Chi-Square Test Categorical Features")] = ch
        st.markdown("**" + L("Uji Chi-Square fitur kategorik vs target",
                             "Chi-Square test of categorical features vs target") + "**")
        U.show_table(ch)
    gs = S.group_summary(data, target, features)
    if not gs.empty:
        tables[L("Ringkasan per Kelas", "Summary per Class")] = gs
        st.markdown("**" + L("Rata-rata fitur per kelas", "Feature means per class") + "**")
        U.show_table(gs)

    # ----------------------------------------------------------- pemodelan
    U.section(L("3. Pelatihan & Evaluasi Model", "3. Model Training & Evaluation"))
    rows, fitted = [], {}
    prog = st.progress(0.0)
    for i, name in enumerate(cfg["models"], start=1):
        pre, num_cols, cat_cols = build_preprocessor(X, cfg["scaling"])
        pipe = Pipeline([("pre", pre),
                         ("model", make_model(name, cfg.get("params", {}).get(name), cfg["rs"]))])
        try:
            pipe.fit(X_tr, y_tr)
        except Exception as e:
            st.error(f"{name}: {e}")
            prog.progress(i / len(cfg["models"]))
            continue
        y_pred = pipe.predict(X_te)
        try:
            proba = pipe.predict_proba(X_te)
        except Exception:
            proba = None
        try:
            n_split = int(min(cfg["cv"], y_tr.value_counts().min()))
            if n_split >= 2:
                cvs = cross_val_score(pipe, X_tr, y_tr, cv=StratifiedKFold(n_split, shuffle=True,
                                      random_state=cfg["rs"]), scoring="accuracy", n_jobs=-1)
            else:
                cvs = np.array([np.nan])
        except Exception:
            cvs = np.array([np.nan])
        auc = np.nan
        try:
            if proba is not None:
                if len(classes) == 2:
                    pos = list(pipe.classes_).index(classes[-1])
                    auc = roc_auc_score((y_te == classes[-1]).astype(int), proba[:, pos])
                else:
                    auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
        except Exception:
            pass
        try:
            ll = log_loss(y_te, proba, labels=list(pipe.classes_)) if proba is not None else np.nan
        except Exception:
            ll = np.nan
        rows.append({
            L("Model", "Model"): name,
            L("Akurasi", "Accuracy"): accuracy_score(y_te, y_pred),
            L("Akurasi Seimbang", "Balanced Accuracy"): balanced_accuracy_score(y_te, y_pred),
            "Precision (macro)": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_te, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "F1 (weighted)": f1_score(y_te, y_pred, average="weighted", zero_division=0),
            "ROC AUC": auc,
            "Cohen Kappa": cohen_kappa_score(y_te, y_pred),
            "MCC": matthews_corrcoef(y_te, y_pred),
            "Log Loss": ll,
            L("CV Akurasi (rata-rata)", "CV Accuracy (mean)"): float(np.nanmean(cvs)),
            L("CV Akurasi (std)", "CV Accuracy (std)"): float(np.nanstd(cvs)),
        })
        fitted[name] = (pipe, y_pred, proba, num_cols, cat_cols)
        prog.progress(i / len(cfg["models"]))
    prog.empty()

    if not rows:
        st.error(L("Tidak ada model yang berhasil dilatih.", "No model trained successfully."))
        return
    ptab = PUI.combined_params_table("classification", cfg["models"], cfg.get("params", {}))
    if not ptab.empty:
        tables[L("Parameter Algoritma", "Algorithm Parameters")] = ptab

    comp = pd.DataFrame(rows).round(4).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    tables[L("Perbandingan Model", "Model Comparison")] = comp
    U.note(L(
        "Akurasi seimbang & F1-macro memberi bobot sama untuk tiap kelas, sehingga lebih adil "
        "pada data tidak seimbang. Kappa dan MCC mengoreksi kebetulan (chance agreement).",
        "Balanced accuracy & macro-F1 weight every class equally, which is fairer on imbalanced "
        "data. Kappa and MCC correct for chance agreement.",
    ))
    U.show_table(comp)
    figs[L("Perbandingan model", "Model comparison")] = V.fig_model_comparison(
        comp, L("Model", "Model"), "F1 (macro)",
        L("Perbandingan Model (F1-macro)", "Model Comparison (macro F1)"))
    st.pyplot(figs[L("Perbandingan model", "Model comparison")], width="stretch")

    best_name = comp.iloc[0][L("Model", "Model")]
    st.success(L(f"Model terbaik: **{best_name}** (F1-macro = {comp.iloc[0]['F1 (macro)']:.4f})",
                 f"Best model: **{best_name}** (macro F1 = {comp.iloc[0]['F1 (macro)']:.4f})"))

    # ------------------------------------------------------ detail terbaik
    U.section(L("4. Detail Model Terbaik", "4. Best Model Detail"))
    sel = st.selectbox(L("Tampilkan detail untuk model", "Show detail for model"),
                       list(fitted), index=list(fitted).index(best_name), key="cls_detail")
    pipe, y_pred, proba, num_cols, cat_cols = fitted[sel]

    rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": L("Kelas / Metrik", "Class / Metric")})
    tables[L("Laporan Klasifikasi", "Classification Report")] = rep_df.round(4)
    st.markdown("**" + L("Laporan klasifikasi per kelas", "Per-class classification report") + "**")
    U.show_table(rep_df.round(4))

    cm = confusion_matrix(y_te, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"{L('Aktual','Actual')}: {c}" for c in classes],
                         columns=[f"{L('Prediksi','Pred')}: {c}" for c in classes])
    tables[L("Matriks Konfusi", "Confusion Matrix")] = cm_df.reset_index().rename(
        columns={"index": L("Kelas", "Class")})
    cc1, cc2 = st.columns(2)
    with cc1:
        figs[L("Matriks konfusi", "Confusion matrix")] = V.fig_confusion(cm, classes)
        st.pyplot(figs[L("Matriks konfusi", "Confusion matrix")], width="stretch")
    with cc2:
        figs[L("Matriks konfusi ternormalisasi", "Confusion matrix normalized")] = V.fig_confusion(cm, classes, True)
        st.pyplot(figs[L("Matriks konfusi ternormalisasi", "Confusion matrix normalized")],
                  width="stretch")

    # kurva ROC / PR
    if proba is not None:
        curves, aucs, pr_curves, aps = {}, {}, {}, {}
        model_classes = list(pipe.classes_)
        for c in classes:
            idx = model_classes.index(c)
            ybin = (y_te == c).astype(int)
            if ybin.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(ybin, proba[:, idx])
            curves[str(c)] = (fpr, tpr)
            aucs[str(c)] = roc_auc_score(ybin, proba[:, idx])
            prec, rec, _ = precision_recall_curve(ybin, proba[:, idx])
            pr_curves[str(c)] = (rec, prec)
            aps[str(c)] = average_precision_score(ybin, proba[:, idx])
        if curves:
            r1, r2 = st.columns(2)
            figs[L("Kurva ROC", "ROC curve")] = V.fig_roc(curves, aucs)
            figs[L("Kurva Precision-Recall", "Precision-Recall curve")] = V.fig_pr_curve(pr_curves, aps)
            r1.pyplot(figs[L("Kurva ROC", "ROC curve")], width="stretch")
            r2.pyplot(figs[L("Kurva Precision-Recall", "Precision-Recall curve")], width="stretch")
            tables[L("Ringkasan AUC per Kelas", "Per-class AUC Summary")] = pd.DataFrame(
                {L("Kelas", "Class"): list(aucs), "ROC AUC": [round(v, 4) for v in aucs.values()],
                 "Average Precision": [round(aps[k], 4) for k in aucs]})

    # kepentingan fitur
    fnames = feature_names_from(pipe.named_steps["pre"], num_cols, cat_cols)
    imp, kind = importance_from(pipe.named_steps["model"], fnames)
    if imp is not None:
        imp_df = pd.DataFrame({L("Fitur", "Feature"): fnames,
                               L("Kepentingan", "Importance"): np.round(imp, 6)})
        imp_df = imp_df.reindex(imp_df[L("Kepentingan", "Importance")].abs()
                                .sort_values(ascending=False).index).reset_index(drop=True)
        tables[L("Kepentingan Fitur", "Feature Importance")] = imp_df
        st.markdown("**" + L(f"Kepentingan fitur ({kind})", f"Feature importance ({kind})") + "**")
        figs[L("Kepentingan fitur", "Feature importance")] = V.fig_importance(
            fnames, imp, L(f"Kepentingan Fitur - {sel}", f"Feature Importance - {sel}"))
        st.pyplot(figs[L("Kepentingan fitur", "Feature importance")], width="stretch")
        U.show_table(imp_df.head(25))

    # kurva pembelajaran
    try:
        n_split = int(min(cfg["cv"], y_tr.value_counts().min()))
        if n_split >= 2 and len(X_tr) >= 30:
            ts, tr_s, te_s = learning_curve(
                pipe, X_tr, y_tr, cv=StratifiedKFold(n_split, shuffle=True, random_state=cfg["rs"]),
                train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy", n_jobs=-1)
            figs[L("Kurva pembelajaran", "Learning curve")] = V.fig_learning_curve(
                ts, tr_s, te_s, L("Akurasi", "Accuracy"))
            st.pyplot(figs[L("Kurva pembelajaran", "Learning curve")], width="stretch")
            tables[L("Kurva Pembelajaran", "Learning Curve")] = pd.DataFrame({
                L("Ukuran latih", "Train size"): ts,
                L("Skor latih", "Train score"): tr_s.mean(axis=1).round(4),
                L("Skor validasi", "Validation score"): te_s.mean(axis=1).round(4)})
    except Exception:
        pass

    # prediksi data uji
    pred_df = X_te.copy()
    pred_df[L("Aktual", "Actual")] = y_te.values
    pred_df[L("Prediksi", "Predicted")] = y_pred
    pred_df[L("Benar?", "Correct?")] = np.where(y_te.values == y_pred, L("Ya", "Yes"), L("Tidak", "No"))
    if proba is not None:
        pred_df[L("Probabilitas Maks", "Max Probability")] = proba.max(axis=1).round(4)
    tables[L("Prediksi Data Uji", "Test Set Predictions")] = pred_df.reset_index(drop=True)
    with st.expander(L("Lihat prediksi data uji", "View test set predictions")):
        U.show_table(pred_df.reset_index(drop=True), height=380)

    # ---------------------------------------------- visualisasi khusus algoritma
    _algo_visuals(sel, pipe, X_tr, y_tr, X_te, y_te, classes, fnames, cfg, tables, figs)

    # ------------------------------------------------------- interpretasi
    U.section(L("6. Interpretasi Otomatis", "6. Automatic Interpretation"))
    best = comp.iloc[0]
    kappa = best["Cohen Kappa"]
    kappa_txt = (L("sangat baik", "very good") if kappa > 0.8 else
                 L("baik", "good") if kappa > 0.6 else
                 L("cukup", "moderate") if kappa > 0.4 else
                 L("lemah", "weak"))
    gap = best[L("CV Akurasi (rata-rata)", "CV Accuracy (mean)")] - best[L("Akurasi", "Accuracy")]
    interp = [
        L(f"Model **{best[L('Model','Model')]}** memberikan performa terbaik dengan akurasi "
          f"**{best[L('Akurasi','Accuracy')]:.2%}**, F1-macro **{best['F1 (macro)']:.4f}**, dan "
          f"Cohen Kappa **{kappa:.4f}** ({kappa_txt}).",
          f"**{best[L('Model','Model')]}** performs best with accuracy "
          f"**{best[L('Akurasi','Accuracy')]:.2%}**, macro-F1 **{best['F1 (macro)']:.4f}**, and "
          f"Cohen Kappa **{kappa:.4f}** ({kappa_txt})."),
        L(f"Validasi silang {cfg['cv']}-fold menghasilkan akurasi "
          f"{best[L('CV Akurasi (rata-rata)','CV Accuracy (mean)')]:.4f} ± "
          f"{best[L('CV Akurasi (std)','CV Accuracy (std)')]:.4f}, "
          + ("selisih terhadap data uji kecil sehingga model stabil (tidak overfitting)."
             if abs(gap) < 0.08 else
             "selisih terhadap data uji cukup besar sehingga perlu diwaspadai overfitting/underfitting."),
          f"{cfg['cv']}-fold cross-validation yields accuracy "
          f"{best[L('CV Akurasi (rata-rata)','CV Accuracy (mean)')]:.4f} ± "
          f"{best[L('CV Akurasi (std)','CV Accuracy (std)')]:.4f}, "
          + ("a small gap versus the test set, so the model is stable (no overfitting)."
             if abs(gap) < 0.08 else
             "a sizeable gap versus the test set — watch for over/underfitting.")),
        L(f"Dataset memiliki {len(classes)} kelas dengan rasio ketidakseimbangan {imb:.2f}:1. "
          + ("Distribusi relatif seimbang sehingga akurasi layak dijadikan acuan."
             if imb <= 1.5 else
             "Karena tidak seimbang, F1-macro dan akurasi seimbang lebih representatif daripada akurasi."),
          f"The dataset has {len(classes)} classes with an imbalance ratio of {imb:.2f}:1. "
          + ("The distribution is fairly balanced, so accuracy is a reasonable headline metric."
             if imb <= 1.5 else
             "Because it is imbalanced, macro-F1 and balanced accuracy are more representative than accuracy.")),
    ]
    if not an.empty:
        topf = an.iloc[0]
        interp.append(L(
            f"Fitur numerik paling diskriminatif secara statistik adalah **{topf[L('Fitur','Feature')]}** "
            f"(F = {topf['F (ANOVA)']:.3f}, p = {topf['p (ANOVA)']:.4g}, Eta² = {topf['Eta²']:.3f}).",
            f"The most statistically discriminative numeric feature is **{topf[L('Fitur','Feature')]}** "
            f"(F = {topf['F (ANOVA)']:.3f}, p = {topf['p (ANOVA)']:.4g}, Eta² = {topf['Eta²']:.3f})."))
    if imp is not None:
        top3 = imp_df.head(3)[L("Fitur", "Feature")].tolist()
        interp.append(L(f"Menurut model, tiga fitur paling berpengaruh adalah: **{', '.join(top3)}**.",
                        f"According to the model, the three most influential features are: **{', '.join(top3)}**."))
    for line in interp:
        st.markdown("- " + line)
    tables[L("Interpretasi", "Interpretation")] = pd.DataFrame(
        {L("No", "No"): range(1, len(interp) + 1),
         L("Interpretasi", "Interpretation"): [t.replace("**", "") for t in interp]})

    # ---------------------------------------------------- fine tuning
    tcfg = cfg.get("tune") or {}
    tuned = None
    if tcfg.get("enable"):
        tuned = TUI.render("classification", tcfg,
                           lambda rs: make_model(tcfg["algo"], cfg.get("params", {}).get(tcfg["algo"]), rs),
                           X_tr, y_tr, X_te, y_te, cfg["scaling"], cfg["rs"],
                           tables, figs, prefix="tune")

    meta = {
        L("Proses", "Process"): L("Klasifikasi", "Classification"),
        L("Kolom target", "Target column"): target,
        L("Jumlah fitur", "Feature count"): len(features),
        L("Fitur", "Features"): ", ".join(features),
        L("Jumlah data", "Row count"): len(data),
        L("Proporsi uji", "Test size"): cfg["test_size"],
        L("Penskalaan", "Scaling"): cfg["scaling"],
        L("Validasi silang", "Cross-validation"): f"{cfg['cv']}-fold",
        L("Model terbaik", "Best model"): best_name,
    }
    if tuned:
        meta[L("Algoritma di-tuning", "Tuned algorithm")] = tcfg["algo"]
        meta[L("Metrik tuning", "Tuning metric")] = tcfg["scoring"]
        meta[L("Parameter terbaik", "Best parameters")] = ", ".join(
            f"{p}={v}" for p, v in tuned["best_params"].items())
    U.export_bar(tables, figs, "klasifikasi", meta)
