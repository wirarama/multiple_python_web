"""Modul Regresi."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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

ALGOS = PR.algos("regression")


def make_model(name: str, params: dict | None, rs: int):
    """Bangun estimator regresi sesuai parameter yang diatur pengguna."""
    return PR.build("regression", name, params, rs)


# Dipertahankan agar kode/uji lama tetap berjalan (parameter bawaan).
MODELS = {name: (lambda rs, _n=name: make_model(_n, None, rs)) for name in ALGOS}


def _mape(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100) if mask.any() else np.nan


def config(df: pd.DataFrame) -> dict | None:
    SET.consume_pending("regression")
    st.markdown("### " + L("Konfigurasi Regresi", "Regression Setup"))
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique(dropna=True) > 10]
    if not num_cols:
        U.warn(L("Tidak ada kolom numerik kontinu yang cocok sebagai target.",
                 "No continuous numeric column suitable as target."))
        return None
    c1, c2 = st.columns(2)
    target = SET.selectbox(L("Kolom target (numerik)", "Target column (numeric)"), num_cols,
                           key="reg_target", index=len(num_cols) - 1, container=c1)
    feats_all = [c for c in df.columns if c != target]
    auto_drop = [c for c in feats_all if df[c].nunique(dropna=True) >= max(len(df) * 0.9, 20)
                 and not pd.api.types.is_numeric_dtype(df[c])]
    features = SET.multiselect(L("Kolom fitur", "Feature columns"), feats_all, key="reg_feats",
                               default=[c for c in feats_all if c not in auto_drop], container=c2)
    if auto_drop:
        st.caption(L(f"Kolom mirip-ID dikecualikan otomatis: {', '.join(auto_drop)}",
                     f"ID-like columns auto-excluded: {', '.join(auto_drop)}"))

    c3, c4, c5, c6 = st.columns(4)
    test_size = SET.slider(L("Proporsi data uji", "Test size"), 0.1, 0.5, 0.25, 0.05,
                           key="reg_ts", container=c3)
    scaling = SET.selectbox(L("Penskalaan fitur", "Feature scaling"),
                            ["standard", "minmax", "none"], key="reg_scale", container=c4)
    cv = SET.slider(L("Lipatan validasi silang", "CV folds"), 2, 10, 5, key="reg_cv", container=c5)
    rs = SET.number_input(L("Random state", "Random state"), 0, 9999, 42, key="reg_rs", container=c6)

    chosen = SET.multiselect(L("Algoritma yang diuji", "Algorithms to evaluate"), ALGOS,
                             key="reg_models",
                             default=["Linear Regression", "Ridge", "Decision Tree",
                                      "Random Forest", "Gradient Boosting"])
    ols = SET.checkbox(L("Sertakan ringkasan OLS (uji-t, p-value, asumsi klasik)",
                         "Include OLS summary (t-test, p-values, classical assumptions)"),
                       True, key="reg_ols")
    if not features or not chosen:
        U.warn(L("Pilih minimal satu fitur dan satu algoritma.", "Select at least one feature and one algorithm."))
        return None

    params = PUI.editor("regression", chosen, "reg")
    tune = TUI.config_panel("regression", chosen, "reg", cv_default=int(cv), seed_default=int(rs))
    cfg = dict(target=target, features=features, test_size=test_size, scaling=scaling,
               cv=cv, rs=int(rs), models=chosen, ols=ols, params=params, tune=tune)
    SET.panel("regression", {k: v for k, v in cfg.items() if k != "tune"})
    return cfg


def _algo_visuals(name, pipe, X_tr, y_tr, X_te, y_te, fnames, cfg, tables, figs):
    """Bagian 5: visualisasi yang menjelaskan cara kerja algoritma regresi terpilih."""
    U.section(L(f"5. Visualisasi Khusus Algoritma — {name}",
                f"5. Algorithm-Specific Visualization — {name}"),
              L("Grafik berikut menggambarkan mekanisme internal algoritma: struktur pohon, "
                "jalur regularisasi, pita epsilon SVR, dan sejenisnya.",
                "The charts below depict the algorithm's internal mechanism: tree structure, "
                "regularization path, the SVR epsilon tube, and the like."))
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    params = cfg.get("params", {}).get(name, {})
    rp = PR.resolve("regression", name, params)
    try:
        Ztr = np.asarray(pre.transform(X_tr), dtype=float)
        Zte = np.asarray(pre.transform(X_te), dtype=float)
    except Exception as e:
        U.warn(L(f"Data tidak dapat ditransformasi untuk visualisasi: {e}",
                 f"The data could not be transformed for visualization: {e}"))
        return
    ytr = np.asarray(y_tr, dtype=float)
    yte = np.asarray(y_te, dtype=float)

    def add(title, fig):
        if fig is None:
            return
        figs[title] = fig
        st.markdown(f"**{title}**")
        st.pyplot(fig, width="stretch")

    if name == "Decision Tree":
        d = SET.slider(L("Kedalaman pohon yang ditampilkan", "Tree depth to display"),
                       1, 6, 3, key="reg_tree_depth")
        U.note(L(
            "Pada pohon regresi, <b>value</b> di tiap simpul adalah rata-rata target dari data "
            "yang sampai di simpul itu, dan <b>squared_error</b> adalah ragam sisanya. "
            "Prediksi akhir sebuah baris adalah nilai daun tempat baris itu berakhir.",
            "In a regression tree, <b>value</b> at each node is the mean target of the rows "
            "reaching it and <b>squared_error</b> is their remaining variance. A row's prediction "
            "is the value of the leaf it ends up in."))
        add(L("Struktur pohon regresi", "Regression tree structure"),
            AV.fig_tree(model, fnames, None, max_depth=int(d)))
        rules = AV.tree_rules_table(model, fnames, None)
        if not rules.empty:
            tables[L("Aturan Pohon Regresi", "Regression Tree Rules")] = rules
            st.markdown("**" + L("Aturan jika-maka yang dipelajari pohon",
                                 "If-then rules learned by the tree") + "**")
            U.show_table(rules.head(30))
        add(L("Penurunan ketidakmurnian per fitur", "Impurity decrease per feature"),
            AV.fig_tree_importance_path(model, fnames))
        if len(Ztr) <= 3000:
            add(L("Jalur pemangkasan cost-complexity", "Cost-complexity pruning path"),
                AV.fig_ccp_path(lambda: make_model(name, params, cfg["rs"]), Ztr, ytr))

    elif name in ("Random Forest", "Extra Trees"):
        n_show = SET.slider(L("Jumlah pohon contoh yang ditampilkan", "Sample trees to display"),
                            1, 4, 3, key="reg_forest_n")
        U.note(L(
            "Prediksi hutan adalah rata-rata prediksi seluruh pohon. Kurva konvergensi "
            "memperlihatkan pada jumlah pohon berapa penambahan pohon tidak lagi memperbaiki R².",
            "A forest's prediction is the average across all trees. The convergence curve shows at "
            "how many trees adding more stops improving R²."))
        add(L("Contoh pohon dalam ansambel", "Sample trees in the ensemble"),
            AV.fig_forest_grid(model, fnames, None, n_trees=int(n_show), max_depth=2))
        add(L("Struktur pohon dalam hutan", "Tree structure across the forest"),
            AV.fig_forest_structure(model))
        add(L("Konvergensi ansambel", "Ensemble convergence"),
            AV.fig_forest_convergence(model, Zte, yte, "regression"))
        add(L("Kestabilan kepentingan fitur", "Feature importance stability"),
            AV.fig_importance_spread(model, fnames))

    elif name == "Gradient Boosting":
        U.note(L(
            "Tiap pohon baru dilatih pada sisa kesalahan (residual) model sebelumnya. Titik "
            "puncak kurva data uji menandai jumlah tahap optimal.",
            "Each new tree is trained on the previous model's residual error. The peak of the test "
            "curve marks the optimal number of stages."))
        add(L("Skor per tahap boosting", "Score per boosting stage"),
            AV.fig_boosting_stages(model, Ztr, ytr, Zte, yte, "regression"))
        add(L("Penurunan kerugian per tahap", "Loss reduction per stage"),
            AV.fig_boosting_loss(model))
        ests = getattr(model, "estimators_", None)
        if ests is not None and len(ests):
            add(L("Pohon lemah pertama", "First weak learner"),
                AV.fig_tree(ests[0][0], fnames, None, max_depth=2))

    elif name == "SVR (RBF)":
        U.note(L(
            "SVR hanya menghukum galat yang keluar dari pita selebar ±epsilon di sekitar garis "
            "prediksi. Titik di luar pita menjadi <b>support vector</b> dan menentukan bentuk model.",
            "SVR only penalizes errors that fall outside a tube of width ±epsilon around the "
            "prediction. Points outside the tube become <b>support vectors</b> and shape the model."))
        eps = float(rp.get("epsilon", 0.1))
        X1, lab1, _ = AV.project_2d(Ztr)
        from sklearn.base import clone

        m1 = clone(model).fit(X1[:, [0]], ytr)
        add(L("Pita epsilon & support vector", "Epsilon tube & support vectors"),
            AV.fig_svr_tube(m1, X1[:, 0], ytr, eps, lab1[0]))
        sv = getattr(model, "support_", None)
        if sv is not None:
            tables[L("Ringkasan Support Vector", "Support Vector Summary")] = pd.DataFrame({
                L("Keterangan", "Item"): [L("Jumlah support vector", "Support vectors"),
                                          L("Persentase data latih (%)", "Share of training data (%)"),
                                          L("Lebar pita epsilon", "Epsilon tube width")],
                L("Nilai", "Value"): [len(sv), round(len(sv) / max(len(Ztr), 1) * 100, 2), eps]})
            U.show_table(tables[L("Ringkasan Support Vector", "Support Vector Summary")])

    elif name in ("Linear Regression", "Ridge", "Lasso", "Elastic Net"):
        U.note(L(
            "Model linear menjumlahkan fitur berbobot. Regularisasi (alpha) menyusutkan bobot: "
            "Ridge menekan semua koefisien mendekati nol, sedangkan Lasso membuat sebagian "
            "koefisien tepat nol sehingga sekaligus menyeleksi fitur.",
            "A linear model sums weighted features. Regularization (alpha) shrinks the weights: "
            "Ridge pushes every coefficient toward zero, while Lasso sets some exactly to zero and "
            "thereby also selects features."))
        coef = np.asarray(getattr(model, "coef_", []), dtype=float).ravel()
        if coef.size:
            add(L("Koefisien model", "Model coefficients"),
                AV.fig_coefficients(fnames, coef,
                                    intercept=float(np.ravel(getattr(model, "intercept_", [0]))[0])))
            nz = int((np.abs(coef) > 1e-10).sum())
            st.caption(L(f"{nz} dari {len(coef)} koefisien bernilai bukan nol.",
                         f"{nz} of {len(coef)} coefficients are non-zero."))
        kind = {"Lasso": "lasso", "Elastic Net": "elasticnet"}.get(name, "ridge")
        if name != "Linear Regression":
            add(L("Jalur regularisasi", "Regularization path"),
                AV.fig_reg_path(kind, Ztr, ytr, fnames, l1_ratio=float(rp.get("l1_ratio", 0.5)),
                                chosen=float(rp.get("alpha", 1.0))))
        else:
            add(L("Jalur regularisasi (pembanding Ridge)", "Regularization path (Ridge comparison)"),
                AV.fig_reg_path("ridge", Ztr, ytr, fnames))

    elif name == "K-Nearest Neighbors":
        U.note(L(
            "k-NN memprediksi nilai target sebagai rata-rata (atau rata-rata terbobot jarak) dari "
            "k tetangga terdekat. Nilai k yang terlalu kecil membuat prediksi bergerigi.",
            "k-NN predicts the target as the (optionally distance-weighted) average of the k nearest "
            "neighbours. Too small a k makes the prediction jagged."))
        add(L("Sensitivitas terhadap k", "Sensitivity to k"),
            AV.fig_k_sensitivity(lambda: make_model(name, params, cfg["rs"]),
                                 Ztr, ytr, Zte, yte, metric_name="R²"))
        X1, lab1, _ = AV.project_2d(Ztr)
        from sklearn.base import clone

        m1 = clone(model).fit(X1[:, [0]], ytr)
        add(L("Bentuk prediksi sepanjang komponen utama",
              "Prediction shape along the principal component"),
            AV.fig_svr_tube(m1, X1[:, 0], ytr, 0.0, lab1[0]))


def run(df: pd.DataFrame, cfg: dict):
    target, features = cfg["target"], cfg["features"]
    data = df[features + [target]].copy()
    data = data[data[target].notna()]
    y = data[target].astype(float)
    X = data[features]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["rs"])

    tables, figs = {}, {}

    U.section(L("1. Analisis Statistik Data", "1. Data Statistical Analysis"))
    t0, f0 = U.descriptive_block(data, "reg")
    tables.update(t0)
    figs.update(f0)

    # ------------------------------------------------ statistik target
    U.section(L("2. Statistik Target & Hubungan dengan Fitur",
                "2. Target Statistics & Relationship with Features"))
    U.metric_row([
        (L("Rata-rata target", "Target mean"), f"{y.mean():,.3f}"),
        (L("Std dev", "Std dev"), f"{y.std():,.3f}"),
        (L("Kemencengan", "Skewness"), f"{y.skew():.3f}"),
        (L("Data latih / uji", "Train / test"), f"{len(X_tr)} / {len(X_te)}"),
    ])
    if abs(y.skew()) > 1:
        U.warn(L(
            f"Target sangat miring (skewness = {y.skew():.2f}). Pertimbangkan transformasi log "
            "atau gunakan model berbasis pohon yang tidak mensyaratkan normalitas.",
            f"The target is strongly skewed (skewness = {y.skew():.2f}). Consider a log transform "
            "or use tree-based models that do not assume normality.",
        ))
    corr_t = S.correlation_pairs(data, "pearson", top=200)
    if not corr_t.empty:
        rel = corr_t[(corr_t[L("Variabel 1", "Variable 1")] == target)
                     | (corr_t[L("Variabel 2", "Variable 2")] == target)].reset_index(drop=True)
        tables[L("Korelasi Fitur-Target", "Feature-Target Correlation")] = rel
        st.markdown("**" + L("Korelasi setiap fitur numerik dengan target",
                             "Correlation of each numeric feature with the target") + "**")
        U.show_table(rel)

    # -------------------------------------------------------- OLS ringkas
    ols_res = None
    if cfg["ols"]:
        try:
            import statsmodels.api as sm

            Xd = pd.get_dummies(X, drop_first=True).apply(pd.to_numeric, errors="coerce")
            Xd = Xd.loc[:, Xd.std() > 0].fillna(Xd.median())
            Xc = sm.add_constant(Xd.astype(float), has_constant="add")
            ols_res = sm.OLS(y.values, Xc.values).fit()
            coef = pd.DataFrame({
                L("Variabel", "Variable"): Xc.columns,
                L("Koefisien", "Coefficient"): np.round(ols_res.params, 6),
                L("Galat Baku", "Std Error"): np.round(ols_res.bse, 6),
                "t": np.round(ols_res.tvalues, 4),
                "p-value": np.round(ols_res.pvalues, 6),
                L("IK 95% Bawah", "95% CI Lower"): np.round(ols_res.conf_int()[:, 0], 6),
                L("IK 95% Atas", "95% CI Upper"): np.round(ols_res.conf_int()[:, 1], 6),
            })
            coef[L("Signifikan (a=0.05)", "Significant (a=0.05)")] = np.where(
                coef["p-value"] < 0.05, L("Ya", "Yes"), L("Tidak", "No"))
            tables[L("Koefisien OLS", "OLS Coefficients")] = coef
            fit_stats = pd.DataFrame([
                (L("R-squared", "R-squared"), round(float(ols_res.rsquared), 4)),
                (L("Adjusted R-squared", "Adjusted R-squared"), round(float(ols_res.rsquared_adj), 4)),
                (L("F-statistic", "F-statistic"), round(float(ols_res.fvalue), 4)),
                (L("Prob (F-statistic)", "Prob (F-statistic)"), f"{float(ols_res.f_pvalue):.6g}"),
                ("AIC", round(float(ols_res.aic), 3)),
                ("BIC", round(float(ols_res.bic), 3)),
                (L("Durbin-Watson", "Durbin-Watson"), round(S.durbin_watson_stat(ols_res.resid), 4)),
                (L("Jumlah observasi", "Observations"), int(ols_res.nobs)),
            ], columns=[L("Metrik", "Metric"), L("Nilai", "Value")])
            tables[L("Statistik Model OLS", "OLS Model Statistics")] = fit_stats
            st.markdown("**" + L("Regresi linear OLS: uji signifikansi koefisien",
                                 "OLS linear regression: coefficient significance") + "**")
            U.note(L(
                "p-value &lt; 0,05 berarti koefisien berbeda nyata dari nol. "
                "Durbin-Watson mendekati 2 menandakan tidak ada autokorelasi residual.",
                "p-value &lt; 0.05 means the coefficient differs significantly from zero. "
                "A Durbin-Watson near 2 indicates no residual autocorrelation.",
            ))
            oc1, oc2 = st.columns([2, 1])
            with oc1:
                U.show_table(coef)
            with oc2:
                U.show_table(fit_stats)
        except Exception as e:
            st.info(L(f"Ringkasan OLS dilewati: {e}", f"OLS summary skipped: {e}"))

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
        yp = pipe.predict(X_te)
        yp_tr = pipe.predict(X_tr)
        try:
            cvs = cross_val_score(pipe, X_tr, y_tr, cv=KFold(cfg["cv"], shuffle=True,
                                  random_state=cfg["rs"]), scoring="r2", n_jobs=-1)
        except Exception:
            cvs = np.array([np.nan])
        n, p = len(y_te), max(len(features), 1)
        r2 = r2_score(y_te, yp)
        adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
        rows.append({
            L("Model", "Model"): name,
            "R²": r2,
            L("R² Disesuaikan", "Adjusted R²"): adj,
            L("R² (data latih)", "R² (train)"): r2_score(y_tr, yp_tr),
            "MAE": mean_absolute_error(y_te, yp),
            "MSE": mean_squared_error(y_te, yp),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, yp))),
            "MAPE (%)": _mape(y_te, yp),
            L("CV R² (rata-rata)", "CV R² (mean)"): float(np.nanmean(cvs)),
            L("CV R² (std)", "CV R² (std)"): float(np.nanstd(cvs)),
        })
        fitted[name] = (pipe, yp, num_cols, cat_cols)
        prog.progress(i / len(cfg["models"]))
    prog.empty()

    if not rows:
        st.error(L("Tidak ada model yang berhasil dilatih.", "No model trained successfully."))
        return
    ptab = PUI.combined_params_table("regression", cfg["models"], cfg.get("params", {}))
    if not ptab.empty:
        tables[L("Parameter Algoritma", "Algorithm Parameters")] = ptab

    comp = pd.DataFrame(rows).round(4).sort_values("R²", ascending=False).reset_index(drop=True)
    tables[L("Perbandingan Model", "Model Comparison")] = comp
    U.note(L(
        "R² = proporsi keragaman target yang dijelaskan model. RMSE dalam satuan target "
        "(semakin kecil semakin baik). MAPE menyatakan galat relatif dalam persen.",
        "R² is the share of target variance explained. RMSE is in target units "
        "(lower is better). MAPE expresses relative error in percent.",
    ))
    U.show_table(comp)
    figs[L("Perbandingan model", "Model comparison")] = V.fig_model_comparison(
        comp, L("Model", "Model"), "R²", L("Perbandingan Model (R²)", "Model Comparison (R²)"))
    figs[L("Perbandingan RMSE", "RMSE comparison")] = V.fig_model_comparison(
        comp, L("Model", "Model"), "RMSE", L("Perbandingan Model (RMSE)", "Model Comparison (RMSE)"),
        higher_better=False)
    mc1, mc2 = st.columns(2)
    mc1.pyplot(figs[L("Perbandingan model", "Model comparison")], width="stretch")
    mc2.pyplot(figs[L("Perbandingan RMSE", "RMSE comparison")], width="stretch")

    best_name = comp.iloc[0][L("Model", "Model")]
    st.success(L(f"Model terbaik: **{best_name}** (R² = {comp.iloc[0]['R²']:.4f}, "
                 f"RMSE = {comp.iloc[0]['RMSE']:,.4f})",
                 f"Best model: **{best_name}** (R² = {comp.iloc[0]['R²']:.4f}, "
                 f"RMSE = {comp.iloc[0]['RMSE']:,.4f})"))

    # ------------------------------------------------------ detail terbaik
    U.section(L("4. Detail Model Terbaik & Diagnostik Residual",
                "4. Best Model Detail & Residual Diagnostics"))
    sel = st.selectbox(L("Tampilkan detail untuk model", "Show detail for model"),
                       list(fitted), index=list(fitted).index(best_name), key="reg_detail")
    pipe, yp, num_cols, cat_cols = fitted[sel]
    resid = np.asarray(y_te) - np.asarray(yp)

    d1, d2 = st.columns(2)
    figs[L("Aktual vs prediksi", "Actual vs predicted")] = V.fig_actual_vs_pred(
        y_te.values, yp, r2_score(y_te, yp))
    figs[L("Plot residual", "Residual plot")] = V.fig_residual_plot(yp, resid)
    d1.pyplot(figs[L("Aktual vs prediksi", "Actual vs predicted")], width="stretch")
    d2.pyplot(figs[L("Plot residual", "Residual plot")], width="stretch")

    figs[L("Distribusi residual", "Residual distribution")] = V.fig_residual_hist(resid)
    st.pyplot(figs[L("Distribusi residual", "Residual distribution")], width="stretch")
    figs[L("Galat per observasi", "Error per observation")] = V.fig_pred_error_dist(y_te.values, yp)
    st.pyplot(figs[L("Galat per observasi", "Error per observation")], width="stretch")

    rd = S.residual_diagnostics(resid)
    dw = S.durbin_watson_stat(resid)
    rd = pd.concat([rd, pd.DataFrame([{
        L("Uji", "Test"): "Durbin-Watson",
        L("Statistik", "Statistic"): round(dw, 4),
        "p-value": "-",
        L("Kesimpulan", "Conclusion"): (
            L("Tidak ada autokorelasi", "No autocorrelation") if 1.5 <= dw <= 2.5
            else L("Indikasi autokorelasi", "Autocorrelation indicated")),
    }])], ignore_index=True)
    tables[L("Diagnostik Residual", "Residual Diagnostics")] = rd
    st.markdown("**" + L("Uji asumsi residual", "Residual assumption tests") + "**")
    U.show_table(rd)

    fnames = feature_names_from(pipe.named_steps["pre"], num_cols, cat_cols)
    imp, kind = importance_from(pipe.named_steps["model"], fnames)
    imp_df = pd.DataFrame()
    if imp is not None:
        imp_df = pd.DataFrame({L("Fitur", "Feature"): fnames,
                               L("Kepentingan", "Importance"): np.round(imp, 6)})
        imp_df = imp_df.reindex(imp_df[L("Kepentingan", "Importance")].abs()
                                .sort_values(ascending=False).index).reset_index(drop=True)
        tables[L("Kepentingan Fitur", "Feature Importance")] = imp_df
        figs[L("Kepentingan fitur", "Feature importance")] = V.fig_importance(
            fnames, imp, L(f"Kepentingan Fitur - {sel} ({kind})", f"Feature Importance - {sel} ({kind})"))
        st.pyplot(figs[L("Kepentingan fitur", "Feature importance")], width="stretch")
        U.show_table(imp_df.head(25))

    pred_df = X_te.copy()
    pred_df[L("Aktual", "Actual")] = y_te.values
    pred_df[L("Prediksi", "Predicted")] = np.round(yp, 4)
    pred_df[L("Residual", "Residual")] = np.round(resid, 4)
    pred_df[L("Galat Absolut (%)", "Absolute Error (%)")] = np.round(
        np.abs(resid) / np.where(np.abs(y_te.values) > 1e-9, np.abs(y_te.values), np.nan) * 100, 2)
    tables[L("Prediksi Data Uji", "Test Set Predictions")] = pred_df.reset_index(drop=True)
    with st.expander(L("Lihat prediksi data uji", "View test set predictions")):
        U.show_table(pred_df.reset_index(drop=True), height=380)

    # -------------------------------------------------------- interpretasi
    _algo_visuals(sel, pipe, X_tr, y_tr, X_te, y_te, fnames, cfg, tables, figs)

    U.section(L("6. Interpretasi Otomatis", "6. Automatic Interpretation"))
    b = comp.iloc[0]
    r2v = b["R²"]
    kualitas = (L("sangat baik", "excellent") if r2v >= 0.9 else
                L("baik", "good") if r2v >= 0.75 else
                L("cukup", "moderate") if r2v >= 0.5 else
                L("lemah", "weak"))
    gap = b[L("R² (data latih)", "R² (train)")] - r2v
    interp = [
        L(f"Model **{b[L('Model','Model')]}** menjelaskan **{r2v:.2%}** keragaman "
          f"variabel **{target}** (kualitas {kualitas}), dengan RMSE {b['RMSE']:,.4f} "
          f"dan MAE {b['MAE']:,.4f}.",
          f"**{b[L('Model','Model')]}** explains **{r2v:.2%}** of the variance in "
          f"**{target}** ({kualitas} fit), with RMSE {b['RMSE']:,.4f} and MAE {b['MAE']:,.4f}."),
        L(f"MAPE sebesar {b['MAPE (%)']:.2f}% berarti rata-rata prediksi meleset "
          f"{b['MAPE (%)']:.2f}% dari nilai sebenarnya.",
          f"A MAPE of {b['MAPE (%)']:.2f}% means predictions miss the true value by "
          f"{b['MAPE (%)']:.2f}% on average."),
        L(f"Selisih R² latih vs uji sebesar {gap:.4f} — "
          + ("model tergeneralisasi dengan baik." if gap < 0.1 else
             "terdapat indikasi overfitting; pertimbangkan regularisasi atau pengurangan fitur."),
          f"The train-test R² gap is {gap:.4f} — "
          + ("the model generalizes well." if gap < 0.1 else
             "overfitting is indicated; consider regularization or fewer features.")),
    ]
    if ols_res is not None:
        sig = coef[(coef["p-value"] < 0.05) & (coef[L("Variabel", "Variable")] != "const")]
        if len(sig):
            names = ", ".join(sig[L("Variabel", "Variable")].head(5).tolist())
            interp.append(L(
                f"Pada model OLS, variabel yang berpengaruh signifikan (p &lt; 0,05): **{names}**. "
                f"Model OLS secara keseluruhan signifikan dengan F = {ols_res.fvalue:.3f} "
                f"(p = {ols_res.f_pvalue:.4g}).",
                f"In the OLS model, significant variables (p &lt; 0.05): **{names}**. "
                f"The overall OLS model is significant with F = {ols_res.fvalue:.3f} "
                f"(p = {ols_res.f_pvalue:.4g})."))
    if len(imp_df):
        top3 = imp_df.head(3)[L("Fitur", "Feature")].tolist()
        interp.append(L(f"Tiga prediktor paling berpengaruh menurut model: **{', '.join(top3)}**.",
                        f"Three most influential predictors: **{', '.join(top3)}**."))
    dw_txt = (L("residual tidak berautokorelasi (asumsi terpenuhi)", "residuals are not autocorrelated (assumption met)")
              if 1.5 <= dw <= 2.5 else
              L("ada indikasi autokorelasi residual (asumsi tidak terpenuhi)",
                "residual autocorrelation is indicated (assumption violated)"))
    interp.append(L(f"Nilai Durbin-Watson {dw:.4f} menunjukkan {dw_txt}.",
                    f"A Durbin-Watson of {dw:.4f} indicates {dw_txt}."))
    for line in interp:
        st.markdown("- " + line)
    tables[L("Interpretasi", "Interpretation")] = pd.DataFrame(
        {L("No", "No"): range(1, len(interp) + 1),
         L("Interpretasi", "Interpretation"): [t.replace("**", "") for t in interp]})

    # ---------------------------------------------------- fine tuning
    tcfg = cfg.get("tune") or {}
    tuned = None
    if tcfg.get("enable"):
        tuned = TUI.render("regression", tcfg,
                           lambda rs: make_model(tcfg["algo"], cfg.get("params", {}).get(tcfg["algo"]), rs),
                           X_tr, y_tr, X_te, y_te, cfg["scaling"], cfg["rs"],
                           tables, figs, prefix="tune")

    meta = {
        L("Proses", "Process"): L("Regresi", "Regression"),
        L("Kolom target", "Target column"): target,
        L("Jumlah fitur", "Feature count"): len(features),
        L("Fitur", "Features"): ", ".join(features),
        L("Jumlah data", "Row count"): len(data),
        L("Proporsi uji", "Test size"): cfg["test_size"],
        L("Penskalaan", "Scaling"): cfg["scaling"],
        L("Model terbaik", "Best model"): best_name,
    }
    if tuned:
        meta[L("Algoritma di-tuning", "Tuned algorithm")] = tcfg["algo"]
        meta[L("Metrik tuning", "Tuning metric")] = tcfg["scoring"]
        meta[L("Parameter terbaik", "Best parameters")] = ", ".join(
            f"{p}={v}" for p, v in tuned["best_params"].items())
    U.export_bar(tables, figs, "regresi", meta)
