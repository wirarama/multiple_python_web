"""Modul Time Series: ARIMA & SARIMA."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from core import algviz as AV
from core import data_io as IO
from core import params as PR
from core import paramui as PUI
from core import settings as SET
from core import stats as S
from core import ui as U
from core import viz as V
from core.i18n import L

FREQ_MAP = {
    "Otomatis | Auto": None,
    "Harian | Daily (D)": "D",
    "Mingguan | Weekly (W)": "W",
    "Bulanan | Monthly (MS)": "MS",
    "Kuartalan | Quarterly (QS)": "QS",
    "Tahunan | Yearly (YS)": "YS",
}
SEASON_DEFAULT = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}


MODE_MANUAL = "manual"
MODE_AUTO = "auto"
ALGO_OF_MODE = {MODE_AUTO: "SARIMAX (grid search)", MODE_MANUAL: "SARIMAX (manual)"}


def config(df: pd.DataFrame) -> dict | None:
    SET.consume_pending("timeseries")
    st.markdown("### " + L("Konfigurasi Time Series", "Time Series Setup"))
    date_cands = IO.datetime_candidates(df) or list(df.columns)
    num_cands = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cands:
        U.warn(L("Tidak ada kolom numerik untuk dijadikan nilai deret.",
                 "No numeric column available as the series value."))
        return None
    c1, c2, c3 = st.columns(3)
    date_col = SET.selectbox(L("Kolom waktu", "Time column"), date_cands, key="ts_date", container=c1)
    value_col = SET.selectbox(L("Kolom nilai", "Value column"), num_cands, key="ts_value",
                              index=len(num_cands) - 1, container=c2)
    freq_label = SET.selectbox(L("Frekuensi data", "Data frequency"), list(FREQ_MAP),
                               key="ts_freq", container=c3)

    c4, c5, c6 = st.columns(3)
    agg = SET.selectbox(L("Agregasi bila tanggal berulang", "Aggregation for duplicate dates"),
                        ["mean", "sum", "median", "max", "min"], key="ts_agg", container=c4)
    test_ratio = SET.slider(L("Proporsi data uji", "Test size"), 0.05, 0.4, 0.2, 0.05,
                            key="ts_ts", container=c5)
    horizon = SET.number_input(L("Jumlah periode ramalan ke depan", "Future forecast periods"),
                               1, 120, 12, key="ts_h", container=c6)

    st.markdown("**" + L("Spesifikasi model", "Model specification") + "**")
    mode_label = {MODE_AUTO: L("Otomatis (grid search AIC)", "Automatic (AIC grid search)"),
                  MODE_MANUAL: L("Manual", "Manual")}
    mode = SET.radio(L("Cara menentukan ordo", "How to determine the order"), list(mode_label),
                     key="ts_mode", horizontal=True, format_func=lambda k: mode_label[k])
    seasonal = SET.checkbox(L("Gunakan komponen musiman (SARIMA)", "Use seasonal component (SARIMA)"),
                            True, key="ts_seas")
    m = SET.number_input(L("Periode musiman (m)", "Seasonal period (m)"), 0, 366, 12,
                         key="ts_m") if seasonal else 0

    algo = ALGO_OF_MODE[mode]
    aparams = PUI.editor("timeseries", [algo], "ts", expanded=True,
                         title=L(f"Parameter model — {algo}", f"Model parameters — {algo}"))
    rp = PR.resolve("timeseries", algo, aparams.get(algo, {}))

    params = dict(date_col=date_col, value_col=value_col, freq=FREQ_MAP[freq_label], agg=agg,
                  test_ratio=test_ratio, horizon=int(horizon), seasonal=seasonal, m=int(m),
                  algo=algo, params=aparams, trend=str(rp["trend"]),
                  enforce_stationarity=bool(rp["enforce_stationarity"]),
                  enforce_invertibility=bool(rp["enforce_invertibility"]))
    if mode == MODE_MANUAL:
        params.update(auto=False, p=int(rp["p"]), d=int(rp["d"]), q=int(rp["q"]),
                      P=int(rp["P"]), D=int(rp["D"]), Q=int(rp["Q"]),
                      concentrate_scale=bool(rp.get("concentrate_scale", False)))
    else:
        params.update(auto=True, max_p=int(rp["max_p"]), max_d=int(rp["max_d"]),
                      max_q=int(rp["max_q"]), max_P=int(rp["max_P"]), max_D=int(rp["max_D"]),
                      max_Q=int(rp["max_Q"]), ic=str(rp["ic"]))
        # kompatibilitas dengan penamaan lama
        params["max_pq"] = max(params["max_p"], params["max_q"])
        params["max_PQ"] = max(params["max_P"], params["max_Q"])
    SET.panel("timeseries", params)
    return params


def _prepare(df: pd.DataFrame, cfg: dict):
    d = df[[cfg["date_col"], cfg["value_col"]]].copy()
    d[cfg["date_col"]] = IO.to_datetime_series(d[cfg["date_col"]])
    d[cfg["value_col"]] = pd.to_numeric(d[cfg["value_col"]], errors="coerce")
    d = d.dropna().sort_values(cfg["date_col"])
    s = d.groupby(cfg["date_col"])[cfg["value_col"]].agg(cfg["agg"])
    freq = cfg["freq"] or pd.infer_freq(s.index)
    if freq:
        try:
            s = s.asfreq(freq)
            s = s.interpolate(method="time").ffill().bfill()
        except Exception:
            freq = None
    return s, freq


def _grid_search(y, seasonal, m, cfg):
    """Cari ordo terbaik menurut kriteria informasi yang dipilih pengguna."""
    ic = str(cfg.get("ic", "aic"))
    trend = cfg.get("trend", "n")
    trend = None if trend in (None, "n") else trend
    es = bool(cfg.get("enforce_stationarity", False))
    ei = bool(cfg.get("enforce_invertibility", False))
    best = (np.inf, None, None)
    orders = list(itertools.product(range(int(cfg.get("max_p", 2)) + 1),
                                    range(int(cfg.get("max_d", 2)) + 1),
                                    range(int(cfg.get("max_q", 2)) + 1)))
    if seasonal and m > 1:
        sorders = list(itertools.product(range(int(cfg.get("max_P", 1)) + 1),
                                         range(int(cfg.get("max_D", 1)) + 1),
                                         range(int(cfg.get("max_Q", 1)) + 1)))
    else:
        sorders = [(0, 0, 0)]
    total = len(orders) * len(sorders)
    prog = st.progress(0.0, text=L(f"Mencari ordo terbaik ({total} kombinasi)...",
                                   f"Searching best order ({total} combinations)..."))
    results = []
    i = 0
    for o in orders:
        for so in sorders:
            i += 1
            if i % 5 == 0 or i == total:
                prog.progress(i / total)
            try:
                kw = dict(order=o, seasonal_order=(*so, m if seasonal and m > 1 else 0),
                          enforce_stationarity=es, enforce_invertibility=ei)
                if trend:
                    kw["trend"] = trend
                res = SARIMAX(y, **kw).fit(disp=False)
                score = float(getattr(res, ic, res.aic))
                if np.isfinite(score):
                    results.append((o, so, float(res.aic), float(res.bic)))
                    if score < best[0]:
                        best = (score, o, so)
            except Exception:
                continue
    prog.empty()
    return best, results


def _fit_sarimax(y, order, seas_order, cfg):
    kw = dict(order=order, seasonal_order=seas_order,
              enforce_stationarity=bool(cfg.get("enforce_stationarity", False)),
              enforce_invertibility=bool(cfg.get("enforce_invertibility", False)))
    trend = cfg.get("trend", "n")
    if trend and trend != "n":
        kw["trend"] = trend
    if cfg.get("concentrate_scale"):
        kw["concentrate_scale"] = True
    return SARIMAX(y, **kw).fit(disp=False)


def _algo_visuals(cfg, res, order, seas_order, grid, tables, figs):
    """Bagian 7: visualisasi yang menjelaskan struktur model ARIMA/SARIMA."""
    U.section(L("7. Visualisasi Khusus Algoritma — ARIMA / SARIMA",
                "7. Algorithm-Specific Visualization — ARIMA / SARIMA"),
              L("Grafik berikut memeriksa struktur internal model: kestabilan akar, signifikansi "
                "tiap koefisien, dampak satu guncangan, dan peta kriteria informasi antar ordo.",
                "The charts below inspect the model's internal structure: root stability, "
                "significance of each coefficient, the effect of a single shock, and the "
                "information-criterion map across orders."))

    def add(title, fig):
        if fig is None:
            return
        figs[title] = fig
        st.markdown(f"**{title}**")
        st.pyplot(fig, width="stretch")

    U.note(L(
        "Sebuah model ARIMA disebut <b>stasioner</b> bila seluruh akar terbalik AR berada di dalam "
        "lingkaran satuan, dan <b>invertibel</b> bila akar terbalik MA juga di dalamnya. Akar yang "
        "berada di luar atau tepat di tepi lingkaran menandakan ordo terlalu tinggi atau deret "
        "belum cukup dibedakan (nilai d/D perlu dinaikkan).",
        "An ARIMA model is <b>stationary</b> when every inverse AR root lies inside the unit circle, "
        "and <b>invertible</b> when the inverse MA roots do too. Roots outside or right on the "
        "circle indicate an over-specified order or a series that needs more differencing "
        "(raise d/D)."))
    add(L("Akar terbalik AR & MA", "Inverse AR & MA roots"), AV.fig_arima_roots(res))
    add(L("Koefisien model & signifikansi", "Model coefficients & significance"),
        AV.fig_arima_coefficients(res))
    add(L("Respons impuls", "Impulse response"),
        AV.fig_impulse_response(res, min(int(cfg.get("horizon", 12)) * 2, 48)))
    if grid:
        ic = str(cfg.get("ic", "aic"))
        add(L(f"Peta {ic.upper()} antar ordo", f"{ic.upper()} map across orders"),
            AV.fig_order_heatmap(grid, ic, best=order))

    try:
        params = res.params
        coef_df = pd.DataFrame({
            L("Parameter", "Parameter"): [str(i) for i in params.index],
            L("Koefisien", "Coefficient"): np.round(np.asarray(params, dtype=float), 6),
            L("Galat baku", "Std. error"): np.round(np.asarray(res.bse, dtype=float), 6),
            L("Statistik z", "z statistic"): np.round(np.asarray(res.tvalues, dtype=float), 4),
            L("p-value", "p-value"): np.round(np.asarray(res.pvalues, dtype=float), 6),
        })
        coef_df[L("Signifikan (α=0,05)", "Significant (α=0.05)")] = np.where(
            coef_df[L("p-value", "p-value")] < 0.05, L("Ya", "Yes"), L("Tidak", "No"))
        tables[L("Koefisien Model", "Model Coefficients")] = coef_df
        U.show_table(coef_df)
    except Exception:
        pass


def run(df: pd.DataFrame, cfg: dict):
    tables, figs = {}, {}
    s, freq = _prepare(df, cfg)
    if len(s) < 12:
        st.error(L("Deret terlalu pendek (minimal 12 titik).", "Series too short (minimum 12 points)."))
        return
    m = cfg["m"] if cfg["seasonal"] else 0
    if cfg["seasonal"] and m <= 1:
        m = SEASON_DEFAULT.get(freq or "MS", 12)

    # ------------------------------------------------ statistik deret waktu
    U.section(L("1. Statistik Deret Waktu", "1. Time Series Statistics"))
    U.metric_row([
        (L("Jumlah titik", "Data points"), f"{len(s):,}"),
        (L("Periode awal", "Start"), f"{s.index.min():%d %b %Y}"),
        (L("Periode akhir", "End"), f"{s.index.max():%d %b %Y}"),
        (L("Frekuensi terdeteksi", "Detected frequency"), str(freq or L("tidak teratur", "irregular"))),
    ])
    tsd = S.ts_descriptive(s)
    tables[L("Statistik Deret Waktu", "Time Series Statistics")] = tsd
    raw = pd.DataFrame({L("Waktu", "Time"): s.index, L("Nilai", "Value"): s.values})
    tables[L("Data Deret Waktu", "Time Series Data")] = raw

    d1, d2 = st.columns([1, 1])
    with d1:
        U.show_table(tsd)
    with d2:
        desc = S.descriptive_numeric(pd.DataFrame({cfg["value_col"]: s.values}))
        tables[L("Deskriptif Nilai", "Value Descriptives")] = desc
        U.show_table(desc.T.reset_index().rename(
            columns={"index": L("Metrik", "Metric"), 0: L("Nilai", "Value")}), hide_index=True)

    roll = max(3, (m if m > 1 else 12))
    figs[L("Plot deret waktu", "Time series plot")] = V.fig_ts_line(
        s, L(f"Deret Waktu: {cfg['value_col']}", f"Time Series: {cfg['value_col']}"), rolling=roll)
    st.pyplot(figs[L("Plot deret waktu", "Time series plot")], width="stretch")

    # ---------------------------------------------------- uji stasioneritas
    U.section(L("2. Uji Stasioneritas & Identifikasi Ordo",
                "2. Stationarity Tests & Order Identification"))
    st_tab = S.stationarity_tests(s)
    tables[L("Uji Stasioneritas (level)", "Stationarity Tests (level)")] = st_tab
    U.note(L(
        "ADF: H0 = tidak stasioner, jadi p &lt; 0,05 berarti <b>stasioner</b>. "
        "KPSS: H0 = stasioner, jadi p &ge; 0,05 berarti <b>stasioner</b>. "
        "Jika belum stasioner, lakukan differencing (naikkan nilai d).",
        "ADF: H0 = non-stationary, so p &lt; 0.05 means <b>stationary</b>. "
        "KPSS: H0 = stationary, so p &ge; 0.05 means <b>stationary</b>. "
        "If not stationary, apply differencing (raise d).",
    ))
    U.show_table(st_tab)

    d_suggest = 0
    tmp = s.copy()
    for i in range(3):
        try:
            from statsmodels.tsa.stattools import adfuller

            if adfuller(tmp.dropna(), autolag="AIC")[1] < 0.05:
                break
        except Exception:
            break
        tmp = tmp.diff().dropna()
        d_suggest += 1
    st.caption(L(f"Saran ordo differencing berdasarkan ADF: **d = {d_suggest}**",
                 f"Suggested differencing order from ADF: **d = {d_suggest}**"))
    if d_suggest > 0:
        st1 = S.stationarity_tests(s.diff(d_suggest).dropna())
        tables[L(f"Uji Stasioneritas (d={d_suggest})", f"Stationarity Tests (d={d_suggest})")] = st1
        st.markdown("**" + L(f"Setelah differencing d = {d_suggest}",
                             f"After differencing d = {d_suggest}") + "**")
        U.show_table(st1)

    figs[L("ACF & PACF", "ACF & PACF")] = V.fig_acf_pacf(s, lags=min(36, len(s) // 2 - 1))
    st.pyplot(figs[L("ACF & PACF", "ACF & PACF")], width="stretch")
    if d_suggest > 0:
        figs[L("ACF & PACF setelah differencing", "ACF & PACF after differencing")] = V.fig_acf_pacf(
            s.diff(d_suggest).dropna(), lags=min(36, len(s) // 2 - 1))
        st.pyplot(figs[L("ACF & PACF setelah differencing", "ACF & PACF after differencing")],
                  width="stretch")
    U.note(L(
        "Ordo <b>q</b> dibaca dari lag terakhir yang signifikan pada ACF; ordo <b>p</b> dari PACF. "
        "Pola berulang tiap m lag menandakan komponen musiman (P/Q).",
        "Order <b>q</b> is read from the last significant lag in the ACF; <b>p</b> from the PACF. "
        "A pattern repeating every m lags indicates a seasonal component (P/Q).",
    ))

    # --------------------------------------------------------- dekomposisi
    if m > 1 and len(s) >= 2 * m:
        U.section(L("3. Dekomposisi Deret Waktu", "3. Time Series Decomposition"))
        model_type = st.radio(L("Model dekomposisi", "Decomposition model"),
                              ["additive", "multiplicative"], horizontal=True, key="ts_dec")
        try:
            if model_type == "multiplicative" and (s <= 0).any():
                st.info(L("Data mengandung nilai ≤ 0, dekomposisi multiplikatif dilewati.",
                          "Data contains values ≤ 0; multiplicative decomposition skipped."))
            else:
                dec = seasonal_decompose(s, model=model_type, period=m)
                figs[L("Dekomposisi", "Decomposition")] = V.fig_decompose(dec, f"({model_type})")
                st.pyplot(figs[L("Dekomposisi", "Decomposition")], width="stretch")
                comp = pd.DataFrame({
                    L("Waktu", "Time"): s.index,
                    L("Observasi", "Observed"): dec.observed.values,
                    L("Tren", "Trend"): dec.trend.values,
                    L("Musiman", "Seasonal"): dec.seasonal.values,
                    L("Residual", "Residual"): dec.resid.values})
                tables[L("Komponen Dekomposisi", "Decomposition Components")] = comp.round(4)
                var_s = float(np.nanvar(dec.seasonal))
                var_r = float(np.nanvar(dec.resid))
                kekuatan = max(0.0, 1 - var_r / (var_r + var_s)) if (var_r + var_s) > 0 else np.nan
                st.caption(L(f"Kekuatan musiman ≈ **{kekuatan:.3f}** (0 = tidak musiman, 1 = sangat musiman)",
                             f"Seasonal strength ≈ **{kekuatan:.3f}** (0 = none, 1 = very seasonal)"))
                tables[L("Kekuatan Musiman", "Seasonal Strength")] = pd.DataFrame(
                    [(L("Kekuatan musiman", "Seasonal strength"), round(kekuatan, 4)),
                     (L("Varians musiman", "Seasonal variance"), round(var_s, 4)),
                     (L("Varians residual", "Residual variance"), round(var_r, 4))],
                    columns=[L("Metrik", "Metric"), L("Nilai", "Value")])
        except Exception as e:
            st.info(L(f"Dekomposisi dilewati: {e}", f"Decomposition skipped: {e}"))
        if freq in ("MS", "M", "QS", "Q", "D", "W"):
            plabel = L("Bulan", "Month") if freq in ("MS", "M") else (
                L("Kuartal", "Quarter") if freq in ("QS", "Q") else L("Hari", "Day"))
            try:
                figs[L("Pola musiman", "Seasonal pattern")] = V.fig_seasonal_boxplot(s, plabel)
                st.pyplot(figs[L("Pola musiman", "Seasonal pattern")], width="stretch")
            except Exception:
                pass

    # -------------------------------------------------------- pemodelan
    U.section(L("4. Pemodelan ARIMA / SARIMA", "4. ARIMA / SARIMA Modeling"))
    n_test = max(2, int(len(s) * cfg["test_ratio"]))
    train, test = s.iloc[:-n_test], s.iloc[-n_test:]

    grid = None
    ptab = PUI.combined_params_table("timeseries", [cfg.get("algo", "SARIMAX (manual)")],
                                     cfg.get("params", {}))
    if not ptab.empty:
        tables[L("Parameter Model", "Model Parameters")] = ptab

    if cfg["auto"]:
        (best_aic, order, sorder), grid = _grid_search(
            train, cfg["seasonal"] and m > 1, m, cfg)
        if order is None:
            st.error(L("Grid search gagal menemukan model yang konvergen.",
                       "Grid search failed to find a converging model."))
            return
        grid_df = pd.DataFrame(
            [{"ARIMA (p,d,q)": str(o), "Seasonal (P,D,Q,m)": f"{so}x{m}" if cfg["seasonal"] and m > 1 else "-",
              "AIC": round(a, 3), "BIC": round(b, 3)} for o, so, a, b in grid]
        ).sort_values("AIC").reset_index(drop=True)
        tables[L("Perbandingan Ordo (Grid Search)", "Order Comparison (Grid Search)")] = grid_df.head(40)
        st.markdown("**" + L("10 kombinasi ordo terbaik menurut AIC",
                             "Top 10 order combinations by AIC") + "**")
        U.show_table(grid_df.head(10))
    else:
        order = (int(cfg["p"]), int(cfg["d"]), int(cfg["q"]))
        sorder = (int(cfg.get("P", 0)), int(cfg.get("D", 0)), int(cfg.get("Q", 0)))

    seas_order = (*sorder, m) if (cfg["seasonal"] and m > 1) else (0, 0, 0, 0)
    model_name = (f"SARIMA{order}x{seas_order}" if seas_order[3] else f"ARIMA{order}")
    st.success(L(f"Model yang digunakan: **{model_name}**", f"Model used: **{model_name}**"))

    try:
        res_tr = _fit_sarimax(train, order, seas_order, cfg)
    except Exception as e:
        st.error(L(f"Model gagal dilatih: {e}", f"Model training failed: {e}"))
        return

    fc = res_tr.get_forecast(steps=len(test))
    pred = pd.Series(np.asarray(fc.predicted_mean), index=test.index)
    ci = fc.conf_int()
    ci = pd.DataFrame(np.asarray(ci), index=test.index, columns=["lower", "upper"])

    mae = mean_absolute_error(test, pred)
    rmse = float(np.sqrt(mean_squared_error(test, pred)))
    mask = np.abs(test.values) > 1e-9
    mape = float(np.mean(np.abs((test.values[mask] - pred.values[mask]) / test.values[mask])) * 100) if mask.any() else np.nan
    r2 = r2_score(test, pred) if len(test) > 1 else np.nan
    naive = np.repeat(train.iloc[-1], len(test))
    mase_den = np.mean(np.abs(np.diff(train.values))) or 1
    mase = float(np.mean(np.abs(test.values - pred.values)) / mase_den)

    U.metric_row([
        ("MAE", f"{mae:,.4f}"),
        ("RMSE", f"{rmse:,.4f}"),
        ("MAPE (%)", f"{mape:.2f}" if np.isfinite(mape) else "-"),
        ("R²", f"{r2:.4f}" if np.isfinite(r2) else "-"),
    ])
    evalt = pd.DataFrame([
        (L("Model", "Model"), model_name),
        (L("Ordo (p,d,q)", "Order (p,d,q)"), str(order)),
        (L("Ordo musiman (P,D,Q,m)", "Seasonal order (P,D,Q,m)"), str(seas_order)),
        ("AIC", round(float(res_tr.aic), 4)),
        ("BIC", round(float(res_tr.bic), 4)),
        ("HQIC", round(float(res_tr.hqic), 4)),
        (L("Log-Likelihood", "Log-Likelihood"), round(float(res_tr.llf), 4)),
        (L("Jumlah data latih", "Training points"), len(train)),
        (L("Jumlah data uji", "Test points"), len(test)),
        ("MAE", round(mae, 6)),
        ("MSE", round(float(mean_squared_error(test, pred)), 6)),
        ("RMSE", round(rmse, 6)),
        ("MAPE (%)", round(mape, 4) if np.isfinite(mape) else "-"),
        ("MASE", round(mase, 4)),
        ("R²", round(float(r2), 4) if np.isfinite(r2) else "-"),
        (L("RMSE naive (baseline)", "Naive RMSE (baseline)"),
         round(float(np.sqrt(mean_squared_error(test, naive))), 6)),
    ], columns=[L("Metrik", "Metric"), L("Nilai", "Value")])
    tables[L("Evaluasi Model", "Model Evaluation")] = evalt
    U.show_table(evalt)

    coef = pd.DataFrame({
        L("Parameter", "Parameter"): res_tr.params.index.astype(str),
        L("Koefisien", "Coefficient"): np.round(res_tr.params.values, 6),
        L("Galat Baku", "Std Error"): np.round(res_tr.bse.values, 6),
        "z": np.round(res_tr.tvalues.values, 4),
        "p-value": np.round(res_tr.pvalues.values, 6),
    })
    coef[L("Signifikan (a=0.05)", "Significant (a=0.05)")] = np.where(
        coef["p-value"] < 0.05, L("Ya", "Yes"), L("Tidak", "No"))
    tables[L("Koefisien Model", "Model Coefficients")] = coef
    st.markdown("**" + L("Signifikansi parameter model", "Model parameter significance") + "**")
    U.show_table(coef)

    fitted = pd.Series(np.asarray(res_tr.fittedvalues), index=train.index)
    figs[L("Fit & validasi", "Fit & validation")] = V.fig_ts_fit_forecast(
        train, test, fitted, pred, ci,
        L(f"{model_name} - Validasi pada Data Uji", f"{model_name} - Validation on Test Set"))
    st.pyplot(figs[L("Fit & validasi", "Fit & validation")], width="stretch")

    val_df = pd.DataFrame({
        L("Waktu", "Time"): test.index,
        L("Aktual", "Actual"): test.values,
        L("Prediksi", "Predicted"): np.round(pred.values, 4),
        L("Residual", "Residual"): np.round(test.values - pred.values, 4),
        L("Galat Absolut (%)", "Absolute Error (%)"): np.round(
            np.abs(test.values - pred.values) / np.where(np.abs(test.values) > 1e-9, np.abs(test.values), np.nan) * 100, 2),
        L("Batas Bawah 95%", "Lower 95%"): np.round(ci["lower"].values, 4),
        L("Batas Atas 95%", "Upper 95%"): np.round(ci["upper"].values, 4),
    })
    tables[L("Validasi Data Uji", "Test Set Validation")] = val_df
    U.show_table(val_df, height=320)

    # ---------------------------------------------------- diagnostik residual
    U.section(L("5. Diagnostik Residual", "5. Residual Diagnostics"))
    resid = pd.Series(np.asarray(res_tr.resid), index=train.index).iloc[max(order[1], 1):]
    lb = S.ljung_box(resid, lags=min(20, max(2, len(resid) // 5)))
    rd = S.residual_diagnostics(resid)
    diag = pd.concat([lb, rd], ignore_index=True)
    tables[L("Diagnostik Residual", "Residual Diagnostics")] = diag
    U.note(L(
        "Model dianggap memadai bila residual bersifat white noise: "
        "Ljung-Box p &ge; 0,05, rata-rata residual mendekati nol, dan ACF residual tidak signifikan.",
        "The model is adequate when residuals are white noise: "
        "Ljung-Box p &ge; 0.05, residual mean near zero, and no significant residual ACF.",
    ))
    U.show_table(diag)
    figs[L("Diagnostik residual", "Residual diagnostics")] = V.fig_ts_residual_diag(resid)
    st.pyplot(figs[L("Diagnostik residual", "Residual diagnostics")], width="stretch")

    # ------------------------------------------------------------ peramalan
    U.section(L("6. Peramalan Periode Mendatang", "6. Future Forecast"))
    try:
        res_full = _fit_sarimax(s, order, seas_order, cfg)
    except Exception:
        res_full = res_tr
    fc2 = res_full.get_forecast(steps=cfg["horizon"])
    fmean = np.asarray(fc2.predicted_mean)
    fci = np.asarray(fc2.conf_int())
    if freq:
        future_idx = pd.date_range(s.index[-1], periods=cfg["horizon"] + 1, freq=freq)[1:]
    else:
        step = (s.index[-1] - s.index[-2]) if len(s) > 1 else pd.Timedelta(days=1)
        future_idx = [s.index[-1] + step * (i + 1) for i in range(cfg["horizon"])]
    future = pd.DataFrame({"forecast": fmean, "lower": fci[:, 0], "upper": fci[:, 1]}, index=future_idx)

    fc_view = pd.DataFrame({
        L("Periode", "Period"): [f"{d:%d %b %Y}" for d in future.index],
        L("Ramalan", "Forecast"): np.round(future["forecast"].values, 4),
        L("Batas Bawah 95%", "Lower 95%"): np.round(future["lower"].values, 4),
        L("Batas Atas 95%", "Upper 95%"): np.round(future["upper"].values, 4),
        L("Lebar Interval", "Interval Width"): np.round(future["upper"].values - future["lower"].values, 4),
    })
    tables[L("Hasil Peramalan", "Forecast Result")] = fc_view
    figs[L("Peramalan", "Forecast")] = V.fig_forecast_table_plot(future, s)
    st.pyplot(figs[L("Peramalan", "Forecast")], width="stretch")
    U.show_table(fc_view, height=340)

    # -------------------------------------------------------- interpretasi
    _algo_visuals(cfg, res_full, order, seas_order, grid, tables, figs)

    U.section(L("8. Interpretasi Otomatis", "8. Automatic Interpretation"))
    adf_p = float(st_tab.iloc[0]["p-value"]) if "p-value" in st_tab.columns else np.nan
    lb_p = float(lb["p-value"].iloc[0])
    akurasi = (L("sangat akurat", "very accurate") if mape < 10 else
               L("akurat", "accurate") if mape < 20 else
               L("cukup akurat", "reasonably accurate") if mape < 50 else
               L("kurang akurat", "inaccurate")) if np.isfinite(mape) else "-"
    trend_dir = L("naik", "upward") if s.iloc[-min(len(s), 12):].mean() > s.iloc[:min(len(s), 12)].mean() else L("turun", "downward")
    interp = [
        L(f"Deret **{cfg['value_col']}** terdiri atas {len(s):,} titik dari "
          f"{s.index.min():%b %Y} sampai {s.index.max():%b %Y} dengan frekuensi {freq or '-'}. "
          f"Rata-rata {s.mean():,.3f} dan simpangan baku {s.std():,.3f}.",
          f"The **{cfg['value_col']}** series has {len(s):,} points from "
          f"{s.index.min():%b %Y} to {s.index.max():%b %Y} at frequency {freq or '-'}. "
          f"Mean {s.mean():,.3f}, standard deviation {s.std():,.3f}."),
        L(f"Uji ADF pada level menghasilkan p = {adf_p:.4g}, sehingga data "
          + ("**sudah stasioner**" if adf_p < 0.05 else "**belum stasioner**")
          + f" dan ordo differencing yang disarankan adalah d = {d_suggest}.",
          f"The ADF test at level gives p = {adf_p:.4g}, so the data is "
          + ("**already stationary**" if adf_p < 0.05 else "**not yet stationary**")
          + f", and the suggested differencing order is d = {d_suggest}."),
        L(f"Model terpilih **{model_name}** dengan AIC {res_tr.aic:.3f} dan BIC {res_tr.bic:.3f}. "
          f"Pada data uji ({len(test)} periode) model menghasilkan MAPE {mape:.2f}% "
          f"({akurasi}), RMSE {rmse:,.4f}, dan MASE {mase:.3f} "
          + ("(lebih baik daripada peramalan naif)." if mase < 1 else "(belum mengungguli peramalan naif)."),
          f"The selected model **{model_name}** has AIC {res_tr.aic:.3f} and BIC {res_tr.bic:.3f}. "
          f"On the test set ({len(test)} periods) it achieves MAPE {mape:.2f}% "
          f"({akurasi}), RMSE {rmse:,.4f}, and MASE {mase:.3f} "
          + ("(better than the naive forecast)." if mase < 1 else "(does not beat the naive forecast).")),
        L(f"Uji Ljung-Box menghasilkan p = {lb_p:.4g}, sehingga residual "
          + ("**bersifat acak (white noise)** dan model dinyatakan memadai."
             if lb_p >= 0.05 else
             "**masih mengandung autokorelasi**; pertimbangkan menaikkan ordo p/q atau P/Q."),
          f"The Ljung-Box test gives p = {lb_p:.4g}, so the residuals are "
          + ("**white noise** and the model is adequate."
             if lb_p >= 0.05 else
             "**still autocorrelated**; consider raising p/q or P/Q.")),
        L(f"Deret menunjukkan kecenderungan {trend_dir}. Ramalan {cfg['horizon']} periode ke depan "
          f"bergerak dari {future['forecast'].iloc[0]:,.3f} menuju {future['forecast'].iloc[-1]:,.3f} "
          f"({(future['forecast'].iloc[-1] / future['forecast'].iloc[0] - 1) * 100:+.2f}%), "
          f"dengan interval keyakinan 95% melebar dari ±{(future['upper'].iloc[0] - future['lower'].iloc[0]) / 2:,.2f} "
          f"menjadi ±{(future['upper'].iloc[-1] - future['lower'].iloc[-1]) / 2:,.2f} seiring jauhnya horizon.",
          f"The series trends {trend_dir}. The {cfg['horizon']}-period forecast moves from "
          f"{future['forecast'].iloc[0]:,.3f} to {future['forecast'].iloc[-1]:,.3f} "
          f"({(future['forecast'].iloc[-1] / future['forecast'].iloc[0] - 1) * 100:+.2f}%), "
          f"with the 95% interval widening from ±{(future['upper'].iloc[0] - future['lower'].iloc[0]) / 2:,.2f} "
          f"to ±{(future['upper'].iloc[-1] - future['lower'].iloc[-1]) / 2:,.2f} as the horizon lengthens."),
    ]
    for line in interp:
        st.markdown("- " + line)
    tables[L("Interpretasi", "Interpretation")] = pd.DataFrame(
        {L("No", "No"): range(1, len(interp) + 1),
         L("Interpretasi", "Interpretation"): [t.replace("**", "") for t in interp]})

    meta = {
        L("Proses", "Process"): "Time Series (ARIMA/SARIMA)",
        L("Kolom waktu", "Time column"): cfg["date_col"],
        L("Kolom nilai", "Value column"): cfg["value_col"],
        L("Frekuensi", "Frequency"): str(freq),
        L("Jumlah titik", "Data points"): len(s),
        L("Model", "Model"): model_name,
        "AIC": round(float(res_tr.aic), 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4) if np.isfinite(mape) else "-",
        L("Horizon ramalan", "Forecast horizon"): cfg["horizon"],
    }
    U.export_bar(tables, figs, "timeseries", meta)
