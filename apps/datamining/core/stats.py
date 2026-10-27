"""Analisis statistik deskriptif, inferensial, dan uji asumsi."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

from .i18n import L  # noqa: E402


def _c(id_t, en_t):
    return L(id_t, en_t)


# ------------------------------------------------------------- ringkasan
def overview(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    rows = [
        (_c("Jumlah baris", "Number of rows"), f"{len(df):,}"),
        (_c("Jumlah kolom", "Number of columns"), f"{df.shape[1]:,}"),
        (_c("Kolom numerik", "Numeric columns"), f"{len(num)}"),
        (_c("Kolom non-numerik", "Non-numeric columns"), f"{df.shape[1] - len(num)}"),
        (_c("Total sel", "Total cells"), f"{df.size:,}"),
        (_c("Sel kosong", "Missing cells"), f"{int(df.isna().sum().sum()):,}"),
        (_c("% sel kosong", "% missing cells"), f"{df.isna().sum().sum() / max(df.size, 1) * 100:.2f}%"),
        (_c("Baris duplikat", "Duplicate rows"), f"{int(df.duplicated().sum()):,}"),
        (_c("Penggunaan memori", "Memory usage"), f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"),
    ]
    return pd.DataFrame(rows, columns=[_c("Metrik", "Metric"), _c("Nilai", "Value")])


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    out = pd.DataFrame(
        {
            _c("Kolom", "Column"): m.index,
            _c("Jumlah Kosong", "Missing Count"): m.values,
            _c("% Kosong", "% Missing"): (m.values / max(len(df), 1) * 100).round(2),
            _c("Terisi", "Filled"): (len(df) - m.values),
        }
    )
    return out.sort_values(_c("Jumlah Kosong", "Missing Count"), ascending=False).reset_index(drop=True)


def descriptive_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append(
            {
                _c("Variabel", "Variable"): c,
                "N": int(s.count()),
                _c("Rata-rata", "Mean"): s.mean(),
                _c("Median", "Median"): s.median(),
                _c("Modus", "Mode"): s.mode().iloc[0] if not s.mode().empty else np.nan,
                _c("Std Dev", "Std Dev"): s.std(),
                _c("Varians", "Variance"): s.var(),
                _c("Min", "Min"): s.min(),
                "Q1": q1,
                "Q3": q3,
                _c("Maks", "Max"): s.max(),
                "IQR": q3 - q1,
                _c("Rentang", "Range"): s.max() - s.min(),
                _c("Kemencengan", "Skewness"): s.skew(),
                _c("Keruncingan", "Kurtosis"): s.kurtosis(),
                _c("Koef. Variasi (%)", "CV (%)"): (s.std() / s.mean() * 100) if s.mean() != 0 else np.nan,
                _c("Galat Baku", "Std Error"): s.std() / np.sqrt(len(s)) if len(s) else np.nan,
                _c("IK 95% Bawah", "95% CI Lower"): s.mean() - 1.96 * s.std() / np.sqrt(len(s)) if len(s) else np.nan,
                _c("IK 95% Atas", "95% CI Upper"): s.mean() + 1.96 * s.std() / np.sqrt(len(s)) if len(s) else np.nan,
            }
        )
    return pd.DataFrame(rows).round(4)


def descriptive_categorical(df: pd.DataFrame, max_unique: int = 60) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        v = s.dropna()
        if v.nunique() > max_unique or len(v) == 0:
            continue
        vc = v.value_counts()
        ent = sps.entropy(vc.values / vc.sum(), base=2)
        rows.append(
            {
                _c("Variabel", "Variable"): c,
                "N": int(len(v)),
                _c("Nilai Unik", "Unique"): int(v.nunique()),
                _c("Modus", "Mode"): str(vc.index[0]),
                _c("Frekuensi Modus", "Mode Freq"): int(vc.iloc[0]),
                _c("% Modus", "% Mode"): round(float(vc.iloc[0] / len(v) * 100), 2),
                _c("Entropi (bit)", "Entropy (bits)"): round(float(ent), 4),
                _c("Distribusi", "Distribution"): " | ".join(
                    f"{k}: {n} ({n / len(v) * 100:.1f}%)" for k, n in vc.head(6).items()
                ),
            }
        )
    return pd.DataFrame(rows)


def frequency_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False)
    out = pd.DataFrame(
        {
            _c("Kategori", "Category"): vc.index.astype(str),
            _c("Frekuensi", "Frequency"): vc.values,
            _c("Proporsi (%)", "Proportion (%)"): (vc.values / vc.values.sum() * 100).round(2),
        }
    )
    out[_c("Kumulatif (%)", "Cumulative (%)")] = out[_c("Proporsi (%)", "Proportion (%)")].cumsum().round(2)
    return out


# --------------------------------------------------------------- korelasi
def correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method).round(4)


def correlation_pairs(df: pd.DataFrame, method: str = "pearson", top: int = 25) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).dropna()
    cols = list(num.columns)
    if len(cols) < 2 or len(num) < 3:
        return pd.DataFrame()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = num[cols[i]], num[cols[j]]
            try:
                if method == "spearman":
                    r, p = sps.spearmanr(a, b)
                elif method == "kendall":
                    r, p = sps.kendalltau(a, b)
                else:
                    r, p = sps.pearsonr(a, b)
            except Exception:
                continue
            ar = abs(r)
            kekuatan = (
                _c("Sangat kuat", "Very strong") if ar >= 0.8
                else _c("Kuat", "Strong") if ar >= 0.6
                else _c("Sedang", "Moderate") if ar >= 0.4
                else _c("Lemah", "Weak") if ar >= 0.2
                else _c("Sangat lemah", "Very weak")
            )
            rows.append(
                {
                    _c("Variabel 1", "Variable 1"): cols[i],
                    _c("Variabel 2", "Variable 2"): cols[j],
                    "r": round(float(r), 4),
                    "p-value": round(float(p), 6),
                    _c("Kekuatan", "Strength"): kekuatan,
                    _c("Arah", "Direction"): _c("Positif", "Positive") if r >= 0 else _c("Negatif", "Negative"),
                    _c("Signifikan (a=0.05)", "Significant (a=0.05)"): _c("Ya", "Yes") if p < 0.05 else _c("Tidak", "No"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


# --------------------------------------------------------- uji normalitas
def normality_tests(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 8 or s.nunique() < 3:
            continue
        try:
            if len(s) <= 5000:
                w, p_sw = sps.shapiro(s)
            else:
                w, p_sw = np.nan, np.nan
        except Exception:
            w, p_sw = np.nan, np.nan
        try:
            ks_stat, p_ks = sps.kstest(sps.zscore(s), "norm")
        except Exception:
            ks_stat, p_ks = np.nan, np.nan
        try:
            da_stat, p_da = sps.normaltest(s)
        except Exception:
            da_stat, p_da = np.nan, np.nan
        p_ref = p_sw if not np.isnan(p_sw) else p_ks
        rows.append(
            {
                _c("Variabel", "Variable"): c,
                "N": int(len(s)),
                "Shapiro-Wilk W": round(float(w), 4) if not np.isnan(w) else "-",
                "Shapiro p": round(float(p_sw), 6) if not np.isnan(p_sw) else "-",
                "K-S D": round(float(ks_stat), 4),
                "K-S p": round(float(p_ks), 6),
                "D'Agostino K2": round(float(da_stat), 4),
                "D'Agostino p": round(float(p_da), 6),
                _c("Kesimpulan (a=0.05)", "Conclusion (a=0.05)"): (
                    _c("Normal", "Normal") if p_ref >= 0.05 else _c("Tidak normal", "Not normal")
                ),
            }
        )
    return pd.DataFrame(rows)


def homogeneity_test(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    groups = [g[value_col].dropna().to_numpy(dtype=float)
              for _, g in df.groupby(group_col) if g[value_col].notna().sum() > 2]
    if len(groups) < 2:
        return pd.DataFrame()
    try:
        lev_s, lev_p = sps.levene(*groups)
    except Exception:
        lev_s, lev_p = np.nan, np.nan
    try:
        bar_s, bar_p = sps.bartlett(*groups)
    except Exception:
        bar_s, bar_p = np.nan, np.nan
    return pd.DataFrame(
        [
            {
                _c("Uji", "Test"): "Levene",
                _c("Statistik", "Statistic"): round(float(lev_s), 4),
                "p-value": round(float(lev_p), 6),
                _c("Kesimpulan", "Conclusion"): _c("Homogen", "Homogeneous") if lev_p >= 0.05 else _c("Tidak homogen", "Not homogeneous"),
            },
            {
                _c("Uji", "Test"): "Bartlett",
                _c("Statistik", "Statistic"): round(float(bar_s), 4),
                "p-value": round(float(bar_p), 6),
                _c("Kesimpulan", "Conclusion"): _c("Homogen", "Homogeneous") if bar_p >= 0.05 else _c("Tidak homogen", "Not homogeneous"),
            },
        ]
    )


# ------------------------------------------------------------- outlier
def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        z = np.abs(sps.zscore(s)) if s.std() > 0 else np.zeros(len(s))
        n_z = int((z > 3).sum())
        rows.append(
            {
                _c("Variabel", "Variable"): c,
                _c("Batas Bawah (IQR)", "Lower Bound (IQR)"): round(float(lo), 4),
                _c("Batas Atas (IQR)", "Upper Bound (IQR)"): round(float(hi), 4),
                _c("Outlier IQR", "IQR Outliers"): n_iqr,
                _c("% Outlier IQR", "% IQR Outliers"): round(n_iqr / len(s) * 100, 2),
                _c("Outlier Z>3", "Z>3 Outliers"): n_z,
                _c("% Outlier Z>3", "% Z>3 Outliers"): round(n_z / len(s) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor untuk deteksi multikolinearitas."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    num = X.select_dtypes(include=[np.number]).dropna()
    num = num.loc[:, num.std() > 0]
    if num.shape[1] < 2 or len(num) < num.shape[1] + 2:
        return pd.DataFrame()
    Xc = add_constant(num, has_constant="add")
    rows = []
    for i, c in enumerate(Xc.columns):
        if c == "const":
            continue
        try:
            v = variance_inflation_factor(Xc.values, i)
        except Exception:
            v = np.nan
        rows.append(
            {
                _c("Variabel", "Variable"): c,
                "VIF": round(float(v), 4) if np.isfinite(v) else np.inf,
                _c("Interpretasi", "Interpretation"): (
                    _c("Tidak ada multikolinearitas", "No multicollinearity") if v < 5
                    else _c("Multikolinearitas sedang", "Moderate multicollinearity") if v < 10
                    else _c("Multikolinearitas tinggi", "High multicollinearity")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


# ------------------------------------------- hubungan fitur vs target
def anova_feature_target(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """Uji beda rerata fitur numerik antar kelas target (ANOVA + Kruskal)."""
    rows = []
    for c in features:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        groups = [g[c].dropna().to_numpy(dtype=float) for _, g in df.groupby(target) if g[c].notna().sum() > 2]
        if len(groups) < 2:
            continue
        try:
            f, p = sps.f_oneway(*groups)
        except Exception:
            f, p = np.nan, np.nan
        try:
            h, pk = sps.kruskal(*groups)
        except Exception:
            h, pk = np.nan, np.nan
        # eta squared
        allv = np.concatenate(groups)
        grand = allv.mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((allv - grand) ** 2).sum()
        eta = ss_b / ss_t if ss_t > 0 else np.nan
        rows.append(
            {
                _c("Fitur", "Feature"): c,
                "F (ANOVA)": round(float(f), 4),
                "p (ANOVA)": round(float(p), 6),
                "H (Kruskal-Wallis)": round(float(h), 4),
                "p (Kruskal)": round(float(pk), 6),
                "Eta²": round(float(eta), 4),
                _c("Diskriminatif (a=0.05)", "Discriminative (a=0.05)"): _c("Ya", "Yes") if p < 0.05 else _c("Tidak", "No"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("p (ANOVA)").reset_index(drop=True) if not out.empty else out


def chi2_feature_target(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """Uji Chi-Square independensi fitur kategorik vs target."""
    rows = []
    for c in features:
        if c == target or pd.api.types.is_numeric_dtype(df[c]):
            continue
        tab = pd.crosstab(df[c], df[target])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        try:
            chi2, p, dof, exp = sps.chi2_contingency(tab)
        except Exception:
            continue
        n = tab.values.sum()
        cramers = np.sqrt(chi2 / (n * (min(tab.shape) - 1))) if n and min(tab.shape) > 1 else np.nan
        rows.append(
            {
                _c("Fitur", "Feature"): c,
                "Chi²": round(float(chi2), 4),
                "df": int(dof),
                "p-value": round(float(p), 6),
                "Cramer's V": round(float(cramers), 4),
                _c("Terkait Target (a=0.05)", "Related to Target (a=0.05)"): _c("Ya", "Yes") if p < 0.05 else _c("Tidak", "No"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("p-value").reset_index(drop=True) if not out.empty else out


def group_summary(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return pd.DataFrame()
    g = df.groupby(target)[num].agg(["mean", "std", "min", "max"]).round(3)
    g.columns = [f"{a} | {b}" for a, b in g.columns]
    return g.reset_index()


# ----------------------------------------------------- uji stasioneritas
def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller, kpss

    s = pd.Series(series).dropna()
    rows = []
    try:
        adf = adfuller(s, autolag="AIC")
        rows.append(
            {
                _c("Uji", "Test"): "ADF (Augmented Dickey-Fuller)",
                _c("H0", "H0"): _c("Ada unit root (tidak stasioner)", "Has unit root (non-stationary)"),
                _c("Statistik", "Statistic"): round(float(adf[0]), 4),
                "p-value": round(float(adf[1]), 6),
                _c("Nilai Kritis 5%", "Critical Value 5%"): round(float(adf[4]["5%"]), 4),
                _c("Kesimpulan", "Conclusion"): _c("Stasioner", "Stationary") if adf[1] < 0.05 else _c("Tidak stasioner", "Non-stationary"),
            }
        )
    except Exception as e:  # pragma: no cover
        rows.append({_c("Uji", "Test"): "ADF", _c("Kesimpulan", "Conclusion"): f"gagal: {e}"})
    try:
        kp = kpss(s, regression="c", nlags="auto")
        rows.append(
            {
                _c("Uji", "Test"): "KPSS",
                _c("H0", "H0"): _c("Deret stasioner", "Series is stationary"),
                _c("Statistik", "Statistic"): round(float(kp[0]), 4),
                "p-value": round(float(kp[1]), 6),
                _c("Nilai Kritis 5%", "Critical Value 5%"): round(float(kp[3]["5%"]), 4),
                _c("Kesimpulan", "Conclusion"): _c("Stasioner", "Stationary") if kp[1] >= 0.05 else _c("Tidak stasioner", "Non-stationary"),
            }
        )
    except Exception as e:  # pragma: no cover
        rows.append({_c("Uji", "Test"): "KPSS", _c("Kesimpulan", "Conclusion"): f"gagal: {e}"})
    return pd.DataFrame(rows)


def ts_descriptive(s: pd.Series) -> pd.DataFrame:
    v = pd.Series(s).dropna()
    diff = v.diff().dropna()
    rows = [
        (_c("Jumlah titik", "Number of points"), f"{len(v):,}"),
        (_c("Rata-rata", "Mean"), f"{v.mean():.4f}"),
        (_c("Std Dev", "Std Dev"), f"{v.std():.4f}"),
        (_c("Minimum", "Minimum"), f"{v.min():.4f}"),
        (_c("Maksimum", "Maximum"), f"{v.max():.4f}"),
        (_c("Kemencengan", "Skewness"), f"{v.skew():.4f}"),
        (_c("Keruncingan", "Kurtosis"), f"{v.kurtosis():.4f}"),
        (_c("Koef. variasi (%)", "CV (%)"), f"{v.std() / v.mean() * 100:.2f}" if v.mean() else "-"),
        (_c("Rata-rata perubahan", "Mean change"), f"{diff.mean():.4f}" if len(diff) else "-"),
        (_c("Std perubahan", "Std of change"), f"{diff.std():.4f}" if len(diff) else "-"),
        (
            _c("Pertumbuhan total (%)", "Total growth (%)"),
            f"{(v.iloc[-1] / v.iloc[0] - 1) * 100:.2f}" if len(v) > 1 and v.iloc[0] != 0 else "-",
        ),
        (_c("Autokorelasi lag-1", "Lag-1 autocorrelation"), f"{v.autocorr(1):.4f}" if len(v) > 2 else "-"),
    ]
    return pd.DataFrame(rows, columns=[_c("Metrik", "Metric"), _c("Nilai", "Value")])


def ljung_box(resid, lags: int = 10) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = pd.Series(resid).dropna()
    lags = int(min(lags, max(1, len(r) // 5)))
    lb = acorr_ljungbox(r, lags=[lags], return_df=True)
    p = float(lb["lb_pvalue"].iloc[0])
    return pd.DataFrame(
        [
            {
                _c("Uji", "Test"): f"Ljung-Box (lag={lags})",
                _c("Statistik", "Statistic"): round(float(lb["lb_stat"].iloc[0]), 4),
                "p-value": round(p, 6),
                _c("Kesimpulan", "Conclusion"): (
                    _c("Residual acak (model memadai)", "Residuals are white noise (model adequate)")
                    if p >= 0.05
                    else _c("Masih ada autokorelasi (model belum memadai)", "Autocorrelation remains (model inadequate)")
                ),
            }
        ]
    )


def residual_diagnostics(resid) -> pd.DataFrame:
    r = pd.Series(resid).dropna()
    rows = []
    try:
        w, p = sps.shapiro(r) if len(r) <= 5000 else (np.nan, np.nan)
        rows.append(
            {
                _c("Uji", "Test"): _c("Normalitas residual (Shapiro-Wilk)", "Residual normality (Shapiro-Wilk)"),
                _c("Statistik", "Statistic"): round(float(w), 4) if np.isfinite(w) else "-",
                "p-value": round(float(p), 6) if np.isfinite(p) else "-",
                _c("Kesimpulan", "Conclusion"): (
                    _c("Residual normal", "Residuals normal") if (np.isfinite(p) and p >= 0.05) else _c("Residual tidak normal", "Residuals not normal")
                ),
            }
        )
    except Exception:
        pass
    try:
        jb, pjb = sps.jarque_bera(r)[:2]
        rows.append(
            {
                _c("Uji", "Test"): "Jarque-Bera",
                _c("Statistik", "Statistic"): round(float(jb), 4),
                "p-value": round(float(pjb), 6),
                _c("Kesimpulan", "Conclusion"): _c("Normal", "Normal") if pjb >= 0.05 else _c("Tidak normal", "Not normal"),
            }
        )
    except Exception:
        pass
    rows.append(
        {
            _c("Uji", "Test"): _c("Rata-rata residual", "Mean of residuals"),
            _c("Statistik", "Statistic"): round(float(r.mean()), 6),
            "p-value": "-",
            _c("Kesimpulan", "Conclusion"): (
                _c("Mendekati nol (baik)", "Close to zero (good)") if abs(r.mean()) < 0.1 * (r.std() + 1e-12) else _c("Ada bias", "Biased")
            ),
        }
    )
    return pd.DataFrame(rows)


def durbin_watson_stat(resid) -> float:
    from statsmodels.stats.stattools import durbin_watson

    return float(durbin_watson(pd.Series(resid).dropna()))
