"""Komponen UI Streamlit yang dipakai bersama oleh semua modul proses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import stats as S
from . import viz as V
from .export import fig_to_png, figs_to_zip, stamp, tables_to_excel
from .i18n import L


def note(text: str):
    st.markdown(f'<div class="dm-note">{text}</div>', unsafe_allow_html=True)


def warn(text: str):
    st.markdown(f'<div class="dm-warn">{text}</div>', unsafe_allow_html=True)


def section(title: str, desc: str | None = None):
    st.markdown(f"### {title}")
    if desc:
        st.markdown(f'<p class="dm-caption">{desc}</p>', unsafe_allow_html=True)


def display_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan tipe kolom campuran menjadi teks.

    Tabel ringkasan sering memuat angka dan teks ('-') dalam satu kolom; Arrow
    menolak kolom seperti itu, jadi kolomnya diubah menjadi teks khusus untuk
    ditampilkan (data asli yang diekspor ke Excel tidak diubah).
    """
    out = df
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) \
                or pd.api.types.is_bool_dtype(s):
            continue
        vals = s.dropna()
        if len(vals) == 0:
            continue
        kinds = {type(v) for v in vals.head(500)}
        if len(kinds) > 1 or not all(k is str for k in kinds):
            if out is df:
                out = df.copy()
            out[c] = s.apply(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v))
                             else str(v))
    return out


def show_table(df: pd.DataFrame, height: int | None = None, hide_index: bool = True):
    if df is None or len(df) == 0:
        st.caption(L("(tidak ada data)", "(no data)"))
        return
    kwargs = {"width": "stretch", "hide_index": hide_index}
    if height:
        kwargs["height"] = int(height)
    st.dataframe(display_safe(df), **kwargs)


def show_fig(name: str, fig, key: str):
    if fig is None:
        return
    st.pyplot(fig, width="stretch")
    st.download_button(
        L("Unduh PNG", "Download PNG"),
        data=fig_to_png(fig),
        file_name=f"{key}.png",
        mime="image/png",
        key=f"png_{key}",
        width="content",
    )


def show_figs(figs: dict, prefix: str = "fig"):
    for i, (name, fig) in enumerate(figs.items()):
        if fig is None:
            continue
        st.markdown(f"**{name}**")
        show_fig(name, fig, f"{prefix}_{i}")
        st.markdown("---")


def export_bar(tables: dict, figs: dict, prefix: str, meta: dict | None = None):
    st.markdown("### " + L("Ekspor Hasil", "Export Results"))
    c1, c2 = st.columns(2)
    with c1:
        try:
            xls = tables_to_excel(tables, meta)
            st.download_button(
                L("Unduh semua tabel (.xlsx)", "Download all tables (.xlsx)"),
                data=xls,
                file_name=stamp(prefix, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                type="primary",
                key=f"xls_{prefix}",
            )
        except Exception as e:
            st.error(L(f"Gagal membuat Excel: {e}", f"Failed to build Excel: {e}"))
    with c2:
        live = {k: v for k, v in figs.items() if v is not None}
        if live:
            st.download_button(
                L(f"Unduh semua grafik ({len(live)} PNG, .zip)", f"Download all charts ({len(live)} PNG, .zip)"),
                data=figs_to_zip(live),
                file_name=stamp(prefix + "_grafik", "zip"),
                mime="application/zip",
                width="stretch",
                key=f"zip_{prefix}",
            )
    st.caption(
        L(
            "File Excel berisi satu sheet per tabel plus sheet 'Informasi'. "
            "File ZIP berisi seluruh grafik beresolusi 200 dpi siap dipakai di laporan.",
            "The Excel file has one sheet per table plus an 'Informasi' sheet. "
            "The ZIP contains all charts at 200 dpi, ready for reports.",
        )
    )


def metric_row(items: list[tuple[str, str]], per_row: int = 4):
    for i in range(0, len(items), per_row):
        chunk = items[i : i + per_row]
        cols = st.columns(len(chunk))
        for col, (label, value) in zip(cols, chunk):
            col.metric(label, value)


# ------------------------------------------------- blok statistik umum
def descriptive_block(df: pd.DataFrame, prefix: str, corr_method: str = "pearson"):
    """Statistik deskriptif + uji asumsi umum. Mengembalikan (tables, figs)."""
    tables: dict[str, pd.DataFrame] = {}
    figs: dict[str, object] = {}

    ov = S.overview(df)
    tables[L("Ringkasan Data", "Data Overview")] = ov

    num_df = df.select_dtypes(include=[np.number])

    tabs = st.tabs([
        L("Ringkasan", "Overview"),
        L("Deskriptif Numerik", "Numeric Descriptives"),
        L("Deskriptif Kategorik", "Categorical Descriptives"),
        L("Korelasi", "Correlation"),
        L("Uji Normalitas", "Normality Tests"),
        L("Outlier & Multikolinearitas", "Outliers & Multicollinearity"),
        L("Grafik Eksplorasi", "Exploratory Charts"),
    ])

    with tabs[0]:
        metric_row([
            (L("Baris", "Rows"), f"{len(df):,}"),
            (L("Kolom", "Columns"), f"{df.shape[1]:,}"),
            (L("Sel kosong", "Missing cells"), f"{int(df.isna().sum().sum()):,}"),
            (L("Duplikat", "Duplicates"), f"{int(df.duplicated().sum()):,}"),
        ])
        show_table(ov)
        miss = S.missing_table(df)
        tables[L("Nilai Kosong", "Missing Values")] = miss
        st.markdown("**" + L("Nilai kosong per kolom", "Missing values per column") + "**")
        show_table(miss, height=min(400, 42 + 35 * len(miss)))

    with tabs[1]:
        dn = S.descriptive_numeric(df)
        tables[L("Deskriptif Numerik", "Numeric Descriptives")] = dn
        if dn.empty:
            st.caption(L("Tidak ada kolom numerik.", "No numeric columns."))
        else:
            note(L(
                "Kemencengan (skewness) |x| &gt; 1 menandakan distribusi sangat miring; "
                "koefisien variasi &gt; 30% menandakan data sangat beragam relatif terhadap rata-ratanya.",
                "Skewness |x| &gt; 1 indicates a strongly skewed distribution; "
                "a coefficient of variation &gt; 30% means the data is highly variable relative to its mean.",
            ))
            show_table(dn)

    with tabs[2]:
        dc = S.descriptive_categorical(df)
        tables[L("Deskriptif Kategorik", "Categorical Descriptives")] = dc
        if dc.empty:
            st.caption(L("Tidak ada kolom kategorik.", "No categorical columns."))
        else:
            show_table(dc)
            cat_cols = list(dc[dc.columns[0]])
            pick = st.selectbox(L("Tabel frekuensi untuk kolom", "Frequency table for column"),
                                cat_cols, key=f"{prefix}_freqcol")
            ft = S.frequency_table(df, pick)
            tables[f"Frekuensi_{pick}"[:31]] = ft
            show_table(ft)
            figs[L(f"Frekuensi kategori: {pick}", f"Category frequency: {pick}")] = V.fig_category_bar(
                ft, ft.columns[0], ft.columns[1],
                L(f"Frekuensi Kategori - {pick}", f"Category Frequency - {pick}")
            )

    with tabs[3]:
        if num_df.shape[1] >= 2:
            corr = S.correlation(df, corr_method)
            pairs = S.correlation_pairs(df, corr_method)
            tables[L("Matriks Korelasi", "Correlation Matrix")] = corr.reset_index().rename(
                columns={"index": L("Variabel", "Variable")})
            tables[L("Korelasi Pasangan", "Pairwise Correlation")] = pairs
            note(L(
                f"Metode: <b>{corr_method}</b>. Nilai r &ge; 0,8 antar fitur mengindikasikan "
                "redundansi (kandidat multikolinearitas).",
                f"Method: <b>{corr_method}</b>. |r| &ge; 0.8 between features indicates "
                "redundancy (multicollinearity candidate).",
            ))
            show_table(pairs)
            figs[L("Matriks korelasi", "Correlation matrix")] = V.fig_corr_heatmap(corr)
            st.pyplot(figs[L("Matriks korelasi", "Correlation matrix")], width="stretch")
        else:
            st.caption(L("Butuh minimal 2 kolom numerik.", "Needs at least 2 numeric columns."))

    with tabs[4]:
        nt = S.normality_tests(df)
        tables[L("Uji Normalitas", "Normality Tests")] = nt
        if nt.empty:
            st.caption(L("Tidak cukup data numerik untuk uji normalitas.",
                         "Not enough numeric data for normality tests."))
        else:
            note(L(
                "H0: data berdistribusi normal. Jika p &lt; 0,05 maka H0 ditolak "
                "(data tidak normal) sehingga uji non-parametrik lebih tepat.",
                "H0: the data is normally distributed. If p &lt; 0.05, H0 is rejected "
                "(not normal) and non-parametric tests are more appropriate.",
            ))
            show_table(nt)

    with tabs[5]:
        ot = S.outlier_table(df)
        tables[L("Deteksi Outlier", "Outlier Detection")] = ot
        st.markdown("**" + L("Deteksi outlier (IQR & skor-z)", "Outlier detection (IQR & z-score)") + "**")
        show_table(ot)
        if num_df.shape[1] >= 2:
            vt = S.vif_table(num_df)
            tables["VIF"] = vt
            st.markdown("**" + L("Multikolinearitas (VIF)", "Multicollinearity (VIF)") + "**")
            note(L(
                "VIF &lt; 5 aman, 5-10 perlu diwaspadai, &gt; 10 menandakan multikolinearitas tinggi.",
                "VIF &lt; 5 is fine, 5-10 warrants caution, &gt; 10 indicates high multicollinearity.",
            ))
            show_table(vt)

    with tabs[6]:
        f_missing = V.fig_missing(df)
        f_hist = V.fig_hist_grid(df)
        f_box = V.fig_box_grid(df)
        figs[L("Peta nilai kosong", "Missing value map")] = f_missing
        figs[L("Distribusi numerik", "Numeric distributions")] = f_hist
        figs[L("Boxplot outlier", "Outlier boxplot")] = f_box
        for i, (nm, fg) in enumerate([
            (L("Nilai kosong", "Missing values"), f_missing),
            (L("Distribusi variabel numerik", "Numeric variable distributions"), f_hist),
            (L("Boxplot terstandardisasi", "Standardized boxplot"), f_box),
        ]):
            if fg is not None:
                st.markdown(f"**{nm}**")
                show_fig(nm, fg, f"{prefix}_expl_{i}")

    return tables, figs
