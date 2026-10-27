"""
WebUI Data Mining & Statistik
=============================
Aplikasi Streamlit untuk klasifikasi, regresi, klastering,
association rule mining, dan time series (ARIMA/SARIMA).

Menjalankan:  streamlit run app.py
"""

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Data Mining & Statistics WebUI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import data_io as IO  # noqa: E402
from core import genui as GEN  # noqa: E402
from core import templates as T  # noqa: E402
from core import ui as U  # noqa: E402
from core.i18n import LANGS, L, PROCESSES, process_desc, process_label  # noqa: E402
from core.theme import APP_CSS  # noqa: E402
from modules import association, classification, clustering, regression, timeseries  # noqa: E402

RUNNERS = {
    "classification": classification,
    "regression": regression,
    "clustering": clustering,
    "association": association,
    "timeseries": timeseries,
}

st.markdown(APP_CSS, unsafe_allow_html=True)
st.session_state.setdefault("lang", "id")


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 📊 Data Mining UI")
    lang = st.radio("Bahasa / Language", list(LANGS), horizontal=True,
                    format_func=lambda k: LANGS[k], key="lang")
    st.markdown("---")

    proc = st.radio(L("Pilih proses analisis", "Choose an analysis process"),
                    list(PROCESSES), key="process", format_func=process_label)
    st.caption(process_desc(proc))
    st.markdown("---")

    st.markdown("**" + L("Sumber data", "Data source") + "**")
    SOURCES = {
        "upload": (L("Unggah file", "Upload file"),
                   L("Unggah berkas Excel/CSV milik Anda.", "Upload your own Excel/CSV file.")),
        "sample": (L("Gunakan data contoh", "Use sample data"),
                   L("Data bawaan siap pakai untuk mencoba.", "Ready-made built-in data to try.")),
        "generate": (L("Bangkitkan data acak", "Generate random data"),
                     L("Buat data sintetis sesuai parameter yang Anda tentukan.",
                       "Create synthetic data from parameters you define.")),
    }
    source = st.radio("", list(SOURCES), key="source", label_visibility="collapsed",
                      format_func=lambda k: SOURCES[k][0])
    st.caption(SOURCES[source][1])
    st.markdown("---")

    st.markdown("**" + L("Pembersihan data", "Data cleaning") + "**")
    drop_dupes = st.checkbox(L("Hapus baris duplikat", "Drop duplicate rows"), False, key="cl_dup")
    miss_label = {
        "none": L("Biarkan apa adanya", "Leave as is"),
        "mean_mode": L("Isi: rata-rata / modus", "Impute: mean / mode"),
        "median_mode": L("Isi: median / modus", "Impute: median / mode"),
        "drop": L("Hapus baris yang kosong", "Drop rows with missing values"),
    }
    missing = st.selectbox(L("Penanganan nilai kosong", "Missing value handling"),
                           list(miss_label), format_func=lambda k: miss_label[k], key="cl_miss")
    st.markdown("---")
    st.caption(L(
        "Semua tabel hasil dapat diunduh sebagai satu berkas Excel multi-sheet, "
        "dan seluruh grafik dapat diunduh sebagai PNG (satuan atau ZIP).",
        "All result tables can be downloaded as a single multi-sheet Excel file, "
        "and every chart can be downloaded as PNG (individually or as a ZIP).",
    ))


# ---------------------------------------------------------------- header
st.markdown(f"# {L('WebUI Data Mining & Statistik', 'Data Mining & Statistics WebUI')}")
st.markdown(
    f'<p class="dm-caption">{L("Analisis statistik lengkap, evaluasi model, visualisasi, dan ekspor hasil ke Excel/PNG — untuk klasifikasi, regresi, klastering, association rule mining, dan peramalan deret waktu.", "Complete statistical analysis, model evaluation, visualization, and export to Excel/PNG — for classification, regression, clustering, association rule mining, and time series forecasting.")}</p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# -------------------------------------------------- langkah 1: template
st.markdown(f"## {L('Langkah 1 — Template Data', 'Step 1 — Data Template')}")
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**{process_label(proc)}** — {process_desc(proc)}")
    st.table(T.guide_df(proc).set_index(T.guide_df(proc).columns[0]))
with c2:
    st.markdown("**" + L("Unduh template", "Download template") + "**")
    st.caption(L(
        "Template berisi contoh data siap pakai beserta lembar petunjuk. "
        "Ganti isinya dengan data Anda, pertahankan strukturnya.",
        "The template contains ready-to-use sample data plus a guide sheet. "
        "Replace the contents with your own data and keep the structure.",
    ))
    st.download_button(
        L("Template Excel (.xlsx)", "Excel template (.xlsx)"),
        data=T.template_xlsx_bytes(proc),
        file_name=f"template_{proc}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        type="primary",
        key=f"tpl_xlsx_{proc}",
    )
    st.download_button(
        L("Template CSV (.csv)", "CSV template (.csv)"),
        data=T.template_csv_bytes(proc),
        file_name=f"template_{proc}.csv",
        mime="text/csv",
        width="stretch",
        key=f"tpl_csv_{proc}",
    )
    with st.expander(L("Pratinjau template", "Preview template")):
        st.dataframe(T.build_template(proc).head(8), width="stretch")

st.markdown("---")

# ------------------------------------------------------ langkah 2: data
st.markdown(f"## {L('Langkah 2 — Muat Data', 'Step 2 — Load Data')}")
df = None
if source == "upload":
    up = st.file_uploader(
        L("Unggah file Excel (.xlsx/.xls) atau CSV", "Upload an Excel (.xlsx/.xls) or CSV file"),
        type=["xlsx", "xls", "xlsm", "csv"], key=f"up_{proc}")
    if up is not None:
        sheets = IO.excel_sheet_names(up)
        sheet = None
        if len(sheets) > 1:
            default = "Data" if "Data" in sheets else sheets[0]
            sheet = st.selectbox(L("Pilih sheet", "Choose sheet"), sheets,
                                 index=sheets.index(default), key=f"sheet_{proc}")
        try:
            df = IO.read_table(up, sheet)
        except Exception as e:
            st.error(L(f"Gagal membaca file: {e}", f"Failed to read the file: {e}"))
    else:
        st.info(L("Belum ada file. Unggah file Anda, atau pilih 'Gunakan data contoh' / "
                  "'Bangkitkan data acak' di sidebar.",
                  "No file yet. Upload one, or pick 'Use sample data' / "
                  "'Generate random data' in the sidebar."))
elif source == "generate":
    df = GEN.render(proc)
else:
    df = T.build_template(proc)
    st.info(L(f"Menggunakan data contoh bawaan untuk proses {process_label(proc)} "
              f"({len(df)} baris, {df.shape[1]} kolom).",
              f"Using the built-in sample data for {process_label(proc)} "
              f"({len(df)} rows, {df.shape[1]} columns)."))

if df is not None and len(df):
    df_clean, report = IO.basic_clean(df, drop_dupes, missing)
    if report and any(v for _, v in report):
        msgs = [f"{k.split(' | ')[0] if lang == 'id' else k.split(' | ')[-1]}: {v}"
                for k, v in report if v]
        if msgs:
            st.caption(" • ".join(msgs))
    df = df_clean

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(L("Baris", "Rows"), f"{len(df):,}")
    m2.metric(L("Kolom", "Columns"), f"{df.shape[1]:,}")
    m3.metric(L("Sel kosong", "Missing cells"), f"{int(df.isna().sum().sum()):,}")
    m4.metric(L("Duplikat", "Duplicates"), f"{int(df.duplicated().sum()):,}")

    holder = (st.expander(L("Pratinjau data siap analisis", "Preview of the analysis-ready data"))
              if source == "generate" else st.container())
    with holder:
        t1, t2 = st.tabs([L("Pratinjau data", "Data preview"), L("Struktur kolom", "Column structure")])
        with t1:
            st.dataframe(IO.safe_sample(df), width="stretch", height=330)
        with t2:
            st.dataframe(IO.infer_type_table(df, lang == "id"), width="stretch", height=330)

    st.markdown("---")

    # -------------------------------------------- langkah 3: konfigurasi
    st.markdown(f"## {L('Langkah 3 — Konfigurasi & Jalankan', 'Step 3 — Configure & Run')}")
    mod = RUNNERS[proc]
    cfg = mod.config(df)

    run_key = f"run_{proc}"
    st.session_state.setdefault(run_key, False)
    b1, b2 = st.columns([1, 4])
    if b1.button("▶ " + L("Jalankan Analisis", "Run Analysis"), type="primary",
                 width="stretch", disabled=cfg is None, key=f"btn_{proc}"):
        st.session_state[run_key] = True
    if b2.button(L("Bersihkan hasil", "Clear results"), width="content", key=f"clr_{proc}"):
        st.session_state[run_key] = False

    if st.session_state[run_key] and cfg is not None:
        st.markdown("---")
        st.markdown(f"## {L('Hasil Analisis', 'Analysis Results')} — {process_label(proc)}")
        with st.spinner(L("Memproses...", "Processing...")):
            try:
                mod.run(df, cfg)
            except Exception as e:
                st.error(L(f"Terjadi kesalahan saat analisis: {e}",
                           f"An error occurred during the analysis: {e}"))
                st.exception(e)
elif df is not None:
    st.warning(L("File terbaca tetapi kosong.", "The file was read but is empty."))

st.markdown("---")
st.caption(L(
    "WebUI Data Mining & Statistik • dibangun dengan Streamlit, scikit-learn, statsmodels, dan mlxtend.",
    "Data Mining & Statistics WebUI • built with Streamlit, scikit-learn, statsmodels, and mlxtend.",
))
