"""Antarmuka Streamlit untuk pembangkit data acak + ekspor/impor konfigurasi JSON."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from . import generator as G
from . import ui as U
from . import viz as V
from .export import stamp, tables_to_excel
from .i18n import L

TIPE_OPSI = ["numerik", "kategorik"]
DIST_OPSI = ["normal", "uniform", "lognormal", "eksponensial", "integer", "poisson", "-"]


# --------------------------------------------------------------- state
def _kcfg(proc):
    return f"gen_cfg_{proc}"


def _kdf(proc):
    return f"gen_df_{proc}"


def _current_cfg(proc) -> dict:
    cfg = st.session_state.get(_kcfg(proc))
    if not cfg:
        cfg = G.default_config(proc)
        st.session_state[_kcfg(proc)] = cfg
    return G.normalize_config(cfg)


def _reset_widgets(proc):
    """Hapus state widget generator agar nilai dari konfigurasi baru dipakai."""
    for k in [k for k in list(st.session_state) if k.startswith(f"g_{proc}_")]:
        del st.session_state[k]


def _w(proc, name):
    return f"g_{proc}_{name}"


# ------------------------------------------------------------ impor JSON
def _import_panel(proc: str):
    up = st.file_uploader(
        L("Impor konfigurasi (.json)", "Import configuration (.json)"),
        type=["json"], key=f"gen_up_{proc}",
        help=L("Berkas JSON hasil ekspor dari aplikasi ini.",
               "A JSON file previously exported from this app."))
    if up is None:
        return
    raw = up.getvalue()
    h = hashlib.md5(raw).hexdigest()
    if st.session_state.get(f"gen_hash_{proc}") == h:
        return
    try:
        cfg = G.json_to_config(raw)
    except Exception as e:
        st.error(L(f"Konfigurasi tidak valid: {e}", f"Invalid configuration: {e}"))
        return
    if cfg["process"] != proc:
        st.error(L(
            f"Konfigurasi ini untuk proses '{cfg['process']}', sedangkan proses aktif '{proc}'. "
            "Pilih proses yang sesuai di sidebar terlebih dahulu.",
            f"This configuration targets process '{cfg['process']}' but the active process is '{proc}'. "
            "Switch process in the sidebar first."))
        return
    st.session_state[f"gen_hash_{proc}"] = h
    st.session_state[_kcfg(proc)] = cfg
    st.session_state.pop(_kdf(proc), None)
    _reset_widgets(proc)
    st.success(L("Konfigurasi berhasil diimpor.", "Configuration imported successfully."))
    st.rerun()


# ------------------------------------------------------ editor spesifikasi
SPEC_HELP_ID = {
    "nama": "Nama kolom pada dataset",
    "tipe": "numerik atau kategorik",
    "distribusi": "Bentuk sebaran nilai numerik",
    "min": "Batas bawah rentang kolom",
    "max": "Batas atas rentang kolom",
    "mean": "Rata-rata (dipakai normal/lognormal/eksponensial/poisson)",
    "std": "Simpangan baku (dipakai normal/lognormal)",
    "desimal": "Jumlah angka di belakang koma (0 = bilangan bulat)",
    "kategori": "Daftar kategori dipisah koma, mis. Rendah, Sedang, Tinggi",
    "bobot_kategori": "Proporsi tiap kategori dipisah koma, mis. 0.5, 0.3, 0.2",
    "persen_kosong": "Persentase sel yang dikosongkan (mengurangi density data)",
    "persen_outlier": "Persentase nilai yang dilempar ke luar rentang",
    "pengaruh": "Bobot pengaruh kolom terhadap target (0 = tidak berpengaruh)",
}
SPEC_HELP_EN = {
    "nama": "Column name in the dataset",
    "tipe": "numeric or categorical",
    "distribusi": "Shape of the numeric distribution",
    "min": "Lower bound of the column range",
    "max": "Upper bound of the column range",
    "mean": "Mean (used by normal/lognormal/exponential/poisson)",
    "std": "Standard deviation (used by normal/lognormal)",
    "desimal": "Decimal places (0 = integer)",
    "kategori": "Comma-separated categories, e.g. Low, Medium, High",
    "bobot_kategori": "Comma-separated proportions, e.g. 0.5, 0.3, 0.2",
    "persen_kosong": "Percentage of cells set to empty (lowers data density)",
    "persen_outlier": "Percentage of values pushed outside the range",
    "pengaruh": "Weight of this column's influence on the target (0 = none)",
}


def _spec_editor(proc: str, cols: list[dict]) -> list[dict]:
    help_map = SPEC_HELP_ID if L("id", "en") == "id" else SPEC_HELP_EN
    frame = pd.DataFrame([{k: c.get(k) for k in G.SPEC_FIELDS} for c in cols])
    cfgcols = {
        "nama": st.column_config.TextColumn(L("Nama kolom", "Column name"), help=help_map["nama"], width="medium"),
        "tipe": st.column_config.SelectboxColumn(L("Tipe", "Type"), options=TIPE_OPSI, help=help_map["tipe"]),
        "distribusi": st.column_config.SelectboxColumn(L("Distribusi", "Distribution"), options=DIST_OPSI,
                                                       help=help_map["distribusi"]),
        "min": st.column_config.NumberColumn("Min", help=help_map["min"], format="%.4g"),
        "max": st.column_config.NumberColumn("Max", help=help_map["max"], format="%.4g"),
        "mean": st.column_config.NumberColumn(L("Rata-rata", "Mean"), help=help_map["mean"], format="%.4g"),
        "std": st.column_config.NumberColumn("Std", help=help_map["std"], format="%.4g"),
        "desimal": st.column_config.NumberColumn(L("Desimal", "Decimals"), min_value=0, max_value=6, step=1,
                                                 help=help_map["desimal"]),
        "kategori": st.column_config.TextColumn(L("Kategori", "Categories"), help=help_map["kategori"], width="medium"),
        "bobot_kategori": st.column_config.TextColumn(L("Bobot kategori", "Category weights"),
                                                      help=help_map["bobot_kategori"]),
        "persen_kosong": st.column_config.NumberColumn(L("% kosong", "% missing"), min_value=0.0, max_value=90.0,
                                                       step=1.0, help=help_map["persen_kosong"]),
        "persen_outlier": st.column_config.NumberColumn(L("% outlier", "% outlier"), min_value=0.0, max_value=90.0,
                                                        step=1.0, help=help_map["persen_outlier"]),
        "pengaruh": st.column_config.NumberColumn(L("Pengaruh target", "Target influence"), min_value=0.0,
                                                  max_value=5.0, step=0.1, help=help_map["pengaruh"]),
    }
    edited = st.data_editor(frame, column_config=cfgcols, num_rows="dynamic",
                            width="stretch", key=_w(proc, "spec"),
                            hide_index=True)
    out = []
    for i, row in enumerate(edited.to_dict("records")):
        if not str(row.get("nama") or "").strip():
            continue
        out.append(G.normalize_column(row, i + 1))
    return out or cols


def _rebuild_columns(proc: str, cfg: dict):
    """Baris kontrol jumlah kolom -> membentuk ulang tabel spesifikasi."""
    c1, c2, c3 = st.columns([1, 1, 1.4])
    n_num = c1.number_input(L("Jumlah kolom numerik", "Numeric column count"), 0, 60,
                            int(sum(1 for c in cfg["params"]["kolom"] if c["tipe"] == "numerik")),
                            key=_w(proc, "n_num"))
    n_cat = c2.number_input(L("Jumlah kolom kategorik", "Categorical column count"), 0, 30,
                            int(sum(1 for c in cfg["params"]["kolom"] if c["tipe"] == "kategorik")),
                            key=_w(proc, "n_cat"))
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button(L("Bentuk ulang tabel kolom", "Rebuild column table"),
                     key=_w(proc, "rebuild"), width="stretch"):
            cfg["params"]["kolom"] = G.default_columns(int(n_num), int(n_cat))
            st.session_state[_kcfg(proc)] = cfg
            st.session_state.pop(_w(proc, "spec"), None)
            st.rerun()
    st.caption(L(
        "Ubah jumlah kolom lalu klik tombol untuk membentuk ulang tabel. Tabel di bawah juga bisa "
        "diedit langsung — tambah/hapus baris memakai ikon di ujung tabel, dan setiap kolom boleh "
        "punya rentang, distribusi, serta tingkat kekosongan sendiri.",
        "Change the counts then click the button to rebuild the table. The table below is directly "
        "editable too — add/remove rows with the icons at its edge, and every column may have its own "
        "range, distribution, and missing rate.",
    ))


def _common_block(proc: str, cfg: dict, params: dict, row_label):
    c1, c2, c3, c4 = st.columns(4)
    params["jumlah_baris"] = int(c1.number_input(row_label, 20, 200_000,
                                                 int(cfg["params"]["jumlah_baris"]), step=50,
                                                 key=_w(proc, "rows")))
    params["persen_kosong_global"] = float(c2.slider(
        L("Nilai kosong global (%)", "Global missing (%)"), 0.0, 50.0,
        float(cfg["params"]["persen_kosong_global"]), 0.5, key=_w(proc, "missglob"),
        help=L("Ditambahkan ke % kosong tiap kolom. Density data = 100% - nilai ini.",
               "Added on top of each column's missing rate. Data density = 100% - this value.")))
    params["persen_duplikat"] = float(c3.slider(
        L("Baris duplikat (%)", "Duplicate rows (%)"), 0.0, 30.0,
        float(cfg["params"]["persen_duplikat"]), 0.5, key=_w(proc, "dup")))
    params["kolom_id"] = bool(c4.checkbox(L("Sertakan kolom ID", "Include ID column"),
                                          bool(cfg["params"]["kolom_id"]), key=_w(proc, "colid")))
    params["prefix_id"] = cfg["params"].get("prefix_id", "ID")


# =============================================================== form/proses
def _form_classification(proc, cfg) -> dict:
    p = dict(cfg["params"])
    _common_block(proc, cfg, p, L("Jumlah baris", "Number of rows"))

    st.markdown("**" + L("Target & persebaran label", "Target & label distribution") + "**")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    p["nama_target"] = c1.text_input(L("Nama kolom target", "Target column name"),
                                     str(cfg["params"]["nama_target"]), key=_w(proc, "tname"))
    p["keterpisahan"] = float(c2.slider(
        L("Keterpisahan kelas", "Class separability"), 0.05, 0.98,
        float(cfg["params"]["keterpisahan"]), 0.01, key=_w(proc, "sep"),
        help=L("Mendekati 1 = kelas mudah dipisahkan model; mendekati 0 = label hampir acak.",
               "Near 1 = classes are easy to separate; near 0 = labels are nearly random.")))
    p["persen_label_bising"] = float(c3.slider(
        L("Label bising (%)", "Label noise (%)"), 0.0, 40.0,
        float(cfg["params"]["persen_label_bising"]), 0.5, key=_w(proc, "lnoise"),
        help=L("Persentase label yang sengaja diacak (simulasi kesalahan anotasi).",
               "Percentage of labels deliberately randomized (annotation error simulation).")))

    kelas = list(cfg["params"]["kelas"])
    bobot = G._weights(cfg["params"].get("bobot_kelas", ""), len(kelas))
    kframe = pd.DataFrame({L("Nama kelas", "Class name"): kelas,
                           L("Proporsi", "Proportion"): np.round(bobot, 4)})
    edited = st.data_editor(
        kframe, num_rows="dynamic", width="stretch", hide_index=True, key=_w(proc, "kelas"),
        column_config={
            L("Nama kelas", "Class name"): st.column_config.TextColumn(width="medium"),
            L("Proporsi", "Proportion"): st.column_config.NumberColumn(
                min_value=0.0, max_value=1.0, step=0.01, format="%.4f",
                help=L("Proporsi tiap kelas; otomatis dinormalkan agar berjumlah 1.",
                       "Proportion per class; automatically normalized to sum to 1.")),
        })
    names = [str(v).strip() for v in edited[L("Nama kelas", "Class name")] if str(v).strip()]
    ws = [float(v) if pd.notna(v) else 0.0 for v in edited[L("Proporsi", "Proportion")]][: len(names)]
    if len(names) >= 2:
        p["kelas"] = names
        p["bobot_kelas"] = ", ".join(str(w) for w in ws) if sum(ws) > 0 else ""
    st.caption(L(f"Total proporsi = {sum(ws):.4f} (akan dinormalkan otomatis).",
                 f"Total proportion = {sum(ws):.4f} (auto-normalized)."))

    st.markdown("**" + L("Spesifikasi kolom fitur", "Feature column specification") + "**")
    _rebuild_columns(proc, cfg)
    p["kolom"] = _spec_editor(proc, cfg["params"]["kolom"])
    return p


def _form_regression(proc, cfg) -> dict:
    p = dict(cfg["params"])
    _common_block(proc, cfg, p, L("Jumlah baris", "Number of rows"))

    st.markdown("**" + L("Target numerik", "Numeric target") + "**")
    c1, c2, c3, c4 = st.columns(4)
    p["nama_target"] = c1.text_input(L("Nama kolom target", "Target column name"),
                                     str(cfg["params"]["nama_target"]), key=_w(proc, "tname"))
    p["target_mean"] = float(c2.number_input(L("Rata-rata target", "Target mean"),
                                             value=float(cfg["params"]["target_mean"]), key=_w(proc, "tmean")))
    p["target_std"] = float(c3.number_input(L("Std dev target", "Target std dev"), min_value=0.0001,
                                            value=float(cfg["params"]["target_std"]), key=_w(proc, "tstd")))
    p["target_desimal"] = int(c4.number_input(L("Desimal target", "Target decimals"), 0, 6,
                                              int(cfg["params"]["target_desimal"]), key=_w(proc, "tdec")))
    c5, c6, c7 = st.columns(3)
    pakai_batas = c5.checkbox(L("Batasi rentang target", "Clamp target range"),
                              cfg["params"].get("target_min") is not None, key=_w(proc, "tclamp"))
    if pakai_batas:
        p["target_min"] = float(c6.number_input(L("Target minimum", "Target minimum"),
                                                value=float(cfg["params"].get("target_min") or 0.0),
                                                key=_w(proc, "tmin")))
        p["target_max"] = float(c7.number_input(L("Target maksimum", "Target maximum"),
                                                value=float(cfg["params"].get("target_max") or 5000.0),
                                                key=_w(proc, "tmax")))
    else:
        p["target_min"] = p["target_max"] = None

    st.markdown("**" + L("Kekuatan & bentuk hubungan", "Relationship strength & shape") + "**")
    c8, c9, c10, c11 = st.columns(4)
    p["r2_target"] = float(c8.slider("R² " + L("yang diinginkan", "target"), 0.05, 0.98,
                                     float(cfg["params"]["r2_target"]), 0.01, key=_w(proc, "r2"),
                                     help=L("Proporsi keragaman target yang dapat dijelaskan fitur.",
                                            "Share of target variance explainable by the features.")))
    p["nonlinearitas"] = float(c9.slider(L("Nonlinearitas", "Nonlinearity"), 0.0, 1.0,
                                         float(cfg["params"]["nonlinearitas"]), 0.05, key=_w(proc, "nl"),
                                         help=L("0 = hubungan linear murni; 1 = sangat melengkung.",
                                                "0 = purely linear; 1 = strongly curved.")))
    p["heteroskedastisitas"] = float(c10.slider(
        L("Heteroskedastisitas", "Heteroscedasticity"), 0.0, 1.0,
        float(cfg["params"]["heteroskedastisitas"]), 0.05, key=_w(proc, "het"),
        help=L("Membuat ragam galat membesar seiring naiknya nilai prediktor.",
               "Makes error variance grow with the predictor value.")))
    p["target_persen_kosong"] = float(c11.slider(L("Target kosong (%)", "Target missing (%)"), 0.0, 30.0,
                                                 float(cfg["params"]["target_persen_kosong"]), 0.5,
                                                 key=_w(proc, "tmiss")))

    st.markdown("**" + L("Spesifikasi kolom fitur", "Feature column specification") + "**")
    _rebuild_columns(proc, cfg)
    p["kolom"] = _spec_editor(proc, cfg["params"]["kolom"])
    return p


def _form_clustering(proc, cfg) -> dict:
    p = dict(cfg["params"])
    _common_block(proc, cfg, p, L("Jumlah baris", "Number of rows"))

    st.markdown("**" + L("Struktur klaster", "Cluster structure") + "**")
    c1, c2, c3, c4 = st.columns(4)
    p["jumlah_klaster"] = int(c1.number_input(L("Jumlah klaster", "Number of clusters"), 1, 20,
                                              int(cfg["params"]["jumlah_klaster"]), key=_w(proc, "k")))
    p["pemisahan"] = float(c2.slider(L("Pemisahan antar klaster", "Cluster separation"), 0.3, 8.0,
                                     float(cfg["params"]["pemisahan"]), 0.1, key=_w(proc, "sep"),
                                     help=L("Jarak antar pusat klaster relatif terhadap sebaran anggotanya.",
                                            "Distance between cluster centers relative to member spread.")))
    p["kerapatan"] = float(c3.slider(L("Kerapatan klaster", "Cluster density"), 0.2, 4.0,
                                     float(cfg["params"]["kerapatan"]), 0.1, key=_w(proc, "dens"),
                                     help=L("Semakin tinggi, anggota klaster makin merapat ke pusatnya.",
                                            "Higher values pull members tighter around the centroid.")))
    p["persen_noise"] = float(c4.slider(L("Titik noise (%)", "Noise points (%)"), 0.0, 40.0,
                                        float(cfg["params"]["persen_noise"]), 0.5, key=_w(proc, "noise"),
                                        help=L("Titik acak di luar struktur klaster (uji DBSCAN).",
                                               "Random points outside the cluster structure (for DBSCAN).")))
    c5, c6, c7 = st.columns([1, 1.2, 1])
    p["bentuk"] = c5.selectbox(L("Bentuk klaster", "Cluster shape"), ["bulat", "lonjong"],
                               index=["bulat", "lonjong"].index(cfg["params"].get("bentuk", "bulat")),
                               key=_w(proc, "shape"))
    p["bobot_klaster"] = c6.text_input(
        L("Proporsi tiap klaster", "Proportion per cluster"), str(cfg["params"].get("bobot_klaster", "")),
        key=_w(proc, "wk"),
        help=L("Kosongkan untuk ukuran seimbang, atau isi mis. 0.5, 0.3, 0.2",
               "Leave empty for balanced sizes, or enter e.g. 0.5, 0.3, 0.2"))
    p["sertakan_label_asli"] = bool(c7.checkbox(
        L("Sertakan label klaster asli", "Include true cluster label"),
        bool(cfg["params"].get("sertakan_label_asli", False)), key=_w(proc, "truth"),
        help=L("Kolom 'klaster_asli' sebagai kunci jawaban. Kecualikan dari fitur saat menganalisis.",
               "Adds a 'klaster_asli' answer-key column. Exclude it from the features when analyzing.")))

    st.markdown("**" + L("Spesifikasi kolom fitur (hanya numerik yang dipakai)",
                         "Feature column specification (numeric only)") + "**")
    _rebuild_columns(proc, cfg)
    p["kolom"] = _spec_editor(proc, cfg["params"]["kolom"])
    return p


def _form_association(proc, cfg) -> dict:
    p = dict(cfg["params"])
    b = cfg["params"]
    st.markdown("**" + L("Volume & variasi transaksi", "Transaction volume & variety") + "**")
    c1, c2, c3, c4 = st.columns(4)
    p["jumlah_transaksi"] = int(c1.number_input(L("Jumlah transaksi", "Transactions"), 30, 200_000,
                                                int(b["jumlah_transaksi"]), step=50, key=_w(proc, "ntrx")))
    p["jumlah_item"] = int(c2.number_input(L("Jumlah item unik", "Unique items"), 3, 300,
                                           int(b["jumlah_item"]), key=_w(proc, "nitem")))
    p["rata_ukuran_keranjang"] = float(c3.slider(L("Rata-rata item/transaksi", "Avg items/transaction"),
                                                 1.0, 30.0, float(b["rata_ukuran_keranjang"]), 0.5,
                                                 key=_w(proc, "bmean")))
    p["std_ukuran_keranjang"] = float(c4.slider(L("Std ukuran keranjang", "Basket size std"), 0.1, 10.0,
                                                float(b["std_ukuran_keranjang"]), 0.1, key=_w(proc, "bstd")))
    c5, c6 = st.columns(2)
    p["kemiringan_popularitas"] = float(c5.slider(
        L("Kemiringan popularitas item", "Item popularity skew"), 0.0, 2.5,
        float(b["kemiringan_popularitas"]), 0.05, key=_w(proc, "skew"),
        help=L("0 = semua item sama populer; makin tinggi, makin sedikit item yang mendominasi.",
               "0 = all items equally popular; higher values make a few items dominate.")))
    p["persen_item_bising"] = float(c6.slider(L("Item acak/bising (%)", "Random noise items (%)"), 0.0, 100.0,
                                              float(b["persen_item_bising"]), 1.0, key=_w(proc, "inoise")))

    st.markdown("**" + L("Aturan yang ditanam", "Planted rules") + "**")
    c7, c8, c9 = st.columns(3)
    p["jumlah_aturan"] = int(c7.number_input(L("Jumlah aturan", "Number of rules"), 0, 30,
                                             int(b["jumlah_aturan"]), key=_w(proc, "nrule")))
    p["ukuran_antecedent"] = int(c8.number_input(L("Ukuran antecedent", "Antecedent size"), 1, 4,
                                                 int(b["ukuran_antecedent"]), key=_w(proc, "asize")))
    p["ukuran_consequent"] = int(c9.number_input(L("Ukuran consequent", "Consequent size"), 1, 3,
                                                 int(b["ukuran_consequent"]), key=_w(proc, "csize")))
    c10, c11 = st.columns(2)
    p["support_aturan"] = float(c10.slider(L("Support tiap aturan", "Support per rule"), 0.01, 0.6,
                                           float(b["support_aturan"]), 0.01, key=_w(proc, "rsup")))
    p["confidence_aturan"] = float(c11.slider(L("Confidence tiap aturan", "Confidence per rule"), 0.1, 1.0,
                                              float(b["confidence_aturan"]), 0.05, key=_w(proc, "rconf")))

    st.markdown("**" + L("Format keluaran", "Output format") + "**")
    fmt_opsi = ["long", "basket", "delimited"]
    fmt_label = {
        "long": L("Panjang (id_transaksi + item)", "Long (id_transaksi + item)"),
        "basket": L("Keranjang (item sebagai kolom 0/1)", "Basket (items as 0/1 columns)"),
        "delimited": L("Daftar item dipisah koma", "Comma-separated item list"),
    }
    c12, c13, c14 = st.columns([1.6, 1, 1])
    p["format"] = c12.selectbox(L("Format", "Format"), fmt_opsi,
                                index=fmt_opsi.index(b.get("format", "long")),
                                format_func=lambda k: fmt_label[k], key=_w(proc, "fmt"))
    if p["format"] == "long":
        p["kolom_tanggal"] = bool(c13.checkbox(L("Kolom tanggal", "Date column"),
                                               bool(b.get("kolom_tanggal", True)), key=_w(proc, "coldate")))
        p["kolom_jumlah"] = bool(c14.checkbox(L("Kolom jumlah", "Quantity column"),
                                              bool(b.get("kolom_jumlah", True)), key=_w(proc, "colqty")))
        c15, c16, c17 = st.columns(3)
        d0 = _to_date(b.get("tanggal_mulai", "2025-01-01"))
        p["tanggal_mulai"] = str(c15.date_input(L("Tanggal mulai", "Start date"), d0, key=_w(proc, "tgl")))
        p["rentang_hari"] = int(c16.number_input(L("Rentang hari transaksi", "Transaction day span"), 1, 3650,
                                                 int(b.get("rentang_hari", 90)), key=_w(proc, "span")))
        p["maks_jumlah"] = int(c17.number_input(L("Jumlah maksimum per item", "Max quantity per item"), 1, 50,
                                                int(b.get("maks_jumlah", 3)), key=_w(proc, "maxqty")))

    daftar = st.text_area(
        L("Daftar item khusus (opsional, dipisah koma)", "Custom item list (optional, comma-separated)"),
        str(b.get("daftar_item", "")), key=_w(proc, "items"), height=80,
        help=L("Kosongkan untuk memakai katalog bawaan. Jika kurang dari jumlah item, sisanya diisi otomatis.",
               "Leave empty to use the built-in catalog. If shorter than the item count, the rest is auto-filled."))
    p["daftar_item"] = daftar
    return p


def _to_date(v):
    try:
        return pd.Timestamp(str(v)).date()
    except Exception:
        return date(2025, 1, 1)


def _form_timeseries(proc, cfg) -> dict:
    p = dict(cfg["params"])
    b = cfg["params"]
    st.markdown("**" + L("Rentang & agregasi waktu", "Time range & aggregation") + "**")
    c1, c2, c3, c4 = st.columns(4)
    p["jumlah_periode"] = int(c1.number_input(L("Jumlah periode", "Number of periods"), 24, 20_000,
                                              int(b["jumlah_periode"]), step=12, key=_w(proc, "n")))
    freqs = list(G.FREQS)
    p["frekuensi"] = c2.selectbox(L("Frekuensi (agregasi)", "Frequency (aggregation)"), freqs,
                                  index=freqs.index(b.get("frekuensi", "MS")),
                                  format_func=lambda k: f"{L(*G.FREQ_LABEL[k])} ({k})", key=_w(proc, "freq"))
    p["tanggal_mulai"] = str(c3.date_input(L("Tanggal mulai", "Start date"),
                                           _to_date(b.get("tanggal_mulai", "2018-01-01")), key=_w(proc, "tgl")))
    p["catatan_per_periode"] = int(c4.number_input(
        L("Catatan per periode", "Records per period"), 1, 50, int(b.get("catatan_per_periode", 1)),
        key=_w(proc, "perper"),
        help=L("1 = satu baris per periode. Lebih dari 1 menghasilkan data mentah sub-periode yang "
               "harus diagregasi (uji fitur agregasi pada modul time series).",
               "1 = one row per period. More than 1 emits raw sub-period records that must be aggregated "
               "(exercises the aggregation option in the time series module).")))
    if p["catatan_per_periode"] > 1:
        p["persen_noise_subperiode"] = float(st.slider(
            L("Ragam antar catatan dalam satu periode (%)", "Within-period record variation (%)"),
            0.0, 50.0, float(b.get("persen_noise_subperiode", 5.0)), 0.5, key=_w(proc, "subnoise")))

    st.markdown("**" + L("Level & tren", "Level & trend") + "**")
    c5, c6, c7, c8 = st.columns(4)
    p["level_dasar"] = float(c5.number_input(L("Level dasar", "Base level"), value=float(b["level_dasar"]),
                                             key=_w(proc, "level")))
    tren_opsi = ["tidak ada", "linear", "eksponensial", "logistik"]
    p["tipe_tren"] = c6.selectbox(L("Tipe tren", "Trend type"), tren_opsi,
                                  index=tren_opsi.index(b.get("tipe_tren", "linear"))
                                  if b.get("tipe_tren") in tren_opsi else 1, key=_w(proc, "trend"))
    p["kekuatan_tren"] = float(c7.slider(L("Kekuatan tren", "Trend strength"), -2.0, 3.0,
                                         float(b["kekuatan_tren"]), 0.05, key=_w(proc, "tstr"),
                                         help=L("Perubahan total relatif terhadap level dasar (negatif = menurun).",
                                                "Total change relative to the base level (negative = declining).")))
    p["persen_noise"] = float(c8.slider(L("Derau acak (%)", "Random noise (%)"), 0.0, 50.0,
                                        float(b["persen_noise"]), 0.5, key=_w(proc, "noise")))

    st.markdown("**" + L("Pola musiman", "Seasonality") + "**")
    c9, c10, c11, c12 = st.columns(4)
    p["periode_musiman"] = int(c9.number_input(L("Periode musiman (m)", "Seasonal period (m)"), 0, 366,
                                               int(b["periode_musiman"]), key=_w(proc, "m")))
    p["amplitudo_musiman"] = float(c10.slider(L("Amplitudo musiman", "Seasonal amplitude"), 0.0, 1.0,
                                              float(b["amplitudo_musiman"]), 0.01, key=_w(proc, "amp"),
                                              help=L("Relatif terhadap level dasar.", "Relative to the base level.")))
    p["amplitudo_harmonik"] = float(c11.slider(L("Amplitudo harmonik ke-2", "2nd harmonic amplitude"), 0.0, 0.5,
                                               float(b["amplitudo_harmonik"]), 0.01, key=_w(proc, "amp2")))
    mode_opsi = ["aditif", "multiplikatif"]
    p["mode_musiman"] = c12.selectbox(L("Mode musiman", "Seasonal mode"), mode_opsi,
                                      index=mode_opsi.index(b.get("mode_musiman", "aditif")),
                                      key=_w(proc, "smode"))
    c13, c14 = st.columns(2)
    p["periode_musiman_2"] = int(c13.number_input(L("Periode musiman kedua (0 = tidak ada)",
                                                    "Second seasonal period (0 = none)"), 0, 366,
                                                  int(b.get("periode_musiman_2", 0)), key=_w(proc, "m2")))
    p["amplitudo_musiman_2"] = float(c14.slider(L("Amplitudo musiman kedua", "Second seasonal amplitude"),
                                                0.0, 1.0, float(b.get("amplitudo_musiman_2", 0.0)), 0.01,
                                                key=_w(proc, "amp3")))

    st.markdown("**" + L("Anomali & perubahan struktur", "Anomalies & structural change") + "**")
    c15, c16, c17, c18 = st.columns(4)
    p["persen_anomali"] = float(c15.slider(L("Tingkat anomali (%)", "Anomaly rate (%)"), 0.0, 25.0,
                                           float(b["persen_anomali"]), 0.5, key=_w(proc, "anom")))
    p["besar_anomali"] = float(c16.slider(L("Besar anomali (× simpangan baku)", "Anomaly magnitude (× std dev)"),
                                          1.0, 12.0, float(b["besar_anomali"]), 0.5, key=_w(proc, "anommag")))
    anom_opsi = ["lonjakan", "penurunan", "keduanya"]
    p["tipe_anomali"] = c17.selectbox(L("Tipe anomali", "Anomaly type"), anom_opsi,
                                      index=anom_opsi.index(b.get("tipe_anomali", "keduanya")),
                                      key=_w(proc, "anomtype"))
    p["sertakan_penanda_anomali"] = bool(c18.checkbox(
        L("Kolom penanda anomali", "Anomaly flag column"), bool(b.get("sertakan_penanda_anomali", False)),
        key=_w(proc, "anomflag"),
        help=L("Menambah kolom 'anomali' berisi 0/1 sebagai kunci jawaban.",
               "Adds an 'anomali' 0/1 answer-key column.")))
    c19, c20, c21, c22 = st.columns(4)
    p["jumlah_pergeseran_level"] = int(c19.number_input(L("Jumlah pergeseran level", "Level shifts"), 0, 10,
                                                        int(b.get("jumlah_pergeseran_level", 0)),
                                                        key=_w(proc, "nshift")))
    p["besar_pergeseran"] = float(c20.slider(L("Besar pergeseran level", "Level shift size"), 0.0, 1.0,
                                             float(b.get("besar_pergeseran", 0.15)), 0.01, key=_w(proc, "shiftmag")))
    p["persen_kosong"] = float(c21.slider(L("Nilai kosong (%)", "Missing values (%)"), 0.0, 40.0,
                                          float(b["persen_kosong"]), 0.5, key=_w(proc, "miss")))
    p["desimal"] = int(c22.number_input(L("Desimal", "Decimals"), 0, 6, int(b["desimal"]), key=_w(proc, "dec")))

    c23, c24, c25 = st.columns(3)
    p["nama_kolom_waktu"] = c23.text_input(L("Nama kolom waktu", "Time column name"),
                                           str(b["nama_kolom_waktu"]), key=_w(proc, "cname1"))
    p["nama_kolom_nilai"] = c24.text_input(L("Nama kolom nilai", "Value column name"),
                                           str(b["nama_kolom_nilai"]), key=_w(proc, "cname2"))
    pakai_min = c25.checkbox(L("Batas bawah nilai", "Floor the value"),
                             b.get("nilai_min") is not None, key=_w(proc, "usemin"))
    p["nilai_min"] = float(st.number_input(L("Nilai minimum", "Minimum value"),
                                           value=float(b.get("nilai_min") or 0.0), key=_w(proc, "vmin"))) \
        if pakai_min else None
    return p


FORMS = {
    "classification": _form_classification,
    "regression": _form_regression,
    "clustering": _form_clustering,
    "association": _form_association,
    "timeseries": _form_timeseries,
}


# ------------------------------------------------------------- pratinjau
def _preview_fig(proc: str, cfg: dict, df: pd.DataFrame):
    try:
        p = cfg["params"]
        if proc == "classification":
            return V.fig_class_balance(df[p["nama_target"]])
        if proc == "regression":
            num = [c["nama"] for c in p["kolom"] if c["tipe"] == "numerik" and c["nama"] in df.columns]
            keep = [p["nama_target"]] + num[:5]
            return V.fig_hist_grid(df[[c for c in keep if c in df.columns]])
        if proc == "clustering":
            num = [c["nama"] for c in p["kolom"] if c["tipe"] == "numerik"]
            if len(num) >= 2:
                lab = df["klaster_asli"].values if "klaster_asli" in df.columns else np.zeros(len(df), dtype=int)
                coords = df[num[:2]].apply(pd.to_numeric, errors="coerce").fillna(0).values
                return V.fig_cluster_scatter(coords, lab, num[0], num[1], None,
                                             L("Pratinjau Data Bangkitan", "Generated Data Preview"))
            return V.fig_hist_grid(df)
        if proc == "association":
            if "item" in df.columns:
                counts = df.groupby("item")["id_transaksi"].nunique()
            elif "daftar_item" in df.columns:
                counts = (df["daftar_item"].str.split(",").explode().str.strip().value_counts())
            else:
                counts = df.drop(columns=["id_transaksi"], errors="ignore").sum(axis=0)
            return V.fig_item_frequency(pd.Series(counts))
        s = pd.Series(pd.to_numeric(df[cfg["params"]["nama_kolom_nilai"]], errors="coerce").values,
                      index=pd.to_datetime(df[cfg["params"]["nama_kolom_waktu"]]))
        s = s.groupby(level=0).mean()
        return V.fig_ts_line(s, L("Pratinjau Deret Bangkitan", "Generated Series Preview"))
    except Exception:
        return None


# ================================================================== render
def render(process: str):
    """Tampilkan panel generator. Kembalikan DataFrame hasil bangkitan (atau None)."""
    cfg = _current_cfg(process)

    st.markdown("### " + L("Pembangkit Data Acak", "Random Data Generator"))
    U.note(L(
        "Atur karakteristik data yang ingin dibangkitkan, lalu klik <b>Bangkitkan Data</b>. "
        "Seluruh konfigurasi dapat diekspor ke berkas JSON dan diimpor kembali kapan saja "
        "sehingga eksperimen mudah diulang persis sama.",
        "Set the characteristics of the data you want, then click <b>Generate Data</b>. "
        "The whole configuration can be exported to JSON and re-imported later so an experiment "
        "can be reproduced exactly.",
    ))

    io_slot = st.container()
    with io_slot:
        ci1, ci2 = st.columns([1.3, 1])
        with ci1:
            _import_panel(process)
        export_slot = ci2.empty()

    seed_c1, seed_c2 = st.columns([1, 3])
    seed = int(seed_c1.number_input(L("Seed acak", "Random seed"), 0, 10**9, int(cfg["seed"]),
                                    key=_w(process, "seed"),
                                    help=L("Seed sama + konfigurasi sama = data identik.",
                                           "Same seed + same configuration = identical data.")))
    if seed_c2.button(L("Acak ulang seed", "Randomize seed"), key=_w(process, "reseed")):
        st.session_state[_w(process, "seed")] = int(np.random.default_rng().integers(0, 10**6))
        st.rerun()

    params = FORMS[process](process, cfg)
    new_cfg = {"schema_version": G.SCHEMA_VERSION, "process": process, "seed": seed,
               "dibuat": datetime.now().strftime("%Y-%m-%d %H:%M"), "params": params}
    try:
        new_cfg = G.normalize_config(new_cfg)
    except Exception as e:
        st.error(str(e))
        return st.session_state.get(_kdf(process))
    st.session_state[_kcfg(process)] = new_cfg

    with export_slot.container():
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        st.download_button(
            L("Ekspor konfigurasi (.json)", "Export configuration (.json)"),
            data=G.config_to_json(new_cfg).encode("utf-8"),
            file_name=f"konfigurasi_{process}_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json", width="stretch", key=f"gen_dl_{process}")

    ok, msgs = G.validate_config(new_cfg)
    for m in msgs:
        U.warn(m)

    st.markdown("---")
    b1, b2 = st.columns([1, 3])
    if b1.button("🎲 " + L("Bangkitkan Data", "Generate Data"), type="primary",
                 width="stretch", key=f"gen_go_{process}"):
        try:
            with st.spinner(L("Membangkitkan data...", "Generating data...")):
                st.session_state[_kdf(process)] = G.generate(new_cfg)
        except Exception as e:
            st.error(L(f"Gagal membangkitkan data: {e}", f"Data generation failed: {e}"))
            st.session_state.pop(_kdf(process), None)
    if b2.button(L("Hapus data bangkitan", "Clear generated data"), key=f"gen_clr_{process}"):
        st.session_state.pop(_kdf(process), None)

    df = st.session_state.get(_kdf(process))
    if df is None or len(df) == 0:
        st.info(L("Belum ada data. Atur parameter lalu klik Bangkitkan Data.",
                  "No data yet. Set the parameters and click Generate Data."))
        return None

    st.success(L(f"Data berhasil dibangkitkan: {len(df):,} baris × {df.shape[1]} kolom.",
                 f"Data generated: {len(df):,} rows × {df.shape[1]} columns."))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(L("Baris", "Rows"), f"{len(df):,}")
    m2.metric(L("Kolom", "Columns"), f"{df.shape[1]:,}")
    m3.metric(L("Sel kosong", "Missing cells"), f"{int(df.isna().sum().sum()):,}")
    m4.metric(L("Density data", "Data density"),
              f"{(1 - df.isna().sum().sum() / max(df.size, 1)) * 100:.1f}%")

    t1, t2, t3 = st.tabs([L("Pratinjau", "Preview"), L("Grafik pratinjau", "Preview chart"),
                          L("Unduh dataset", "Download dataset")])
    with t1:
        st.dataframe(df.head(200), width="stretch", height=340)
    with t2:
        fig = _preview_fig(process, new_cfg, df)
        if fig is None:
            st.caption(L("Pratinjau grafik tidak tersedia untuk konfigurasi ini.",
                         "No preview chart available for this configuration."))
        else:
            U.show_fig("preview", fig, f"gen_prev_{process}")
    with t3:
        d1, d2 = st.columns(2)
        d1.download_button(
            L("Unduh dataset (.xlsx)", "Download dataset (.xlsx)"),
            data=tables_to_excel({L("Data", "Data"): df,
                                  L("Konfigurasi", "Configuration"): G.summarize(new_cfg, df)},
                                 {L("Proses", "Process"): process,
                                  L("Seed", "Seed"): new_cfg["seed"],
                                  L("Dibangkitkan", "Generated at"): new_cfg["dibuat"]}),
            file_name=stamp(f"data_bangkitan_{process}", "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch", key=f"gen_xlsx_{process}")
        d2.download_button(
            L("Unduh dataset (.csv)", "Download dataset (.csv)"),
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=stamp(f"data_bangkitan_{process}", "csv"),
            mime="text/csv", width="stretch", key=f"gen_csv_{process}")
        st.caption(L(
            "Berkas Excel menyertakan sheet 'Konfigurasi' berisi seluruh parameter yang dipakai.",
            "The Excel file includes a 'Konfigurasi' sheet listing every parameter used."))

    return df
