"""Generator template data (XLSX & CSV) untuk tiap proses analitik."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2024)


# ----------------------------------------------------------- data contoh
def _classification() -> pd.DataFrame:
    n = 120
    r = np.random.default_rng(11)
    usia = r.integers(21, 60, n)
    pendapatan = np.round(r.normal(7.5, 2.6, n) * 1_000_000, -4).clip(2_000_000, None)
    lama_kerja = r.integers(0, 25, n)
    jumlah_tanggungan = r.integers(0, 5, n)
    pinjaman = np.round(r.normal(60, 25, n) * 1_000_000, -5).clip(5_000_000, None)
    skor_kredit = r.integers(300, 850, n)
    status_rumah = r.choice(["Milik Sendiri", "Sewa", "Keluarga"], n, p=[0.5, 0.3, 0.2])
    pendidikan = r.choice(["SMA", "D3", "S1", "S2"], n, p=[0.3, 0.2, 0.4, 0.1])
    skor = (
        (skor_kredit - 500) / 120
        + (pendapatan / 1_000_000 - 7) / 2.2
        + lama_kerja / 9
        - (pinjaman / 1_000_000 - 60) / 22
        - jumlah_tanggungan / 3.5
        + (status_rumah == "Milik Sendiri") * 0.7
        + r.normal(0, 0.7, n)
    )
    kelas = np.where(skor > 0.35, "Layak", "Tidak Layak")
    return pd.DataFrame(
        {
            "id_nasabah": [f"N{i:04d}" for i in range(1, n + 1)],
            "usia": usia,
            "pendapatan_bulanan": pendapatan.astype(int),
            "lama_kerja_tahun": lama_kerja,
            "jumlah_tanggungan": jumlah_tanggungan,
            "jumlah_pinjaman": pinjaman.astype(int),
            "skor_kredit": skor_kredit,
            "status_rumah": status_rumah,
            "pendidikan": pendidikan,
            "status_kelayakan": kelas,
        }
    )


def _regression() -> pd.DataFrame:
    n = 120
    r = np.random.default_rng(22)
    luas_tanah = r.integers(60, 500, n)
    luas_bangunan = (luas_tanah * r.uniform(0.4, 0.9, n)).round().astype(int)
    kamar_tidur = np.clip((luas_bangunan / 45).round(), 1, 6).astype(int)
    kamar_mandi = np.clip((kamar_tidur * r.uniform(0.5, 1.0, n)).round(), 1, 4).astype(int)
    umur = r.integers(0, 35, n)
    jarak = np.round(r.uniform(0.5, 25, n), 1)
    daya_listrik = r.choice([900, 1300, 2200, 3500], n, p=[0.2, 0.4, 0.3, 0.1])
    harga = (
        luas_tanah * 4.2e6
        + luas_bangunan * 6.1e6
        + kamar_tidur * 3.5e7
        + daya_listrik * 9_000
        - umur * 1.6e7
        - jarak * 2.4e7
        + r.normal(0, 1.1e8, n)
    )
    harga = np.clip(harga, 1.5e8, None)
    return pd.DataFrame(
        {
            "id_properti": [f"P{i:04d}" for i in range(1, n + 1)],
            "luas_tanah_m2": luas_tanah,
            "luas_bangunan_m2": luas_bangunan,
            "jumlah_kamar_tidur": kamar_tidur,
            "jumlah_kamar_mandi": kamar_mandi,
            "umur_bangunan_tahun": umur,
            "jarak_pusat_kota_km": jarak,
            "daya_listrik_va": daya_listrik,
            "harga_jual": np.round(harga, -6).astype("int64"),
        }
    )


def _clustering() -> pd.DataFrame:
    n_per = 40
    r = np.random.default_rng(33)
    centers = [(27, 42, 78, 24, 260), (45, 95, 38, 9, 720), (36, 62, 55, 15, 430)]
    rows = []
    for cx in centers:
        rows.append(
            np.column_stack(
                [
                    r.normal(cx[0], 5, n_per),
                    r.normal(cx[1], 12, n_per),
                    r.normal(cx[2], 9, n_per),
                    r.normal(cx[3], 4, n_per),
                    r.normal(cx[4], 70, n_per),
                ]
            )
        )
    m = np.vstack(rows)
    df = pd.DataFrame(
        m,
        columns=[
            "usia",
            "pendapatan_tahunan_juta",
            "skor_pengeluaran",
            "frekuensi_belanja_bulanan",
            "rata_nilai_transaksi_ribu",
        ],
    )
    df = df.round(1)
    df.insert(0, "id_pelanggan", [f"C{i:04d}" for i in range(1, len(df) + 1)])
    df["usia"] = df["usia"].round().astype(int)
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def _association_long() -> pd.DataFrame:
    r = np.random.default_rng(44)
    katalog = [
        "Roti Tawar", "Susu UHT", "Telur", "Kopi Bubuk", "Gula Pasir",
        "Mie Instan", "Minyak Goreng", "Beras", "Teh Celup", "Sabun Mandi",
        "Shampo", "Deterjen", "Air Mineral", "Biskuit", "Cokelat",
    ]
    aturan = [
        (["Roti Tawar", "Susu UHT", "Telur"], 0.30),
        (["Kopi Bubuk", "Gula Pasir", "Susu UHT"], 0.24),
        (["Mie Instan", "Telur", "Air Mineral"], 0.22),
        (["Sabun Mandi", "Shampo", "Deterjen"], 0.20),
        (["Beras", "Minyak Goreng", "Gula Pasir"], 0.18),
        (["Teh Celup", "Biskuit", "Gula Pasir"], 0.16),
    ]
    rows = []
    for i in range(1, 301):
        tid = f"TRX{i:04d}"
        items = set()
        for pola, p in aturan:
            if r.random() < p:
                items.update(r.choice(pola, size=r.integers(2, len(pola) + 1), replace=False))
        while len(items) < 2:
            items.add(str(r.choice(katalog)))
        for _ in range(int(r.integers(0, 3))):
            items.add(str(r.choice(katalog)))
        for it in sorted(items):
            rows.append({"id_transaksi": tid, "item": it, "jumlah": int(r.integers(1, 4))})
    return pd.DataFrame(rows)


def _association_basket() -> pd.DataFrame:
    long = _association_long()
    piv = (
        long.assign(v=1)
        .pivot_table(index="id_transaksi", columns="item", values="v", aggfunc="max", fill_value=0)
        .reset_index()
    )
    piv.columns.name = None
    return piv.head(60)


def _timeseries() -> pd.DataFrame:
    n = 84  # 7 tahun bulanan
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    t = np.arange(n)
    r = np.random.default_rng(55)
    tren = 1200 + 9.5 * t
    musim = 210 * np.sin(2 * np.pi * (t % 12) / 12) + 90 * np.cos(2 * np.pi * (t % 12) / 6)
    noise = r.normal(0, 55, n)
    nilai = tren + musim + noise
    return pd.DataFrame(
        {
            "tanggal": idx.strftime("%Y-%m-%d"),
            "nilai": np.round(nilai, 2),
        }
    )


BUILDERS = {
    "classification": _classification,
    "regression": _regression,
    "clustering": _clustering,
    "association": _association_long,
    "timeseries": _timeseries,
}


# ------------------------------------------------------------- petunjuk
GUIDE = {
    "classification": [
        ("Tujuan | Purpose", "Memprediksi kelas kategorik | Predict a categorical class"),
        ("Kolom target | Target column", "status_kelayakan — wajib kategorik (2+ kelas) | must be categorical (2+ classes)"),
        ("Kolom fitur | Feature columns", "Boleh numerik & kategorik. Kolom ID sebaiknya dikecualikan | Numeric & categorical allowed; exclude ID columns"),
        ("Minimal baris | Minimum rows", "30 baris, idealnya >= 100 dan tiap kelas >= 10 baris | 30 rows, ideally >= 100 with >= 10 rows per class"),
        ("Catatan | Notes", "Nama kolom bebas; ganti isi dengan data Anda, jangan ubah struktur satu-baris-satu-observasi | Column names are free; keep one row = one observation"),
    ],
    "regression": [
        ("Tujuan | Purpose", "Memprediksi nilai numerik kontinu | Predict a continuous numeric value"),
        ("Kolom target | Target column", "harga_jual — wajib numerik kontinu | must be continuous numeric"),
        ("Kolom fitur | Feature columns", "Numerik & kategorik (otomatis di-encode) | Numeric & categorical (auto-encoded)"),
        ("Minimal baris | Minimum rows", "30 baris, idealnya >= 100 | 30 rows, ideally >= 100"),
        ("Catatan | Notes", "Hindari kolom yang merupakan turunan langsung dari target (kebocoran data) | Avoid columns derived from the target (leakage)"),
    ],
    "clustering": [
        ("Tujuan | Purpose", "Mengelompokkan data tanpa label | Group unlabeled data"),
        ("Kolom target | Target column", "TIDAK ADA — klastering bersifat unsupervised | NONE — clustering is unsupervised"),
        ("Kolom fitur | Feature columns", "Minimal 2 kolom numerik | At least 2 numeric columns"),
        ("Minimal baris | Minimum rows", "20 baris, idealnya >= 50 | 20 rows, ideally >= 50"),
        ("Catatan | Notes", "Skala fitur berbeda ditangani otomatis lewat standardisasi | Different feature scales are handled by standardization"),
    ],
    "association": [
        ("Tujuan | Purpose", "Menemukan pola beli-bersama antar item | Find co-occurrence patterns between items"),
        ("Format 1 (disarankan) | Format 1 (recommended)", "Format panjang: kolom id_transaksi + kolom item, satu baris satu item | Long format: id_transaksi + item, one row per item"),
        ("Format 2 | Format 2", "Format keranjang: satu baris satu transaksi, tiap item jadi kolom 0/1 (lihat sheet 'Contoh_Format_Keranjang') | Basket format: one row per transaction, items as 0/1 columns"),
        ("Minimal baris | Minimum rows", "50 transaksi agar aturan bermakna | 50 transactions for meaningful rules"),
        ("Catatan | Notes", "Kolom 'jumlah' opsional dan tidak dipakai dalam perhitungan aturan | The 'jumlah' column is optional and unused in rule mining"),
    ],
    "timeseries": [
        ("Tujuan | Purpose", "Memodelkan & meramalkan deret waktu | Model & forecast a time series"),
        ("Kolom waktu | Time column", "tanggal — format YYYY-MM-DD, urut, interval tetap | tanggal — YYYY-MM-DD, sorted, fixed interval"),
        ("Kolom nilai | Value column", "nilai — wajib numerik | nilai — must be numeric"),
        ("Minimal baris | Minimum rows", "30 titik; untuk SARIMA minimal 2-3 siklus musiman (mis. 24-36 bulan) | 30 points; SARIMA needs 2-3 seasonal cycles"),
        ("Catatan | Notes", "Hanya satu deret per file. Interval boleh harian/mingguan/bulanan/kuartalan/tahunan | One series per file; daily/weekly/monthly/quarterly/yearly"),
    ],
}


def guide_df(process: str) -> pd.DataFrame:
    return pd.DataFrame(GUIDE[process], columns=["Aspek | Aspect", "Ketentuan | Requirement"])


def build_template(process: str) -> pd.DataFrame:
    return BUILDERS[process]()


def template_csv_bytes(process: str) -> bytes:
    return build_template(process).to_csv(index=False).encode("utf-8-sig")


def template_xlsx_bytes(process: str) -> bytes:
    buf = io.BytesIO()
    data = build_template(process)
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        data.to_excel(xw, sheet_name="Data", index=False)
        guide_df(process).to_excel(xw, sheet_name="Petunjuk_Guide", index=False)
        if process == "association":
            _association_basket().to_excel(xw, sheet_name="Contoh_Format_Keranjang", index=False)

        book = xw.book
        head = book.add_format(
            {"bold": True, "bg_color": "#2a78d6", "font_color": "#FFFFFF", "border": 1, "border_color": "#FFFFFF", "align": "center", "valign": "vcenter", "text_wrap": True}
        )
        wrap = book.add_format({"text_wrap": True, "valign": "top"})
        for sheet_name, frame in [
            ("Data", data),
            ("Petunjuk_Guide", guide_df(process)),
        ] + ([("Contoh_Format_Keranjang", _association_basket())] if process == "association" else []):
            ws = xw.sheets[sheet_name]
            for j, col in enumerate(frame.columns):
                width = max(12, min(46, int(frame[col].astype(str).str.len().max() if len(frame) else 12) + 4, ))
                width = max(width, len(str(col)) + 3)
                ws.set_column(j, j, width, wrap if sheet_name != "Data" else None)
                ws.write(0, j, str(col), head)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
    return buf.getvalue()
