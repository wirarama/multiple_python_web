"""Pembangkit data acak (random data generator) per proses analitik.

Seluruh konfigurasi berupa dict yang JSON-serializable sehingga dapat
diekspor/diimpor lewat berkas .json.

Struktur konfigurasi:
    {
      "schema_version": 1,
      "process": "classification" | "regression" | "clustering" |
                 "association" | "timeseries",
      "seed": 42,
      "params": { ... spesifik proses ... }
    }
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

PROCESSES = ["classification", "regression", "clustering", "association", "timeseries"]

DIST_NUMERIC = ["normal", "uniform", "lognormal", "eksponensial", "integer", "poisson"]
COL_TYPES = ["numerik", "kategorik"]

# kolom wajib pada tabel spesifikasi fitur
SPEC_FIELDS = [
    "nama", "tipe", "distribusi", "min", "max", "mean", "std", "desimal",
    "kategori", "bobot_kategori", "persen_kosong", "persen_outlier", "pengaruh",
]

ITEM_CATALOG = [
    "Roti Tawar", "Susu UHT", "Telur", "Kopi Bubuk", "Gula Pasir", "Mie Instan",
    "Minyak Goreng", "Beras", "Teh Celup", "Sabun Mandi", "Shampo", "Deterjen",
    "Air Mineral", "Biskuit", "Cokelat", "Keju", "Mentega", "Sereal", "Yoghurt",
    "Saus Sambal", "Kecap Manis", "Garam", "Tepung Terigu", "Santan", "Bawang Merah",
    "Bawang Putih", "Cabai", "Tomat", "Kentang", "Wortel", "Ayam Potong", "Ikan Segar",
    "Daging Sapi", "Tisu", "Pasta Gigi", "Sikat Gigi", "Pembersih Lantai", "Spons Cuci",
    "Kopi Sachet", "Susu Kental Manis",
]

FREQS = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}
FREQ_LABEL = {
    "D": ("Harian", "Daily"),
    "W": ("Mingguan", "Weekly"),
    "MS": ("Bulanan", "Monthly"),
    "QS": ("Kuartalan", "Quarterly"),
    "YS": ("Tahunan", "Yearly"),
}


# ===================================================================== util
def _rng(seed):
    return np.random.default_rng(int(seed) if seed is not None else None)


def _parse_list(text, cast=str):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [cast(x) for x in text]
    parts = [p.strip() for p in str(text).replace(";", ",").split(",")]
    out = []
    for p in parts:
        if p == "":
            continue
        try:
            out.append(cast(p))
        except Exception:
            continue
    return out


def _weights(raw, k):
    w = _parse_list(raw, float)
    if len(w) != k or sum(w) <= 0:
        return np.full(k, 1.0 / k)
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def _zscore(v):
    v = np.asarray(v, dtype=float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 1e-12 else 1.0)


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s).strip().lower()).strip("_") or "kolom"


# ============================================================ spesifikasi
def default_column(i: int, tipe: str = "numerik", nama: str | None = None) -> dict:
    if tipe == "kategorik":
        return {
            "nama": nama or f"kategori_{i}", "tipe": "kategorik", "distribusi": "-",
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "desimal": 0,
            "kategori": "A, B, C", "bobot_kategori": "0.5, 0.3, 0.2",
            "persen_kosong": 0.0, "persen_outlier": 0.0, "pengaruh": 1.0,
        }
    return {
        "nama": nama or f"fitur_{i}", "tipe": "numerik", "distribusi": "normal",
        "min": 0.0, "max": 100.0, "mean": 50.0, "std": 15.0, "desimal": 2,
        "kategori": "", "bobot_kategori": "",
        "persen_kosong": 0.0, "persen_outlier": 0.0, "pengaruh": 1.0,
    }


def default_columns(n_numeric: int = 5, n_categorical: int = 2) -> list[dict]:
    cols = [default_column(i + 1) for i in range(int(n_numeric))]
    cols += [default_column(i + 1, "kategorik") for i in range(int(n_categorical))]
    return cols


def normalize_column(spec: dict, idx: int = 1) -> dict:
    base = default_column(idx, str(spec.get("tipe", "numerik")).lower().strip() or "numerik")
    out = dict(base)
    for k in SPEC_FIELDS:
        if k in spec and spec[k] is not None and not (isinstance(spec[k], float) and math.isnan(spec[k])):
            out[k] = spec[k]
    out["nama"] = _slug(out["nama"])
    out["tipe"] = "kategorik" if str(out["tipe"]).lower().startswith("kat") else "numerik"
    if out["tipe"] == "numerik":
        dist = str(out["distribusi"]).lower().strip()
        out["distribusi"] = dist if dist in DIST_NUMERIC else "normal"
        out["min"], out["max"] = float(out["min"]), float(out["max"])
        if out["max"] <= out["min"]:
            out["max"] = out["min"] + 1.0
        out["mean"] = float(np.clip(float(out["mean"]), out["min"], out["max"]))
        out["std"] = max(float(out["std"]), 1e-6)
        out["desimal"] = int(np.clip(int(out["desimal"]), 0, 6))
    else:
        out["distribusi"] = "-"
        if not _parse_list(out["kategori"]):
            out["kategori"] = "A, B, C"
    for k in ("persen_kosong", "persen_outlier"):
        out[k] = float(np.clip(float(out[k]), 0.0, 90.0))
    out["pengaruh"] = float(np.clip(float(out["pengaruh"]), 0.0, 5.0))
    return out


def _unique_names(cols: list[dict]) -> list[dict]:
    seen, out = {}, []
    for c in cols:
        n = c["nama"]
        if n in seen:
            seen[n] += 1
            c = dict(c, nama=f"{n}_{seen[n]}")
        else:
            seen[n] = 0
        out.append(c)
    return out


# ================================================== pembangkit satu kolom
def _gen_numeric(spec, n, rng):
    lo, hi = spec["min"], spec["max"]
    mean, std, dist = spec["mean"], spec["std"], spec["distribusi"]
    if dist == "uniform":
        v = rng.uniform(lo, hi, n)
    elif dist == "lognormal":
        m = max(abs(mean), 1e-6)
        sigma = float(np.clip(std / m, 0.05, 1.4))
        v = rng.lognormal(math.log(m) - sigma ** 2 / 2, sigma, n)
    elif dist == "eksponensial":
        v = lo + rng.exponential(max(mean - lo, 1e-6), n)
    elif dist == "poisson":
        v = rng.poisson(max(mean, 0.0), n).astype(float)
    elif dist == "integer":
        v = rng.integers(int(math.floor(lo)), int(math.ceil(hi)) + 1, n).astype(float)
    else:  # normal
        v = rng.normal(mean, std, n)
    return np.clip(v, lo, hi)


def _apply_outliers(v, spec, rng):
    p = spec["persen_outlier"] / 100.0
    if p <= 0:
        return v
    n = len(v)
    k = max(1, int(round(n * p)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    span = max(spec["max"] - spec["min"], 1e-6)
    direction = rng.choice([-1.0, 1.0], size=len(idx))
    magnitude = rng.uniform(1.6, 3.2, size=len(idx))
    v = v.copy()
    v[idx] = np.where(direction > 0, spec["max"] + span * (magnitude - 1.0),
                      spec["min"] - span * (magnitude - 1.0))
    return v


def _apply_missing(series: pd.Series, pct, rng):
    p = float(pct) / 100.0
    if p <= 0:
        return series
    n = len(series)
    k = int(round(n * p))
    if k <= 0:
        return series
    idx = rng.choice(n, size=min(k, n), replace=False)
    out = series.copy()
    out.iloc[idx] = np.nan
    return out


def _finish_numeric(v, spec, rng, missing_extra=0.0):
    s = pd.Series(np.round(v, spec["desimal"]))
    s = _apply_missing(s, spec["persen_kosong"] + missing_extra, rng)
    if spec["desimal"] == 0 and s.notna().all():
        s = s.astype("int64")
    return s


def build_features(columns: list[dict], n: int, rng, missing_extra: float = 0.0):
    """Bangkitkan seluruh kolom fitur. Kembalikan (DataFrame, latent_score)."""
    cols = _unique_names([normalize_column(c, i + 1) for i, c in enumerate(columns)])
    data, contributions = {}, []
    for spec in cols:
        if spec["tipe"] == "numerik":
            raw = _gen_numeric(spec, n, rng)
            raw = _apply_outliers(raw, spec, rng)
            data[spec["nama"]] = _finish_numeric(raw, spec, rng, missing_extra)
            if spec["pengaruh"] > 0:
                contributions.append(spec["pengaruh"] * _zscore(np.nan_to_num(raw, nan=np.nanmean(raw))))
        else:
            cats = _parse_list(spec["kategori"])
            w = _weights(spec["bobot_kategori"], len(cats))
            vals = rng.choice(cats, size=n, p=w)
            s = pd.Series(vals, dtype=object)
            data[spec["nama"]] = _apply_missing(s, spec["persen_kosong"] + missing_extra, rng)
            if spec["pengaruh"] > 0 and len(cats) > 1:
                effect = {c: e for c, e in zip(cats, rng.normal(0, 1, len(cats)))}
                contributions.append(spec["pengaruh"] * _zscore([effect[v] for v in vals]))
    df = pd.DataFrame(data)
    latent = np.sum(contributions, axis=0) if contributions else rng.normal(0, 1, n)
    return df, _zscore(latent), cols


def _post_process(df: pd.DataFrame, params: dict, rng) -> pd.DataFrame:
    """Duplikat baris & kolom ID opsional."""
    dup = float(params.get("persen_duplikat", 0.0)) / 100.0
    if dup > 0 and len(df) > 1:
        k = max(1, int(round(len(df) * dup)))
        idx = rng.choice(len(df), size=min(k, len(df)), replace=True)
        df = pd.concat([df, df.iloc[idx]], ignore_index=True)
        df = df.sample(frac=1.0, random_state=int(rng.integers(0, 10**6))).reset_index(drop=True)
    if params.get("kolom_id", True):
        prefix = str(params.get("prefix_id", "ID"))
        df.insert(0, "id_data", [f"{prefix}{i:05d}" for i in range(1, len(df) + 1)])
    return df


# ============================================================== KLASIFIKASI
def _gen_classification(p: dict, rng) -> pd.DataFrame:
    n = int(p["jumlah_baris"])
    df, latent, _ = build_features(p["kolom"], n, rng, p.get("persen_kosong_global", 0.0))

    kelas = [str(k) for k in p.get("kelas", ["Kelas A", "Kelas B"])]
    w = _weights(p.get("bobot_kelas", ""), len(kelas))
    # keterpisahan 0..1 -> rasio sinyal/derau. 0.5 = seimbang, mendekati 1 = hampir
    # sempurna terpisah, mendekati 0 = kelas nyaris acak terhadap fitur.
    sep = float(np.clip(p.get("keterpisahan", 0.7), 0.02, 0.99))
    noise_scale = (1.0 / sep) - 1.0
    noise = _zscore(rng.normal(0, 1, n))
    score = latent + noise_scale * noise

    cuts = np.cumsum(w)[:-1]
    thresholds = np.quantile(score, cuts)
    labels = np.array(kelas)[np.searchsorted(thresholds, score, side="right")]

    flip = float(np.clip(p.get("persen_label_bising", 0.0), 0, 50)) / 100.0
    if flip > 0:
        k = int(round(n * flip))
        idx = rng.choice(n, size=min(k, n), replace=False)
        labels = labels.copy()
        labels[idx] = rng.choice(kelas, size=len(idx))

    df[str(p.get("nama_target", "target"))] = labels
    return _post_process(df, p, rng)


# ================================================================= REGRESI
def _gen_regression(p: dict, rng) -> pd.DataFrame:
    n = int(p["jumlah_baris"])
    df, latent, _ = build_features(p["kolom"], n, rng, p.get("persen_kosong_global", 0.0))

    nl = float(np.clip(p.get("nonlinearitas", 0.0), 0.0, 1.0))
    if nl > 0:
        latent = _zscore((1 - nl) * latent + nl * (latent ** 2 - 1.0))

    r2 = float(np.clip(p.get("r2_target", 0.75), 0.01, 0.99))
    r = math.sqrt(r2)
    noise = _zscore(rng.normal(0, 1, n))
    hetero = float(np.clip(p.get("heteroskedastisitas", 0.0), 0.0, 1.0))
    if hetero > 0:
        scale = 1.0 + hetero * (latent - latent.min()) / max(np.ptp(latent), 1e-9) * 2.0
        noise = _zscore(noise * scale)

    z = r * latent + math.sqrt(max(1 - r2, 0.0)) * noise
    mean_t = float(p.get("target_mean", 100.0))
    std_t = float(max(p.get("target_std", 20.0), 1e-6))
    y = mean_t + std_t * _zscore(z)

    lo, hi = p.get("target_min"), p.get("target_max")
    if lo is not None and hi is not None and float(hi) > float(lo):
        y = np.clip(y, float(lo), float(hi))
    dec = int(np.clip(int(p.get("target_desimal", 2)), 0, 6))
    y = np.round(y, dec)

    name = str(p.get("nama_target", "target"))
    df[name] = y.astype("int64") if dec == 0 else y
    miss_t = float(p.get("target_persen_kosong", 0.0))
    if miss_t > 0:
        df[name] = _apply_missing(df[name].astype(float), miss_t, rng)
    return _post_process(df, p, rng)


# =============================================================== KLASTERING
def _maximin_centers(k: int, d: int, rng, tries: int = 40) -> np.ndarray:
    """Pusat klaster pada [0.12, 0.88]^d dengan jarak minimum antar pusat maksimal."""
    best, best_score = None, -1.0
    for _ in range(tries):
        # Latin hypercube: tiap dimensi dibagi k strata, satu pusat per strata
        cand = np.empty((k, d))
        for j in range(d):
            strata = (rng.permutation(k) + rng.random(k)) / k
            cand[:, j] = strata
        if k > 1:
            diff = cand[:, None, :] - cand[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            score = float(dist.min())
        else:
            score = 0.0
        if score > best_score:
            best, best_score = cand, score
    return 0.12 + 0.76 * best


def _gen_clustering(p: dict, rng) -> pd.DataFrame:
    n = int(p["jumlah_baris"])
    cols = _unique_names([normalize_column(c, i + 1) for i, c in enumerate(p["kolom"])])
    cols = [c for c in cols if c["tipe"] == "numerik"] or [default_column(1), default_column(2)]
    d = len(cols)
    k = int(np.clip(int(p.get("jumlah_klaster", 3)), 1, 20))

    w = _weights(p.get("bobot_klaster", ""), k)
    counts = np.floor(w * n).astype(int)
    counts[: n - counts.sum()] += 1 if n - counts.sum() > 0 else 0
    while counts.sum() < n:
        counts[int(rng.integers(0, k))] += 1
    while counts.sum() > n:
        j = int(np.argmax(counts))
        counts[j] -= 1

    pemisahan = float(np.clip(p.get("pemisahan", 3.0), 0.2, 10.0))
    kerapatan = float(np.clip(p.get("kerapatan", 1.0), 0.1, 5.0))
    bentuk = str(p.get("bentuk", "bulat"))

    # Pusat klaster ditempatkan dengan rancangan maximin Latin-hypercube supaya
    # setiap dimensi ikut memisahkan klaster (pusat acak di permukaan bola
    # menghasilkan pasangan pusat yang terlalu dekat).
    centers = _maximin_centers(k, d, rng)
    spacing = (1.0 / max(k, 1)) ** (1.0 / max(d, 1))
    sigma = 0.76 * spacing / (1.4 * pemisahan * kerapatan)

    points, labels = [], []
    for c in range(k):
        m = int(counts[c])
        if m <= 0:
            continue
        spread = np.full(d, sigma)
        if bentuk == "lonjong":
            spread = spread * rng.uniform(0.35, 2.2, d)
        pts = centers[c] + rng.normal(0, 1, (m, d)) * spread
        points.append(pts)
        labels += [c] * m
    X = np.vstack(points)
    labels = np.asarray(labels)

    noise_pct = float(np.clip(p.get("persen_noise", 0.0), 0, 40)) / 100.0
    if noise_pct > 0:
        m = max(1, int(round(len(X) * noise_pct)))
        span = X.max(axis=0) - X.min(axis=0)
        extra = rng.uniform(X.min(axis=0) - 0.15 * span, X.max(axis=0) + 0.15 * span, (m, d))
        X = np.vstack([X, extra])
        labels = np.concatenate([labels, np.full(m, -1)])

    order = rng.permutation(len(X))
    X, labels = X[order], labels[order]

    # satu peta afin global (bukan per kolom) agar geometri antar klaster utuh
    g_lo, g_hi = X.min(), X.max()
    X = (X - g_lo) / (g_hi - g_lo if g_hi - g_lo > 1e-12 else 1.0)

    data = {}
    for j, spec in enumerate(cols):
        v = spec["min"] + X[:, j] * (spec["max"] - spec["min"])
        v = _apply_outliers(v, spec, rng)
        data[spec["nama"]] = _finish_numeric(v, spec, rng, p.get("persen_kosong_global", 0.0))
    df = pd.DataFrame(data)
    if p.get("sertakan_label_asli", False):
        df["klaster_asli"] = labels
    return _post_process(df, p, rng)


# ================================================================ ASOSIASI
def _gen_association(p: dict, rng) -> pd.DataFrame:
    n_trx = int(p["jumlah_transaksi"])
    n_items = int(np.clip(int(p.get("jumlah_item", 15)), 2, 300))
    nama_item = _parse_list(p.get("daftar_item", ""))
    if len(nama_item) < n_items:
        pool = list(ITEM_CATALOG)
        nama_item = nama_item + [i for i in pool if i not in nama_item]
        while len(nama_item) < n_items:
            nama_item.append(f"Item {len(nama_item) + 1:03d}")
    items = nama_item[:n_items]

    skew = float(np.clip(p.get("kemiringan_popularitas", 0.8), 0.0, 3.0))
    pop = 1.0 / np.power(np.arange(1, n_items + 1), skew)
    pop = pop / pop.sum()

    n_rules = int(np.clip(int(p.get("jumlah_aturan", 4)), 0, 30))
    a_size = int(np.clip(int(p.get("ukuran_antecedent", 2)), 1, 4))
    c_size = int(np.clip(int(p.get("ukuran_consequent", 1)), 1, 3))
    sup = float(np.clip(p.get("support_aturan", 0.15), 0.01, 0.9))
    conf = float(np.clip(p.get("confidence_aturan", 0.8), 0.05, 1.0))

    rules = []
    for _ in range(n_rules):
        pick = rng.choice(n_items, size=min(a_size + c_size, n_items), replace=False, p=pop)
        rules.append((list(pick[:a_size]), list(pick[a_size:a_size + c_size])))

    mean_size = float(np.clip(p.get("rata_ukuran_keranjang", 4.0), 1.0, n_items))
    std_size = float(max(p.get("std_ukuran_keranjang", 1.5), 0.1))
    noise_rate = float(np.clip(p.get("persen_item_bising", 10.0), 0, 100)) / 100.0

    baskets = []
    for _ in range(n_trx):
        chosen = set()
        for ante, cons in rules:
            if rng.random() < sup:
                chosen.update(ante)
                if rng.random() < conf:
                    chosen.update(cons)
        target = int(np.clip(round(rng.normal(mean_size, std_size)), 1, n_items))
        guard = 0
        while len(chosen) < target and guard < 200:
            guard += 1
            if rng.random() < noise_rate:
                chosen.add(int(rng.integers(0, n_items)))
            else:
                chosen.add(int(rng.choice(n_items, p=pop)))
        baskets.append(sorted(chosen))

    fmt = str(p.get("format", "long"))
    start = pd.Timestamp(str(p.get("tanggal_mulai", "2025-01-01")))
    if fmt == "basket":
        mat = np.zeros((n_trx, n_items), dtype=int)
        for i, b in enumerate(baskets):
            mat[i, b] = 1
        df = pd.DataFrame(mat, columns=items)
        df.insert(0, "id_transaksi", [f"TRX{i:05d}" for i in range(1, n_trx + 1)])
        return df
    if fmt == "delimited":
        return pd.DataFrame({
            "id_transaksi": [f"TRX{i:05d}" for i in range(1, n_trx + 1)],
            "daftar_item": [", ".join(items[j] for j in b) for b in baskets],
        })
    rows = []
    for i, b in enumerate(baskets, start=1):
        tanggal = start + pd.Timedelta(days=int(rng.integers(0, max(int(p.get("rentang_hari", 90)), 1))))
        for j in b:
            row = {"id_transaksi": f"TRX{i:05d}", "item": items[j]}
            if p.get("kolom_tanggal", True):
                row["tanggal"] = tanggal.strftime("%Y-%m-%d")
            if p.get("kolom_jumlah", True):
                row["jumlah"] = int(rng.integers(1, int(p.get("maks_jumlah", 3)) + 1))
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================== TIME SERIES
def _gen_timeseries(p: dict, rng) -> pd.DataFrame:
    n = int(p["jumlah_periode"])
    freq = str(p.get("frekuensi", "MS"))
    freq = freq if freq in FREQS else "MS"
    start = pd.Timestamp(str(p.get("tanggal_mulai", "2020-01-01")))
    idx = pd.date_range(start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)

    level = float(p.get("level_dasar", 1000.0))
    tren_tipe = str(p.get("tipe_tren", "linear"))
    kekuatan = float(p.get("kekuatan_tren", 0.5))
    if tren_tipe == "linear":
        tren = level * kekuatan * t / max(n, 1)
    elif tren_tipe == "eksponensial":
        tren = level * (np.power(1.0 + kekuatan / max(n, 1), t) - 1.0)
    elif tren_tipe == "logistik":
        mid, k = n / 2.0, 10.0 / max(n, 1)
        tren = level * kekuatan / (1.0 + np.exp(-k * (t - mid)))
    else:
        tren = np.zeros(n)

    m = int(p.get("periode_musiman", FREQS.get(freq, 12)))
    amp1 = float(p.get("amplitudo_musiman", 0.15)) * level
    amp2 = float(p.get("amplitudo_harmonik", 0.05)) * level
    musim = np.zeros(n)
    if m > 1 and (amp1 or amp2):
        musim = amp1 * np.sin(2 * np.pi * t / m) + amp2 * np.cos(4 * np.pi * t / m)
    m2 = int(p.get("periode_musiman_2", 0))
    if m2 > 1:
        musim = musim + float(p.get("amplitudo_musiman_2", 0.0)) * level * np.sin(2 * np.pi * t / m2)

    noise_std = float(p.get("persen_noise", 3.0)) / 100.0 * level
    noise = rng.normal(0, max(noise_std, 1e-9), n)

    mode = str(p.get("mode_musiman", "aditif"))
    if mode == "multiplikatif":
        base = level + tren
        nilai = base * (1.0 + musim / max(level, 1e-9)) + noise
    else:
        nilai = level + tren + musim + noise

    # perubahan level (level shift / structural break)
    n_shift = int(np.clip(int(p.get("jumlah_pergeseran_level", 0)), 0, 10))
    shift_info = []
    if n_shift > 0 and n > 4:
        pts = sorted(rng.choice(np.arange(n // 10 + 1, n - 1), size=min(n_shift, n - 2), replace=False))
        for cp in pts:
            delta = level * float(p.get("besar_pergeseran", 0.15)) * rng.choice([-1.0, 1.0])
            nilai[cp:] += delta
            shift_info.append((int(cp), float(delta)))

    # anomali
    anom_pct = float(np.clip(p.get("persen_anomali", 2.0), 0, 30)) / 100.0
    anom_mag = float(p.get("besar_anomali", 4.0))
    tipe_anom = str(p.get("tipe_anomali", "keduanya"))
    anom_idx = np.array([], dtype=int)
    if anom_pct > 0:
        k = max(1, int(round(n * anom_pct)))
        anom_idx = rng.choice(n, size=min(k, n), replace=False)
        sigma = max(np.std(nilai), 1e-9)
        arah = (np.ones(len(anom_idx)) if tipe_anom == "lonjakan"
                else -np.ones(len(anom_idx)) if tipe_anom == "penurunan"
                else rng.choice([-1.0, 1.0], size=len(anom_idx)))
        nilai[anom_idx] += arah * anom_mag * sigma * rng.uniform(0.7, 1.3, len(anom_idx))

    lo = p.get("nilai_min")
    if lo is not None:
        nilai = np.maximum(nilai, float(lo))
    dec = int(np.clip(int(p.get("desimal", 2)), 0, 6))

    date_col = str(p.get("nama_kolom_waktu", "tanggal"))
    val_col = str(p.get("nama_kolom_nilai", "nilai"))

    per_period = int(np.clip(int(p.get("catatan_per_periode", 1)), 1, 50))
    if per_period == 1:
        df = pd.DataFrame({date_col: idx.strftime("%Y-%m-%d"), val_col: np.round(nilai, dec)})
    else:
        # data mentah sub-periode -> perlu diagregasi oleh modul time series
        step = pd.tseries.frequencies.to_offset(freq)
        rows_d, rows_v = [], []
        sub_noise = float(p.get("persen_noise_subperiode", 5.0)) / 100.0 * level
        for i, ts in enumerate(idx):
            try:
                nxt = ts + step
                span = max((nxt - ts).days, 1)
            except Exception:
                span = 1
            for _ in range(per_period):
                offset = int(rng.integers(0, span))
                rows_d.append((ts + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"))
                rows_v.append(nilai[i] + rng.normal(0, max(sub_noise, 1e-9)))
        df = pd.DataFrame({date_col: rows_d, val_col: np.round(rows_v, dec)})

    miss = float(np.clip(p.get("persen_kosong", 0.0), 0, 50))
    if miss > 0:
        df[val_col] = _apply_missing(df[val_col].astype(float), miss, rng)
    if dec == 0 and df[val_col].notna().all():
        df[val_col] = df[val_col].astype("int64")
    if p.get("sertakan_penanda_anomali", False) and per_period == 1:
        flag = np.zeros(n, dtype=int)
        flag[anom_idx] = 1
        df["anomali"] = flag
    return df


# =============================================================== dispatcher
BUILDERS = {
    "classification": _gen_classification,
    "regression": _gen_regression,
    "clustering": _gen_clustering,
    "association": _gen_association,
    "timeseries": _gen_timeseries,
}


def generate(config: dict) -> pd.DataFrame:
    cfg = normalize_config(config)
    rng = _rng(cfg["seed"])
    return BUILDERS[cfg["process"]](cfg["params"], rng)


# ========================================================= konfigurasi baku
def default_params(process: str) -> dict:
    if process == "classification":
        return {
            "jumlah_baris": 500,
            "kolom": default_columns(5, 2),
            "nama_target": "kelas_target",
            "kelas": ["Kelas A", "Kelas B"],
            "bobot_kelas": "0.6, 0.4",
            "keterpisahan": 0.7,
            "persen_label_bising": 2.0,
            "persen_kosong_global": 0.0,
            "persen_duplikat": 0.0,
            "kolom_id": True,
            "prefix_id": "ID",
        }
    if process == "regression":
        return {
            "jumlah_baris": 500,
            "kolom": default_columns(5, 1),
            "nama_target": "nilai_target",
            "target_mean": 1000.0,
            "target_std": 250.0,
            "target_min": 0.0,
            "target_max": 5000.0,
            "target_desimal": 2,
            "target_persen_kosong": 0.0,
            "r2_target": 0.75,
            "nonlinearitas": 0.0,
            "heteroskedastisitas": 0.0,
            "persen_kosong_global": 0.0,
            "persen_duplikat": 0.0,
            "kolom_id": True,
            "prefix_id": "ID",
        }
    if process == "clustering":
        return {
            "jumlah_baris": 400,
            "kolom": default_columns(4, 0),
            "jumlah_klaster": 3,
            "bobot_klaster": "",
            "pemisahan": 3.0,
            "kerapatan": 1.0,
            "bentuk": "bulat",
            "persen_noise": 3.0,
            "persen_kosong_global": 0.0,
            "persen_duplikat": 0.0,
            "sertakan_label_asli": False,
            "kolom_id": True,
            "prefix_id": "C",
        }
    if process == "association":
        return {
            "jumlah_transaksi": 800,
            "jumlah_item": 20,
            "daftar_item": "",
            "kemiringan_popularitas": 0.8,
            "jumlah_aturan": 5,
            "ukuran_antecedent": 2,
            "ukuran_consequent": 1,
            "support_aturan": 0.15,
            "confidence_aturan": 0.8,
            "rata_ukuran_keranjang": 4.0,
            "std_ukuran_keranjang": 1.5,
            "persen_item_bising": 10.0,
            "format": "long",
            "kolom_tanggal": True,
            "kolom_jumlah": True,
            "maks_jumlah": 3,
            "tanggal_mulai": "2025-01-01",
            "rentang_hari": 90,
        }
    return {
        "jumlah_periode": 120,
        "frekuensi": "MS",
        "tanggal_mulai": "2018-01-01",
        "level_dasar": 1000.0,
        "tipe_tren": "linear",
        "kekuatan_tren": 0.6,
        "periode_musiman": 12,
        "amplitudo_musiman": 0.15,
        "amplitudo_harmonik": 0.05,
        "periode_musiman_2": 0,
        "amplitudo_musiman_2": 0.0,
        "mode_musiman": "aditif",
        "persen_noise": 3.0,
        "persen_anomali": 2.0,
        "besar_anomali": 4.0,
        "tipe_anomali": "keduanya",
        "jumlah_pergeseran_level": 0,
        "besar_pergeseran": 0.15,
        "persen_kosong": 0.0,
        "catatan_per_periode": 1,
        "persen_noise_subperiode": 5.0,
        "desimal": 2,
        "nilai_min": 0.0,
        "nama_kolom_waktu": "tanggal",
        "nama_kolom_nilai": "nilai",
        "sertakan_penanda_anomali": False,
    }


def default_config(process: str, seed: int = 42) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "process": process,
        "seed": int(seed),
        "dibuat": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": default_params(process),
    }


def normalize_config(config: dict) -> dict:
    """Lengkapi field yang hilang & paksa tipe agar konfigurasi selalu valid."""
    if not isinstance(config, dict):
        raise ValueError("Konfigurasi harus berupa objek JSON / Configuration must be a JSON object")
    process = str(config.get("process", "classification"))
    if process not in PROCESSES:
        raise ValueError(f"Proses tidak dikenal / unknown process: {process}")
    base = default_params(process)
    params = dict(base)
    params.update({k: v for k, v in dict(config.get("params", {})).items() if k in base or k == "kolom"})
    if "kolom" in base:
        raw_cols = params.get("kolom") or base["kolom"]
        params["kolom"] = _unique_names(
            [normalize_column(c, i + 1) for i, c in enumerate(raw_cols)]) or base["kolom"]
    if process == "classification":
        kelas = [str(k) for k in (params.get("kelas") or base["kelas"])]
        params["kelas"] = kelas if len(kelas) >= 2 else base["kelas"]
    seed = config.get("seed", 42)
    try:
        seed = int(seed)
    except Exception:
        seed = 42
    return {"schema_version": SCHEMA_VERSION, "process": process, "seed": seed,
            "dibuat": config.get("dibuat", datetime.now().strftime("%Y-%m-%d %H:%M")),
            "params": params}


def validate_config(config: dict) -> tuple[bool, list[str]]:
    msgs = []
    try:
        cfg = normalize_config(config)
    except Exception as e:
        return False, [str(e)]
    p, proc = cfg["params"], cfg["process"]
    if proc in ("classification", "regression", "clustering"):
        if int(p["jumlah_baris"]) < 20:
            msgs.append("Jumlah baris minimal 20 / At least 20 rows required")
        if len(p["kolom"]) < 1:
            msgs.append("Minimal 1 kolom fitur / At least 1 feature column required")
    if proc == "clustering":
        num = [c for c in p["kolom"] if c["tipe"] == "numerik"]
        if len(num) < 2:
            msgs.append("Klastering butuh minimal 2 kolom numerik / Clustering needs 2+ numeric columns")
        if int(p["jumlah_klaster"]) > int(p["jumlah_baris"]) / 5:
            msgs.append("Jumlah klaster terlalu besar dibanding jumlah baris / Too many clusters for the row count")
    if proc == "classification" and len(p["kelas"]) < 2:
        msgs.append("Butuh minimal 2 kelas / At least 2 classes required")
    if proc == "association":
        if int(p["jumlah_transaksi"]) < 30:
            msgs.append("Minimal 30 transaksi / At least 30 transactions")
        if float(p["rata_ukuran_keranjang"]) > int(p["jumlah_item"]):
            msgs.append("Rata-rata ukuran keranjang melebihi jumlah item / Basket size exceeds item count")
    if proc == "timeseries":
        if int(p["jumlah_periode"]) < 24:
            msgs.append("Minimal 24 periode agar model musiman dapat dilatih / At least 24 periods for seasonal models")
        if int(p["periode_musiman"]) > int(p["jumlah_periode"]) / 2:
            msgs.append("Periode musiman terlalu panjang dibanding jumlah data / Seasonal period too long for the series")
    return len(msgs) == 0, msgs


# ============================================================ JSON I/O
def config_to_json(config: dict) -> str:
    return json.dumps(normalize_config(config), indent=2, ensure_ascii=False)


def json_to_config(text: str | bytes) -> dict:
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", errors="replace")
    data = json.loads(text)
    return normalize_config(data)


def summarize(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan singkat konfigurasi + hasil, untuk sheet Informasi."""
    cfg = normalize_config(config)
    rows = [("Proses / Process", cfg["process"]), ("Seed", cfg["seed"]),
            ("Baris dihasilkan / Rows generated", len(df)),
            ("Kolom dihasilkan / Columns generated", df.shape[1])]
    for k, v in cfg["params"].items():
        if k == "kolom":
            rows.append(("Jumlah spesifikasi kolom / Column specs", len(v)))
            continue
        rows.append((k, ", ".join(map(str, v)) if isinstance(v, list) else str(v)))
    return pd.DataFrame(rows, columns=["Parameter", "Nilai / Value"])
