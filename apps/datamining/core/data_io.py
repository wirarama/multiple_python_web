"""Pembacaan & pengenalan tipe data dari CSV/Excel."""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

MAX_PREVIEW = 200


def is_text_dtype(s: pd.Series) -> bool:
    """Kolom teks — tahan terhadap pandas 2.x (object) maupun 3.x (str)."""
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype))


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols, seen = [], {}
    for c in df.columns:
        name = re.sub(r"\s+", " ", str(c)).strip()
        if name == "" or name.lower().startswith("unnamed:"):
            name = f"kolom_{len(cols) + 1}"
        base = name
        if name in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        cols.append(name)
    df.columns = cols
    return df


def _strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if is_text_dtype(df[c]):
            df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def read_table(file, sheet_name=None) -> pd.DataFrame:
    """Baca CSV/XLSX dari file-like Streamlit atau path."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(file, sheet_name=sheet_name or 0)
        if isinstance(df, dict):
            df = list(df.values())[0]
    else:
        raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
        text = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        sep = ";" if text.count(";") > text.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        # angka gaya Indonesia: 1.234,56
        if sep == ";":
            for c in df.columns:
                if is_text_dtype(df[c]):
                    s = df[c].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                    conv = pd.to_numeric(s, errors="coerce")
                    if conv.notna().mean() > 0.9:
                        df[c] = conv
    return _strip_object_cols(_clean_columns(df))


def excel_sheet_names(file) -> list[str]:
    name = getattr(file, "name", str(file)).lower()
    if not name.endswith((".xlsx", ".xls", ".xlsm")):
        return []
    try:
        xl = pd.ExcelFile(file)
        names = list(xl.sheet_names)
        try:
            file.seek(0)
        except Exception:
            pass
        return names
    except Exception:
        return []


# ------------------------------------------------------------ tipe kolom
def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_cols(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            if len(vals) and vals.nunique() <= 10 and bool(np.all(np.mod(vals.to_numpy(dtype=float), 1) == 0)):
                out.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            continue
        else:
            if s.dropna().nunique() <= max_unique:
                out.append(c)
    return out


def all_categorical_like(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


ISO_RE = r"^\s*\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?"


def to_datetime_series(s: pd.Series) -> pd.Series:
    """Parse tanggal secara aman.

    Format ISO (YYYY-MM-DD) diparse apa adanya; format lain (DD/MM/YYYY yang
    lazim di Indonesia) diparse dengan dayfirst=True. Memakai dayfirst pada
    tanggal ISO menyebabkan 2025-12-01 salah terbaca sebagai 12 Januari 2025.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    txt = s.astype(str).str.strip()
    sample = txt[txt.str.lower().ne("nan")].head(200)
    if len(sample) == 0:
        return pd.to_datetime(s, errors="coerce")
    is_iso = bool(sample.str.match(ISO_RE).mean() > 0.8)
    attempts = (
        [{"format": "ISO8601"}, {"dayfirst": False, "format": "mixed"}]
        if is_iso
        else [{"dayfirst": True, "format": "mixed"}, {"dayfirst": False, "format": "mixed"}]
    )
    for kw in attempts:
        try:
            out = pd.to_datetime(txt, errors="coerce", **kw)
        except Exception:
            continue
        if out.notna().mean() > 0.85:
            return out
    return pd.to_datetime(s, errors="coerce")


def datetime_candidates(df: pd.DataFrame) -> list[str]:
    """Kolom yang bisa di-parse sebagai tanggal."""
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        sample = s.dropna().head(60)
        if len(sample) == 0:
            continue
        if to_datetime_series(sample).notna().mean() > 0.85:
            out.append(c)
    return out


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def infer_type_table(df: pd.DataFrame, lang_id: bool = True) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            kind = "Numerik" if lang_id else "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "Tanggal" if lang_id else "Datetime"
        elif s.dropna().nunique() <= 20:
            kind = "Kategorik" if lang_id else "Categorical"
        else:
            kind = "Teks" if lang_id else "Text"
        rows.append(
            {
                ("Kolom" if lang_id else "Column"): c,
                ("Tipe Data" if lang_id else "Dtype"): str(s.dtype),
                ("Kategori" if lang_id else "Kind"): kind,
                ("Nilai Unik" if lang_id else "Unique"): int(s.nunique(dropna=True)),
                ("Kosong" if lang_id else "Missing"): int(s.isna().sum()),
                ("% Kosong" if lang_id else "% Missing"): round(float(s.isna().mean() * 100), 2),
                ("Contoh Nilai" if lang_id else "Sample"): ", ".join(
                    map(lambda v: str(v)[:18], s.dropna().unique()[:3])
                ),
            }
        )
    return pd.DataFrame(rows)


def safe_sample(df: pd.DataFrame, n: int = MAX_PREVIEW) -> pd.DataFrame:
    return df.head(n)


def basic_clean(df: pd.DataFrame, drop_dupes: bool, missing_strategy: str, target: str | None = None):
    """Bersihkan data. missing_strategy: none|drop|mean_mode|median_mode."""
    report = []
    out = df.copy()
    n0 = len(out)
    if drop_dupes:
        out = out.drop_duplicates()
        report.append(("Baris duplikat dihapus / Duplicate rows removed", n0 - len(out)))
    if target and target in out.columns:
        before = len(out)
        out = out[out[target].notna()]
        report.append(("Baris tanpa target dihapus / Rows with empty target removed", before - len(out)))
    if missing_strategy == "drop":
        before = len(out)
        out = out.dropna()
        report.append(("Baris dengan nilai kosong dihapus / Rows with missing values removed", before - len(out)))
    elif missing_strategy in ("mean_mode", "median_mode"):
        filled = 0
        for c in out.columns:
            if out[c].isna().sum() == 0:
                continue
            filled += int(out[c].isna().sum())
            if pd.api.types.is_numeric_dtype(out[c]):
                val = out[c].mean() if missing_strategy == "mean_mode" else out[c].median()
                out[c] = out[c].fillna(val)
            else:
                mode = out[c].mode(dropna=True)
                out[c] = out[c].fillna(mode.iloc[0] if len(mode) else "TIDAK_DIKETAHUI")
        report.append(("Sel kosong diisi / Missing cells imputed", filled))
    return out, report
