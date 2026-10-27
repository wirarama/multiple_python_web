"""Ekspor hasil: Excel multi-sheet, PNG per gambar, dan ZIP semua gambar."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime

import pandas as pd

from .i18n import L

HEADER_BG = "#2a78d6"


def _safe_sheet(name: str, used: set) -> str:
    s = re.sub(r"[\[\]\:\*\?\/\\]", "-", str(name))[:31].strip() or "Sheet"
    base = s
    i = 1
    while s.lower() in used:
        suffix = f"_{i}"
        s = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(s.lower())
    return s


def tables_to_excel(tables: dict[str, pd.DataFrame], meta: dict | None = None) -> bytes:
    """tables: {nama_sheet: DataFrame} -> bytes file .xlsx dengan format rapi."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        book = xw.book
        f_head = book.add_format({
            "bold": True, "bg_color": HEADER_BG, "font_color": "#FFFFFF",
            "border": 1, "border_color": "#FFFFFF", "align": "center",
            "valign": "vcenter", "text_wrap": True,
        })
        f_txt = book.add_format({"valign": "top", "text_wrap": True})
        f_num = book.add_format({"num_format": "#,##0.0000", "valign": "top"})
        f_int = book.add_format({"num_format": "#,##0", "valign": "top"})
        f_title = book.add_format({"bold": True, "font_size": 14, "font_color": "#1F3864"})
        f_sub = book.add_format({"font_size": 10, "font_color": "#52514e"})

        # --- sheet informasi
        if meta:
            ws = book.add_worksheet("Informasi")
            ws.set_column(0, 0, 34, f_txt)
            ws.set_column(1, 1, 74, f_txt)
            ws.write(0, 0, L("LAPORAN ANALISIS DATA MINING", "DATA MINING ANALYSIS REPORT"), f_title)
            ws.write(1, 0, datetime.now().strftime("%d %B %Y, %H:%M"), f_sub)
            r = 3
            for k, v in meta.items():
                ws.write(r, 0, str(k), f_head)
                ws.write(r, 1, str(v), f_txt)
                r += 1

        used = {"informasi"}
        for name, df in tables.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            frame = df.copy()
            if isinstance(frame.index, pd.MultiIndex) or (
                frame.index.name is not None and frame.index.name not in frame.columns
            ):
                frame = frame.reset_index()
            frame.columns = [str(c) for c in frame.columns]
            sheet = _safe_sheet(name, used)
            frame.to_excel(xw, sheet_name=sheet, index=False, startrow=0)
            ws = xw.sheets[sheet]
            for j, col in enumerate(frame.columns):
                ws.write(0, j, col, f_head)
                try:
                    body_len = int(frame[col].astype(str).str.len().max())
                except Exception:
                    body_len = 12
                width = min(52, max(11, max(body_len, len(col)) + 3))
                if pd.api.types.is_float_dtype(frame[col]):
                    fmt = f_num
                elif pd.api.types.is_integer_dtype(frame[col]):
                    fmt = f_int
                else:
                    fmt = f_txt
                ws.set_column(j, j, width, fmt)
            ws.freeze_panes(1, 0)
            if len(frame):
                ws.autofilter(0, 0, len(frame), max(len(frame.columns) - 1, 0))
    return buf.getvalue()


def fig_to_png(fig, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()


def figs_to_zip(figs: dict, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (name, fig) in enumerate(figs.items(), start=1):
            if fig is None:
                continue
            fname = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_").lower()
            zf.writestr(f"{i:02d}_{fname}.png", fig_to_png(fig, dpi))
    return buf.getvalue()


def stamp(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M}.{ext}"
