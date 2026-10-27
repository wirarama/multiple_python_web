"""Simpan & muat seluruh konfigurasi proses sebagai berkas JSON.

Cara kerja
----------
Semua widget konfigurasi memakai kunci `st.session_state` berawalan tetap per
proses (`cls_`, `reg_`, `clu_`, `ar_`, `ts_`). Menyimpan konfigurasi = menulis
seluruh pasangan kunci-nilai itu ke JSON; memuat konfigurasi = menaruhnya
kembali ke `st.session_state` SEBELUM widget dibuat, lalu `st.rerun()`.

Karena nilai lama bisa tidak sah untuk data yang sedang dibuka (mis. nama kolom
tidak ada), setiap widget dibuat lewat pembungkus di modul ini yang membuang
nilai tak sah lebih dulu, sehingga aplikasi tidak pernah gagal karena berkas
konfigurasi yang tidak cocok.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import streamlit as st

from .i18n import L

SCHEMA_VERSION = 1
APP_ID = "dataminingui"

PREFIX = {
    "classification": "cls_",
    "regression": "reg_",
    "clustering": "clu_",
    "association": "ar_",
    "timeseries": "ts_",
}

PROC_LABEL = {
    "classification": ("Klasifikasi", "Classification"),
    "regression": ("Regresi", "Regression"),
    "clustering": ("Klastering", "Clustering"),
    "association": ("Association Rule Mining", "Association Rule Mining"),
    "timeseries": ("Time Series", "Time Series"),
}


def _pending_key(proc: str) -> str:
    return f"_pending_settings_{proc}"


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------- pembungkus
def _drop_if(key: str, ok) -> None:
    if key in st.session_state:
        try:
            if not ok(st.session_state[key]):
                del st.session_state[key]
        except Exception:
            try:
                del st.session_state[key]
            except Exception:
                pass


def _c(container):
    return container if container is not None else st


def selectbox(label, options, key, index=0, container=None, **kw):
    options = list(options)
    _drop_if(key, lambda v: v in options)
    if not options:
        return None
    index = min(max(int(index), 0), len(options) - 1)
    return _c(container).selectbox(label, options, index=index, key=key, **kw)


def radio(label, options, key, index=0, container=None, **kw):
    options = list(options)
    _drop_if(key, lambda v: v in options)
    index = min(max(int(index), 0), len(options) - 1)
    return _c(container).radio(label, options, index=index, key=key, **kw)


def multiselect(label, options, key, default=None, container=None, **kw):
    options = list(options)
    opt_set = set(options)
    if key in st.session_state:
        cur = st.session_state[key]
        if isinstance(cur, (list, tuple)):
            keep = [v for v in cur if v in opt_set]
            if keep:
                st.session_state[key] = keep
            else:
                del st.session_state[key]
        else:
            del st.session_state[key]
    default = [v for v in (default or []) if v in opt_set]
    return _c(container).multiselect(label, options, default=default, key=key, **kw)


def _num_ok(lo, hi):
    def ok(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        return lo <= v <= hi
    return ok


def _retype(key, is_int):
    if key in st.session_state:
        try:
            st.session_state[key] = (int(round(float(st.session_state[key]))) if is_int
                                     else float(st.session_state[key]))
        except (TypeError, ValueError):
            del st.session_state[key]


def slider(label, lo, hi, value, step=None, key=None, container=None, **kw):
    is_int = isinstance(lo, int) and isinstance(hi, int) and (step is None or isinstance(step, int))
    _drop_if(key, _num_ok(lo, hi))
    _retype(key, is_int)
    args = dict(min_value=lo, max_value=hi, value=value, key=key, **kw)
    if step is not None:
        args["step"] = step
    return _c(container).slider(label, **args)


def number_input(label, lo, hi, value, step=None, key=None, container=None, **kw):
    is_int = isinstance(lo, int) and isinstance(hi, int) and isinstance(value, int)
    _drop_if(key, _num_ok(lo, hi))
    _retype(key, is_int)
    args = dict(min_value=lo, max_value=hi, value=value, key=key, **kw)
    if step is not None:
        args["step"] = step
    return _c(container).number_input(label, **args)


def checkbox(label, value, key, container=None, **kw):
    _drop_if(key, lambda v: isinstance(v, bool))
    return _c(container).checkbox(label, value, key=key, **kw)


def text_input(label, value, key, container=None, **kw):
    _drop_if(key, lambda v: isinstance(v, str))
    return _c(container).text_input(label, value, key=key, **kw)


# --------------------------------------------------------- ambil & terapkan
def snapshot(proc: str) -> dict:
    """Seluruh nilai widget milik proses ini yang dapat ditulis ke JSON."""
    pre = PREFIX[proc]
    out = {}
    for k in list(st.session_state):
        if not isinstance(k, str) or not k.startswith(pre):
            continue
        v = st.session_state[k]
        if _jsonable(v):
            out[k] = v
    return dict(sorted(out.items()))


def to_json(proc: str, cfg: dict | None = None, extra: dict | None = None) -> bytes:
    doc = {
        "app": APP_ID,
        "schema_version": SCHEMA_VERSION,
        "process": proc,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "widgets": snapshot(proc),
        "config": cfg if _jsonable(cfg) else None,
    }
    if extra:
        doc.update({k: v for k, v in extra.items() if _jsonable(v)})
    return json.dumps(doc, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def parse(raw: bytes | str) -> dict:
    doc = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if not isinstance(doc, dict):
        raise ValueError("format bukan objek JSON")
    if doc.get("app") != APP_ID:
        raise ValueError("berkas ini bukan konfigurasi WebUI Data Mining")
    if doc.get("process") not in PREFIX:
        raise ValueError(f"proses tidak dikenal: {doc.get('process')!r}")
    if not isinstance(doc.get("widgets"), dict):
        raise ValueError("bagian 'widgets' tidak ditemukan")
    return doc


def stage(proc: str, widgets: dict) -> None:
    """Simpan nilai untuk diterapkan pada rerun berikutnya."""
    pre = PREFIX[proc]
    st.session_state[_pending_key(proc)] = {
        k: v for k, v in widgets.items() if isinstance(k, str) and k.startswith(pre)
    }


def consume_pending(proc: str) -> int:
    """Terapkan konfigurasi tertunda. Panggil di awal `config()` tiap modul."""
    pend = st.session_state.pop(_pending_key(proc), None)
    if not pend:
        return 0
    pre = PREFIX[proc]
    for k in [k for k in list(st.session_state)
              if isinstance(k, str) and k.startswith(pre) and k not in pend]:
        try:
            del st.session_state[k]
        except Exception:
            pass
    n = 0
    for k, v in pend.items():
        try:
            st.session_state[k] = v
            n += 1
        except Exception:
            continue
    return n


def reset(proc: str) -> None:
    pre = PREFIX[proc]
    for k in [k for k in list(st.session_state) if isinstance(k, str) and k.startswith(pre)]:
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.session_state.pop(_pending_key(proc), None)
    st.session_state.pop(f"_settings_hash_{proc}", None)


# -------------------------------------------------------------- panel UI
def panel(proc: str, cfg: dict | None = None, extra: dict | None = None) -> None:
    """Panel simpan/muat konfigurasi. Letakkan di akhir `config()`."""
    label = L(*PROC_LABEL[proc])
    with st.expander("💾 " + L("Simpan / muat pengaturan (JSON)",
                               "Save / load settings (JSON)"), expanded=False):
        blurb = L(
            "Seluruh isian pada halaman ini — kolom target, fitur, proporsi uji, penskalaan, "
            "validasi silang, algoritma yang dipilih, <b>dan seluruh parameter tiap algoritma</b> — "
            "disimpan dalam satu berkas JSON. Muat kembali berkas itu untuk memulihkan persis "
            "pengaturan yang sama, atau bagikan kepada orang lain agar analisisnya dapat "
            "direproduksi.",
            "Every field on this page — target column, features, test size, scaling, "
            "cross-validation, chosen algorithms, <b>and every algorithm parameter</b> — is stored "
            "in a single JSON file. Load it back to restore exactly the same setup, or share it "
            "so someone else can reproduce the analysis.")
        st.markdown(f'<div class="dm-note">{blurb}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button(
                "⬇ " + L("Unduh pengaturan (.json)", "Download settings (.json)"),
                data=to_json(proc, cfg, extra),
                file_name=f"pengaturan_{proc}_{datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json",
                width="stretch",
                key=f"dl_set_{proc}",
            )
            st.caption(L(f"Berisi {len(snapshot(proc))} isian untuk proses {label}.",
                         f"Contains {len(snapshot(proc))} fields for the {label} process."))
        with c2:
            if st.button("↺ " + L("Kembalikan ke pengaturan bawaan", "Reset to defaults"),
                         width="stretch", key=f"rst_set_{proc}"):
                reset(proc)
                st.rerun()

        up = st.file_uploader(
            L("Muat pengaturan dari berkas .json", "Load settings from a .json file"),
            type=["json"], key=f"up_set_{proc}",
            help=L("Berkas JSON hasil unduhan dari panel ini.",
                   "A JSON file previously downloaded from this panel."))
        if up is None:
            return
        raw = up.getvalue()
        h = hashlib.md5(raw).hexdigest()
        if st.session_state.get(f"_settings_hash_{proc}") == h:
            st.caption(L("Berkas ini sudah diterapkan.", "This file has already been applied."))
            return
        try:
            doc = parse(raw)
        except Exception as e:
            st.error(L(f"Konfigurasi tidak valid: {e}", f"Invalid configuration: {e}"))
            return
        if doc["process"] != proc:
            other = L(*PROC_LABEL[doc["process"]])
            st.error(L(
                f"Berkas ini untuk proses '{other}', sedangkan proses aktif '{label}'. "
                "Ganti proses di sidebar lebih dulu.",
                f"This file targets the '{other}' process while the active one is '{label}'. "
                "Switch process in the sidebar first."))
            return
        st.session_state[f"_settings_hash_{proc}"] = h
        stage(proc, doc["widgets"])
        st.success(L(f"{len(doc['widgets'])} isian dimuat — memuat ulang halaman...",
                     f"{len(doc['widgets'])} fields loaded — reloading the page..."))
        st.rerun()
