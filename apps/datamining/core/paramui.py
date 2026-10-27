"""Editor parameter algoritma: setiap hiperparameter dapat diubah dari antarmuka.

Nilai disimpan pada `st.session_state` dengan kunci berawalan proses sehingga
ikut terbawa saat pengaturan disimpan/dimuat sebagai JSON (lihat `core.settings`).
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from . import params as P
from . import settings as SET
from . import ui as U
from .i18n import L


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def key_of(prefix: str, algo: str, param: str) -> str:
    return f"{prefix}_p_{slug(algo)}_{param}"


def _help(sp: dict) -> str:
    return L(*sp["help"])


def _is_sci(sp: dict) -> bool:
    return sp["kind"] == "float" and str(sp.get("fmt", "")).endswith("e")


def _widget(prefix: str, algo: str, name: str, sp: dict, container):
    """Bangun satu widget parameter dan kembalikan nilainya."""
    k = key_of(prefix, algo, name)
    hint = _help(sp)
    with container:
        if sp["kind"] == "bool":
            return SET.checkbox(name, bool(sp["default"]), key=k, help=hint)
        if sp["kind"] == "choice":
            opts = list(sp["options"])
            idx = opts.index(sp["default"]) if sp["default"] in opts else 0
            return SET.selectbox(name, opts, key=k, index=idx, help=hint)
        if sp["kind"] == "numstr":
            extra = ""
            if sp.get("options"):
                extra = L(f" Pilihan umum: {', '.join(map(str, sp['options']))}.",
                          f" Common values: {', '.join(map(str, sp['options']))}.")
            return SET.text_input(name, str(sp["default"]), key=k, help=hint + extra)
        if sp["kind"] == "int":
            note = ""
            if sp.get("none_at") is not None:
                note = L(f" ({sp['none_at']} = tanpa batas)", f" ({sp['none_at']} = unlimited)")
            return SET.number_input(name, int(sp["min"]), int(sp["max"]), int(sp["default"]),
                                    step=int(sp.get("step", 1)), key=k, help=hint + note)
        if sp["kind"] == "float":
            if _is_sci(sp):
                return SET.text_input(name, f"{float(sp['default']):.1e}", key=k,
                                      help=hint + L(" Format ilmiah, mis. 1e-9.",
                                                    " Scientific notation, e.g. 1e-9."))
            note = ""
            if sp.get("none_at") is not None:
                note = L(f" ({sp['none_at']:g} = otomatis/seluruh data)",
                         f" ({sp['none_at']:g} = automatic/all rows)")
            return SET.number_input(name, float(sp["min"]), float(sp["max"]), float(sp["default"]),
                                    step=float(sp.get("step", 0.01)), key=k,
                                    format=str(sp.get("fmt", "%.4f")), help=hint + note)
        return SET.text_input(name, str(sp["default"]), key=k, help=hint)


def read(task: str, algo: str, prefix: str) -> dict:
    """Baca nilai parameter satu algoritma dari session_state (tanpa membuat widget)."""
    sp = P.spec(task, algo)
    raw = {n: st.session_state.get(key_of(prefix, algo, n), s["default"]) for n, s in sp.items()}
    return P.coerce_all(task, algo, raw)


def reset_algo(prefix: str, task: str, algo: str) -> None:
    for n in P.spec(task, algo):
        st.session_state.pop(key_of(prefix, algo, n), None)


def params_table(task: str, algo: str, values: dict) -> pd.DataFrame:
    sp = P.spec(task, algo)
    vals = P.coerce_all(task, algo, values)
    rows = []
    for n, s in sp.items():
        rows.append({
            L("Parameter", "Parameter"): n,
            L("Nilai dipakai", "Value used"): str(vals[n]),
            L("Bawaan", "Default"): str(s["default"]),
            L("Diubah?", "Changed?"): L("Ya", "Yes") if vals[n] != s["default"] else "",
            L("Keterangan", "Description"): _help(s),
        })
    return pd.DataFrame(rows)


def _summary(task: str, algo: str, values: dict) -> str:
    ch = P.changed_from_default(task, algo, values)
    if not ch:
        return L("Semua parameter memakai nilai bawaan.", "All parameters are at their defaults.")
    return L("Diubah dari bawaan: ", "Changed from defaults: ") + ", ".join(
        f"`{k}={v}`" for k, v in ch.items())


def algo_block(task: str, algo: str, prefix: str, per_row: int = 3) -> dict:
    """Render seluruh widget parameter untuk satu algoritma."""
    sp = P.spec(task, algo)
    if not sp:
        st.caption(L("Algoritma ini tidak memiliki parameter yang dapat diubah.",
                     "This algorithm has no editable parameters."))
        return {}
    names = list(sp)
    values = {}
    for i in range(0, len(names), per_row):
        cols = st.columns(per_row)
        for col, n in zip(cols, names[i:i + per_row]):
            values[n] = _widget(prefix, algo, n, sp[n], col)
    vals = P.coerce_all(task, algo, values)
    c1, c2 = st.columns([4, 1])
    c1.caption(_summary(task, algo, vals))
    if c2.button(L("Bawaan", "Defaults"), key=f"{prefix}_reset_{slug(algo)}", width="stretch",
                 help=L(f"Kembalikan parameter {algo} ke nilai bawaan.",
                        f"Reset {algo} parameters to their defaults.")):
        reset_algo(prefix, task, algo)
        st.rerun()
    return vals


def editor(task: str, algo_list: list[str], prefix: str, expanded: bool = False,
           title: str | None = None) -> dict:
    """Panel parameter untuk daftar algoritma. Mengembalikan {algoritma: {param: nilai}}."""
    algo_list = [a for a in algo_list if a in P.space(task)]
    if not algo_list:
        return {}
    head = title or L("Parameter algoritma", "Algorithm parameters")
    with st.expander("⚙ " + head, expanded=expanded):
        U.note(L(
            "Setiap parameter di bawah ini dapat diubah dan langsung dipakai saat analisis "
            "dijalankan. Arahkan kursor ke tanda tanya pada tiap isian untuk membaca artinya. "
            "Seluruh nilai ikut tersimpan saat Anda mengunduh pengaturan JSON.",
            "Every parameter below can be edited and is applied when the analysis runs. "
            "Hover the question mark on each field to read what it does. All values are included "
            "when you download the JSON settings."))
        out = {}
        if len(algo_list) == 1:
            out[algo_list[0]] = algo_block(task, algo_list[0], prefix)
        else:
            tabs = st.tabs(algo_list)
            for tab, algo in zip(tabs, algo_list):
                with tab:
                    out[algo] = algo_block(task, algo, prefix)
        return out


def combined_params_table(task: str, algo_list: list[str], per_algo: dict) -> pd.DataFrame:
    """Satu tabel berisi parameter seluruh algoritma yang dipakai (untuk diekspor)."""
    frames = []
    for a in algo_list:
        t = params_table(task, a, (per_algo or {}).get(a, {}))
        if t.empty:
            continue
        t.insert(0, L("Algoritma", "Algorithm"), a)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def projection_controls(prefix: str, feature_names: list[str]):
    """Pilihan bidang 2D untuk menggambar batas keputusan / hyperplane."""
    names = list(feature_names)
    modes = {
        "pca": L("PCA 2 komponen (semua fitur terwakili)",
                 "2 PCA components (all features represented)"),
        "pair": L("Dua fitur asli pilihan saya", "Two original features I choose"),
    }
    c1, c2, c3 = st.columns([2, 1, 1])
    mode = SET.selectbox(L("Bidang penggambaran", "Drawing plane"), list(modes),
                         key=f"{prefix}_viz_mode", index=0, container=c1,
                         format_func=lambda k: modes[k],
                         help=L("Batas keputusan hanya dapat digambar pada bidang dua dimensi. "
                                "PCA memampatkan seluruh fitur menjadi dua sumbu; pilihan kedua "
                                "menggambar pada dua fitur asli sehingga lebih mudah ditafsirkan.",
                                "A decision boundary can only be drawn on a 2-D plane. PCA compresses "
                                "all features into two axes; the second option draws on two original "
                                "features, which is easier to interpret."))
    cols = None
    if mode == "pair" and len(names) >= 2:
        i = SET.selectbox(L("Sumbu X", "X axis"), names, key=f"{prefix}_viz_x", index=0, container=c2)
        j = SET.selectbox(L("Sumbu Y", "Y axis"), names, key=f"{prefix}_viz_y",
                          index=min(1, len(names) - 1), container=c3)
        cols = (names.index(i), names.index(j))
    elif mode == "pair":
        st.caption(L("Butuh minimal dua fitur — memakai PCA.",
                     "At least two features are required — falling back to PCA."))
        mode = "pca"
    return mode, cols
