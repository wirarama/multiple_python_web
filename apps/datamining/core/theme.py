"""Tema visual terpusat: palet tervalidasi CVD + gaya matplotlib.

Palet kategorik dipakai dalam urutan tetap (tidak pernah diputar/cycled).
Sequential = satu hue terang->gelap. Diverging = biru<->merah dengan abu netral.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ palet
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
# Untuk bentuk "semua pasangan" (scatter/bubble) hanya 3 slot pertama yang
# lolos gate CVD all-pairs -> sisanya dilipat jadi "Lainnya"/difacet.
CATEGORICAL_SAFE3 = CATEGORICAL[:3]

SURFACE = "#fcfcfb"
GRID = "#e3e2de"
INK = "#0b0b0b"
INK_MUTED = "#52514e"

STATUS = {
    "good": "#008300",
    "warning": "#eda100",
    "serious": "#eb6834",
    "critical": "#e34948",
}

SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("dmui_seq", SEQ_STEPS)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "dmui_div", ["#0d366b", "#3987e5", "#cde2fb", "#f0efec", "#f6c3c2", "#e34948", "#8f1f1e"]
)


def color(i: int) -> str:
    """Warna kategorik slot ke-i (urutan tetap, di-clamp bukan di-cycle)."""
    return CATEGORICAL[i % len(CATEGORICAL)]


# ------------------------------------------------------------ rcParams
def apply_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
        }
    )


APP_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e3e2de; border-radius: 10px;
    padding: 12px 14px 10px 14px;
}
section[data-testid="stSidebar"] {background: #f7f7f5; border-right: 1px solid #e3e2de;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {padding: 8px 14px; border-radius: 8px 8px 0 0;}
.dm-card {
    background:#ffffff; border:1px solid #e3e2de; border-radius:12px;
    padding:16px 18px; margin-bottom:14px;
}
.dm-note {
    background:#f3f7fd; border-left:3px solid #2a78d6; border-radius:6px;
    padding:10px 14px; margin:8px 0 14px 0; font-size:0.9rem; color:#1f2937;
}
.dm-warn {
    background:#fdf6e8; border-left:3px solid #eda100; border-radius:6px;
    padding:10px 14px; margin:8px 0 14px 0; font-size:0.9rem; color:#1f2937;
}
.dm-caption {color:#52514e; font-size:0.85rem;}
</style>
"""
