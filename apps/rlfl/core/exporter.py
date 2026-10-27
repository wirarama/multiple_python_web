"""Membangun paket unduhan: CSV jaringan, konfigurasi FL (JSON), script Python
standalone (01/02/03 + run_all) yang dapat dijalankan di luar aplikasi ini.

Script standalone dibuat dengan menyalin (inline) source modul inti pada folder
core/ -- jadi logika yang dijalankan script IDENTIK dengan yang dijalankan
aplikasi web (hasil dengan seed sama akan sama persis)."""
import io
import json
import os
import zipfile
from datetime import datetime

from . import network as netmod
from .federated import DEFAULT_SUPERVISED
from .qrouting import DEFAULT_QROUTING
from .rl_reward import DEFAULT_RL, rl_tables
from .results_io import results_to_tables

CORE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SIMULATION = {"seed": 42, "time_steps": 48, "test_ratio": 0.2, "eval_hour": 19,
                      "noise_pct": 5.0}

SCRIPT_SPECS = {
    "01_data_jaringan.py": {
        "modules": ["network.py"],
        "main": "main_01_data.py",
        "title": "LANGKAH 1 - Data konfigurasi jaringan & simulasi data trafik lokal",
        "desc": ("Membaca data/nodes.csv, data/links.csv, data/clients.csv (atau membangkitkan\n"
                 "topologi baru dengan --generate), memvalidasi, lalu mensimulasikan pengukuran\n"
                 "trafik lokal tiap klien -> data/measurements.csv dan data/snapshot.csv."),
        "usage": "python 01_data_jaringan.py            (atau: python 01_data_jaringan.py --generate)",
        "requires": "numpy, networkx",
    },
    "02_federated_learning.py": {
        "modules": ["network.py", "routing.py", "federated.py", "qrouting.py", "results_io.py"],
        "main": "main_02_fl.py",
        "title": "LANGKAH 2 - Pelatihan Federated Learning untuk optimasi jalur jaringan",
        "desc": ("Membaca CSV jaringan + config/fl_config.json, menjalankan:\n"
                 "  - supervised : FL prediksi delay link (MLP numpy) + routing Dijkstra\n"
                 "  - qrouting   : Federated Reinforcement Learning (Q-Routing)\n"
                 "dan membandingkan model GLOBAL vs LOKAL-saja vs TERPUSAT vs ORACLE.\n"
                 "Hasil CSV/JSON ditulis ke results/<mode>/."),
        "usage": "python 02_federated_learning.py       (opsional: --mode supervised|qrouting|both)",
        "requires": "numpy, networkx",
    },
    "03_visualisasi.py": {
        "modules": ["network.py", "routing.py", "results_io.py", "viz.py"],
        "main": "main_03_viz.py",
        "title": "LANGKAH 3 - Visualisasi jalur teroptimasi (lokal & global) dan metrik",
        "desc": ("Membaca results/<mode>/*.csv lalu membuat gambar PNG: jalur tiap model\n"
                 "(oracle/global/terpusat/lokal), evolusi jalur global per ronde, kurva\n"
                 "pembelajaran, perbandingan model, peta delay link, dan sebaran data non-IID."),
        "usage": "python 03_visualisasi.py --source N1 --target N10",
        "requires": "numpy, networkx, matplotlib",
    },
    "04_reinforcement_learning.py": {
        "modules": ["network.py", "routing.py", "viz.py", "rl_reward.py", "rl_viz.py"],
        "main": "main_04_rl.py",
        "title": "LANGKAH 4 - Reinforcement Learning routing: reward plus/minus per langkah",
        "desc": ("Melatih satu agen Q-learning untuk merutekan paket source -> target dengan\n"
                 "fungsi reward terurai (+ tiba, +/- kemajuan, - delay, - hop, - loop, - kongesti,\n"
                 "- timeout). Menulis results/rl/episodes.csv, steps.csv (rincian reward setiap\n"
                 "langkah setiap episode), q_table.csv, greedy_vs_optimal.csv dan gambar PNG."),
        "usage": "python 04_reinforcement_learning.py --source N1 --target N24 --episode 5",
        "requires": "numpy, networkx, matplotlib",
    },
}


def default_config(net=None, mode="both"):
    ids = [n["node_id"] for n in net["nodes"]] if net else ["N1", "N2"]
    return {
        "mode": mode,
        "simulation": dict(DEFAULT_SIMULATION),
        "supervised": dict(DEFAULT_SUPERVISED),
        "qrouting": dict(DEFAULT_QROUTING),
        "rl": dict(DEFAULT_RL, source=ids[0], target=ids[-1]),
        "evaluation": {"num_pairs": 30, "viz_source": ids[0], "viz_target": ids[-1]},
        "topology": {"num_nodes": len(ids), "area_width": 1000.0, "area_height": 1000.0,
                     "k_neighbors": 3, "num_clients": 4, "seed": 42},
    }


def _module_source(name):
    with open(os.path.join(CORE_DIR, name), encoding="utf-8") as f:
        lines = f.read().splitlines()
    return "\n".join(l for l in lines if "@@internal" not in l)


def build_script(name):
    spec = SCRIPT_SPECS[name]
    head = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        '"""\n'
        f"{spec['title']}\n"
        f"{'=' * len(spec['title'])}\n"
        f"{spec['desc']}\n\n"
        f"Cara pakai : {spec['usage']}\n"
        f"Kebutuhan  : Python 3.9+, {spec['requires']}\n"
        "             pip install -r requirements.txt\n"
        "Struktur folder yang diharapkan (relatif terhadap script ini):\n"
        "  data/nodes.csv, data/links.csv, data/clients.csv   (konfigurasi jaringan)\n"
        "  config/fl_config.json                              (konfigurasi federated learning)\n"
        "  results/                                           (keluaran)\n\n"
        "Script ini STANDALONE: seluruh fungsi yang dibutuhkan sudah disalin di bawah,\n"
        "identik dengan engine aplikasi web FedNet Edu.\n"
        f"Dibangkitkan: {datetime.now().strftime('%Y-%m-%d %H:%M')}  -  Wirarama Wedashwara, Universitas Mataram\n"
        '"""\n'
    )
    parts = [head]
    for mod in spec["modules"]:
        parts.append(f"\n\n# >>>>>>>>>>>>>>>>>>>>>>>>>>>> {mod} >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        parts.append(_module_source(mod))
    parts.append("\n\n")
    parts.append(_module_source(os.path.join("script_mains", spec["main"])))
    return "\n".join(parts) + "\n"


RUN_ALL = '''#!/usr/bin/env python3
"""Menjalankan seluruh langkah berurutan: 01 data -> 02 federated learning -> 03 visualisasi.

    python run_all.py                  # memakai data/*.csv dan config/fl_config.json
    python run_all.py --generate       # bangkitkan ulang topologi acak dari config
    python run_all.py --mode supervised --source N1 --target N10
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    print("$ python " + " ".join(args))
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--mode", choices=["supervised", "qrouting", "both"], default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--target", default=None)
    a = ap.parse_args()
    run(["01_data_jaringan.py"] + (["--generate"] if a.generate else []))
    run(["02_federated_learning.py"] + (["--mode", a.mode] if a.mode else []))
    v = ["03_visualisasi.py"]
    if a.mode:
        v += ["--mode", a.mode]
    if a.source and a.target:
        v += ["--source", a.source, "--target", a.target]
    run(v)
    run(["04_reinforcement_learning.py"] + (["--source", a.source, "--target", a.target] if a.source and a.target else []))
    print("\\nSelesai. Lihat folder results/ untuk CSV dan gambar PNG.")
'''

REQUIREMENTS = "numpy>=1.24\nnetworkx>=3.0\nmatplotlib>=3.7\n"

README = """# Paket Script Federated Learning untuk Optimasi Jaringan

Dibangkitkan dari aplikasi **FedNet Edu** pada {date}.

## Isi
| File | Fungsi |
|------|--------|
| `data/nodes.csv` | Node: `node_id, x, y, region` (region = klien federated yang mengelola node) |
| `data/links.csv` | Link: `link_id, source, target, distance_m, base_delay_ms, bandwidth_mbps, loss, traffic_weight` |
| `data/clients.csv` | Klien/region: `client_id, name, traffic_intensity, burstiness, peak_hour, dest_locality, color` |
| `data/measurements.csv` | Data pengukuran lokal tiap klien (hasil langkah 1) |
| `data/snapshot.csv` | Kondisi jaringan saat evaluasi (bobot delay sebenarnya / oracle) |
| `config/fl_config.json` | Konfigurasi simulasi, federated learning (supervised & Q-routing), evaluasi |
| `01_data_jaringan.py` | Langkah 1: validasi CSV jaringan, simulasi data trafik lokal (non-IID) |
| `02_federated_learning.py` | Langkah 2: latih FL, bandingkan global vs lokal vs terpusat vs oracle |
| `03_visualisasi.py` | Langkah 3: gambar jalur teroptimasi tiap model + kurva metrik |
| `04_reinforcement_learning.py` | Langkah 4: agen RL dengan reward plus/minus per langkah (results/rl/) |
| `run_all.py` | Menjalankan langkah 1-4 berurutan |
| `results/<mode>/` | Hasil (CSV, JSON, PNG) |

## Menjalankan
```bash
pip install -r requirements.txt
python run_all.py
# atau per langkah:
python 01_data_jaringan.py
python 02_federated_learning.py --mode both
python 03_visualisasi.py --source {src} --target {dst}
python 04_reinforcement_learning.py --source {src} --target {dst} --episode 5
```

## Eksperimen yang disarankan
1. Ubah `traffic_intensity` klien di `data/clients.csv` agar makin berbeda -> data makin non-IID.
2. Ganti `supervised.aggregation` menjadi `fedprox`, `fedmedian`, atau `trimmed_mean`.
3. Turunkan `supervised.client_fraction` atau naikkan `dropout_prob` (klien tidak selalu ikut).
4. Aktifkan privasi: `dp_clip` = 1.0 dan `dp_noise` = 0.05 lalu amati turunnya akurasi.
5. Q-routing: ubah `dest_locality` klien dan `episodes_per_round`, amati cakupan model lokal.

Semua script standalone (tidak bergantung pada aplikasi web) dan memakai seed yang sama,
sehingga hasil identik dengan yang ditampilkan aplikasi.
"""


def build_export_files(net, cfg, measurements=None, snapshot=None, results=None, figures=None):
    """Kembalikan dict {path_relatif: str|bytes}."""
    files = {}
    files["data/nodes.csv"] = netmod.rows_to_csv_text(netmod.NODE_FIELDS, net["nodes"])
    files["data/links.csv"] = netmod.rows_to_csv_text(netmod.LINK_FIELDS, net["links"])
    files["data/clients.csv"] = netmod.rows_to_csv_text(netmod.CLIENT_FIELDS, net["clients"])
    if measurements:
        files["data/measurements.csv"] = netmod.rows_to_csv_text(netmod.MEAS_FIELDS, measurements)
    if snapshot:
        files["data/snapshot.csv"] = netmod.rows_to_csv_text(netmod.SNAP_FIELDS, snapshot)
    files["config/fl_config.json"] = json.dumps(cfg, indent=2, ensure_ascii=False)
    for name in SCRIPT_SPECS:
        files[name] = build_script(name)
    files["run_all.py"] = RUN_ALL
    files["requirements.txt"] = REQUIREMENTS
    ev = cfg.get("evaluation", {})
    files["README.md"] = README.format(date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                       src=ev.get("viz_source", "N1"), dst=ev.get("viz_target", "N2"))
    for mode, res in (results or {}).items():
        if res and mode == "rl":
            ep, st, q, g = rl_tables(res)
            for name, rows in (("episodes", ep), ("steps", st), ("q_table", q), ("greedy_vs_optimal", g)):
                files[f"results/rl/{name}.csv"] = rows_csv(rows)
        elif res:
            for name, text in results_to_tables(res, net).items():
                files[f"results/{mode}/{name}"] = text
    for path, data in (figures or {}).items():
        files[path] = data
    return files


def rows_csv(rows):
    import csv as _csv
    buf = io.StringIO()
    if rows:
        w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


def build_zip(files, root="fednet_scripts"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in sorted(files.items()):
            z.writestr(f"{root}/{path}", data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8"))
    buf.seek(0)
    return buf
