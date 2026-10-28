"""Halaman portal: daftar semua aplikasi beserta statusnya."""
from __future__ import annotations

import os
import urllib.request

from flask import Flask, jsonify, render_template

APPS = [
    {"key": "sdn", "url": "/sdn/", "title": "SDN Simulator",
     "tag": "Jaringan", "desc": "Simulasi routing SDN dengan 10 algoritma (Dijkstra, A*, ACO, GA, PSO, "
     "Q-Learning, dst.), load balancing, QoS, dan intrusion detection. Ekspor hasil ke Excel."},
    {"key": "smarthome", "url": "/smarthome/", "title": "Smart Home Simulator",
     "tag": "IoT / STEAM", "desc": "Simulator rumah pintar: denah, komponen listrik, rule otomasi, "
     "sumber energi (PLN, surya, baterai, genset), studi kasus, dan laporan energi."},
    {"key": "rlfl", "url": "/rlfl/", "title": "RL & Federated Learning SDN",
     "tag": "Machine Learning", "desc": "Federated learning (supervised & Q-routing) dan reinforcement "
     "learning untuk routing SDN, lengkap dengan visualisasi dan paket script standalone."},
    {"key": "pyspark", "url": "/pyspark/", "title": "PySpark ML Script Generator",
     "tag": "Big Data", "desc": "Unggah CSV, pilih tugas dan algoritma Spark MLlib, lalu bangkitkan "
     "script PySpark siap dijalankan."},
]


def _streamlit_status():
    port = os.environ.get("STREAMLIT_PORT", "8501")
    base = os.environ.get("STREAMLIT_BASE_PATH", "datamining").strip("/")
    url = f"http://127.0.0.1:{port}/{base + '/' if base else ''}_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=0.6) as r:
            return r.status == 200
    except Exception:
        return False


def create_portal(loaded, failed):
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask("mpw_portal", root_path=here, static_url_path="/portal/static")

    @app.route("/")
    @app.route("/portal")
    @app.route("/portal/")
    def index():
        apps = []
        for a in APPS:
            item = dict(a)
            item["ok"] = a["key"] in loaded
            item["error"] = failed.get(a["key"])
            item["disabled"] = not item["ok"] and not item["error"]
            apps.append(item)
        dm = {
            "key": "datamining", "title": "Data Mining UI", "tag": "Streamlit",
            "desc": "Klasifikasi, regresi, klastering, asosiasi, dan deret waktu dengan "
                    "pembangkit data, tuning parameter, dan visualisasi algoritma.",
            "ok": _streamlit_status(),
            "enabled": os.environ.get("ENABLE_DATAMINING", "1") != "0",
            "note": os.environ.get("DATAMINING_NOTE", ""),
            "url": os.environ.get("DATAMINING_URL", ""),
            "port": os.environ.get("STREAMLIT_PORT", "8501"),
            "base": os.environ.get("STREAMLIT_BASE_PATH", "datamining").strip("/"),
        }
        return render_template("portal.html", apps=apps, dm=dm)

    @app.route("/healthz")
    def healthz():
        return jsonify(status="ok", loaded=loaded, failed=failed,
                       datamining_enabled=os.environ.get("ENABLE_DATAMINING", "1") != "0",
                       datamining=_streamlit_status())

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("404.html"), 404

    return app
