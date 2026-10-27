# -*- coding: utf-8 -*-
"""
PySpark MLlib Script Generator - Flask backend.

Antarmuka web untuk menyusun script PySpark MLlib (classification,
regression, clustering, association rule mining) secara interaktif:
pilih task, sumber data (upload CSV atau data acak), algoritma, dan
hyperparameter, lalu unduh atau salin script Python yang siap dijalankan.

Catatan: aplikasi Flask ini TIDAK menjalankan PySpark sama sekali - ia
hanya menghasilkan (generate) kode Python yang memanggil PySpark MLlib.
Jalankan file .py hasil generate di lingkungan yang sudah terpasang
PySpark (mis. `spark-submit nama_script.py` atau `python nama_script.py`
jika paket `pyspark` sudah terinstal).
"""

import math
import os
import uuid

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from algo_config import ALGORITHMS, TASKS, get_algorithms
from script_generator import generate_script

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"csv"}
MAX_PREVIEW_ROWS = 5
MAX_SAMPLE_ROWS_FOR_DTYPE = 5000

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
app.secret_key = os.environ.get("SECRET_KEY", "pyspark-mllib-generator-dev-key")

# menyimpan pemetaan file_id -> path fisik file CSV di server (in-memory,
# cukup untuk kebutuhan sesi lokal/pengembangan; untuk deployment multi-
# worker sebaiknya diganti dengan penyimpanan eksternal / database)
UPLOADED_FILES = {}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _classify_dtype(series):
    """Klasifikasikan kolom pandas sebagai 'numeric' atau 'categorical'."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _clean_for_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


@app.route("/")
def index():
    return render_template("index.html", tasks=TASKS)


@app.route("/api/algorithms/<task>")
def api_algorithms(task):
    algos = get_algorithms(task)
    if not algos:
        return jsonify({"error": f"Task tidak dikenal: {task}"}), 404
    result = []
    for key, definition in algos.items():
        result.append({
            "key": key,
            "label": definition["label"],
            "params": definition["params"],
            "notes": definition.get("notes", ""),
        })
    return jsonify({"task": task, "algorithms": result})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "Hanya file .csv yang didukung."}), 400

    original_filename = secure_filename(file.filename)
    file_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    file.save(saved_path)

    try:
        df_sample = pd.read_csv(saved_path, nrows=MAX_SAMPLE_ROWS_FOR_DTYPE)
    except Exception as exc:  # noqa: BLE001
        os.remove(saved_path)
        return jsonify({"error": f"Gagal membaca CSV: {exc}"}), 400

    if df_sample.shape[1] == 0:
        os.remove(saved_path)
        return jsonify({"error": "File CSV tidak memiliki kolom."}), 400

    columns = []
    for col_name in df_sample.columns:
        series = df_sample[col_name]
        columns.append({
            "name": str(col_name),
            "dtype": _classify_dtype(series),
            "pandas_dtype": str(series.dtype),
            "n_unique": int(series.nunique(dropna=True)),
        })

    preview_records = df_sample.head(MAX_PREVIEW_ROWS).to_dict(orient="records")
    preview = [
        {k: _clean_for_json(v) for k, v in record.items()}
        for record in preview_records
    ]

    try:
        total_rows = sum(1 for _ in open(saved_path, "r", encoding="utf-8", errors="ignore")) - 1
    except Exception:  # noqa: BLE001
        total_rows = None

    UPLOADED_FILES[file_id] = {
        "path": saved_path,
        "original_filename": original_filename,
    }

    return jsonify({
        "file_id": file_id,
        "filename": original_filename,
        "columns": columns,
        "preview": preview,
        "n_rows_sampled": int(df_sample.shape[0]),
        "n_rows_total": total_rows,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Body request harus berupa JSON."}), 400

    task = payload.get("task")
    if task not in TASKS:
        return jsonify({"error": f"Task tidak valid: {task}"}), 400

    algorithm = payload.get("algorithm")
    if algorithm not in ALGORITHMS.get(task, {}):
        return jsonify({"error": f"Algoritma tidak valid untuk task '{task}': {algorithm}"}), 400

    data_source = payload.get("data_source")
    if data_source not in ("csv", "random"):
        return jsonify({"error": "data_source harus 'csv' atau 'random'."}), 400

    config = {
        "task": task,
        "algorithm": algorithm,
        "params": payload.get("params", {}),
        "data_source": data_source,
        "app_name": payload.get("app_name") or f"PySpark MLlib - {TASKS[task]}",
        "train_ratio": float(payload.get("train_ratio", 0.8)),
        "split_seed": int(payload.get("split_seed", 42)),
    }

    if data_source == "csv":
        file_id = payload.get("file_id")
        file_info = UPLOADED_FILES.get(file_id)
        if not file_info:
            return jsonify({"error": "File CSV tidak ditemukan. Silakan upload ulang."}), 400

        config["csv_filename"] = file_info["original_filename"]

        if task == "association":
            items_column = payload.get("items_column")
            if not items_column:
                return jsonify({"error": "items_column wajib diisi untuk association rule mining."}), 400
            config["items_column"] = items_column
            config["delimiter"] = payload.get("delimiter", ",")
        else:
            feature_columns = payload.get("feature_columns") or []
            target_column = payload.get("target_column")
            column_types = payload.get("column_types") or {}

            if not feature_columns:
                return jsonify({"error": "Pilih minimal satu kolom fitur."}), 400
            if task in ("classification", "regression") and not target_column:
                return jsonify({"error": "Pilih kolom target/label."}), 400
            if task in ("classification", "regression") and target_column in feature_columns:
                return jsonify({"error": "Kolom target tidak boleh sama dengan kolom fitur."}), 400

            config["feature_columns"] = feature_columns
            config["target_column"] = target_column
            config["column_types"] = column_types
    else:
        config["random_params"] = payload.get("random_params", {})

    try:
        script_text = generate_script(config)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Gagal membuat script: {exc}"}), 500

    algo_label = ALGORITHMS[task][algorithm]["label"]
    safe_algo = algorithm.replace(" ", "_")
    filename = f"pyspark_{task}_{safe_algo}.py"

    return jsonify({
        "script": script_text,
        "filename": filename,
        "task_label": TASKS[task],
        "algorithm_label": algo_label,
    })


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
