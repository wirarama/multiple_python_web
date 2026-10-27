# -*- coding: utf-8 -*-
"""
Generator script PySpark MLlib.

Modul ini menyusun script Python (.py) yang berdiri sendiri (standalone),
siap dijalankan dengan `spark-submit` atau `python`, berdasarkan pilihan
pengguna di antarmuka web: jenis task (classification / regression /
clustering / association rule mining), sumber data (upload CSV atau data
acak), algoritma, dan hyperparameter.

Pendekatan templating memakai `string.Template` ($nama_variabel) agar kurung
kurawal `{ }` pada kode PySpark hasil generate (f-string, dict, dsb.) tidak
perlu di-escape.
"""

from string import Template

from algo_config import get_algorithm


# ----------------------------------------------------------------------
# Helper umum
# ----------------------------------------------------------------------

def _py_list_literal(items):
    """Render list Python dari daftar string, mis. ['a', 'b'] -> '["a", "b"]'."""
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def _params_kwargs_block(params, indent="    "):
    """Render baris-baris kwargs 'nama=nilai,' untuk dimasukkan ke constructor."""
    lines = []
    for name, value in params.items():
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        elif isinstance(value, (int, float)):
            rendered = repr(value)
        else:
            rendered = f'"{value}"'
        lines.append(f"{indent}{name}={rendered},")
    return "\n".join(lines)


def _notes_comment(notes):
    if not notes:
        return ""
    return f"# Catatan: {notes}\n"


def _algo_params(algo_def, user_params):
    """Gabungkan default param algoritma dengan nilai yang dikirim pengguna."""
    result = {}
    for spec in algo_def["params"]:
        name = spec["name"]
        default = spec["default"]
        value = user_params.get(name, default) if user_params else default
        if spec["type"] == "int":
            value = int(value)
        elif spec["type"] == "float":
            value = float(value)
        result[name] = value
    return result


HEADER_TPL = Template('''"""
Script PySpark MLlib - $task_label
Algoritma      : $algo_label
Sumber data    : $data_source_label
Dibuat otomatis oleh PySpark MLlib Script Generator
"""
$notes_comment
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
''')

SPARK_SESSION_TPL = Template('''
# ============================================================
# 1. Inisialisasi Spark Session
# ============================================================
spark = SparkSession.builder \\
    .appName("$app_name") \\
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
''')

FOOTER_TPL = Template('''
# ============================================================
# $step_num. Selesai
# ============================================================
spark.stop()
''')


# ----------------------------------------------------------------------
# Blok: pemuatan data CSV (classification / regression / clustering)
# ----------------------------------------------------------------------

CSV_LOAD_TPL = Template('''
# ============================================================
# 2. Load Data dari CSV
# ============================================================
CSV_PATH = "$csv_filename"  # TODO: sesuaikan path file CSV Anda

df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

print("Jumlah baris:", df.count())
print("Kolom:", df.columns)
df.printSchema()
df.show(5, truncate=False)
''')


# ----------------------------------------------------------------------
# Blok: pembuatan data acak
# ----------------------------------------------------------------------

RANDOM_CLASSIFICATION_TPL = Template('''
# ============================================================
# 2. Membuat Data Acak (Classification)
# ============================================================
import numpy as np
import pandas as pd

RANDOM_SEED = $seed
rng = np.random.default_rng(RANDOM_SEED)

N_SAMPLES = $n_samples
N_FEATURES = $n_features
N_CLASSES = $n_classes
CLASS_SEP = $class_sep

samples_per_class = N_SAMPLES // N_CLASSES
rows = []
for c in range(N_CLASSES):
    mean = rng.uniform(-CLASS_SEP, CLASS_SEP, size=N_FEATURES) * (c + 1)
    cov = np.eye(N_FEATURES)
    X_c = rng.multivariate_normal(mean, cov, size=samples_per_class)
    y_c = np.full((samples_per_class, 1), c)
    rows.append(np.hstack([X_c, y_c]))

# jika N_SAMPLES tidak habis dibagi N_CLASSES, sisanya masuk kelas terakhir
remainder = N_SAMPLES - samples_per_class * N_CLASSES
if remainder > 0:
    mean = rng.uniform(-CLASS_SEP, CLASS_SEP, size=N_FEATURES) * N_CLASSES
    cov = np.eye(N_FEATURES)
    X_extra = rng.multivariate_normal(mean, cov, size=remainder)
    y_extra = np.full((remainder, 1), N_CLASSES - 1)
    rows.append(np.hstack([X_extra, y_extra]))

data = np.vstack(rows)
rng.shuffle(data)

feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
columns = feature_names + ["label"]
pdf = pd.DataFrame(data, columns=columns)
pdf["label"] = pdf["label"].astype(int)

df = spark.createDataFrame(pdf)

print(f"Data acak berhasil dibuat: {df.count()} baris, {N_FEATURES} fitur, {N_CLASSES} kelas")
df.show(5)
''')

RANDOM_REGRESSION_TPL = Template('''
# ============================================================
# 2. Membuat Data Acak (Regression)
# ============================================================
import numpy as np
import pandas as pd

RANDOM_SEED = $seed
rng = np.random.default_rng(RANDOM_SEED)

N_SAMPLES = $n_samples
N_FEATURES = $n_features
NOISE_STD = $noise

X = rng.uniform(-10, 10, size=(N_SAMPLES, N_FEATURES))
true_coef = rng.uniform(-5, 5, size=N_FEATURES)
true_intercept = rng.uniform(-10, 10)
noise = rng.normal(0, NOISE_STD, size=N_SAMPLES)
y = X.dot(true_coef) + true_intercept + noise

feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
pdf = pd.DataFrame(X, columns=feature_names)
pdf["label"] = y

df = spark.createDataFrame(pdf)

print(f"Data acak berhasil dibuat: {df.count()} baris, {N_FEATURES} fitur")
print("Koefisien asli (ground truth):", dict(zip(feature_names, true_coef.round(3))))
df.show(5)
''')

RANDOM_CLUSTERING_TPL = Template('''
# ============================================================
# 2. Membuat Data Acak (Clustering)
# ============================================================
import numpy as np
import pandas as pd

RANDOM_SEED = $seed
rng = np.random.default_rng(RANDOM_SEED)

N_SAMPLES = $n_samples
N_FEATURES = $n_features
N_CLUSTERS = $n_clusters
CLUSTER_STD = $cluster_std

samples_per_cluster = N_SAMPLES // N_CLUSTERS
centers = rng.uniform(-20, 20, size=(N_CLUSTERS, N_FEATURES))
rows = []
for c in range(N_CLUSTERS):
    cov = np.eye(N_FEATURES) * (CLUSTER_STD ** 2)
    X_c = rng.multivariate_normal(centers[c], cov, size=samples_per_cluster)
    rows.append(X_c)

remainder = N_SAMPLES - samples_per_cluster * N_CLUSTERS
if remainder > 0:
    cov = np.eye(N_FEATURES) * (CLUSTER_STD ** 2)
    X_extra = rng.multivariate_normal(centers[-1], cov, size=remainder)
    rows.append(X_extra)

data = np.vstack(rows)
rng.shuffle(data)

feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
pdf = pd.DataFrame(data, columns=feature_names)

df = spark.createDataFrame(pdf)

print(f"Data acak berhasil dibuat: {df.count()} baris, {N_FEATURES} fitur, {N_CLUSTERS} cluster (ground truth)")
df.show(5)
''')

RANDOM_ASSOCIATION_TPL = Template('''
# ============================================================
# 2. Membuat Data Transaksi Acak
# ============================================================
import numpy as np
import pandas as pd

RANDOM_SEED = $seed
rng = np.random.default_rng(RANDOM_SEED)

N_TRANSACTIONS = $n_transactions
N_ITEMS = $n_items
MIN_ITEMS_PER_TX = $min_items
MAX_ITEMS_PER_TX = $max_items

items_pool = [f"item_{i}" for i in range(N_ITEMS)]

transactions = []
for tx_id in range(N_TRANSACTIONS):
    basket_size = rng.integers(MIN_ITEMS_PER_TX, MAX_ITEMS_PER_TX + 1)
    basket_size = min(basket_size, N_ITEMS)
    basket = list(rng.choice(items_pool, size=basket_size, replace=False))
    transactions.append((tx_id, basket))

pdf = pd.DataFrame(transactions, columns=["id", "items"])
df = spark.createDataFrame(pdf)

print(f"Data transaksi acak berhasil dibuat: {df.count()} transaksi, {N_ITEMS} item unik")
df.show(5, truncate=False)
''')

CSV_LOAD_ASSOCIATION_TPL = Template('''
# ============================================================
# 2. Load Data Transaksi dari CSV
# ============================================================
from pyspark.sql.functions import split, trim, monotonically_increasing_id

CSV_PATH = "$csv_filename"     # TODO: sesuaikan path file CSV Anda
ITEMS_COLUMN = "$items_column"  # kolom berisi daftar item per transaksi
DELIMITER = "$delimiter"        # pemisah antar item dalam satu baris

raw_df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
raw_df.show(5, truncate=False)

# ubah kolom string "item1,item2,item3" menjadi array [item1, item2, item3]
df = raw_df.select(
    monotonically_increasing_id().alias("id"),
    split(trim(col(ITEMS_COLUMN)), DELIMITER).alias("items")
)

print("Jumlah transaksi:", df.count())
df.show(5, truncate=False)
''')


# ----------------------------------------------------------------------
# Blok: praproses (classification / regression)
# ----------------------------------------------------------------------

PREPROCESS_RANDOM_TPL = Template('''
# ============================================================
# 3. Praproses: Vector Assembler & Train-Test Split
# ============================================================
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
df_assembled = assembler.transform(df)

train_df, test_df = df_assembled.randomSplit([$train_ratio, $test_ratio], seed=$split_seed)
print("Jumlah data latih:", train_df.count())
print("Jumlah data uji  :", test_df.count())
''')

PREPROCESS_CLUSTERING_RANDOM_TPL = Template('''
# ============================================================
# 3. Praproses: Vector Assembler
# ============================================================
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
df_assembled = assembler.transform(df)
''')


def _preprocess_csv_block(feature_columns, column_types, target_column, task, train_ratio, test_ratio, split_seed):
    """
    Bangun blok praproses untuk data CSV: StringIndexer + OneHotEncoder untuk
    fitur kategorikal, VectorAssembler untuk menggabungkan semua fitur, dan
    (untuk classification/regression) penyiapan kolom label serta
    train-test split.
    """
    categorical_features = [c for c in feature_columns if column_types.get(c) == "categorical"]
    numeric_features = [c for c in feature_columns if c not in categorical_features]

    lines = []
    lines.append("# ============================================================")
    lines.append("# 3. Praproses: Encoding Fitur, Vector Assembler" + (", Label" if task in ("classification", "regression") else "") + (" & Train-Test Split" if task in ("classification", "regression") else ""))
    lines.append("# ============================================================")
    lines.append("from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder")
    lines.append("from pyspark.ml import Pipeline")
    lines.append("")
    lines.append("stages = []")
    lines.append("")

    assembler_inputs = list(numeric_features)
    if categorical_features:
        lines.append("# encoding kolom fitur kategorikal (StringIndexer -> OneHotEncoder)")
        for c in categorical_features:
            idx_col = f"{c}_idx"
            vec_col = f"{c}_vec"
            lines.append(f'stages.append(StringIndexer(inputCol="{c}", outputCol="{idx_col}", handleInvalid="keep"))')
            lines.append(f'stages.append(OneHotEncoder(inputCol="{idx_col}", outputCol="{vec_col}"))')
            assembler_inputs.append(vec_col)
        lines.append("")

    lines.append(f"feature_inputs = {_py_list_literal(assembler_inputs)}")
    lines.append('stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="features"))')
    lines.append("")

    if task == "classification":
        is_target_categorical = column_types.get(target_column) == "categorical"
        if is_target_categorical:
            lines.append("# label bertipe kategorikal -> indeks-kan menjadi angka 0..n_kelas-1")
            lines.append(f'stages.append(StringIndexer(inputCol="{target_column}", outputCol="label", handleInvalid="keep"))')
        else:
            lines.append(f'df = df.withColumn("label", col("{target_column}").cast("double"))')
    elif task == "regression":
        lines.append(f'df = df.withColumn("label", col("{target_column}").cast("double"))')

    lines.append("")
    lines.append("pipeline = Pipeline(stages=stages)")
    lines.append("pipeline_model = pipeline.fit(df)")
    lines.append("df_assembled = pipeline_model.transform(df)")

    if task in ("classification", "regression"):
        lines.append("")
        lines.append(f"train_df, test_df = df_assembled.randomSplit([{train_ratio}, {test_ratio}], seed={split_seed})")
        lines.append('print("Jumlah data latih:", train_df.count())')
        lines.append('print("Jumlah data uji  :", test_df.count())')

    return "\n" + "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Blok: model & training
# ----------------------------------------------------------------------

def _model_block_classification(algo_def, params, step_num):
    kwargs = _params_kwargs_block(params)
    show_cols = '"label", "prediction"'
    if algo_def.get("has_probability"):
        show_cols += ', "probability"'
    return f'''
# ============================================================
# {step_num}. Definisi dan Pelatihan Model: {algo_def["label"]}
# ============================================================
{algo_def["import_stmt"]}

model = {algo_def["class_name"]}(
    featuresCol="features",
    labelCol="label",
{kwargs}
)

trained_model = model.fit(train_df)
predictions = trained_model.transform(test_df)

print("Contoh hasil prediksi:")
predictions.select({show_cols}).show(10)
'''


def _model_block_regression(algo_def, params, step_num):
    kwargs = _params_kwargs_block(params)
    extra = ""
    if algo_def["class_name"] == "GeneralizedLinearRegression":
        extra = '    family="gaussian",\n    link="identity",\n'
    return f'''
# ============================================================
# {step_num}. Definisi dan Pelatihan Model: {algo_def["label"]}
# ============================================================
{algo_def["import_stmt"]}

model = {algo_def["class_name"]}(
    featuresCol="features",
    labelCol="label",
{extra}{kwargs}
)

trained_model = model.fit(train_df)
predictions = trained_model.transform(test_df)

print("Contoh hasil prediksi:")
predictions.select("label", "prediction").show(10)
'''


def _model_block_clustering(algo_def, params, step_num):
    kwargs = _params_kwargs_block(params)
    return f'''
# ============================================================
# {step_num}. Definisi dan Pelatihan Model: {algo_def["label"]}
# ============================================================
{algo_def["import_stmt"]}

model = {algo_def["class_name"]}(
    featuresCol="features",
    predictionCol="prediction",
{kwargs}
)

trained_model = model.fit(df_assembled)
predictions = trained_model.transform(df_assembled)

print("Contoh hasil prediksi cluster:")
predictions.select("features", "prediction").show(10, truncate=False)
'''


# ----------------------------------------------------------------------
# Blok: evaluasi
# ----------------------------------------------------------------------

def _eval_block_classification(step_num):
    return f'''
# ============================================================
# {step_num}. Evaluasi Model
# ============================================================
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)

print("=" * 50)
print("HASIL EVALUASI MODEL KLASIFIKASI")
print("=" * 50)
print(f"Accuracy             : {{accuracy:.4f}}")
print(f"F1-Score (weighted)  : {{f1_score:.4f}}")
print(f"Precision (weighted) : {{precision:.4f}}")
print(f"Recall (weighted)    : {{recall:.4f}}")

print("\\nConfusion Matrix (label vs prediction):")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()
'''


def _eval_block_regression(step_num):
    return f'''
# ============================================================
# {step_num}. Evaluasi Model
# ============================================================
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("=" * 50)
print("HASIL EVALUASI MODEL REGRESI")
print("=" * 50)
print(f"RMSE : {{rmse:.4f}}")
print(f"MAE  : {{mae:.4f}}")
print(f"R2   : {{r2:.4f}}")
'''


def _eval_block_clustering(step_num):
    return f'''
# ============================================================
# {step_num}. Evaluasi Model
# ============================================================
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean",
)
silhouette = evaluator.evaluate(predictions)

print("=" * 50)
print("HASIL EVALUASI MODEL CLUSTERING")
print("=" * 50)
print(f"Silhouette Score : {{silhouette:.4f}}")
print("(mendekati 1.0 = cluster terpisah baik, mendekati 0 = tumpang tindih, negatif = kemungkinan salah cluster)")

print("\\nJumlah anggota tiap cluster:")
predictions.groupBy("prediction").count().orderBy("prediction").show()

if hasattr(trained_model, "clusterCenters"):
    print("\\nPusat Cluster (Cluster Centers):")
    for i, center in enumerate(trained_model.clusterCenters()):
        print(f"  Cluster {{i}}: {{center}}")
'''


def _association_block(algo_def, params, step_num):
    kwargs = _params_kwargs_block(params)
    return f'''
# ============================================================
# {step_num}. Definisi dan Pelatihan Model: {algo_def["label"]}
# ============================================================
{algo_def["import_stmt"]}

fp = FPGrowth(
    itemsCol="items",
{kwargs}
)

model = fp.fit(df)

# ============================================================
# {step_num + 1}. Evaluasi / Hasil
# ============================================================
print("=" * 50)
print("FREQUENT ITEMSETS")
print("=" * 50)
freq_itemsets = model.freqItemsets.orderBy(col("freq").desc())
freq_itemsets.show(20, truncate=False)
print("Jumlah frequent itemsets:", freq_itemsets.count())

print()
print("=" * 50)
print("ASSOCIATION RULES")
print("=" * 50)
rules = model.associationRules.orderBy(col("confidence").desc())
rules.show(20, truncate=False)
n_rules = rules.count()
print("Jumlah association rules:", n_rules)

if n_rules > 0:
    avg_confidence = rules.agg({{"confidence": "avg"}}).collect()[0][0]
    avg_lift = rules.agg({{"lift": "avg"}}).collect()[0][0]
    print(f"Rata-rata Confidence : {{avg_confidence:.4f}}")
    print(f"Rata-rata Lift       : {{avg_lift:.4f}}")
else:
    print("Tidak ada rule yang memenuhi minSupport/minConfidence. Coba turunkan nilainya.")

print()
print("Prediksi item tambahan untuk tiap transaksi (berdasarkan rule yang ditemukan):")
model.transform(df).show(10, truncate=False)
'''


# ----------------------------------------------------------------------
# Fungsi utama per task
# ----------------------------------------------------------------------

def _build_common_header(task_label, algo_def, data_source_label):
    return HEADER_TPL.substitute(
        task_label=task_label,
        algo_label=algo_def["label"],
        data_source_label=data_source_label,
        notes_comment=_notes_comment(algo_def.get("notes", "")),
    )


def build_classification_script(config):
    algo_def = get_algorithm("classification", config["algorithm"])
    params = _algo_params(algo_def, config.get("params"))
    data_source = config["data_source"]
    train_ratio = config.get("train_ratio", 0.8)
    test_ratio = round(1 - train_ratio, 4)
    split_seed = config.get("split_seed", 42)

    parts = [_build_common_header("Classification", algo_def, "CSV Upload" if data_source == "csv" else "Data Acak (Random)")]
    parts.append(SPARK_SESSION_TPL.substitute(app_name=config.get("app_name", "PySpark MLlib - Classification")))

    if data_source == "csv":
        parts.append(CSV_LOAD_TPL.substitute(csv_filename=config["csv_filename"]))
        parts.append(_preprocess_csv_block(
            feature_columns=config["feature_columns"],
            column_types=config["column_types"],
            target_column=config["target_column"],
            task="classification",
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        ))
    else:
        rp = config["random_params"]
        parts.append(RANDOM_CLASSIFICATION_TPL.substitute(
            seed=int(rp.get("seed", 42)),
            n_samples=int(rp.get("n_samples", 1000)),
            n_features=int(rp.get("n_features", 5)),
            n_classes=int(rp.get("n_classes", 2)),
            class_sep=float(rp.get("class_sep", 2.0)),
        ))
        parts.append(PREPROCESS_RANDOM_TPL.substitute(
            train_ratio=train_ratio, test_ratio=test_ratio, split_seed=split_seed,
        ))

    parts.append(_model_block_classification(algo_def, params, step_num=4))
    parts.append(_eval_block_classification(step_num=5))
    parts.append(FOOTER_TPL.substitute(step_num=6))

    return "".join(parts)


def build_regression_script(config):
    algo_def = get_algorithm("regression", config["algorithm"])
    params = _algo_params(algo_def, config.get("params"))
    data_source = config["data_source"]
    train_ratio = config.get("train_ratio", 0.8)
    test_ratio = round(1 - train_ratio, 4)
    split_seed = config.get("split_seed", 42)

    parts = [_build_common_header("Regression", algo_def, "CSV Upload" if data_source == "csv" else "Data Acak (Random)")]
    parts.append(SPARK_SESSION_TPL.substitute(app_name=config.get("app_name", "PySpark MLlib - Regression")))

    if data_source == "csv":
        parts.append(CSV_LOAD_TPL.substitute(csv_filename=config["csv_filename"]))
        parts.append(_preprocess_csv_block(
            feature_columns=config["feature_columns"],
            column_types=config["column_types"],
            target_column=config["target_column"],
            task="regression",
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        ))
    else:
        rp = config["random_params"]
        parts.append(RANDOM_REGRESSION_TPL.substitute(
            seed=int(rp.get("seed", 42)),
            n_samples=int(rp.get("n_samples", 1000)),
            n_features=int(rp.get("n_features", 5)),
            noise=float(rp.get("noise", 1.0)),
        ))
        parts.append(PREPROCESS_RANDOM_TPL.substitute(
            train_ratio=train_ratio, test_ratio=test_ratio, split_seed=split_seed,
        ))

    parts.append(_model_block_regression(algo_def, params, step_num=4))
    parts.append(_eval_block_regression(step_num=5))
    parts.append(FOOTER_TPL.substitute(step_num=6))

    return "".join(parts)


def build_clustering_script(config):
    algo_def = get_algorithm("clustering", config["algorithm"])
    params = _algo_params(algo_def, config.get("params"))
    data_source = config["data_source"]

    parts = [_build_common_header("Clustering", algo_def, "CSV Upload" if data_source == "csv" else "Data Acak (Random)")]
    parts.append(SPARK_SESSION_TPL.substitute(app_name=config.get("app_name", "PySpark MLlib - Clustering")))

    if data_source == "csv":
        parts.append(CSV_LOAD_TPL.substitute(csv_filename=config["csv_filename"]))
        parts.append(_preprocess_csv_block(
            feature_columns=config["feature_columns"],
            column_types=config["column_types"],
            target_column=None,
            task="clustering",
            train_ratio=0.8,
            test_ratio=0.2,
            split_seed=42,
        ))
    else:
        rp = config["random_params"]
        parts.append(RANDOM_CLUSTERING_TPL.substitute(
            seed=int(rp.get("seed", 42)),
            n_samples=int(rp.get("n_samples", 1000)),
            n_features=int(rp.get("n_features", 2)),
            n_clusters=int(rp.get("n_clusters", 3)),
            cluster_std=float(rp.get("cluster_std", 1.0)),
        ))
        parts.append(PREPROCESS_CLUSTERING_RANDOM_TPL.substitute())

    parts.append(_model_block_clustering(algo_def, params, step_num=4))
    parts.append(_eval_block_clustering(step_num=5))
    parts.append(FOOTER_TPL.substitute(step_num=6))

    return "".join(parts)


def build_association_script(config):
    algo_def = get_algorithm("association", config["algorithm"])
    params = _algo_params(algo_def, config.get("params"))
    data_source = config["data_source"]

    parts = [_build_common_header("Association Rule Mining", algo_def, "CSV Upload" if data_source == "csv" else "Data Acak (Random)")]
    parts.append(SPARK_SESSION_TPL.substitute(app_name=config.get("app_name", "PySpark MLlib - Association Rule Mining")))

    if data_source == "csv":
        parts.append(CSV_LOAD_ASSOCIATION_TPL.substitute(
            csv_filename=config["csv_filename"],
            items_column=config["items_column"],
            delimiter=config.get("delimiter", ","),
        ))
    else:
        rp = config["random_params"]
        parts.append(RANDOM_ASSOCIATION_TPL.substitute(
            seed=int(rp.get("seed", 42)),
            n_transactions=int(rp.get("n_transactions", 1000)),
            n_items=int(rp.get("n_items", 20)),
            min_items=int(rp.get("min_items", 1)),
            max_items=int(rp.get("max_items", 5)),
        ))

    parts.append(_association_block(algo_def, params, step_num=3))
    parts.append(FOOTER_TPL.substitute(step_num=5))

    return "".join(parts)


BUILDERS = {
    "classification": build_classification_script,
    "regression": build_regression_script,
    "clustering": build_clustering_script,
    "association": build_association_script,
}


def generate_script(config):
    task = config["task"]
    builder = BUILDERS.get(task)
    if builder is None:
        raise ValueError(f"Task tidak dikenal: {task}")
    return builder(config)
