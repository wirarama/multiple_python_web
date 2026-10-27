# -*- coding: utf-8 -*-
"""
Definisi algoritma PySpark MLlib beserta hyperparameter yang bisa diatur
pengguna dari antarmuka web. Satu sumber kebenaran (single source of truth)
yang dipakai baik oleh backend Flask (untuk membangun form & validasi)
maupun oleh generator script (untuk menuliskan kode PySpark yang sesuai).
"""

TASKS = {
    "classification": "Classification",
    "regression": "Regression",
    "clustering": "Clustering",
    "association": "Association Rule Mining",
}

# Setiap algoritma didefinisikan dengan:
#   label       : nama tampilan di UI
#   class_name  : nama class PySpark ML
#   import_stmt : baris import yang dituliskan di script hasil generate
#   params      : daftar hyperparameter yang bisa diatur pengguna
#   notes       : catatan tambahan (ditulis sebagai komentar di script)
ALGORITHMS = {
    "classification": {
        "logistic_regression": {
            "label": "Logistic Regression",
            "class_name": "LogisticRegression",
            "import_stmt": "from pyspark.ml.classification import LogisticRegression",
            "params": [
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 100, "min": 1, "max": 1000},
                {"name": "regParam", "label": "Regularization Param", "type": "float", "default": 0.0, "min": 0, "max": 10, "step": 0.01},
                {"name": "elasticNetParam", "label": "ElasticNet Mixing Param", "type": "float", "default": 0.0, "min": 0, "max": 1, "step": 0.01},
            ],
            "notes": "Mendukung binary maupun multiclass classification.",
            "has_probability": True,
        },
        "decision_tree": {
            "label": "Decision Tree Classifier",
            "class_name": "DecisionTreeClassifier",
            "import_stmt": "from pyspark.ml.classification import DecisionTreeClassifier",
            "params": [
                {"name": "maxDepth", "label": "Max Depth", "type": "int", "default": 5, "min": 1, "max": 30},
                {"name": "maxBins", "label": "Max Bins", "type": "int", "default": 32, "min": 2, "max": 512},
            ],
            "notes": "",
            "has_probability": True,
        },
        "random_forest": {
            "label": "Random Forest Classifier",
            "class_name": "RandomForestClassifier",
            "import_stmt": "from pyspark.ml.classification import RandomForestClassifier",
            "params": [
                {"name": "numTrees", "label": "Number of Trees", "type": "int", "default": 20, "min": 1, "max": 500},
                {"name": "maxDepth", "label": "Max Depth", "type": "int", "default": 5, "min": 1, "max": 30},
            ],
            "notes": "",
            "has_probability": True,
        },
        "gbt": {
            "label": "Gradient-Boosted Tree Classifier",
            "class_name": "GBTClassifier",
            "import_stmt": "from pyspark.ml.classification import GBTClassifier",
            "params": [
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 20, "min": 1, "max": 200},
                {"name": "maxDepth", "label": "Max Depth", "type": "int", "default": 5, "min": 1, "max": 30},
            ],
            "notes": "Catatan: GBTClassifier di PySpark MLlib hanya mendukung binary classification (2 kelas).",
            "has_probability": True,
        },
        "naive_bayes": {
            "label": "Naive Bayes",
            "class_name": "NaiveBayes",
            "import_stmt": "from pyspark.ml.classification import NaiveBayes",
            "params": [
                {"name": "smoothing", "label": "Smoothing", "type": "float", "default": 1.0, "min": 0, "max": 10, "step": 0.1},
            ],
            "notes": "Membutuhkan fitur non-negatif (cocok untuk data hitungan/frekuensi).",
            "has_probability": True,
        },
        "linear_svc": {
            "label": "Linear SVM",
            "class_name": "LinearSVC",
            "import_stmt": "from pyspark.ml.classification import LinearSVC",
            "params": [
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 100, "min": 1, "max": 1000},
                {"name": "regParam", "label": "Regularization Param", "type": "float", "default": 0.0, "min": 0, "max": 10, "step": 0.01},
            ],
            "notes": "Catatan: LinearSVC di PySpark MLlib hanya mendukung binary classification (2 kelas).",
            "has_probability": False,
        },
    },
    "regression": {
        "linear_regression": {
            "label": "Linear Regression",
            "class_name": "LinearRegression",
            "import_stmt": "from pyspark.ml.regression import LinearRegression",
            "params": [
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 100, "min": 1, "max": 1000},
                {"name": "regParam", "label": "Regularization Param", "type": "float", "default": 0.0, "min": 0, "max": 10, "step": 0.01},
                {"name": "elasticNetParam", "label": "ElasticNet Mixing Param", "type": "float", "default": 0.0, "min": 0, "max": 1, "step": 0.01},
            ],
            "notes": "",
        },
        "decision_tree": {
            "label": "Decision Tree Regressor",
            "class_name": "DecisionTreeRegressor",
            "import_stmt": "from pyspark.ml.regression import DecisionTreeRegressor",
            "params": [
                {"name": "maxDepth", "label": "Max Depth", "type": "int", "default": 5, "min": 1, "max": 30},
                {"name": "maxBins", "label": "Max Bins", "type": "int", "default": 32, "min": 2, "max": 512},
            ],
            "notes": "",
        },
        "random_forest": {
            "label": "Random Forest Regressor",
            "class_name": "RandomForestRegressor",
            "import_stmt": "from pyspark.ml.regression import RandomForestRegressor",
            "params": [
                {"name": "numTrees", "label": "Number of Trees", "type": "int", "default": 20, "min": 1, "max": 500},
                {"name": "maxDepth", "label": "Max Depth", "type": "int", "default": 5, "min": 1, "max": 30},
            ],
            "notes": "",
        },
        "gbt": {
            "label": "Gradient-Boosted Tree Regressor",
            "class_name": "GBTRegressor",
            "import_stmt": "from pyspark.ml.regression import GBTRegressor",
            "params": [
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 20, "min": 1, "max": 200},
                {"name": "maxDepth", "label": "Max Depth", "type": "int", "default": 5, "min": 1, "max": 30},
            ],
            "notes": "",
        },
        "glr": {
            "label": "Generalized Linear Regression",
            "class_name": "GeneralizedLinearRegression",
            "import_stmt": "from pyspark.ml.regression import GeneralizedLinearRegression",
            "params": [
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 25, "min": 1, "max": 500},
                {"name": "regParam", "label": "Regularization Param", "type": "float", "default": 0.0, "min": 0, "max": 10, "step": 0.01},
            ],
            "notes": "Menggunakan family='gaussian', link='identity' (setara linear regression klasik).",
        },
    },
    "clustering": {
        "kmeans": {
            "label": "K-Means",
            "class_name": "KMeans",
            "import_stmt": "from pyspark.ml.clustering import KMeans",
            "params": [
                {"name": "k", "label": "Number of Clusters (k)", "type": "int", "default": 3, "min": 2, "max": 50},
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 20, "min": 1, "max": 500},
                {"name": "seed", "label": "Random Seed", "type": "int", "default": 42, "min": 0, "max": 999999},
            ],
            "notes": "",
        },
        "bisecting_kmeans": {
            "label": "Bisecting K-Means",
            "class_name": "BisectingKMeans",
            "import_stmt": "from pyspark.ml.clustering import BisectingKMeans",
            "params": [
                {"name": "k", "label": "Number of Clusters (k)", "type": "int", "default": 3, "min": 2, "max": 50},
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 20, "min": 1, "max": 500},
            ],
            "notes": "",
        },
        "gaussian_mixture": {
            "label": "Gaussian Mixture Model",
            "class_name": "GaussianMixture",
            "import_stmt": "from pyspark.ml.clustering import GaussianMixture",
            "params": [
                {"name": "k", "label": "Number of Components (k)", "type": "int", "default": 3, "min": 2, "max": 50},
                {"name": "maxIter", "label": "Max Iterations", "type": "int", "default": 100, "min": 1, "max": 500},
            ],
            "notes": "",
        },
    },
    "association": {
        "fpgrowth": {
            "label": "FP-Growth",
            "class_name": "FPGrowth",
            "import_stmt": "from pyspark.ml.fpm import FPGrowth",
            "params": [
                {"name": "minSupport", "label": "Min Support", "type": "float", "default": 0.1, "min": 0.001, "max": 1, "step": 0.001},
                {"name": "minConfidence", "label": "Min Confidence", "type": "float", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
            ],
            "notes": "Satu-satunya algoritma association rule mining yang tersedia di PySpark MLlib.",
        },
    },
}


def get_algorithms(task):
    return ALGORITHMS.get(task, {})


def get_algorithm(task, algo_key):
    return ALGORITHMS.get(task, {}).get(algo_key)
