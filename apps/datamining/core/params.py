"""Registri parameter seluruh algoritma + pembangun estimator.

Satu sumber kebenaran untuk:
  * daftar hiperparameter yang boleh diedit pengguna (beserta batas & penjelasan),
  * nilai bawaan,
  * konversi nilai (JSON <-> Python) sehingga konfigurasi dapat disimpan/dimuat,
  * pembuatan objek estimator dari dict parameter.

Setiap spesifikasi parameter berbentuk dict:
    {"kind": "int"|"float"|"choice"|"bool"|"numstr"|"text",
     "default": ..., "min": ..., "max": ..., "step": ...,
     "options": [...], "none_at": <nilai yang berarti None>,
     "fmt": "%.4f", "help": (teks_id, teks_en)}
"""

from __future__ import annotations

import numpy as np

# ============================================================ helper spec
def I(default, lo, hi, step=1, id_="", en_="", none_at=None):
    return {"kind": "int", "default": default, "min": lo, "max": hi, "step": step,
            "none_at": none_at, "help": (id_, en_)}


def F(default, lo, hi, step=0.01, id_="", en_="", fmt="%.4f", none_at=None):
    return {"kind": "float", "default": default, "min": lo, "max": hi, "step": step,
            "fmt": fmt, "none_at": none_at, "help": (id_, en_)}


def C(default, options, id_="", en_=""):
    return {"kind": "choice", "default": default, "options": list(options), "help": (id_, en_)}


def B(default, id_="", en_=""):
    return {"kind": "bool", "default": default, "help": (id_, en_)}


def NS(default, id_="", en_="", options=()):
    """Nilai bisa teks ('scale', 'sqrt', 'None') atau angka."""
    return {"kind": "numstr", "default": default, "options": list(options), "help": (id_, en_)}


def T(default, id_="", en_=""):
    return {"kind": "text", "default": default, "help": (id_, en_)}


# ================================================== ruang parameter: klasifikasi
_TREE_CLF = {
    "criterion": C("gini", ["gini", "entropy", "log_loss"],
                   "Ukuran ketidakmurnian simpul saat memilih pemecahan terbaik.",
                   "Impurity measure used to pick the best split."),
    "splitter": C("best", ["best", "random"],
                  "'best' memilih pemecahan optimal, 'random' memilih acak terbaik (lebih cepat, lebih beragam).",
                  "'best' picks the optimal split, 'random' the best random one (faster, more diverse)."),
    "max_depth": I(0, 0, 60, 1,
                   "Kedalaman maksimum pohon. 0 = tanpa batas (rawan overfitting).",
                   "Maximum tree depth. 0 = unlimited (overfitting-prone).", none_at=0),
    "min_samples_split": I(2, 2, 200, 1,
                           "Minimum sampel agar sebuah simpul boleh dipecah.",
                           "Minimum samples required to split an internal node."),
    "min_samples_leaf": I(1, 1, 200, 1,
                          "Minimum sampel pada setiap daun. Besar = pohon lebih halus.",
                          "Minimum samples at each leaf. Larger = smoother tree."),
    "max_features": NS("None",
                       "Jumlah fitur yang dipertimbangkan tiap pemecahan: None, sqrt, log2, atau angka.",
                       "Features considered per split: None, sqrt, log2, or a number.",
                       options=["None", "sqrt", "log2"]),
    "max_leaf_nodes": I(0, 0, 500, 1,
                        "Batas jumlah daun. 0 = tanpa batas.",
                        "Cap on the number of leaves. 0 = unlimited.", none_at=0),
    "min_impurity_decrease": F(0.0, 0.0, 0.5, 0.001,
                               "Pemecahan hanya dilakukan bila menurunkan ketidakmurnian minimal sebesar ini.",
                               "A split happens only if it decreases impurity by at least this much."),
    "ccp_alpha": F(0.0, 0.0, 0.1, 0.001,
                   "Parameter pemangkasan cost-complexity. Besar = pohon lebih kecil.",
                   "Cost-complexity pruning parameter. Larger = smaller tree."),
    "class_weight": C("None", ["None", "balanced"],
                      "'balanced' memberi bobot lebih besar pada kelas minoritas.",
                      "'balanced' up-weights minority classes."),
}

_FOREST_COMMON = {
    "n_estimators": I(300, 10, 1000, 10,
                      "Jumlah pohon dalam ansambel. Lebih banyak = lebih stabil tetapi lebih lambat.",
                      "Number of trees in the ensemble. More = more stable but slower."),
    "criterion": C("gini", ["gini", "entropy", "log_loss"],
                   "Ukuran ketidakmurnian simpul.", "Node impurity measure."),
    "max_depth": I(0, 0, 60, 1, "Kedalaman maksimum tiap pohon. 0 = tanpa batas.",
                   "Maximum depth of each tree. 0 = unlimited.", none_at=0),
    "min_samples_split": I(2, 2, 200, 1, "Minimum sampel untuk memecah simpul.",
                           "Minimum samples to split a node."),
    "min_samples_leaf": I(1, 1, 200, 1, "Minimum sampel per daun.", "Minimum samples per leaf."),
    "max_features": NS("sqrt", "Fitur yang dipertimbangkan tiap pemecahan.",
                       "Features considered per split.", options=["sqrt", "log2", "None"]),
    "bootstrap": B(True, "Ambil sampel dengan pengembalian untuk tiap pohon (bagging).",
                   "Sample with replacement for each tree (bagging)."),
    "oob_score": B(False, "Hitung skor out-of-bag sebagai validasi gratis (butuh bootstrap aktif).",
                   "Compute the out-of-bag score as free validation (requires bootstrap)."),
    "max_samples": F(0.0, 0.0, 1.0, 0.05,
                     "Proporsi data untuk tiap pohon. 0 = pakai seluruh data.",
                     "Fraction of rows per tree. 0 = use all rows.", none_at=0.0),
    "ccp_alpha": F(0.0, 0.0, 0.1, 0.001, "Pemangkasan cost-complexity.",
                   "Cost-complexity pruning."),
    "class_weight": C("None", ["None", "balanced", "balanced_subsample"],
                      "Penyeimbang bobot kelas.", "Class weight balancing."),
}

PARAMS_CLF = {
    "Logistic Regression": {
        "C": F(1.0, 0.001, 1000.0, 0.01,
               "Kebalikan kekuatan regularisasi. Kecil = regularisasi kuat (model sederhana).",
               "Inverse regularization strength. Small = strong regularization (simpler model)."),
        "penalty": C("l2", ["l2", "l1", "elasticnet", "none"],
                     "Jenis penalti koefisien. l1 menyeleksi fitur, l2 menyusutkan koefisien.",
                     "Coefficient penalty. l1 selects features, l2 shrinks coefficients."),
        "solver": C("lbfgs", ["lbfgs", "liblinear", "saga", "newton-cg", "sag"],
                    "Algoritma optimisasi. liblinear/saga mendukung penalti l1.",
                    "Optimization algorithm. liblinear/saga support the l1 penalty."),
        "l1_ratio": F(0.5, 0.0, 1.0, 0.05, "Campuran L1:L2 khusus penalti elasticnet.",
                      "L1:L2 mix, used only with the elasticnet penalty."),
        "max_iter": I(2000, 50, 20000, 50, "Batas iterasi optimisasi.",
                      "Maximum optimization iterations."),
        "tol": F(1e-4, 1e-8, 0.1, 1e-4, "Ambang konvergensi.", "Convergence tolerance.", fmt="%.1e"),
        "fit_intercept": B(True, "Sertakan konstanta (intersep).", "Include the intercept term."),
        "class_weight": C("None", ["None", "balanced"], "Penyeimbang bobot kelas.",
                          "Class weight balancing."),
    },
    "Decision Tree": dict(_TREE_CLF),
    "Random Forest": dict(_FOREST_COMMON),
    "Extra Trees": {**_FOREST_COMMON, "bootstrap": B(False,
                    "Extra Trees biasanya memakai seluruh data (tanpa bootstrap).",
                    "Extra Trees normally uses the whole sample (no bootstrap).")},
    "Gradient Boosting": {
        "loss": C("log_loss", ["log_loss", "exponential"],
                  "Fungsi kerugian yang diminimalkan.", "Loss function being minimized."),
        "n_estimators": I(100, 10, 1000, 10, "Jumlah tahap boosting (pohon berurutan).",
                          "Number of boosting stages (sequential trees)."),
        "learning_rate": F(0.1, 0.001, 1.0, 0.01,
                           "Laju belajar. Kecil butuh lebih banyak pohon tetapi lebih tahan overfitting.",
                           "Learning rate. Smaller needs more trees but generalizes better."),
        "max_depth": I(3, 1, 20, 1, "Kedalaman tiap pohon lemah (weak learner).",
                       "Depth of each weak learner."),
        "min_samples_split": I(2, 2, 200, 1, "Minimum sampel untuk memecah simpul.",
                               "Minimum samples to split a node."),
        "min_samples_leaf": I(1, 1, 200, 1, "Minimum sampel per daun.", "Minimum samples per leaf."),
        "subsample": F(1.0, 0.1, 1.0, 0.05, "Proporsi sampel tiap tahap (<1 = stochastic boosting).",
                       "Row fraction per stage (<1 = stochastic boosting)."),
        "max_features": NS("None", "Fitur per pemecahan.", "Features per split.",
                           options=["None", "sqrt", "log2"]),
        "ccp_alpha": F(0.0, 0.0, 0.1, 0.001, "Pemangkasan cost-complexity.", "Cost-complexity pruning."),
    },
    "SVM (RBF)": {
        "C": F(1.0, 0.001, 1000.0, 0.1,
               "Penalti kesalahan. Besar = margin sempit, model lebih kompleks.",
               "Error penalty. Larger = narrower margin, more complex model."),
        "kernel": C("rbf", ["rbf", "linear", "poly", "sigmoid"],
                    "Fungsi kernel yang memetakan data ke ruang berdimensi lebih tinggi.",
                    "Kernel function mapping data into a higher-dimensional space."),
        "gamma": NS("scale", "Jangkauan pengaruh satu sampel: scale, auto, atau angka.",
                    "Reach of a single sample: scale, auto, or a number.", options=["scale", "auto"]),
        "degree": I(3, 1, 8, 1, "Derajat polinomial (khusus kernel poly).",
                    "Polynomial degree (poly kernel only)."),
        "coef0": F(0.0, -10.0, 10.0, 0.1, "Suku bebas kernel poly/sigmoid.",
                   "Independent term for the poly/sigmoid kernel."),
        "shrinking": B(True, "Gunakan heuristik penyusutan agar pelatihan lebih cepat.",
                       "Use the shrinking heuristic to speed up training."),
        "tol": F(1e-3, 1e-8, 0.1, 1e-4, "Ambang konvergensi.", "Convergence tolerance.", fmt="%.1e"),
        "max_iter": I(-1, -1, 1000000, 100, "Batas iterasi. -1 = tanpa batas.",
                      "Iteration cap. -1 = unlimited."),
        "class_weight": C("None", ["None", "balanced"], "Penyeimbang bobot kelas.",
                          "Class weight balancing."),
    },
    "K-Nearest Neighbors": {
        "n_neighbors": I(5, 1, 100, 1,
                         "Jumlah tetangga. Kecil = batas keputusan berlekuk (varians tinggi).",
                         "Number of neighbours. Small = jagged boundary (high variance)."),
        "weights": C("uniform", ["uniform", "distance"],
                     "Bobot tetangga: seragam atau berbanding terbalik dengan jarak.",
                     "Neighbour weighting: uniform or inverse-distance."),
        "algorithm": C("auto", ["auto", "ball_tree", "kd_tree", "brute"],
                       "Struktur data pencarian tetangga.", "Neighbour search data structure."),
        "leaf_size": I(30, 1, 200, 1, "Ukuran daun pada ball_tree/kd_tree.",
                       "Leaf size for ball_tree/kd_tree."),
        "p": I(2, 1, 5, 1, "1 = jarak Manhattan, 2 = jarak Euclid.",
               "1 = Manhattan distance, 2 = Euclidean distance."),
        "metric": C("minkowski", ["minkowski", "euclidean", "manhattan", "chebyshev", "cosine"],
                    "Ukuran jarak antar titik.", "Distance metric between points."),
    },
    "Naive Bayes": {
        "var_smoothing": F(1e-9, 1e-15, 1e-3, 1e-9,
                           "Penghalus varians untuk kestabilan numerik.",
                           "Variance smoothing for numerical stability.", fmt="%.1e"),
    },
}


# ==================================================== ruang parameter: regresi
_TREE_REG = {
    "criterion": C("squared_error", ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                   "Ukuran galat untuk memilih pemecahan.", "Error measure used to choose splits."),
    "splitter": C("best", ["best", "random"], "Cara memilih pemecahan.", "How splits are chosen."),
    "max_depth": I(0, 0, 60, 1, "Kedalaman maksimum pohon. 0 = tanpa batas.",
                   "Maximum tree depth. 0 = unlimited.", none_at=0),
    "min_samples_split": I(2, 2, 200, 1, "Minimum sampel untuk memecah simpul.",
                           "Minimum samples to split a node."),
    "min_samples_leaf": I(1, 1, 200, 1, "Minimum sampel per daun.", "Minimum samples per leaf."),
    "max_features": NS("None", "Fitur per pemecahan.", "Features per split.",
                       options=["None", "sqrt", "log2"]),
    "max_leaf_nodes": I(0, 0, 500, 1, "Batas jumlah daun. 0 = tanpa batas.",
                        "Cap on leaves. 0 = unlimited.", none_at=0),
    "min_impurity_decrease": F(0.0, 0.0, 0.5, 0.001, "Ambang penurunan ketidakmurnian.",
                               "Minimum impurity decrease."),
    "ccp_alpha": F(0.0, 0.0, 0.1, 0.001, "Pemangkasan cost-complexity.", "Cost-complexity pruning."),
}

_FOREST_REG = {
    "n_estimators": I(300, 10, 1000, 10, "Jumlah pohon.", "Number of trees."),
    "criterion": C("squared_error", ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                   "Ukuran galat pemecahan.", "Split error measure."),
    "max_depth": I(0, 0, 60, 1, "Kedalaman maksimum. 0 = tanpa batas.",
                   "Maximum depth. 0 = unlimited.", none_at=0),
    "min_samples_split": I(2, 2, 200, 1, "Minimum sampel untuk memecah simpul.",
                           "Minimum samples to split a node."),
    "min_samples_leaf": I(1, 1, 200, 1, "Minimum sampel per daun.", "Minimum samples per leaf."),
    "max_features": NS("1.0", "Fitur per pemecahan: angka, sqrt, log2, atau None.",
                       "Features per split: a number, sqrt, log2, or None.",
                       options=["1.0", "sqrt", "log2", "None"]),
    "bootstrap": B(True, "Sampel dengan pengembalian (bagging).", "Sample with replacement (bagging)."),
    "oob_score": B(False, "Hitung skor out-of-bag.", "Compute the out-of-bag score."),
    "max_samples": F(0.0, 0.0, 1.0, 0.05, "Proporsi data tiap pohon. 0 = seluruh data.",
                     "Row fraction per tree. 0 = all rows.", none_at=0.0),
    "ccp_alpha": F(0.0, 0.0, 0.1, 0.001, "Pemangkasan cost-complexity.", "Cost-complexity pruning."),
}

PARAMS_REG = {
    "Linear Regression": {
        "fit_intercept": B(True, "Sertakan konstanta (intersep) dalam model.",
                           "Include the intercept term."),
        "positive": B(False, "Paksa seluruh koefisien bernilai positif.",
                      "Force all coefficients to be positive."),
    },
    "Ridge": {
        "alpha": F(1.0, 0.0001, 1000.0, 0.01,
                   "Kekuatan regularisasi L2. Besar = koefisien makin ditekan ke nol.",
                   "L2 regularization strength. Larger = coefficients shrunk harder."),
        "solver": C("auto", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                    "Metode penyelesaian sistem persamaan.", "Solver for the linear system."),
        "fit_intercept": B(True, "Sertakan intersep.", "Include intercept."),
        "max_iter": I(0, 0, 100000, 100, "Batas iterasi (0 = bawaan solver).",
                      "Iteration cap (0 = solver default).", none_at=0),
        "tol": F(1e-4, 1e-10, 0.1, 1e-4, "Ambang konvergensi.", "Convergence tolerance.", fmt="%.1e"),
    },
    "Lasso": {
        "alpha": F(0.01, 0.00001, 100.0, 0.001,
                   "Kekuatan regularisasi L1; besar = makin banyak koefisien menjadi nol.",
                   "L1 strength; larger drives more coefficients to exactly zero."),
        "fit_intercept": B(True, "Sertakan intersep.", "Include intercept."),
        "max_iter": I(10000, 100, 200000, 100, "Batas iterasi koordinat turun.",
                      "Coordinate-descent iteration cap."),
        "tol": F(1e-4, 1e-10, 0.1, 1e-4, "Ambang konvergensi.", "Convergence tolerance.", fmt="%.1e"),
        "selection": C("cyclic", ["cyclic", "random"], "Urutan pembaruan koefisien.",
                       "Coefficient update order."),
        "positive": B(False, "Paksa koefisien positif.", "Force positive coefficients."),
    },
    "Elastic Net": {
        "alpha": F(0.01, 0.00001, 100.0, 0.001, "Kekuatan regularisasi total.",
                   "Overall regularization strength."),
        "l1_ratio": F(0.5, 0.0, 1.0, 0.05,
                      "Proporsi L1 terhadap L2. 1 = Lasso murni, 0 = Ridge murni.",
                      "L1 vs L2 mix. 1 = pure Lasso, 0 = pure Ridge."),
        "fit_intercept": B(True, "Sertakan intersep.", "Include intercept."),
        "max_iter": I(10000, 100, 200000, 100, "Batas iterasi.", "Iteration cap."),
        "selection": C("cyclic", ["cyclic", "random"], "Urutan pembaruan koefisien.",
                       "Coefficient update order."),
    },
    "Decision Tree": dict(_TREE_REG),
    "Random Forest": dict(_FOREST_REG),
    "Extra Trees": {**_FOREST_REG, "bootstrap": B(False, "Tanpa bootstrap (bawaan Extra Trees).",
                                                  "No bootstrap (Extra Trees default).")},
    "Gradient Boosting": {
        "loss": C("squared_error", ["squared_error", "absolute_error", "huber", "quantile"],
                  "Fungsi kerugian.", "Loss function."),
        "n_estimators": I(100, 10, 1000, 10, "Jumlah tahap boosting.", "Boosting stages."),
        "learning_rate": F(0.1, 0.001, 1.0, 0.01, "Laju belajar.", "Learning rate."),
        "max_depth": I(3, 1, 20, 1, "Kedalaman pohon lemah.", "Weak learner depth."),
        "min_samples_split": I(2, 2, 200, 1, "Minimum sampel memecah simpul.",
                               "Minimum samples to split."),
        "min_samples_leaf": I(1, 1, 200, 1, "Minimum sampel per daun.", "Minimum samples per leaf."),
        "subsample": F(1.0, 0.1, 1.0, 0.05, "Proporsi sampel tiap tahap.", "Row fraction per stage."),
        "max_features": NS("None", "Fitur per pemecahan.", "Features per split.",
                           options=["None", "sqrt", "log2"]),
        "alpha": F(0.9, 0.01, 0.99, 0.01, "Kuantil untuk loss huber/quantile.",
                   "Quantile used by the huber/quantile loss."),
    },
    "SVR (RBF)": {
        "C": F(1.0, 0.001, 1000.0, 0.1, "Penalti kesalahan.", "Error penalty."),
        "kernel": C("rbf", ["rbf", "linear", "poly", "sigmoid"], "Fungsi kernel.", "Kernel function."),
        "gamma": NS("scale", "Jangkauan pengaruh satu sampel.", "Reach of a single sample.",
                    options=["scale", "auto"]),
        "epsilon": F(0.1, 0.0, 10.0, 0.01,
                     "Lebar pita toleransi galat yang tidak dikenai penalti.",
                     "Width of the no-penalty error tube."),
        "degree": I(3, 1, 8, 1, "Derajat polinomial (kernel poly).", "Polynomial degree (poly kernel)."),
        "coef0": F(0.0, -10.0, 10.0, 0.1, "Suku bebas kernel poly/sigmoid.",
                   "Independent term for poly/sigmoid."),
        "tol": F(1e-3, 1e-8, 0.1, 1e-4, "Ambang konvergensi.", "Convergence tolerance.", fmt="%.1e"),
        "max_iter": I(-1, -1, 1000000, 100, "Batas iterasi. -1 = tanpa batas.",
                      "Iteration cap. -1 = unlimited."),
    },
    "K-Nearest Neighbors": {
        "n_neighbors": I(5, 1, 100, 1, "Jumlah tetangga.", "Number of neighbours."),
        "weights": C("uniform", ["uniform", "distance"], "Bobot tetangga.", "Neighbour weighting."),
        "algorithm": C("auto", ["auto", "ball_tree", "kd_tree", "brute"],
                       "Struktur pencarian tetangga.", "Neighbour search structure."),
        "leaf_size": I(30, 1, 200, 1, "Ukuran daun pohon pencarian.", "Search tree leaf size."),
        "p": I(2, 1, 5, 1, "1 = Manhattan, 2 = Euclid.", "1 = Manhattan, 2 = Euclidean."),
        "metric": C("minkowski", ["minkowski", "euclidean", "manhattan", "chebyshev", "cosine"],
                    "Ukuran jarak.", "Distance metric."),
    },
}


# ================================================= ruang parameter: klastering
PARAMS_CLU = {
    "K-Means": {
        "n_clusters": I(3, 2, 30, 1, "Jumlah klaster (k) yang dibentuk.",
                        "Number of clusters (k) to form."),
        "init": C("k-means++", ["k-means++", "random"],
                  "Cara memilih pusat awal. k-means++ menyebar pusat awal agar konvergensi stabil.",
                  "Initial centroid strategy. k-means++ spreads seeds for stable convergence."),
        "n_init": I(10, 1, 50, 1, "Jumlah pengulangan dengan pusat awal berbeda; hasil terbaik dipakai.",
                    "Number of restarts with different seeds; the best result is kept."),
        "max_iter": I(300, 10, 5000, 10, "Batas iterasi tiap pengulangan.",
                      "Iteration cap per restart."),
        "tol": F(1e-4, 1e-10, 0.1, 1e-4, "Ambang pergeseran pusat untuk berhenti.",
                 "Centroid shift tolerance for stopping.", fmt="%.1e"),
        "algorithm": C("lloyd", ["lloyd", "elkan"], "Varian algoritma K-Means.",
                       "K-Means algorithm variant."),
    },
    "Agglomerative (Hierarchical)": {
        "n_clusters": I(3, 2, 30, 1, "Jumlah klaster hasil pemotongan dendrogram.",
                        "Number of clusters obtained by cutting the dendrogram."),
        "linkage": C("ward", ["ward", "complete", "average", "single"],
                     "Cara mengukur jarak antar klaster saat digabung.",
                     "How distance between clusters is measured when merging."),
        "metric": C("euclidean", ["euclidean", "manhattan", "cosine", "l1", "l2"],
                    "Ukuran jarak antar titik (ward hanya mendukung euclidean).",
                    "Distance metric between points (ward supports euclidean only)."),
    },
    "DBSCAN": {
        "eps": F(0.8, 0.01, 20.0, 0.05,
                 "Radius ketetanggaan. Besar = klaster makin menyatu.",
                 "Neighbourhood radius. Larger = clusters merge more."),
        "min_samples": I(5, 2, 100, 1,
                         "Minimum tetangga agar sebuah titik menjadi titik inti (core point).",
                         "Minimum neighbours for a point to be a core point."),
        "metric": C("euclidean", ["euclidean", "manhattan", "chebyshev", "cosine"],
                    "Ukuran jarak.", "Distance metric."),
        "algorithm": C("auto", ["auto", "ball_tree", "kd_tree", "brute"],
                       "Struktur pencarian tetangga.", "Neighbour search structure."),
        "leaf_size": I(30, 1, 200, 1, "Ukuran daun pohon pencarian.", "Search tree leaf size."),
        "p": I(2, 1, 5, 1, "Orde Minkowski.", "Minkowski order."),
    },
    "Gaussian Mixture": {
        "n_components": I(3, 1, 30, 1, "Jumlah komponen Gaussian (klaster).",
                          "Number of Gaussian components (clusters)."),
        "covariance_type": C("full", ["full", "tied", "diag", "spherical"],
                             "Bentuk matriks kovarians tiap komponen (full = elips bebas).",
                             "Covariance shape per component (full = free-form ellipse)."),
        "n_init": I(1, 1, 20, 1, "Jumlah inisialisasi ulang.", "Number of initializations."),
        "max_iter": I(100, 10, 2000, 10, "Batas iterasi EM.", "EM iteration cap."),
        "tol": F(1e-3, 1e-10, 0.1, 1e-4, "Ambang konvergensi log-likelihood.",
                 "Log-likelihood convergence tolerance.", fmt="%.1e"),
        "reg_covar": F(1e-6, 1e-12, 0.1, 1e-6, "Penambah diagonal kovarians agar tidak singular.",
                       "Diagonal added to covariance to keep it non-singular.", fmt="%.1e"),
        "init_params": C("kmeans", ["kmeans", "k-means++", "random", "random_from_data"],
                         "Cara menentukan nilai awal komponen.", "Component initialization method."),
    },
}


# =============================================== ruang parameter: association
_AR_COMMON = {
    "min_support": F(0.05, 0.001, 1.0, 0.005,
                     "Proporsi minimum transaksi yang memuat sebuah itemset.",
                     "Minimum fraction of transactions containing an itemset."),
    "metric": C("confidence", ["confidence", "lift", "leverage", "conviction", "support"],
                "Metrik penyaring aturan utama.", "Primary metric used to filter rules."),
    "min_threshold": F(0.4, 0.0, 10.0, 0.05, "Ambang minimum untuk metrik di atas.",
                       "Minimum threshold for the metric above."),
    "min_lift": F(1.0, 0.0, 20.0, 0.1,
                  "Ambang lift. Lift > 1 berarti kaitan lebih kuat daripada kebetulan.",
                  "Lift threshold. Lift > 1 means stronger than chance association."),
    "max_len": I(3, 1, 10, 1, "Panjang maksimum itemset yang dicari.",
                 "Maximum itemset length to search."),
}

PARAMS_AR = {
    "Apriori": {**_AR_COMMON,
                "low_memory": B(False,
                                "Mode hemat memori (lebih lambat, untuk data sangat besar).",
                                "Memory-saving mode (slower, for very large data).")},
    "FP-Growth": dict(_AR_COMMON),
}


# ================================================ ruang parameter: time series
PARAMS_TS = {
    "SARIMAX (manual)": {
        "p": I(1, 0, 12, 1, "Ordo autoregresif (AR): berapa lag masa lalu dipakai.",
               "Autoregressive order (AR): how many past lags are used."),
        "d": I(1, 0, 3, 1, "Ordo pembedaan (differencing) untuk menstasionerkan deret.",
               "Differencing order used to make the series stationary."),
        "q": I(1, 0, 12, 1, "Ordo rata-rata bergerak (MA): berapa lag galat dipakai.",
               "Moving-average order (MA): how many error lags are used."),
        "P": I(1, 0, 4, 1, "Ordo AR musiman.", "Seasonal AR order."),
        "D": I(1, 0, 2, 1, "Ordo pembedaan musiman.", "Seasonal differencing order."),
        "Q": I(1, 0, 4, 1, "Ordo MA musiman.", "Seasonal MA order."),
        "trend": C("n", ["n", "c", "t", "ct"],
                   "Komponen tren deterministik: n=tidak ada, c=konstanta, t=linear, ct=keduanya.",
                   "Deterministic trend: n=none, c=constant, t=linear, ct=both."),
        "enforce_stationarity": B(False, "Paksa parameter AR memenuhi syarat stasioneritas.",
                                  "Force AR parameters to satisfy stationarity."),
        "enforce_invertibility": B(False, "Paksa parameter MA memenuhi syarat invertibilitas.",
                                   "Force MA parameters to satisfy invertibility."),
        "concentrate_scale": B(False, "Konsentrasikan skala galat agar estimasi lebih cepat.",
                               "Concentrate the error scale for faster estimation."),
    },
    "SARIMAX (grid search)": {
        "max_p": I(2, 0, 6, 1, "Batas atas p yang dicoba.", "Upper bound of p searched."),
        "max_d": I(2, 0, 3, 1, "Batas atas d yang dicoba.", "Upper bound of d searched."),
        "max_q": I(2, 0, 6, 1, "Batas atas q yang dicoba.", "Upper bound of q searched."),
        "max_P": I(1, 0, 3, 1, "Batas atas P musiman.", "Upper bound of seasonal P."),
        "max_D": I(1, 0, 2, 1, "Batas atas D musiman.", "Upper bound of seasonal D."),
        "max_Q": I(1, 0, 3, 1, "Batas atas Q musiman.", "Upper bound of seasonal Q."),
        "ic": C("aic", ["aic", "bic", "hqic"],
                "Kriteria informasi untuk memilih ordo terbaik.",
                "Information criterion used to pick the best order."),
        "trend": C("n", ["n", "c", "t", "ct"], "Komponen tren deterministik.",
                   "Deterministic trend component."),
        "enforce_stationarity": B(False, "Paksa syarat stasioneritas.", "Enforce stationarity."),
        "enforce_invertibility": B(False, "Paksa syarat invertibilitas.", "Enforce invertibility."),
    },
}

SPACES = {
    "classification": PARAMS_CLF,
    "regression": PARAMS_REG,
    "clustering": PARAMS_CLU,
    "association": PARAMS_AR,
    "timeseries": PARAMS_TS,
}


# ================================================================= akses
def space(task: str) -> dict:
    return SPACES.get(task, {})


def algos(task: str) -> list[str]:
    return list(space(task))


def spec(task: str, algo: str) -> dict:
    return space(task).get(algo, {})


def defaults(task: str, algo: str) -> dict:
    return {k: v["default"] for k, v in spec(task, algo).items()}


def all_defaults(task: str) -> dict:
    return {a: defaults(task, a) for a in algos(task)}


# ============================================================== konversi
_TRUE = {"true", "1", "yes", "ya", "benar"}
_FALSE = {"false", "0", "no", "tidak", "salah"}


def _as_number(text: str):
    t = str(text).strip()
    try:
        if any(c in t for c in ".eE") and t.lower() not in ("inf", "-inf"):
            return float(t)
        return int(t)
    except ValueError:
        try:
            return float(t)
        except ValueError:
            return None


def coerce(sp: dict, raw):
    """Ubah nilai mentah (mis. dari JSON) menjadi nilai widget yang sah."""
    kind = sp["kind"]
    if raw is None:
        if kind in ("choice", "numstr"):
            return "None"
        if sp.get("none_at") is not None:
            return sp["none_at"]
        return sp["default"]
    if kind == "int":
        try:
            v = int(round(float(raw)))
        except (TypeError, ValueError):
            return sp["default"]
        return int(min(max(v, sp["min"]), sp["max"]))
    if kind == "float":
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return sp["default"]
        if not np.isfinite(v):
            return sp["default"]
        return float(min(max(v, sp["min"]), sp["max"]))
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        return bool(sp["default"])
    if kind == "choice":
        s = "None" if raw is None else str(raw)
        return s if s in sp["options"] else sp["default"]
    if kind == "numstr":
        if isinstance(raw, bool):
            return sp["default"]
        if isinstance(raw, (int, float)):
            return str(raw)
        return str(raw).strip() or sp["default"]
    return str(raw)


def coerce_all(task: str, algo: str, values: dict | None) -> dict:
    sp = spec(task, algo)
    values = values or {}
    return {k: coerce(s, values.get(k, s["default"])) for k, s in sp.items()}


def resolve(task: str, algo: str, values: dict | None) -> dict:
    """Nilai widget -> kwargs siap dipakai estimator (None dikembalikan sebagai None)."""
    sp = spec(task, algo)
    vals = coerce_all(task, algo, values)
    out = {}
    for k, s in sp.items():
        v = vals[k]
        if s["kind"] in ("int", "float"):
            if s.get("none_at") is not None and v == s["none_at"]:
                v = None
        elif s["kind"] == "choice":
            if v in ("None", "none"):
                v = None
        elif s["kind"] == "numstr":
            txt = str(v).strip()
            if txt in ("None", "none", ""):
                v = None
            else:
                num = _as_number(txt)
                v = txt if num is None else num
        out[k] = v
    return out


def as_json(task: str, per_algo: dict | None) -> dict:
    """Bentuk yang aman disimpan ke JSON (hanya tipe dasar)."""
    per_algo = per_algo or {}
    return {a: coerce_all(task, a, per_algo.get(a)) for a in algos(task)}


def changed_from_default(task: str, algo: str, values: dict) -> dict:
    d = defaults(task, algo)
    cur = coerce_all(task, algo, values)
    return {k: v for k, v in cur.items() if v != d.get(k)}


def _keep_supported(cls, kwargs: dict) -> dict:
    """Buang argumen yang tidak dikenal versi scikit-learn yang terpasang."""
    import inspect

    try:
        allowed = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


# ============================================================== pembangun
def _fix_logreg(kw: dict) -> dict:
    kw = dict(kw)
    pen = kw.get("penalty")
    solver = kw.get("solver", "lbfgs")
    if pen in (None, "none"):
        kw["penalty"] = None
        if solver == "liblinear":
            kw["solver"] = "lbfgs"
    elif pen == "l1" and solver not in ("liblinear", "saga"):
        kw["solver"] = "liblinear"
    elif pen == "elasticnet":
        kw["solver"] = "saga"
    if kw.get("penalty") != "elasticnet":
        kw.pop("l1_ratio", None)
    return kw


def _fix_agglo(kw: dict) -> dict:
    kw = dict(kw)
    if kw.get("linkage") == "ward":
        kw["metric"] = "euclidean"
    return kw


def build(task: str, algo: str, values: dict | None, rs: int = 42):
    """Buat objek estimator sesuai algoritma dan parameter pengguna."""
    kw = resolve(task, algo, values)

    if task == "classification":
        from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        if algo == "Logistic Regression":
            return LogisticRegression(random_state=rs, **_keep_supported(
                LogisticRegression, _fix_logreg(kw)))
        if algo == "Decision Tree":
            return DecisionTreeClassifier(random_state=rs,
                                          **_keep_supported(DecisionTreeClassifier, kw))
        if algo == "Random Forest":
            if not kw.get("bootstrap"):
                kw["oob_score"] = False
                kw["max_samples"] = None
            return RandomForestClassifier(random_state=rs, n_jobs=-1,
                                          **_keep_supported(RandomForestClassifier, kw))
        if algo == "Extra Trees":
            if not kw.get("bootstrap"):
                kw["oob_score"] = False
                kw["max_samples"] = None
            return ExtraTreesClassifier(random_state=rs, n_jobs=-1,
                                        **_keep_supported(ExtraTreesClassifier, kw))
        if algo == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=rs,
                                              **_keep_supported(GradientBoostingClassifier, kw))
        if algo == "SVM (RBF)":
            return SVC(probability=True, random_state=rs, **_keep_supported(SVC, kw))
        if algo == "K-Nearest Neighbors":
            if kw.get("metric") != "minkowski":
                kw.pop("p", None)
            return KNeighborsClassifier(**_keep_supported(KNeighborsClassifier, kw))
        if algo == "Naive Bayes":
            return GaussianNB(**_keep_supported(GaussianNB, kw))

    if task == "regression":
        from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                                      RandomForestRegressor)
        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor

        if algo == "Linear Regression":
            return LinearRegression(**_keep_supported(LinearRegression, kw))
        if algo == "Ridge":
            return Ridge(random_state=rs, **_keep_supported(Ridge, kw))
        if algo == "Lasso":
            return Lasso(random_state=rs, **_keep_supported(Lasso, kw))
        if algo == "Elastic Net":
            return ElasticNet(random_state=rs, **_keep_supported(ElasticNet, kw))
        if algo == "Decision Tree":
            return DecisionTreeRegressor(random_state=rs,
                                         **_keep_supported(DecisionTreeRegressor, kw))
        if algo == "Random Forest":
            if not kw.get("bootstrap"):
                kw["oob_score"] = False
                kw["max_samples"] = None
            return RandomForestRegressor(random_state=rs, n_jobs=-1,
                                         **_keep_supported(RandomForestRegressor, kw))
        if algo == "Extra Trees":
            if not kw.get("bootstrap"):
                kw["oob_score"] = False
                kw["max_samples"] = None
            return ExtraTreesRegressor(random_state=rs, n_jobs=-1,
                                       **_keep_supported(ExtraTreesRegressor, kw))
        if algo == "Gradient Boosting":
            if kw.get("loss") not in ("huber", "quantile"):
                kw.pop("alpha", None)
            return GradientBoostingRegressor(random_state=rs,
                                             **_keep_supported(GradientBoostingRegressor, kw))
        if algo == "SVR (RBF)":
            return SVR(**_keep_supported(SVR, kw))
        if algo == "K-Nearest Neighbors":
            if kw.get("metric") != "minkowski":
                kw.pop("p", None)
            return KNeighborsRegressor(**_keep_supported(KNeighborsRegressor, kw))

    if task == "clustering":
        from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
        from sklearn.mixture import GaussianMixture

        if algo == "K-Means":
            return KMeans(random_state=rs, **_keep_supported(KMeans, kw))
        if algo == "Agglomerative (Hierarchical)":
            kw = _fix_agglo(kw)
            keep = _keep_supported(AgglomerativeClustering, kw)
            if "metric" not in keep and "affinity" in dir(AgglomerativeClustering):
                keep["affinity"] = kw.get("metric", "euclidean")
            return AgglomerativeClustering(**keep)
        if algo == "DBSCAN":
            if kw.get("metric") != "minkowski":
                kw.pop("p", None)
            return DBSCAN(**_keep_supported(DBSCAN, kw))
        if algo == "Gaussian Mixture":
            return GaussianMixture(random_state=rs, **_keep_supported(GaussianMixture, kw))

    raise ValueError(f"Algoritma tidak dikenal: {task}/{algo}")
