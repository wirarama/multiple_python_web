"""Fine tuning hiperparameter + analisis perbandingan antar parameter.

Modul ini menjawab dua pertanyaan sekaligus:
  1. Kombinasi parameter mana yang terbaik?
  2. MENGAPA kombinasi itu terbaik — parameter mana yang benar-benar
     berpengaruh, seberapa besar pengaruhnya, dan apakah keunggulannya
     signifikan secara statistik atau hanya kebetulan.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from .i18n import L  # noqa: E402
from .mlutils import build_preprocessor  # noqa: E402

MAX_FITS_WARN = 1500


# ======================================================== ruang parameter
def _p(values, id_help, en_help):
    return {"values": values, "help": (id_help, en_help)}


SPACE_CLF = {
    "Logistic Regression": {
        "C": _p([0.01, 0.1, 1, 10, 100],
                "Kebalikan kekuatan regularisasi. Kecil = regularisasi kuat (model sederhana).",
                "Inverse regularization strength. Small = strong regularization (simpler model)."),
        "solver": _p(["lbfgs", "liblinear"],
                     "Algoritma optimisasi. liblinear cocok untuk data kecil.",
                     "Optimization algorithm. liblinear suits small datasets."),
        "class_weight": _p([None, "balanced"],
                           "'balanced' memberi bobot lebih pada kelas minoritas.",
                           "'balanced' up-weights minority classes."),
    },
    "Decision Tree": {
        "max_depth": _p([None, 3, 5, 10, 20],
                        "Kedalaman maksimum pohon. None = tumbuh penuh (rawan overfitting).",
                        "Maximum tree depth. None = grow fully (overfitting-prone)."),
        "min_samples_split": _p([2, 5, 10, 20],
                                "Minimum sampel agar sebuah simpul boleh dipecah.",
                                "Minimum samples required to split a node."),
        "min_samples_leaf": _p([1, 2, 5, 10],
                               "Minimum sampel pada tiap daun. Besar = pohon lebih halus.",
                               "Minimum samples at each leaf. Larger = smoother tree."),
        "criterion": _p(["gini", "entropy"],
                        "Ukuran ketidakmurnian simpul.",
                        "Node impurity measure."),
        "class_weight": _p([None, "balanced"],
                           "Penyeimbang bobot kelas.",
                           "Class weight balancing."),
    },
    "Random Forest": {
        "n_estimators": _p([100, 200, 400],
                           "Jumlah pohon. Lebih banyak = lebih stabil tapi lebih lambat.",
                           "Number of trees. More = more stable but slower."),
        "max_depth": _p([None, 5, 10, 20],
                        "Kedalaman maksimum tiap pohon.",
                        "Maximum depth of each tree."),
        "min_samples_leaf": _p([1, 2, 4],
                               "Minimum sampel per daun.",
                               "Minimum samples per leaf."),
        "max_features": _p(["sqrt", "log2", None],
                           "Jumlah fitur yang dipertimbangkan tiap pemecahan.",
                           "Number of features considered at each split."),
        "class_weight": _p([None, "balanced"],
                           "Penyeimbang bobot kelas.",
                           "Class weight balancing."),
    },
    "Extra Trees": {
        "n_estimators": _p([100, 200, 400], "Jumlah pohon.", "Number of trees."),
        "max_depth": _p([None, 5, 10, 20], "Kedalaman maksimum.", "Maximum depth."),
        "min_samples_leaf": _p([1, 2, 4], "Minimum sampel per daun.", "Minimum samples per leaf."),
        "max_features": _p(["sqrt", "log2", None], "Fitur per pemecahan.", "Features per split."),
    },
    "Gradient Boosting": {
        "n_estimators": _p([100, 200, 300],
                           "Jumlah tahap boosting.", "Number of boosting stages."),
        "learning_rate": _p([0.01, 0.05, 0.1, 0.2],
                            "Laju belajar. Kecil butuh lebih banyak pohon tapi lebih akurat.",
                            "Learning rate. Smaller needs more trees but generalizes better."),
        "max_depth": _p([2, 3, 5],
                        "Kedalaman tiap pohon lemah.", "Depth of each weak learner."),
        "subsample": _p([0.8, 1.0],
                        "Proporsi sampel per tahap (<1 = stochastic boosting).",
                        "Sample fraction per stage (<1 = stochastic boosting)."),
    },
    "SVM (RBF)": {
        "C": _p([0.1, 1, 10, 100],
                "Penalti kesalahan. Besar = margin sempit, model lebih kompleks.",
                "Error penalty. Larger = narrower margin, more complex model."),
        "gamma": _p(["scale", "auto", 0.01, 0.1],
                    "Jangkauan pengaruh satu sampel. Besar = model sangat lokal.",
                    "Reach of a single sample. Larger = very local model."),
        "kernel": _p(["rbf", "linear", "poly"],
                     "Fungsi kernel.", "Kernel function."),
        "class_weight": _p([None, "balanced"], "Penyeimbang kelas.", "Class balancing."),
    },
    "K-Nearest Neighbors": {
        "n_neighbors": _p([3, 5, 7, 9, 11, 15],
                          "Jumlah tetangga. Kecil = batas keputusan berlekuk (varians tinggi).",
                          "Number of neighbours. Small = jagged boundary (high variance)."),
        "weights": _p(["uniform", "distance"],
                      "Bobot tetangga: seragam atau berdasarkan jarak.",
                      "Neighbour weighting: uniform or distance-based."),
        "p": _p([1, 2],
                "1 = jarak Manhattan, 2 = jarak Euclid.",
                "1 = Manhattan distance, 2 = Euclidean distance."),
    },
    "Naive Bayes": {
        "var_smoothing": _p([1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
                            "Penghalus varians untuk kestabilan numerik.",
                            "Variance smoothing for numerical stability."),
    },
}

SPACE_REG = {
    "Linear Regression": {
        "fit_intercept": _p([True, False],
                            "Sertakan konstanta (intersep) dalam model.",
                            "Include the intercept term."),
        "positive": _p([False, True],
                       "Paksa seluruh koefisien bernilai positif.",
                       "Force all coefficients to be positive."),
    },
    "Ridge": {
        "alpha": _p([0.01, 0.1, 1, 10, 100],
                    "Kekuatan regularisasi L2. Besar = koefisien makin ditekan.",
                    "L2 regularization strength. Larger = coefficients shrunk more."),
        "solver": _p(["auto", "svd", "cholesky", "lsqr"],
                     "Metode penyelesaian sistem persamaan.",
                     "Solver used for the linear system."),
        "fit_intercept": _p([True, False], "Sertakan intersep.", "Include intercept."),
    },
    "Lasso": {
        "alpha": _p([0.001, 0.01, 0.1, 1, 10],
                    "Kekuatan regularisasi L1; besar = makin banyak koefisien jadi nol.",
                    "L1 strength; larger drives more coefficients to zero."),
        "selection": _p(["cyclic", "random"],
                        "Urutan pembaruan koefisien.", "Coefficient update order."),
    },
    "Elastic Net": {
        "alpha": _p([0.001, 0.01, 0.1, 1],
                    "Kekuatan regularisasi total.", "Overall regularization strength."),
        "l1_ratio": _p([0.1, 0.3, 0.5, 0.7, 0.9],
                       "Proporsi L1 terhadap L2. 1 = Lasso murni, 0 = Ridge murni.",
                       "L1 vs L2 mix. 1 = pure Lasso, 0 = pure Ridge."),
    },
    "Decision Tree": {
        "max_depth": _p([None, 3, 5, 10, 20], "Kedalaman maksimum pohon.", "Maximum tree depth."),
        "min_samples_split": _p([2, 5, 10, 20], "Minimum sampel untuk memecah simpul.",
                                "Minimum samples to split a node."),
        "min_samples_leaf": _p([1, 2, 5, 10], "Minimum sampel per daun.",
                               "Minimum samples per leaf."),
        "criterion": _p(["squared_error", "friedman_mse", "absolute_error"],
                        "Ukuran galat untuk memilih pemecahan.",
                        "Error measure used to choose splits."),
    },
    "Random Forest": {
        "n_estimators": _p([100, 200, 400], "Jumlah pohon.", "Number of trees."),
        "max_depth": _p([None, 5, 10, 20], "Kedalaman maksimum.", "Maximum depth."),
        "min_samples_leaf": _p([1, 2, 4], "Minimum sampel per daun.", "Minimum samples per leaf."),
        "max_features": _p([1.0, "sqrt", "log2"], "Fitur per pemecahan.", "Features per split."),
    },
    "Extra Trees": {
        "n_estimators": _p([100, 200, 400], "Jumlah pohon.", "Number of trees."),
        "max_depth": _p([None, 5, 10, 20], "Kedalaman maksimum.", "Maximum depth."),
        "min_samples_leaf": _p([1, 2, 4], "Minimum sampel per daun.", "Minimum samples per leaf."),
        "max_features": _p([1.0, "sqrt", "log2"], "Fitur per pemecahan.", "Features per split."),
    },
    "Gradient Boosting": {
        "n_estimators": _p([100, 200, 300], "Jumlah tahap boosting.", "Boosting stages."),
        "learning_rate": _p([0.01, 0.05, 0.1, 0.2], "Laju belajar.", "Learning rate."),
        "max_depth": _p([2, 3, 5], "Kedalaman pohon lemah.", "Weak learner depth."),
        "subsample": _p([0.8, 1.0], "Proporsi sampel per tahap.", "Sample fraction per stage."),
    },
    "SVR (RBF)": {
        "C": _p([0.1, 1, 10, 100], "Penalti kesalahan.", "Error penalty."),
        "gamma": _p(["scale", "auto", 0.01, 0.1], "Jangkauan pengaruh sampel.",
                    "Reach of a single sample."),
        "epsilon": _p([0.01, 0.1, 0.2, 0.5],
                      "Lebar pita toleransi galat (tidak dikenai penalti).",
                      "Width of the no-penalty error tube."),
        "kernel": _p(["rbf", "linear"], "Fungsi kernel.", "Kernel function."),
    },
    "K-Nearest Neighbors": {
        "n_neighbors": _p([3, 5, 7, 9, 11, 15], "Jumlah tetangga.", "Number of neighbours."),
        "weights": _p(["uniform", "distance"], "Bobot tetangga.", "Neighbour weighting."),
        "p": _p([1, 2], "1 = Manhattan, 2 = Euclid.", "1 = Manhattan, 2 = Euclidean."),
    },
}

SCORING_CLF = {
    "f1_macro": ("F1-macro (seimbang antar kelas)", "Macro F1 (equal weight per class)"),
    "accuracy": ("Akurasi", "Accuracy"),
    "balanced_accuracy": ("Akurasi seimbang", "Balanced accuracy"),
    "f1_weighted": ("F1-weighted", "Weighted F1"),
    "precision_macro": ("Precision-macro", "Macro precision"),
    "recall_macro": ("Recall-macro", "Macro recall"),
    "roc_auc_ovr_weighted": ("ROC AUC (OvR, weighted)", "ROC AUC (OvR, weighted)"),
}
SCORING_REG = {
    "r2": ("R²", "R²"),
    "neg_root_mean_squared_error": ("RMSE (makin kecil makin baik)", "RMSE (lower is better)"),
    "neg_mean_absolute_error": ("MAE (makin kecil makin baik)", "MAE (lower is better)"),
    "neg_mean_absolute_percentage_error": ("MAPE (makin kecil makin baik)", "MAPE (lower is better)"),
}


def space(task: str) -> dict:
    return SPACE_CLF if task == "classification" else SPACE_REG


def scorings(task: str) -> dict:
    return SCORING_CLF if task == "classification" else SCORING_REG


def metric_label(scoring: str) -> str:
    d = {**SCORING_CLF, **SCORING_REG}
    return L(*d[scoring]) if scoring in d else scoring


def is_negated(scoring: str) -> bool:
    return scoring.startswith("neg_")


def positive_name(scoring: str) -> str:
    return scoring.replace("neg_", "").replace("root_mean_squared_error", "RMSE") \
        .replace("mean_absolute_percentage_error", "MAPE") \
        .replace("mean_absolute_error", "MAE")


# ============================================================ parsing grid
def parse_scalar(text):
    """Ubah teks menjadi nilai Python: None / bool / int / float / str."""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    s = text.strip()
    low = s.lower()
    if low in ("none", "null", "nan", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if any(ch in s for ch in ".eE") and low not in ("e",):
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def format_scalar(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def default_grid_frame(task: str, algo: str) -> pd.DataFrame:
    sp = space(task).get(algo, {})
    rows = []
    for name, meta in sp.items():
        rows.append({
            "parameter": name,
            "aktif": True,
            "nilai_diuji": ", ".join(format_scalar(v) for v in meta["values"]),
            "keterangan": L(*meta["help"]),
        })
    return pd.DataFrame(rows, columns=["parameter", "aktif", "nilai_diuji", "keterangan"])


def parse_grid(frame: pd.DataFrame) -> dict[str, list]:
    grid = {}
    for row in frame.to_dict("records"):
        name = str(row.get("parameter") or "").strip()
        if not name or not bool(row.get("aktif", False)):
            continue
        raw = str(row.get("nilai_diuji") or "")
        vals, seen = [], set()
        for part in raw.split(","):
            v = parse_scalar(part)
            key = format_scalar(v)
            if part.strip() == "" or key in seen:
                continue
            seen.add(key)
            vals.append(v)
        if vals:
            grid[name] = vals
    return grid


def count_combos(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= max(len(v), 1)
    return n


# ============================================================== pencarian
def make_cv(task: str, y, folds: int, seed: int):
    if task == "classification":
        n = int(min(folds, pd.Series(y).value_counts().min()))
        n = max(n, 2)
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return KFold(n_splits=max(int(folds), 2), shuffle=True, random_state=seed)


def build_pipeline(model_factory, X, scaling, seed):
    pre, num_cols, cat_cols = build_preprocessor(X, scaling)
    return Pipeline([("pre", pre), ("model", model_factory(seed))]), num_cols, cat_cols


def run_search(model_factory, X, y, grid, task, method="grid", n_iter=40,
               folds=5, scoring="f1_macro", scaling="standard", seed=42):
    """Jalankan pencarian hiperparameter. Kembalikan (search, cv, pipeline_dasar)."""
    pipe, num_cols, cat_cols = build_pipeline(model_factory, X, scaling, seed)
    cv = make_cv(task, y, folds, seed)
    param_grid = {f"model__{k}": v for k, v in grid.items()}
    common = dict(scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True,
                  error_score=np.nan, refit=True)
    if method == "random" and count_combos(grid) > n_iter:
        search = RandomizedSearchCV(pipe, param_grid, n_iter=int(n_iter),
                                    random_state=seed, **common)
    else:
        search = GridSearchCV(pipe, param_grid, **common)
    search.fit(X, y)
    return search, cv, (num_cols, cat_cols)


# ====================================================== tabel hasil & analisis
def _fold_cols(cv_results) -> list[str]:
    return sorted([c for c in cv_results if c.startswith("split") and c.endswith("_test_score")],
                  key=lambda c: int(c.split("split")[1].split("_")[0]))


def results_table(search, grid: dict, scoring: str) -> pd.DataFrame:
    """Semua kombinasi yang diuji beserta skor CV, skor latih, dan selisihnya."""
    r = search.cv_results_
    neg = is_negated(scoring)
    sign = -1.0 if neg else 1.0
    data = {}
    for p in grid:
        data[p] = [format_scalar(v) for v in r[f"param_model__{p}"]]
    out = pd.DataFrame(data)
    out.insert(0, L("Peringkat", "Rank"), r["rank_test_score"])
    out[L("Skor CV", "CV score")] = np.round(sign * r["mean_test_score"], 6)
    out[L("Std CV", "CV std")] = np.round(r["std_test_score"], 6)
    out[L("Skor latih", "Train score")] = np.round(sign * r["mean_train_score"], 6)
    out[L("Selisih latih-CV", "Train-CV gap")] = np.round(
        np.abs(r["mean_train_score"] - r["mean_test_score"]), 6)
    out[L("Waktu fit (detik)", "Fit time (s)")] = np.round(r["mean_fit_time"], 4)
    out = out.sort_values(L("Peringkat", "Rank")).reset_index(drop=True)
    return out


def fold_scores(search) -> np.ndarray:
    """Matriks (n_kandidat x n_fold) berisi skor tiap lipatan."""
    r = search.cv_results_
    cols = _fold_cols(r)
    return np.vstack([np.asarray(r[c], dtype=float) for c in cols]).T


def param_effects(search, grid: dict, scoring: str) -> dict[str, pd.DataFrame]:
    """Efek marginal tiap parameter: rata-rata skor CV & latih per nilai."""
    r = search.cv_results_
    sign = -1.0 if is_negated(scoring) else 1.0
    out = {}
    for p in grid:
        vals = [format_scalar(v) for v in r[f"param_model__{p}"]]
        d = pd.DataFrame({
            "nilai": vals,
            "cv": sign * np.asarray(r["mean_test_score"], dtype=float),
            "train": sign * np.asarray(r["mean_train_score"], dtype=float),
            "std": np.asarray(r["std_test_score"], dtype=float),
        }).dropna(subset=["cv"])
        if d.empty:
            continue
        g = d.groupby("nilai", sort=False).agg(
            n=("cv", "size"), cv_mean=("cv", "mean"), cv_std=("cv", "std"),
            cv_best=("cv", "max" if not is_negated(scoring) else "min"),
            train_mean=("train", "mean")).reset_index()
        g["cv_std"] = g["cv_std"].fillna(0.0)
        order = _value_order(p, grid[p])
        g["_ord"] = g["nilai"].map(order).fillna(9e9)
        g = g.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
        g.columns = [L("Nilai", "Value"), L("Jumlah kombinasi", "Combinations"),
                     L("Rata-rata skor CV", "Mean CV score"), L("Std antar kombinasi", "Std across combos"),
                     L("Skor CV terbaik", "Best CV score"), L("Rata-rata skor latih", "Mean train score")]
        out[p] = g.round(6)
    return out


def _value_order(param: str, values: list) -> dict:
    """Urutan tampilan nilai parameter (numerik menaik, kategorik apa adanya)."""
    def key(v):
        c = _complexity_value(param, v)
        return c if c is not None else float("inf")
    try:
        ordered = sorted(values, key=key)
    except Exception:
        ordered = values
    return {format_scalar(v): i for i, v in enumerate(ordered)}


def param_importance(search, grid: dict, scoring: str) -> pd.DataFrame:
    """Seberapa besar tiap parameter menjelaskan variasi skor (ANOVA + eta²)."""
    r = search.cv_results_
    sign = -1.0 if is_negated(scoring) else 1.0
    scores = sign * np.asarray(r["mean_test_score"], dtype=float)
    valid = np.isfinite(scores)
    rows = []
    for p in grid:
        vals = np.array([format_scalar(v) for v in r[f"param_model__{p}"]])[valid]
        sc = scores[valid]
        pairs = [(u, sc[vals == u]) for u in pd.unique(vals)]
        pairs = [(u, g) for u, g in pairs if len(g) > 0]
        if len(pairs) < 2:
            continue
        labels_u = [u for u, _ in pairs]
        groups = [g for _, g in pairs]
        means = np.array([g.mean() for g in groups])
        grand = sc.mean()
        ss_b = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
        ss_t = float(((sc - grand) ** 2).sum())
        eta2 = ss_b / ss_t if ss_t > 1e-15 else 0.0
        try:
            f, pv = sps.f_oneway(*groups) if all(len(g) > 1 for g in groups) else (np.nan, np.nan)
        except Exception:
            f, pv = np.nan, np.nan
        best_idx = int(np.argmax(means)) if not is_negated(scoring) else int(np.argmin(means))
        best_val = labels_u[best_idx]
        rows.append({
            L("Parameter", "Parameter"): p,
            L("Nilai diuji", "Values tested"): len(groups),
            L("Rentang pengaruh", "Effect range"): round(float(means.max() - means.min()), 6),
            "Eta²": round(float(eta2), 4),
            "F": round(float(f), 4) if np.isfinite(f) else "-",
            "p-value": round(float(pv), 6) if np.isfinite(pv) else "-",
            L("Nilai terbaik", "Best value"): best_val,
            L("Kesimpulan", "Verdict"): (
                L("Sangat berpengaruh", "Highly influential") if eta2 >= 0.14 else
                L("Cukup berpengaruh", "Moderately influential") if eta2 >= 0.06 else
                L("Pengaruh kecil", "Small influence")),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Eta²", ascending=False).reset_index(drop=True)


def significance_vs_best(search, grid: dict, scoring: str, top: int = 12) -> pd.DataFrame:
    """Uji-t berpasangan antar lipatan: apakah konfigurasi lain benar-benar kalah?"""
    r = search.cv_results_
    folds = fold_scores(search)
    sign = -1.0 if is_negated(scoring) else 1.0
    ranks = np.asarray(r["rank_test_score"])
    order = np.argsort(ranks)[:top]
    best_i = int(np.nanargmax(np.asarray(r["mean_test_score"], dtype=float)))
    best_folds = folds[best_i]
    rows = []
    for i in order:
        params = ", ".join(f"{p}={format_scalar(r[f'param_model__{p}'][i])}" for p in grid)
        f_i = folds[i]
        if i == best_i:
            stat, pv, note = np.nan, np.nan, L("(konfigurasi terbaik)", "(best configuration)")
        else:
            diff = best_folds - f_i
            if np.allclose(diff, 0) or not np.isfinite(diff).all():
                stat, pv = np.nan, 1.0
            else:
                stat, pv = sps.ttest_rel(best_folds, f_i)
            note = (L("Setara secara statistik", "Statistically tied") if (not np.isfinite(pv) or pv >= 0.05)
                    else L("Kalah signifikan", "Significantly worse"))
        rows.append({
            L("Peringkat", "Rank"): int(ranks[i]),
            L("Konfigurasi", "Configuration"): params,
            L("Skor CV", "CV score"): round(float(sign * r["mean_test_score"][i]), 6),
            L("Std antar lipatan", "Std across folds"): round(float(r["std_test_score"][i]), 6),
            L("Selisih dari terbaik", "Difference from best"): round(
                float(sign * (r["mean_test_score"][i] - r["mean_test_score"][best_i])), 6),
            "t": round(float(stat), 4) if np.isfinite(stat) else "-",
            "p-value": round(float(pv), 6) if np.isfinite(pv) else "-",
            L("Status", "Status"): note,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ kompleksitas
COMPLEX_DIR = {
    "n_estimators": 1, "max_depth": 1, "min_samples_split": -1, "min_samples_leaf": -1,
    "max_features": 1, "learning_rate": 1, "subsample": 1, "C": 1, "gamma": 1,
    "alpha": -1, "n_neighbors": -1, "var_smoothing": -1, "epsilon": -1, "degree": 1,
}
_ORDINAL = {
    "max_features": {"log2": 1.0, "sqrt": 2.0, "None": 3.0, "1.0": 3.0, "auto": 2.0},
    "gamma": {"scale": 1.0, "auto": 1.0},
    "kernel": {"linear": 1.0, "poly": 2.0, "rbf": 2.0},
    "weights": {"uniform": 1.0, "distance": 2.0},
}


def _complexity_value(param: str, value):
    """Nilai numerik untuk mengurutkan kompleksitas; None bila netral."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return 1e6 if param == "max_depth" else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value)
    if param in _ORDINAL and s in _ORDINAL[param]:
        return _ORDINAL[param][s]
    return None


def complexity_score(search, grid: dict) -> np.ndarray:
    """Skor kompleksitas relatif (0 = paling sederhana) untuk tiap kandidat."""
    r = search.cv_results_
    n = len(r["mean_test_score"])
    total = np.zeros(n)
    used = 0
    for p in grid:
        direction = COMPLEX_DIR.get(p, 0)
        if direction == 0:
            continue
        raw = [_complexity_value(p, v) for v in r[f"param_model__{p}"]]
        if any(v is None for v in raw):
            continue
        arr = np.asarray(raw, dtype=float)
        span = arr.max() - arr.min()
        if span <= 0:
            continue
        norm = (arr - arr.min()) / span
        total += norm if direction > 0 else (1.0 - norm)
        used += 1
    return total / used if used else np.zeros(n)


def one_se_rule(search, grid: dict, scoring: str) -> dict:
    """Kaidah one-standard-error: konfigurasi paling sederhana yang masih
    berada dalam 1 galat baku dari skor terbaik."""
    r = search.cv_results_
    mean = np.asarray(r["mean_test_score"], dtype=float)
    std = np.asarray(r["std_test_score"], dtype=float)
    n_folds = len(_fold_cols(r))
    se = std / max(np.sqrt(n_folds), 1)
    best_i = int(np.nanargmax(mean))
    threshold = mean[best_i] - se[best_i]
    within = np.where(np.isfinite(mean) & (mean >= threshold))[0]
    comp = complexity_score(search, grid)
    pick = int(within[np.argmin(comp[within])]) if len(within) else best_i
    sign = -1.0 if is_negated(scoring) else 1.0
    return {
        "best_index": best_i,
        "pick_index": pick,
        "n_within": int(len(within)),
        "threshold": float(sign * threshold),
        "se_best": float(se[best_i]),
        "best_params": {p: r[f"param_model__{p}"][best_i] for p in grid},
        "pick_params": {p: r[f"param_model__{p}"][pick] for p in grid},
        "best_score": float(sign * mean[best_i]),
        "pick_score": float(sign * mean[pick]),
        "best_time": float(r["mean_fit_time"][best_i]),
        "pick_time": float(r["mean_fit_time"][pick]),
        "simpler": bool(comp[pick] < comp[best_i] - 1e-9),
    }


def baseline_comparison(model_factory, X, y, cv, scoring, scaling, seed, search):
    """Bandingkan parameter bawaan vs hasil tuning pada lipatan CV yang sama."""
    pipe, _, _ = build_pipeline(model_factory, X, scaling, seed)
    base = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                          return_train_score=True, error_score=np.nan)
    base_folds = np.asarray(base["test_score"], dtype=float)
    r = search.cv_results_
    best_i = int(np.nanargmax(np.asarray(r["mean_test_score"], dtype=float)))
    tuned_folds = fold_scores(search)[best_i]
    try:
        stat, pv = sps.ttest_rel(tuned_folds, base_folds)
    except Exception:
        stat, pv = np.nan, np.nan
    diff = tuned_folds - base_folds
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    cohen = float(diff.mean() / sd) if sd > 1e-12 else np.nan
    sign = -1.0 if is_negated(scoring) else 1.0
    return {
        "base_folds": base_folds,
        "tuned_folds": tuned_folds,
        "base_mean": float(sign * np.nanmean(base_folds)),
        "tuned_mean": float(sign * np.nanmean(tuned_folds)),
        "base_std": float(np.nanstd(base_folds)),
        "tuned_std": float(np.nanstd(tuned_folds)),
        "delta": float(sign * (np.nanmean(tuned_folds) - np.nanmean(base_folds))),
        "t": float(stat) if np.isfinite(stat) else np.nan,
        "p": float(pv) if np.isfinite(pv) else np.nan,
        "cohen_d": cohen,
        "significant": bool(np.isfinite(pv) and pv < 0.05 and np.nanmean(tuned_folds) > np.nanmean(base_folds)),
    }


def baseline_table(cmp: dict, scoring: str) -> pd.DataFrame:
    name = metric_label(scoring)
    arah = (L("nilai lebih kecil lebih baik", "lower is better") if is_negated(scoring)
            else L("nilai lebih besar lebih baik", "higher is better"))
    return pd.DataFrame([
        (L("Metrik yang dioptimalkan", "Optimized metric"), f"{name} ({arah})"),
        (L("Skor CV parameter bawaan", "CV score with default params"), round(cmp["base_mean"], 6)),
        (L("Skor CV setelah tuning", "CV score after tuning"), round(cmp["tuned_mean"], 6)),
        (L("Selisih (sesudah − sebelum)", "Difference (after − before)"), round(cmp["delta"], 6)),
        (L("Perubahan relatif (%)", "Relative change (%)"),
         round(cmp["delta"] / abs(cmp["base_mean"]) * 100, 4) if abs(cmp["base_mean"]) > 1e-12 else "-"),
        (L("Uji-t berpasangan (t)", "Paired t-test (t)"),
         round(cmp["t"], 4) if np.isfinite(cmp["t"]) else "-"),
        (L("p-value", "p-value"), round(cmp["p"], 6) if np.isfinite(cmp["p"]) else "-"),
        (L("Ukuran efek (Cohen's d)", "Effect size (Cohen's d)"),
         round(cmp["cohen_d"], 4) if np.isfinite(cmp["cohen_d"]) else "-"),
        (L("Kesimpulan", "Conclusion"),
         L("Tuning memberi perbaikan signifikan", "Tuning gives a significant improvement")
         if cmp["significant"] else
         L("Perbaikan tidak signifikan secara statistik",
           "Improvement is not statistically significant")),
    ], columns=[L("Aspek", "Aspect"), L("Nilai", "Value")])


def interaction_pivot(search, grid: dict, p1: str, p2: str, scoring: str) -> pd.DataFrame:
    r = search.cv_results_
    sign = -1.0 if is_negated(scoring) else 1.0
    d = pd.DataFrame({
        p1: [format_scalar(v) for v in r[f"param_model__{p1}"]],
        p2: [format_scalar(v) for v in r[f"param_model__{p2}"]],
        "score": sign * np.asarray(r["mean_test_score"], dtype=float),
    }).dropna()
    if d.empty:
        return pd.DataFrame()
    piv = d.pivot_table(index=p1, columns=p2, values="score",
                        aggfunc="max" if not is_negated(scoring) else "min")
    o1, o2 = _value_order(p1, grid[p1]), _value_order(p2, grid[p2])
    piv = piv.reindex(index=sorted(piv.index, key=lambda v: o1.get(v, 9e9)),
                      columns=sorted(piv.columns, key=lambda v: o2.get(v, 9e9)))
    return piv.round(6)


# ============================================================== narasi
def narrative(search, grid, scoring, imp: pd.DataFrame, sig: pd.DataFrame,
              ose: dict, cmp: dict, effects: dict, task: str, algo: str) -> list[str]:
    """Susun penjelasan naratif: mengapa parameter tersebut yang terbaik."""
    r = search.cv_results_
    n_cand = len(r["mean_test_score"])
    n_folds = len(_fold_cols(r))
    mlab = metric_label(scoring)
    best_str = ", ".join(f"**{p} = {format_scalar(v)}**" for p, v in ose["best_params"].items())
    lines = []

    lines.append(L(
        f"Sebanyak **{n_cand} kombinasi parameter** {algo} diuji dengan validasi silang "
        f"{n_folds} lipatan, dioptimalkan terhadap metrik **{mlab}**. "
        f"Kombinasi terbaik adalah {best_str} dengan skor CV "
        f"**{ose['best_score']:.4f}** (galat baku {ose['se_best']:.4f}).",
        f"**{n_cand} parameter combinations** of {algo} were evaluated with "
        f"{n_folds}-fold cross-validation, optimizing **{mlab}**. "
        f"The best combination is {best_str} with a CV score of "
        f"**{ose['best_score']:.4f}** (standard error {ose['se_best']:.4f})."))

    if not imp.empty:
        top = imp.iloc[0]
        eta = top["Eta²"]
        pv = top["p-value"]
        sig_txt = (L("signifikan", "significant") if isinstance(pv, (int, float)) and pv < 0.05
                   else L("tidak signifikan", "not significant"))
        lines.append(L(
            f"Parameter yang paling menentukan adalah **{top[L('Parameter', 'Parameter')]}**: "
            f"nilainya menjelaskan **{eta:.1%}** variasi skor antar kombinasi "
            f"(Eta² = {eta:.4f}, p = {pv}, {sig_txt}), dengan rentang pengaruh "
            f"{top[L('Rentang pengaruh', 'Effect range')]:.4f} poin metrik. "
            f"Nilai terbaiknya adalah **{top[L('Nilai terbaik', 'Best value')]}**.",
            f"The most decisive parameter is **{top[L('Parameter', 'Parameter')]}**: "
            f"its value explains **{eta:.1%}** of the score variance across combinations "
            f"(Eta² = {eta:.4f}, p = {pv}, {sig_txt}), spanning "
            f"{top[L('Rentang pengaruh', 'Effect range')]:.4f} metric points. "
            f"Its best value is **{top[L('Nilai terbaik', 'Best value')]}**."))

        weak = imp[imp["Eta²"] < 0.06]
        if len(weak):
            names = ", ".join(weak[L("Parameter", "Parameter")].head(4).tolist())
            lines.append(L(
                f"Sebaliknya, {names} hampir tidak memengaruhi hasil (Eta² &lt; 0,06). "
                "Untuk pencarian berikutnya, parameter ini dapat dikunci pada nilai bawaan "
                "agar ruang pencarian lebih kecil dan lebih cepat.",
                f"In contrast, {names} barely affect the outcome (Eta² &lt; 0.06). "
                "In a follow-up search these can be fixed at their defaults to shrink "
                "the search space and speed it up."))

    # penjelasan arah efek untuk parameter terpenting
    if not imp.empty and imp.iloc[0][L("Parameter", "Parameter")] in effects:
        pname = imp.iloc[0][L("Parameter", "Parameter")]
        e = effects[pname]
        col_val = L("Nilai", "Value")
        col_cv = L("Rata-rata skor CV", "Mean CV score")
        col_tr = L("Rata-rata skor latih", "Mean train score")
        best_row = e.loc[e[col_cv].idxmax() if not is_negated(scoring) else e[col_cv].idxmin()]
        gap_best = abs(best_row[col_tr] - best_row[col_cv])
        gaps = (e[col_tr] - e[col_cv]).abs()
        worst_gap_row = e.loc[gaps.idxmax()]
        if gaps.max() > gap_best * 1.5 and gaps.max() > 1e-6:
            lines.append(L(
                f"Kurva validasi {pname} menjelaskan alasannya: pada "
                f"{pname} = {worst_gap_row[col_val]} skor latih dan skor CV berselisih "
                f"{gaps.max():.4f} (model menghafal data latih), sedangkan pada nilai terpilih "
                f"{pname} = {best_row[col_val]} selisihnya hanya {gap_best:.4f}. "
                "Nilai terpilih adalah titik keseimbangan antara bias dan varians.",
                f"The {pname} validation curve explains why: at "
                f"{pname} = {worst_gap_row[col_val]} the train and CV scores differ by "
                f"{gaps.max():.4f} (the model memorizes the training data), whereas at the chosen "
                f"{pname} = {best_row[col_val]} the gap is only {gap_best:.4f}. "
                "The chosen value sits at the bias-variance sweet spot."))

    # keunggulan signifikan atau tidak
    if not sig.empty:
        tied = sig[sig[L("Status", "Status")] == L("Setara secara statistik", "Statistically tied")]
        n_tied = max(len(tied) - 1, 0)
        if n_tied > 0:
            lines.append(L(
                f"Uji-t berpasangan antar lipatan menunjukkan **{n_tied} konfigurasi lain "
                "setara secara statistik** dengan yang terbaik (p ≥ 0,05). Artinya selisih "
                "skornya masih dalam rentang keacakan pembagian data, bukan keunggulan nyata.",
                f"Paired t-tests across folds show **{n_tied} other configurations are "
                "statistically tied** with the best one (p ≥ 0.05). Their score differences fall "
                "within the randomness of the data split rather than being a real advantage."))
        else:
            lines.append(L(
                "Uji-t berpasangan antar lipatan menunjukkan konfigurasi terbaik unggul "
                "signifikan (p &lt; 0,05) atas seluruh konfigurasi pesaing di peringkat teratas.",
                "Paired t-tests across folds show the best configuration significantly "
                "outperforms (p &lt; 0.05) every runner-up configuration."))

    # kaidah one-standard-error
    if ose["simpler"]:
        pick_str = ", ".join(f"**{p} = {format_scalar(v)}**" for p, v in ose["pick_params"].items())
        speed = (ose["best_time"] / ose["pick_time"]) if ose["pick_time"] > 1e-9 else 1.0
        lines.append(L(
            f"Menurut kaidah *one-standard-error*, ada {ose['n_within']} konfigurasi yang skornya "
            f"masih berada dalam 1 galat baku dari yang terbaik. Yang paling sederhana di antaranya "
            f"adalah {pick_str} dengan skor {ose['pick_score']:.4f} dan waktu latih "
            f"{speed:.1f}× lebih cepat. Konfigurasi ini layak dipilih bila kesederhanaan dan "
            "ketahanan terhadap data baru lebih diutamakan daripada skor tertinggi.",
            f"By the *one-standard-error* rule, {ose['n_within']} configurations score within one "
            f"standard error of the best. The simplest among them is {pick_str} with a score of "
            f"{ose['pick_score']:.4f} and a {speed:.1f}× faster fit time. It is the better pick "
            "when simplicity and robustness on new data matter more than the top score."))
    else:
        lines.append(L(
            f"Kaidah *one-standard-error* tidak menemukan alternatif yang lebih sederhana: "
            f"dari {ose['n_within']} konfigurasi yang skornya setara, konfigurasi terbaik "
            "sekaligus yang paling sederhana — jadi tidak ada kompleksitas yang terbuang.",
            f"The *one-standard-error* rule finds no simpler alternative: among the "
            f"{ose['n_within']} configurations with comparable scores, the best one is also the "
            "simplest, so no complexity is wasted."))

    # dibanding parameter bawaan
    arah = L("naik", "up") if cmp["delta"] >= 0 else L("turun", "down")
    lines.append(L(
        f"Dibandingkan parameter bawaan, skor CV {arah} dari {cmp['base_mean']:.4f} menjadi "
        f"{cmp['tuned_mean']:.4f} (selisih {cmp['delta']:+.4f}; uji-t berpasangan t = "
        f"{cmp['t']:.3f}, p = {cmp['p']:.4g}; Cohen's d = {cmp['cohen_d']:.3f}). "
        + (L("Perbaikan ini signifikan secara statistik sehingga tuning terbukti berguna.",
             "This improvement is statistically significant, so tuning demonstrably helped.")
           if cmp["significant"] else
           L("Perbaikan ini belum signifikan; parameter bawaan sudah cukup baik untuk data ini "
             "dan tuning tidak wajib dipertahankan.",
             "This improvement is not significant; the defaults are already good enough for this "
             "dataset and tuning is not essential.")),
        f"Compared with the default parameters the CV score moves {arah} from {cmp['base_mean']:.4f} "
        f"to {cmp['tuned_mean']:.4f} (difference {cmp['delta']:+.4f}; paired t = {cmp['t']:.3f}, "
        f"p = {cmp['p']:.4g}; Cohen's d = {cmp['cohen_d']:.3f}). "
        + (L("Perbaikan ini signifikan secara statistik sehingga tuning terbukti berguna.",
             "This improvement is statistically significant, so tuning demonstrably helped.")
           if cmp["significant"] else
           L("Perbaikan ini belum signifikan; parameter bawaan sudah cukup baik untuk data ini.",
             "This improvement is not significant; the defaults are already good enough here."))))

    return lines
