"""Utilitas pemodelan bersama (pipeline preprocessing & nama fitur)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, scaling: str = "standard"):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scaling == "standard":
        num_steps.append(("sc", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("sc", MinMaxScaler()))

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(
            ("cat",
             Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]),
             cat_cols)
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    return pre, num_cols, cat_cols


def feature_names_from(pre, num_cols, cat_cols) -> list[str]:
    names = list(num_cols)
    if cat_cols:
        try:
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            names += list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            names += list(cat_cols)
    return names


def importance_from(model, feature_names) -> tuple[np.ndarray | None, str]:
    """Ambil kepentingan fitur dari model apa pun yang mendukungnya."""
    if hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_, dtype=float)
        if len(v) == len(feature_names):
            return v, "impurity"
    if hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=float)
        v = np.abs(c).mean(axis=0) if c.ndim > 1 else np.asarray(c, dtype=float)
        if len(v) == len(feature_names):
            return v, "coefficient"
    return None, ""
