"""Préparation : espace de variables (imputation, encodage, normalisation) + découpage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.data import label


def split_columns(X: pd.DataFrame):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return num, [c for c in X.columns if c not in num]


def split_data(X, y, test_size=0.25, seed=42, groups=None):
    """Découpage stratifié ; si `groups` (patients) est donné, aucun patient n'est dans les deux jeux."""
    if groups is not None and groups.nunique() < len(groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


class FeatureSpace:
    """Transforme les lignes brutes en matrice numérique ; garde la correspondance des colonnes.

    - `transform` : brut -> matrice du modèle (imputée, encodée, normalisée)
    - `to_display` / `from_display` : repasse les variables numériques en unités d'origine
      (pour des graphiques SHAP / LIME lisibles) et inversement.
    """

    def __init__(self, num_impute="median", cat_impute="constant", scale=True, min_freq=0.01):
        self.num_impute, self.cat_impute, self.scale, self.min_freq = num_impute, cat_impute, scale, min_freq

    def fit(self, X: pd.DataFrame):
        self.num_cols, self.cat_cols = split_columns(X)
        self.raw_columns = list(X.columns)
        num_steps = [("imp", SimpleImputer(strategy=self.num_impute))]
        if self.scale:
            num_steps.append(("sc", StandardScaler()))
        cat_imp = (SimpleImputer(strategy="most_frequent") if self.cat_impute == "most_frequent"
                   else SimpleImputer(strategy="constant", fill_value="Manquant"))
        cat_steps = [("imp", cat_imp),
                     ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=self.min_freq, sparse_output=False))]
        tf = []
        if self.num_cols:
            tf.append(("num", Pipeline(num_steps), self.num_cols))
        if self.cat_cols:
            tf.append(("cat", Pipeline(cat_steps), self.cat_cols))
        self.preprocessor = ColumnTransformer(tf).fit(X)
        self.n_num = len(self.num_cols)
        self.scaler = self.preprocessor.named_transformers_["num"].named_steps.get("sc") if self.num_cols else None
        self._build_meta()
        return self

    def _build_meta(self):
        by_len = sorted(self.cat_cols, key=len, reverse=True)
        self.meta = []
        for n in self.preprocessor.get_feature_names_out():
            body = n.split("__", 1)[1]
            if n.startswith("num__"):
                self.meta.append(dict(name=body, source=body, kind="num", category=None))
            else:
                src = next(c for c in by_len if body.startswith(c + "_"))
                cat = body[len(src) + 1:].replace("infrequent_sklearn", "autres")
                self.meta.append(dict(name=f"{src}={cat}", source=src, kind="cat", category=cat))
        self.names = [m["name"] for m in self.meta]
        self.cat_idx = [i for i, m in enumerate(self.meta) if m["kind"] == "cat"]

    # -- transformations ----------------------------------------------------- #
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(X[self.raw_columns]), dtype=float)

    def to_display(self, Xt: np.ndarray) -> np.ndarray:
        D = np.array(Xt, dtype=float, copy=True)
        if self.scaler is not None and self.n_num:
            D[:, :self.n_num] = self.scaler.inverse_transform(D[:, :self.n_num])
        return D

    def from_display(self, D: np.ndarray) -> np.ndarray:
        Xt = np.array(D, dtype=float, copy=True)
        if self.scaler is not None and self.n_num:
            Xt[:, :self.n_num] = self.scaler.transform(Xt[:, :self.n_num])
        return Xt

    # -- lisibilité ----------------------------------------------------------- #
    def describe_var(self, src: str, raw_row: pd.Series) -> str:
        v = raw_row[src]
        if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
            v = f"{v:g}"
        return f"{label(src)} ({v})"
