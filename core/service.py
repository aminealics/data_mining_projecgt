"""Couche « service » partagée par l'application Streamlit et l'API FastAPI.

Un *bundle* (joblib) contient tout ce qu'il faut pour prédire et expliquer un individu.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core import explain as ex
from core.data import label

ART = Path(__file__).resolve().parent.parent / "artifacts"
BUNDLE_PATH = ART / "bundle.joblib"


def make_bundle(space, model, model_name: str, threshold: float, X_train_raw: pd.DataFrame,
                Xt_train: np.ndarray, Xt_test: np.ndarray, metrics: dict | None = None,
                seed: int = 0, n_global: int | None = None, with_global: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    defaults, schema = {}, []
    for c in space.raw_columns:
        col = X_train_raw[c]
        if c in space.num_cols:
            defaults[c] = float(col.median())
            schema.append(dict(name=c, label=label(c), type="number", default=defaults[c],
                               min=float(col.min()), max=float(col.max())))
        else:
            vc = col.value_counts()
            defaults[c] = str(vc.index[0])
            schema.append(dict(name=c, label=label(c), type="category", default=defaults[c],
                               options=[str(v) for v in vc.index[:30]]))
    bg = Xt_train[rng.choice(len(Xt_train), min(100, len(Xt_train)), replace=False)]
    lime_train = space.to_display(Xt_train[rng.choice(len(Xt_train), min(1500, len(Xt_train)), replace=False)])
    gi = []
    if with_global:  # importance globale SHAP (par variable d'origine) sur un échantillon du jeu de test
        n_g = n_global or (60 if model_name == ex.KNN else 400)
        te = Xt_test[rng.choice(len(Xt_test), min(n_g, len(Xt_test)), replace=False)]
        e = ex.make_shap_explainer(model_name, model, bg)
        exp = ex.shap_explain(model_name, e, te, space.to_display(te), space.names)
        gi = ex.global_by_variable(exp.values, space.meta).to_dict("records")
    return dict(space=space, model=model, model_name=model_name, threshold=float(threshold),
                defaults=defaults, schema=schema, shap_background=bg, lime_train=lime_train,
                global_importance=gi, metrics=metrics or {},
                created=time.strftime("%Y-%m-%d %H:%M:%S"))


def save_bundle(bundle: dict, path: Path = BUNDLE_PATH) -> Path:
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    return joblib.load(path)


class Explainer:
    """Prédit et explique un individu à partir d'un bundle."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.space = bundle["space"]
        self.model = bundle["model"]
        self.name = bundle["model_name"]
        self.shap = ex.make_shap_explainer(self.name, self.model, bundle["shap_background"])
        self.lime = ex.make_lime(self.space, bundle["lime_train"])

    # -- entrée ---------------------------------------------------------------- #
    def frame(self, values: dict) -> pd.DataFrame:
        unknown = set(values) - set(self.space.raw_columns)
        if unknown:
            raise ValueError(f"Variables inconnues : {sorted(unknown)}")
        row = {**self.b["defaults"], **values}
        df = pd.DataFrame([row])[self.space.raw_columns]
        for c in self.space.num_cols:
            df[c] = pd.to_numeric(df[c], errors="raise")
        for c in self.space.cat_cols:
            df[c] = df[c].astype(str)
        return df

    # -- sorties --------------------------------------------------------------- #
    def predict(self, values: dict, threshold: float | None = None) -> dict:
        thr = self.b["threshold"] if threshold is None else float(threshold)
        p = float(self.model.predict_proba(self.space.transform(self.frame(values)))[0, 1])
        return dict(probability=p, threshold=thr, decision=bool(p >= thr),
                    risk_level="élevé" if p >= thr else ("modéré" if p >= 0.6 * thr else "faible"))

    def explain(self, values: dict, threshold: float | None = None, top: int = 12,
                lime_samples: int = 2000, with_lime: bool = True) -> dict:
        df = self.frame(values)
        Xt = self.space.transform(df)
        D = self.space.to_display(Xt)
        out = self.predict(values, threshold)
        exp = ex.shap_explain(self.name, self.shap, Xt, D, self.space.names)
        sv = exp.values[0]
        order = np.argsort(-np.abs(sv))[:top]
        out["shap"] = dict(
            space=ex.shap_space(self.name), base_value=float(exp.base_values[0]),
            contributions=[dict(feature=self.space.names[i], variable=self.space.meta[i]["source"],
                                value=float(D[0, i]), shap=float(sv[i])) for i in order],
            by_variable=[dict(variable=k, label=label(k), shap=float(v)) for k, v in
                         ex.aggregate_by_source(sv, self.space.meta).iloc[0]
                         .sort_values(key=np.abs, ascending=False).head(top).items()])
        out["sentence"] = ex.summary_sentence(self.space, df.iloc[0], sv, out["probability"], out["threshold"])
        if with_lime:
            w, rules, _ = ex.lime_explain(self.lime, self.space, self.model, D[0], lime_samples)
            cmp = ex.compare_shap_lime(sv, w, self.space.meta)
            out["lime"] = dict(
                rules=[dict(rule=r, weight=float(v)) for r, v in rules[:top]],
                by_variable=[dict(variable=k, label=label(k), weight=float(v)) for k, v in
                             cmp["lime_by_var"].sort_values(key=np.abs, ascending=False).head(top).items()])
            out["comparison"] = dict(spearman=cmp["spearman"], top_k_overlap=cmp["top_k_overlap"],
                                     same_sign_share=cmp["same_sign_share"], k=cmp["k"])
        # objets bruts pour l'application (non sérialisés par l'API)
        out["_raw"] = dict(exp=exp, lime_w=w if with_lime else None, df=df)
        return out
