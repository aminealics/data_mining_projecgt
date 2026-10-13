"""Explicabilité : importances, SHAP (global/local), LIME, phrase de synthèse."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from core.data import SENSITIVE, label
from core.models import TREE_BASED

LINEAR = "Régression logistique"
KNN = "k plus proches voisins"
CLASS_NAMES = ["Pas de réadmission", "Réadmission < 30 j"]


def shap_space(name: str) -> str:
    """Unité des valeurs SHAP : log-odds (LR, XGBoost) ou probabilité (autres)."""
    return "log-odds" if name in (LINEAR, "XGBoost") else "probabilité"


# ----------------------------------------------------------------------------- #
# Importances globales propres au modèle / permutation
# ----------------------------------------------------------------------------- #
def native_importance(name: str, model, names: list[str]) -> pd.DataFrame | None:
    if name == LINEAR:
        v = model.coef_.ravel()
        return pd.DataFrame({"feature": names, "importance": v, "abs": np.abs(v)}).sort_values("abs", ascending=False)
    if hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_)
        return pd.DataFrame({"feature": names, "importance": v, "abs": np.abs(v)}).sort_values("abs", ascending=False)
    return None  # kNN : pas d'importance native


def permutation_by_variable(model, space, X_raw: pd.DataFrame, y, n: int = 1500, repeats: int = 3,
                            seed: int = 0) -> pd.DataFrame:
    """Importance par permutation sur les variables d'ORIGINE (comparable entre modèles) — baisse d'AUC."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_raw), size=min(n, len(X_raw)), replace=False)
    pipe = Pipeline([("prep", space.preprocessor), ("clf", model)])
    r = permutation_importance(pipe, X_raw.iloc[idx], np.asarray(y)[idx], scoring="roc_auc",
                               n_repeats=repeats, random_state=seed, n_jobs=1)
    return pd.DataFrame({"variable": X_raw.columns, "importance": r.importances_mean,
                         "std": r.importances_std}).sort_values("importance", ascending=False)


# ----------------------------------------------------------------------------- #
# SHAP
# ----------------------------------------------------------------------------- #
def _pos(a):
    if isinstance(a, list):
        return np.asarray(a[1])
    a = np.asarray(a)
    return a[..., 1] if a.ndim == 3 else a


def make_shap_explainer(name: str, model, background: np.ndarray):
    if name == LINEAR:
        return shap.LinearExplainer(model, background)
    if name in TREE_BASED:
        return shap.TreeExplainer(model)
    return shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], shap.sample(background, 40, random_state=0))


def shap_explain(name: str, explainer, Xt: np.ndarray, display: np.ndarray, names: list[str]) -> shap.Explanation:
    """Explication SHAP (classe positive). `display` : valeurs en unités d'origine pour les graphiques."""
    if name == KNN:
        sv = explainer.shap_values(Xt, nsamples=150, silent=True)
    elif name in ("Random Forest", "Arbre de décision"):
        sv = explainer.shap_values(Xt, check_additivity=False)
    else:
        sv = explainer.shap_values(Xt)
    sv = _pos(sv)
    ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    base = float(ev[1] if len(ev) > 1 else ev[0])
    return shap.Explanation(values=sv, base_values=np.full(len(sv), base), data=display, feature_names=names)


def aggregate_by_source(values: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    """Somme les contributions des colonnes one-hot d'une même variable d'origine."""
    src = pd.Series([m["source"] for m in meta])
    vals = pd.DataFrame(np.atleast_2d(values))
    return vals.T.groupby(src.values).sum().T


def global_by_variable(values: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    agg = aggregate_by_source(values, meta)
    out = agg.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    out.columns = ["variable", "mean_abs_shap"]
    out["sensible"] = out["variable"].isin(SENSITIVE)
    out["part"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out


# ----------------------------------------------------------------------------- #
# LIME (espace « affichage » : unités d'origine, règles lisibles)
# ----------------------------------------------------------------------------- #
def make_lime(space, display_train: np.ndarray) -> LimeTabularExplainer:
    return LimeTabularExplainer(display_train, feature_names=space.names, class_names=CLASS_NAMES,
                                categorical_features=space.cat_idx, discretize_continuous=True,
                                mode="classification", random_state=0)


def lime_explain(lime_exp, space, model, display_row: np.ndarray, num_samples: int = 2000):
    """Retourne (vecteur complet de poids, liste (règle, poids) triée, proba locale)."""
    fn = lambda z: model.predict_proba(space.from_display(z))
    e = lime_exp.explain_instance(display_row, fn, num_features=len(space.names),
                                  num_samples=num_samples, labels=(1,))
    w = np.zeros(len(space.names))
    for i, v in e.as_map()[1]:
        w[i] = v
    return w, e.as_list(label=1), float(e.local_pred[0]) if hasattr(e, "local_pred") else float("nan")


def compare_shap_lime(shap_vals: np.ndarray, lime_w: np.ndarray, meta: list[dict], k: int = 5) -> dict:
    """Cohérence SHAP/LIME au niveau des VARIABLES d'origine (les colonnes one-hot sont sommées)."""
    s = aggregate_by_source(shap_vals, meta).iloc[0]
    l = aggregate_by_source(lime_w, meta).iloc[0].reindex(s.index)
    rho = spearmanr(s, l).statistic
    ts = set(s.abs().nlargest(k).index); tl = set(l.abs().nlargest(k).index)
    both = list(ts | tl)
    same = float(np.mean(np.sign(s[both]) == np.sign(l[both])))
    return dict(spearman=float(rho) if rho == rho else 0.0, top_k_overlap=len(ts & tl) / k,
                same_sign_share=same, k=k, shap_by_var=s, lime_by_var=l)


# ----------------------------------------------------------------------------- #
# Phrase de synthèse en langage décideur
# ----------------------------------------------------------------------------- #
def summary_sentence(space, raw_row: pd.Series, shap_vals: np.ndarray, proba: float, thr: float,
                     k: int = 3) -> str:
    agg = aggregate_by_source(shap_vals, space.meta).iloc[0].sort_values(ascending=False)
    ups = [s for s, v in agg.items() if v > 0][:k]
    downs = [s for s, v in agg.sort_values().items() if v < 0][:2]
    level = "élevé" if proba >= thr else ("modéré" if proba >= 0.6 * thr else "faible")
    txt = f"Le risque de réadmission sous 30 jours est {level} (probabilité {proba:.1%}, seuil {thr:.0%})"
    if ups:
        txt += ", principalement à cause de : " + " ; ".join(space.describe_var(s, raw_row) for s in ups)
    txt += "."
    if downs:
        txt += " À l'inverse, " + " et ".join(space.describe_var(s, raw_row) for s in downs) + \
               (" tendent" if len(downs) > 1 else " tend") + " à réduire ce risque."
    sens = [s for s in ups if s in SENSITIVE]
    if sens:
        txt += " ⚠ Une variable sensible (" + ", ".join(label(s) for s in sens) + ") pèse dans cette décision."
    return txt
