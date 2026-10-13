"""Les cinq modèles de classification, leurs hyperparamètres (UI) et l'optimisation."""
from __future__ import annotations

import time

from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_NAMES = ["Régression logistique", "Arbre de décision", "k plus proches voisins",
               "Random Forest", "XGBoost"]
TREE_BASED = {"Arbre de décision", "Random Forest", "XGBoost"}

# Paramètres exposés dans l'interface : (nom, type, min, max, défaut) ou (nom, "choice", options, défaut)
PARAM_UI = {
    "Régression logistique": [dict(name="C", kind="log", lo=1e-3, hi=1e2, default=1.0)],
    "Arbre de décision": [
        dict(name="max_depth", kind="int", lo=2, hi=20, default=6),
        dict(name="min_samples_leaf", kind="int", lo=1, hi=200, default=50),
        dict(name="criterion", kind="choice", choices=["gini", "entropy"], default="gini")],
    "k plus proches voisins": [
        dict(name="n_neighbors", kind="int", lo=3, hi=201, default=51),
        dict(name="weights", kind="choice", choices=["uniform", "distance"], default="uniform")],
    "Random Forest": [
        dict(name="n_estimators", kind="int", lo=50, hi=600, default=200),
        dict(name="max_depth", kind="int", lo=3, hi=30, default=14),
        dict(name="min_samples_leaf", kind="int", lo=1, hi=100, default=10)],
    "XGBoost": [
        dict(name="n_estimators", kind="int", lo=50, hi=800, default=300),
        dict(name="max_depth", kind="int", lo=2, hi=10, default=4),
        dict(name="learning_rate", kind="log", lo=0.01, hi=0.5, default=0.05),
        dict(name="subsample", kind="float", lo=0.5, hi=1.0, default=0.8)],
}

SEARCH_SPACE = {
    "Régression logistique": {"C": loguniform(1e-3, 1e2)},
    "Arbre de décision": {"max_depth": randint(3, 12), "min_samples_leaf": randint(5, 120),
                          "criterion": ["gini", "entropy"]},
    "k plus proches voisins": {"n_neighbors": randint(15, 121), "weights": ["uniform", "distance"]},
    "Random Forest": {"n_estimators": randint(100, 400), "max_depth": randint(6, 25),
                      "min_samples_leaf": randint(2, 40)},
    "XGBoost": {"n_estimators": randint(100, 500), "max_depth": randint(2, 8),
                "learning_rate": loguniform(0.02, 0.3), "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 20)},
}


def defaults(name: str) -> dict:
    return {p["name"]: p["default"] for p in PARAM_UI[name]}


def build_model(name: str, params: dict | None = None, balanced: bool = True,
                pos_weight: float = 1.0, seed: int = 42):
    p = {**defaults(name), **(params or {})}
    cw = "balanced" if balanced else None
    if name == "Régression logistique":
        return LogisticRegression(max_iter=2000, class_weight=cw, **p)
    if name == "Arbre de décision":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed, **p)
    if name == "k plus proches voisins":
        return KNeighborsClassifier(n_jobs=-1, **p)
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced_subsample" if balanced else None,
                                      n_jobs=-1, random_state=seed, **p)
    if name == "XGBoost":
        return XGBClassifier(scale_pos_weight=pos_weight if balanced else 1.0, eval_metric="logloss",
                             tree_method="hist", n_jobs=-1, random_state=seed, **p)
    raise ValueError(name)


def train(name: str, Xt, y, params: dict | None = None, balanced: bool = True, tune: bool = False,
          n_iter: int = 8, cv: int = 3, scoring: str = "roc_auc", seed: int = 42) -> dict:
    """Entraîne un modèle (avec RandomizedSearchCV si `tune`). Retourne modèle + infos."""
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    base = build_model(name, params, balanced, pos_weight, seed)
    t0 = time.time()
    if tune:
        space = SEARCH_SPACE[name]
        if name == "Régression logistique":
            n_iter = min(n_iter, 6)
        search = RandomizedSearchCV(base, space, n_iter=n_iter, scoring=scoring,
                                    cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                                    random_state=seed, n_jobs=1, refit=True)
        search.fit(Xt, y)
        model, best, cv_score = search.best_estimator_, search.best_params_, float(search.best_score_)
    else:
        model = base.fit(Xt, y)
        best = {**defaults(name), **(params or {})}
        cv_score = None
    return dict(model=model, params={k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
                cv_score=cv_score, fit_seconds=time.time() - t0, tuned=tune)
