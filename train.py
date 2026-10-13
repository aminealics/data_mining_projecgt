"""Pipeline complet hors interface : données -> modèles -> évaluation -> explications -> équité.

    python train.py                       # jeu UCI réel, optimisation des hyperparamètres
    python train.py --n-iter 4 --max-train 20000      # version rapide
    USE_SYNTHETIC_DEMO=1 python train.py --out /tmp/demo   # test hors-ligne (données SYNTHÉTIQUES)

Produit dans --out (défaut : artifacts/) : results.json, bundle.joblib (modèle pour l'API), figs/*.png.
Ces fichiers alimentent generate_onepager.py (one-pager PDF).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

from core import data as D
from core import evaluation as EV
from core import explain as EX
from core import models as M
from core import service as SV
from core.prep import FeatureSpace, split_data


def direction(src, meta, values, display):
    idx = [i for i, m in enumerate(meta) if m["source"] == src]
    if meta[idx[0]]["kind"] == "num":
        r = np.corrcoef(display[:, idx[0]], values[:, idx[0]])[0, 1]
        return "valeur élevée → risque ↑" if r > 0.05 else ("valeur élevée → risque ↓" if r < -0.05 else "effet non monotone")
    best, bm = None, -9
    for i in idx:
        on = display[:, i] > 0.5
        if on.sum() >= 30 and values[on, i].mean() > bm:
            best, bm = meta[i]["category"], values[on, i].mean()
    return f"catégorie la plus à risque : {best}" if best else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=M.MODEL_NAMES)
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    out, figs = Path(a.out), Path(a.out) / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    synthetic = os.getenv("USE_SYNTHETIC_DEMO") == "1"

    # 1-2. données et préparation ------------------------------------------------
    print("→ données…")
    raw = D.fetch_diabetes()
    X, y, groups, info = D.prepare_diabetes(raw)
    Xtr, Xte, ytr, yte = split_data(X, y, a.test_size, a.seed, groups)
    if len(Xtr) > a.max_train:
        Xtr = Xtr.sample(a.max_train, random_state=a.seed)
        ytr = ytr.loc[Xtr.index]
    fs = FeatureSpace().fit(Xtr)
    A, B = fs.transform(Xtr), fs.transform(Xte)
    print(f"  train {A.shape} · test {B.shape} · prévalence {y.mean():.3f}")

    # 3. modèles -------------------------------------------------------------------
    res = {}
    for n in a.models:
        print(f"→ {n}…")
        r = M.train(n, A, ytr, balanced=True, tune=not a.no_tune, n_iter=a.n_iter, cv=a.cv, seed=a.seed)
        r["proba"] = r["model"].predict_proba(B)[:, 1]
        res[n] = r
    table = EV.comparison_table(res, yte)
    print(table.iloc[:, :9].round(3).to_string())
    key = "AUC (CV)" if not a.no_tune else "AUC-ROC"
    best = table.sort_values(key, ascending=False).iloc[0]["Modèle"]
    bm, proba = res[best]["model"], res[best]["proba"]

    # 4. seuil choisi sur le jeu d'ENTRAÎNEMENT (prédictions hors-pli), évalué sur le test ------
    oof = cross_val_predict(clone(bm), A, ytr, cv=3, method="predict_proba")[:, 1]
    thr = EV.best_threshold(ytr, oof)
    thr_recall = EV.best_threshold(ytr, oof, min_recall=0.6)
    at_thr, at_recall = EV.metrics_at(yte, proba, thr), EV.metrics_at(yte, proba, thr_recall)
    rk = EV.ranking_metrics(yte, proba)

    # 5. explications globales -----------------------------------------------------
    print("→ SHAP…")
    rng = np.random.default_rng(0)
    n_shap = 100 if best == EX.KNN else 1500
    idx = rng.choice(len(B), min(n_shap, len(B)), replace=False)
    ex_ = EX.make_shap_explainer(best, bm, A[rng.choice(len(A), 100, replace=False)])
    disp = fs.to_display(B[idx])
    exp = EX.shap_explain(best, ex_, B[idx], disp, fs.names)
    gv = EX.global_by_variable(exp.values, fs.meta)
    top_vars = [dict(variable=r.variable, label=D.label(r.variable), part=float(r.part), sensible=bool(r.sensible),
                     direction=direction(r.variable, fs.meta, exp.values, disp)) for r in gv.head(8).itertuples()]
    perm = EX.permutation_by_variable(bm, fs, Xte, yte, 2000 if best != EX.KNN else 500)
    lr = res.get(EX.LINEAR)
    lr_coef = None
    if lr is not None:
        nat = EX.native_importance(EX.LINEAR, lr["model"], fs.names).head(10)
        lr_coef = [dict(feature=r.feature, coef=float(r.importance)) for r in nat.itertuples()]

    # figures
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for n, r in res.items():
        f, t, _ = roc_curve(yte, r["proba"])
        ax.plot(f, t, lw=2.2 if n == best else 1.1, label=f"{n.split(' ')[0]} ({EV.ranking_metrics(yte, r['proba'])['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=.8)
    ax.set(xlabel="Faux positifs", ylabel="Rappel", title="Courbes ROC (jeu de test)")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(figs / "roc.png", dpi=200); plt.close(fig)
    shap.plots.beeswarm(exp, max_display=10, show=False)
    plt.gcf().set_size_inches(5.2, 3.6); plt.tight_layout(); plt.savefig(figs / "beeswarm.png", dpi=200, bbox_inches="tight"); plt.close()
    g8 = gv.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.barh([D.label(v) for v in g8["variable"]], g8["mean_abs_shap"], color=["#DC2626" if s else "#0F766E" for s in g8["sensible"]])
    ax.set(title="Poids moyen |SHAP|", xlabel=f"unité : {EX.shap_space(best)}"); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(figs / "shap_bar.png", dpi=200); plt.close(fig)

    # 6. cohérence SHAP / LIME sur 20 individus ---------------------------------------
    print("→ LIME (20 individus)…")
    E = SV.Explainer(SV.make_bundle(fs, bm, best, thr, Xtr, A, B, with_global=False))
    sel = rng.choice(len(Xte), 20, replace=False)
    sp, ov, sg = [], [], []
    example = None
    for j in sel:
        o = E.explain({k: v for k, v in Xte.iloc[j].items() if not pd.isna(v)}, thr, lime_samples=1500)
        sp.append(o["comparison"]["spearman"]); ov.append(o["comparison"]["top_k_overlap"]); sg.append(o["comparison"]["same_sign_share"])
        if o["decision"] and (example is None or o["probability"] > example["probability"]):
            example = dict(probability=o["probability"], sentence=o["sentence"], truth=int(yte.iloc[j]),
                           lime_top=o["lime"]["by_variable"][:3], shap_top=o["shap"]["by_variable"][:3])
    agreement = dict(n=len(sel), spearman_mean=float(np.mean(sp)), spearman_min=float(np.min(sp)),
                     top5_overlap_mean=float(np.mean(ov)), same_sign_mean=float(np.mean(sg)))

    # 7. équité --------------------------------------------------------------------
    audit = X.loc[Xte.index, [c for c in D.SENSITIVE if c in X]].copy()
    audit["age"] = EV.age_band(audit["age"])
    fair = {}
    for c in audit:
        gm = EV.group_metrics(yte, proba, thr, audit[c].to_numpy())
        fair[c] = dict(groups=gm.round(4).to_dict("records"), gaps=EV.fairness_gaps(gm))
    # ablation : même modèle sans variables sensibles
    keep = [c for c in Xtr.columns if c not in D.SENSITIVE]
    fs2 = FeatureSpace().fit(Xtr[keep])
    ab = clone(bm).fit(fs2.transform(Xtr[keep]), ytr)
    auc_wo = EV.ranking_metrics(yte, ab.predict_proba(fs2.transform(Xte[keep]))[:, 1])["auc"]
    sens_part = float(gv.loc[gv["sensible"], "part"].sum())

    # 8. sauvegarde -----------------------------------------------------------------
    SV.save_bundle(SV.make_bundle(fs, bm, best, thr, Xtr, A, B, metrics=rk), out / "bundle.joblib")
    results = dict(
        data_source="synthetic" if synthetic else "uci-296", created=time.strftime("%Y-%m-%d %H:%M"),
        config=dict(vars(a), tuned=not a.no_tune), versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, shap=shap.__version__),
        dataset=dict(n_raw=info["n_raw"], n_after_expired=info["n_after_expired"], n_final=info["n_final"],
                     positive_rate=info["positive_rate"], n_train=len(Xtr), n_test=len(Xte), n_features=Xtr.shape[1],
                     n_columns_model=A.shape[1], test_positive_rate=float(yte.mean())),
        models=table.round(4).to_dict("records"), best=best, ranking=rk,
        threshold=dict(value=float(thr), at_test=at_thr, recall60_value=float(thr_recall), recall60_at_test=at_recall),
        top_variables=top_vars, permutation_top=perm.head(8).round(4).to_dict("records"), lr_coefficients=lr_coef,
        agreement=agreement, example=example, fairness=fair,
        sensitive=dict(shap_share=sens_part, auc_with=rk["auc"], auc_without=float(auc_wo)),
        runtime_min=round((time.time() - t_start) / 60, 1))
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    print(f"✔ terminé en {results['runtime_min']} min → {out}/results.json, bundle.joblib, figs/")


if __name__ == "__main__":
    main()
