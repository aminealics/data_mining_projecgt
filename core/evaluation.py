"""Métriques, courbes (Plotly), choix du seuil et analyse d'équité."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

CLASS_NAMES = ["Pas de réadmission < 30 j", "Réadmission < 30 j"]


def metrics_at(y, proba, thr: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(threshold=float(thr), precision=prec, recall=rec, f1=f1,
                specificity=tn / (tn + fp) if tn + fp else 0.0,
                flagged_rate=float(pred.mean()), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def ranking_metrics(y, proba) -> dict:
    auc = roc_auc_score(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)
    return dict(auc=float(auc), pr_auc=float(average_precision_score(y, proba)),
                gini=float(2 * auc - 1), ks=float(np.max(tpr - fpr)))


def best_threshold(y, proba, metric="f1", min_recall: float | None = None) -> float:
    """Seuil maximisant F1 ; ou, si min_recall, le seuil le plus élevé qui atteint ce rappel."""
    p, r, t = precision_recall_curve(y, proba)
    p, r = p[:-1], r[:-1]
    if min_recall is not None:
        ok = np.where(r >= min_recall)[0]
        return float(t[ok[-1]]) if len(ok) else float(t[0])
    f1 = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0)
    return float(t[int(np.argmax(f1))])


def comparison_table(results: dict, yte) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rk = ranking_metrics(yte, r["proba"])
        thr = best_threshold(yte, r["proba"])
        m = metrics_at(yte, r["proba"], thr)
        rows.append({"Modèle": name, "AUC-ROC": rk["auc"], "PR-AUC": rk["pr_auc"], "Gini": rk["gini"],
                     "KS": rk["ks"], "Seuil F1-opt.": thr, "Précision": m["precision"],
                     "Rappel": m["recall"], "F1": m["f1"],
                     "AUC (CV)": r.get("cv_score"), "Temps (s)": r["fit_seconds"],
                     "Hyperparamètres": ", ".join(f"{k}={v}" for k, v in r["params"].items())})
    return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- #
# Figures
# ----------------------------------------------------------------------------- #
def roc_fig(results: dict, yte) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(yte, r["proba"])
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc_score(yte, r['proba']):.3f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Hasard")
    fig.update_layout(title="Courbes ROC superposées", xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs (rappel)", height=450, legend=dict(y=0.05, x=0.45))
    return fig


def pr_fig(results: dict, yte) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        p, rc, _ = precision_recall_curve(yte, r["proba"])
        fig.add_scatter(x=rc, y=p, mode="lines",
                        name=f"{name} (PR-AUC={average_precision_score(yte, r['proba']):.3f})")
    fig.add_hline(y=float(np.mean(yte)), line_dash="dash", line_color="grey",
                  annotation_text="Prévalence (modèle aléatoire)")
    fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel", yaxis_title="Précision", height=450)
    return fig


def confusion_fig(m: dict) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Prédit : non", "Prédit : oui"], y=["Réel : non", "Réel : oui"],
                               text=[[f"{v:,}".replace(",", " ") for v in row] for row in z],
                               texttemplate="%{text}", colorscale="Teal", showscale=False))
    fig.update_layout(title="Matrice de confusion", height=360, yaxis=dict(autorange="reversed"))
    return fig


def threshold_fig(y, proba, thr: float) -> go.Figure:
    ts = np.linspace(0.01, min(0.99, float(np.max(proba))), 80)
    ms = [metrics_at(y, proba, t) for t in ts]
    fig = go.Figure()
    for key, nm in (("precision", "Précision"), ("recall", "Rappel"), ("f1", "F1")):
        fig.add_scatter(x=ts, y=[m[key] for m in ms], mode="lines", name=nm)
    fig.add_vline(x=thr, line_dash="dot", line_color="black")
    fig.update_layout(title="Effet du seuil de décision", xaxis_title="Seuil", yaxis_title="Valeur", height=360)
    return fig


# ----------------------------------------------------------------------------- #
# Équité
# ----------------------------------------------------------------------------- #
def group_metrics(y, proba, thr: float, groups: pd.Series, min_n: int = 100) -> pd.DataFrame:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= thr).astype(int)
    g = pd.Series(np.asarray(groups)).astype(str)
    rows = []
    for k in sorted(g.unique()):
        m = (g == k).to_numpy()
        if m.sum() < min_n:
            continue
        pos = y[m] == 1
        rows.append({"Groupe": k, "n": int(m.sum()), "Taux réel de réadmission": float(y[m].mean()),
                     "Taux signalé par le modèle": float(pred[m].mean()),
                     "Rappel": float(pred[m][pos].mean()) if pos.any() else np.nan,
                     "Précision": float(y[m][pred[m] == 1].mean()) if (pred[m] == 1).any() else np.nan,
                     "Faux positifs (taux)": float(pred[m][~pos].mean()) if (~pos).any() else np.nan})
    return pd.DataFrame(rows)


def fairness_gaps(gm: pd.DataFrame) -> dict:
    """Écarts max-min entre groupes : parité démographique et égalité des chances."""
    if len(gm) < 2:
        return {}
    return dict(demographic_parity_gap=float(gm["Taux signalé par le modèle"].max() - gm["Taux signalé par le modèle"].min()),
                equal_opportunity_gap=float(gm["Rappel"].max() - gm["Rappel"].min()),
                fpr_gap=float(gm["Faux positifs (taux)"].max() - gm["Faux positifs (taux)"].min()))


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 40, 60, 70, 80, 120], labels=["< 40", "40-59", "60-69", "70-79", "≥ 80"],
                  right=False).astype(str)
