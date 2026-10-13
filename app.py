"""Application Streamlit — classification explicable de la réadmission hospitalière à 30 jours
(cas n°12 : Diabetes 130-US hospitals, UCI). Six modules : données, préparation, modélisation,
évaluation, IA explicable, API d'interprétation."""
from __future__ import annotations

import os
import threading
import time

import httpx
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from core import data as D  # noqa: E402
from core import evaluation as EV  # noqa: E402
from core import explain as EX  # noqa: E402
from core import models as M  # noqa: E402
from core import service as SV  # noqa: E402
from core.prep import FeatureSpace, split_data  # noqa: E402

st.set_page_config(page_title="Réadmission & IA explicable", page_icon="🏥", layout="wide")
SS = st.session_state
LOCAL_API_PORT = int(os.getenv("LOCAL_API_PORT", "8008"))

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 25%, #1e3a8a 100%);
        color: #e2e8f0;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 0.9rem;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.20);
    }
    .modern-card {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.18);
        color: #bfdbfe;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================== #
# Utilitaires
# =============================================================================== #

def find_available_port(start_port: int, max_attempts: int = 12) -> int:
    import socket

    for offset in range(max_attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


@st.cache_data(show_spinner="Téléchargement du jeu UCI #296 (Diabetes 130-US hospitals)…")
def load_uci():
    return D.fetch_diabetes()


@st.cache_data(show_spinner=False)
def cached_clean(key, mode, drop_expired, first_enc, target, positive, _df):
    if mode == "diabetes":
        return D.prepare_diabetes(_df, drop_expired, first_enc)
    return D.prepare_generic(_df, target, positive)


def reset_downstream(keep_data=True):
    for k in ("prep", "results", "shap_cache", "perm_cache", "explainers"):
        SS.pop(k, None)
    if not keep_data:
        for k in ("raw", "mode", "generic", "source"):
            SS.pop(k, None)


def get_clean():
    o = SS.get("clean_opts", dict(drop_expired=True, first_encounter=True))
    g = SS.get("generic", (None, None))
    key = f"{SS['source']}-{len(SS['raw'])}"
    return cached_clean(key, SS["mode"], o["drop_expired"], o["first_encounter"], g[0], g[1], SS["raw"])


def need(*keys):
    msgs = {"raw": "Module 1 (charger les données)", "prep": "Module 2 (préparer les données)",
            "results": "Module 3 (entraîner au moins un modèle)"}
    for k in keys:
        if k not in SS:
            st.warning(f"Terminez d'abord le {msgs[k]}.")
            st.stop()


def fmt_pct(x):
    return f"{100 * x:.1f} %"


def lab(c):
    return D.label(c)


def ensure_local_api():
    """Si aucune API distante n'est configurée, lance l'API FastAPI dans un thread (même conteneur)."""
    @st.cache_resource
    def _start(port):
        import uvicorn
        from api.main import app as api_app
        actual_port = find_available_port(port)
        server = uvicorn.Server(uvicorn.Config(api_app, host="127.0.0.1", port=actual_port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(60):
            if server.started:
                break
            time.sleep(0.1)
        return actual_port
    return f"http://127.0.0.1:{_start(LOCAL_API_PORT)}"


def api_url():
    url = os.getenv("API_URL")
    if not url:
        try:
            url = st.secrets.get("API_URL")
        except Exception:
            url = None
    return (url.rstrip("/"), True) if url else (ensure_local_api(), False)


# =============================================================================== #
# Vue d'ensemble
# =============================================================================== #
def page_overview():
    st.header("Vue d'ensemble du projet")
    st.markdown('<div class="pill">Clinical AI dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Objectif** : prioriser les patients à risque de réadmission < 30 jours avec un workflow transparent et auditable.")

    if "prep" in SS and "results" in SS:
        P, R = SS["prep"], SS["results"]
        table = EV.comparison_table(R, P["yte"]).sort_values("AUC-ROC", ascending=False)
        best = table.iloc[0]
        best_name = best["Modèle"]
        scenario = st.selectbox("Seuil d'alerte à tester", [0.10, 0.15, 0.20, 0.30, 0.40], index=1)
        m = EV.metrics_at(P["yte"], R[best_name]["proba"], scenario)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best_name)
        c2.metric("AUC-ROC", f"{best['AUC-ROC']:.3f}")
        c3.metric("PR-AUC", f"{best['PR-AUC']:.3f}")
        c4.metric("Prévalence test", f"{100 * P['yte'].mean():.1f}%")

        st.subheader("Résumé de décision")
        s1, s2, s3 = st.columns(3)
        s1.metric("Rappel à ce seuil", fmt_pct(m["recall"]))
        s2.metric("Précision", fmt_pct(m["precision"]))
        s3.metric("Patients signalés", fmt_pct(m["flagged_rate"]))

        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.caption("Recommandation courte")
            recommendation = (
                f"Le modèle {best_name} reste robuste en AUC-ROC ({best['AUC-ROC']:.3f}) ; "
                f"au seuil de {scenario:.2f}, il détecte {fmt_pct(m['recall'])} des cas positifs "
                f"en signalant {fmt_pct(m['flagged_rate'])} des patients. "
                f"Pour un usage opérationnel, il est conseillé d'examiner le seuil associé au F1-optimal puis de valider "
                "la priorisation clinique sur un jeu de production."
            )
            st.write(recommendation)
            st.markdown('</div>', unsafe_allow_html=True)

        a, b = st.columns([2, 1])
        with a:
            st.plotly_chart(EV.roc_fig(R, P["yte"]), width="stretch")
        with b:
            st.dataframe(table.head(5).rename(columns={"Modèle": "Modèle", "AUC-ROC": "AUC-ROC", "PR-AUC": "PR-AUC"}),
                         width="stretch", hide_index=True)

        st.subheader("Téléchargement du résumé")
        st.download_button("Exporter le tableau comparatif", data=table.to_csv(index=False), file_name="model_summary.csv",
                           mime="text/csv")
    else:
        st.info("Complétez les modules 1 à 3 pour populater ce tableau de bord. Les cartes de suivi et le synthèse de décision s'actualiseront automatiquement.")
        st.markdown(
            """
            <div class="modern-card">
                <h4>Workflow clinique guidé</h4>
                <p>1. Charger les données et valider la cohérence du jeu.<br>
                2. Préparer le cadre de modélisation et définir le seuil d'alerte.<br>
                3. Comparer les modèles, expliquer les prédictions et auditer l'équité.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================== #
# Module 1 — Données
# =============================================================================== #
def page_data():
    st.header("Module 1 — Données")
    src = st.radio("Source des données", ["Cas n°12 — UCI « Diabetes 130-US hospitals »", "Importer mon propre CSV"],
                   horizontal=True)
    if src.startswith("Cas"):
        if SS.get("mode") != "diabetes":
            try:
                SS.update(raw=load_uci(), mode="diabetes", source="uci")
                reset_downstream()
            except Exception as e:  # réseau indisponible, etc.
                st.error(f"Téléchargement impossible : {e}")
                st.info("Vérifiez la connexion, ou importez le CSV dans l'autre mode.")
                st.stop()
    else:
        up = st.file_uploader("Fichier CSV", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            tgt = st.selectbox("Variable cible", df.columns, index=len(df.columns) - 1)
            vals = sorted(df[tgt].dropna().astype(str).unique())[:50]
            pos = st.selectbox("Classe d'intérêt (positive)", vals)
            if st.button("Valider cet import", type="primary"):
                SS.update(raw=df, mode="generic", generic=(tgt, pos), source=f"csv-{up.name}")
                reset_downstream()
        if SS.get("mode") != "generic":
            st.info("Importez un CSV et indiquez la cible pour continuer.")
            st.stop()

    raw = SS["raw"]
    X, y, groups, info = get_clean()
    c = st.columns(4)
    c[0].metric("Lignes brutes", f"{len(raw):,}".replace(",", " "))
    c[1].metric("Lignes après nettoyage", f"{len(X):,}".replace(",", " "))
    c[2].metric("Variables explicatives", X.shape[1])
    c[3].metric("Classe positive (réadmission < 30 j)" if SS["mode"] == "diabetes" else "Classe positive",
                fmt_pct(y.mean()))
    if SS["mode"] == "diabetes":
        st.caption(f"Nettoyage : {info['n_raw'] - info['n_after_expired']} décès/soins palliatifs retirés ; "
                   "une seule rencontre par patient ; poids et code payeur écartés (trop de valeurs manquantes). "
                   "Options modifiables au Module 2.")

    t1, t2, t3, t4, t5 = st.tabs(["Aperçu", "Statistiques", "Valeurs manquantes", "Classes", "Exploration"])
    with t1:
        st.dataframe(raw.head(30), width="stretch")
        st.caption("Données brutes téléchargées (le « ? » désigne une valeur manquante).")
    with t2:
        st.dataframe(X.describe(include="all").T, width="stretch")
    with t3:
        miss = raw.replace("?", np.nan).isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss):
            st.plotly_chart(px.bar(miss.rename("part").reset_index(), x="index", y="part",
                                   labels={"index": "Variable", "part": "Part manquante"},
                                   title="Valeurs manquantes (données brutes)"), width="stretch")
        else:
            st.success("Aucune valeur manquante.")
    with t4:
        a, b = st.columns(2)
        if SS["mode"] == "diabetes" and "readmitted" in raw:
            vc = raw["readmitted"].value_counts().rename_axis("classe").reset_index(name="n")
            a.plotly_chart(px.bar(vc, x="classe", y="n", title="Cible d'origine (3 classes)"), width="stretch")
        yb = y.map({0: "Non", 1: "Oui"}).value_counts().rename_axis("classe").reset_index(name="n")
        b.plotly_chart(px.pie(yb, names="classe", values="n", title="Cible binaire utilisée", hole=.45),
                       width="stretch")
        st.info(f"Classes déséquilibrées : {fmt_pct(y.mean())} de cas positifs → on privilégiera PR-AUC, "
                "rappel et précision plutôt que l'exactitude.")
    with t5:
        mode = st.radio("Visualisation", ["Distributions", "Corrélations", "Taux de la classe positive par variable"],
                        horizontal=True)
        num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        df_e = X.copy()
        df_e["cible"] = y.map({0: "Non", 1: "Oui"})
        df_e = df_e.sample(min(20000, len(df_e)), random_state=0)
        if mode == "Distributions":
            v = st.selectbox("Variable", list(X.columns), format_func=lab)
            if v in num:
                fig = px.histogram(df_e, x=v, color="cible", barmode="overlay", opacity=.6, histnorm="percent",
                                   marginal="box", labels={v: lab(v)})
            else:
                fig = px.histogram(df_e, x=v, color="cible", barmode="group", histnorm="percent", labels={v: lab(v)})
            st.plotly_chart(fig, width="stretch")
        elif mode == "Corrélations":
            cm = X[num].assign(cible=y).corr()
            st.plotly_chart(px.imshow(cm, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                                      title="Matrice de corrélation (variables numériques + cible)"),
                            width="stretch")
            ct = cm["cible"].drop("cible").sort_values()
            st.plotly_chart(px.bar(ct, orientation="h", title="Corrélation avec la cible", labels={"value": "r", "index": ""}),
                            width="stretch")
        else:
            v = st.selectbox("Variable", list(X.columns), format_func=lab, key="rate_var")
            s = X[v]
            if v in num and s.nunique() > 12:
                s = pd.qcut(s, 10, duplicates="drop").astype(str)
            g = pd.DataFrame({"v": s.astype(str), "y": y}).groupby("v")["y"].agg(["mean", "size"]).reset_index()
            fig = px.bar(g, x="v", y="mean", text="size", labels={"v": lab(v), "mean": "Taux de réadmission < 30 j"})
            fig.add_hline(y=float(y.mean()), line_dash="dash", annotation_text="Taux global")
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            st.plotly_chart(fig, width="stretch")


# =============================================================================== #
# Module 2 — Préparation
# =============================================================================== #
def page_prep():
    st.header("Module 2 — Préparation")
    need("raw")
    X0, y0, g0, _ = get_clean()
    with st.form("prep_form"):
        c1, c2 = st.columns(2)
        with c1:
            o = SS.get("clean_opts", dict(drop_expired=True, first_encounter=True))
            if SS["mode"] == "diabetes":
                drop_exp = st.checkbox("Retirer les patients décédés / en soins palliatifs", o["drop_expired"],
                                       help="Ils ne peuvent pas être réadmis : les garder biaiserait la cible.")
                first = st.checkbox("Une seule rencontre par patient (la première)", o["first_encounter"],
                                    help="Sinon, le découpage est fait par patient pour éviter les fuites.")
            else:
                drop_exp, first = True, True
            excl = st.checkbox("Exclure les variables sensibles (âge, origine, sexe)", False,
                               help="L'audit d'équité du Module 5 reste possible.")
            feats = st.multiselect("Variables explicatives", list(X0.columns), default=list(X0.columns),
                                   format_func=lab)
        with c2:
            num_imp = st.selectbox("Valeurs manquantes — numériques", ["median", "mean", "Supprimer les lignes"])
            cat_imp = st.selectbox("Valeurs manquantes — catégorielles",
                                   ["Catégorie « Manquant »", "Valeur la plus fréquente", "Supprimer les lignes"])
            scale = st.checkbox("Normaliser les variables numériques (z-score)", True)
            test_size = st.slider("Part du jeu de test", 0.10, 0.40, 0.25, 0.05)
            seed = st.number_input("Graine aléatoire", 0, 9999, 42)
            max_train = st.slider("Taille maximale du jeu d'entraînement", 5000, 100000, 30000, 5000,
                                  help="Limite le temps de calcul (kNN, SHAP).")
            balanced = st.checkbox("Pondérer les classes (déséquilibre)", True)
        go_ = st.form_submit_button("Préparer les données", type="primary")
    if go_:
        SS["clean_opts"] = dict(drop_expired=drop_exp, first_encounter=first)
        Xf, y, groups, _ = get_clean()
        cols = [c for c in feats if not (excl and c in D.SENSITIVE)]
        if not cols:
            st.error("Sélectionnez au moins une variable.")
            st.stop()
        X = Xf[cols]
        mask = pd.Series(True, index=X.index)
        if num_imp == "Supprimer les lignes":
            mask &= X[[c for c in cols if pd.api.types.is_numeric_dtype(X[c])]].notna().all(axis=1)
        if cat_imp == "Supprimer les lignes":
            mask &= X[[c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]].notna().all(axis=1)
        X, y = X[mask], y[mask]
        groups = groups[mask] if groups is not None else None
        Xtr, Xte, ytr, yte = split_data(X, y, test_size, int(seed), groups)
        if len(Xtr) > max_train:
            Xtr = Xtr.sample(max_train, random_state=int(seed))
            ytr = ytr.loc[Xtr.index]
        with st.spinner("Encodage et normalisation…"):
            fs = FeatureSpace("median" if num_imp != "mean" else "mean",
                              "most_frequent" if cat_imp == "Valeur la plus fréquente" else "constant", scale).fit(Xtr)
            A, B = fs.transform(Xtr), fs.transform(Xte)
        reset_downstream()
        SS["prep"] = dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, fs=fs, A=A, B=B, balanced=balanced, seed=int(seed),
                          audit=Xf.loc[Xte.index, [c for c in D.SENSITIVE if c in Xf]])
    if "prep" in SS:
        P = SS["prep"]
        c = st.columns(4)
        c[0].metric("Entraînement", f"{len(P['Xtr']):,}".replace(",", " "))
        c[1].metric("Test", f"{len(P['Xte']):,}".replace(",", " "))
        c[2].metric("Variables d'origine → colonnes modèle", f"{P['Xtr'].shape[1]} → {P['A'].shape[1]}")
        c[3].metric("Positifs train / test", f"{fmt_pct(P['ytr'].mean())} / {fmt_pct(P['yte'].mean())}")
        st.success("Données prêtes : passez au Module 3.")


# =============================================================================== #
# Module 3 — Modélisation
# =============================================================================== #
def param_widgets(name):
    out = {}
    for p in M.PARAM_UI[name]:
        k = f"{name}-{p['name']}"
        if p["kind"] == "int":
            out[p["name"]] = st.slider(p["name"], p["lo"], p["hi"], p["default"], key=k)
        elif p["kind"] == "float":
            out[p["name"]] = st.slider(p["name"], float(p["lo"]), float(p["hi"]), float(p["default"]), key=k)
        elif p["kind"] == "log":
            opts = [float(f"{v:.4g}") for v in np.logspace(np.log10(p["lo"]), np.log10(p["hi"]), 30)]
            near = min(opts, key=lambda v: abs(np.log(v) - np.log(p["default"])))
            out[p["name"]] = st.select_slider(p["name"], opts, near, key=k)
        else:
            out[p["name"]] = st.selectbox(p["name"], p["choices"], key=k)
    return out


def page_models():
    st.header("Module 3 — Modélisation et comparaison")
    need("prep")
    P = SS["prep"]
    chosen = st.multiselect("Modèles à entraîner", M.MODEL_NAMES,
                            default=[m for m in M.MODEL_NAMES if m != "k plus proches voisins"])
    if "k plus proches voisins" in chosen:
        st.caption("⏱ kNN est lent (prédiction sur tout le jeu d'entraînement) ; SHAP/LIME y sont aussi plus lents.")
    c = st.columns(4)
    tune = c[0].checkbox("Optimiser (RandomizedSearchCV)", False)
    n_iter = c[1].slider("Essais", 3, 30, 8, disabled=not tune)
    cv = c[2].slider("Plis de validation croisée", 2, 5, 3, disabled=not tune)
    scoring = c[3].selectbox("Critère", ["roc_auc", "average_precision"], disabled=not tune)
    params = {}
    for n in chosen:
        with st.expander(f"Hyperparamètres — {n}" + (" (point de départ si optimisation)" if tune else "")):
            params[n] = param_widgets(n)
    if st.button("Entraîner les modèles", type="primary", disabled=not chosen):
        res = {}
        bar = st.progress(0.0)
        for i, n in enumerate(chosen):
            with st.spinner(f"Entraînement : {n}…"):
                r = M.train(n, P["A"], P["ytr"], params[n], P["balanced"], tune, n_iter, cv, scoring, P["seed"])
                r["proba"] = r["model"].predict_proba(P["B"])[:, 1]
                res[n] = r
            bar.progress((i + 1) / len(chosen))
        SS["results"] = res
        for k in ("shap_cache", "perm_cache", "explainers"):
            SS.pop(k, None)
    if "results" in SS:
        T = EV.comparison_table(SS["results"], P["yte"])
        st.subheader("Tableau comparatif")
        st.dataframe(T.style.format({c: "{:.3f}" for c in ["AUC-ROC", "PR-AUC", "Gini", "KS", "Seuil F1-opt.", "Précision",
                                                           "Rappel", "F1", "AUC (CV)"] if c in T}, na_rep="—")
                     .background_gradient(subset=["AUC-ROC", "PR-AUC"], cmap="Greens"), width="stretch")
        st.caption("Les métriques de classement (AUC, PR-AUC, Gini, KS) ne dépendent pas du seuil. Précision / rappel / F1 "
                   "sont donnés au seuil maximisant le F1 **sur le jeu de test** (légèrement optimiste) ; "
                   "le seuil se choisit dans le Module 4.")
        st.plotly_chart(px.bar(T, x="Modèle", y=["AUC-ROC", "PR-AUC"], barmode="group", title="Comparaison des modèles"),
                        width="stretch")


# =============================================================================== #
# Module 4 — Évaluation
# =============================================================================== #
def page_eval():
    st.header("Module 4 — Évaluation")
    need("prep", "results")
    P, R = SS["prep"], SS["results"]
    yte = P["yte"]
    a, b = st.columns(2)
    a.plotly_chart(EV.roc_fig(R, yte), width="stretch")
    b.plotly_chart(EV.pr_fig(R, yte), width="stretch")
    st.subheader("Seuil de décision")
    name = st.selectbox("Modèle", list(R), key="eval_model")
    proba = R[name]["proba"]
    best = EV.best_threshold(yte, proba)
    c = st.columns([3, 1])
    c[0].slider("Seuil (probabilité au-delà de laquelle on signale un patient)", 0.01, 0.99,
                SS.get("thr", 0.15), 0.01, key="thr")
    c[1].button("Seuil F1-optimal", on_click=lambda: SS.update(thr=float(round(best, 2))))
    m = EV.metrics_at(yte, proba, SS["thr"])
    k = st.columns(4)
    k[0].metric("Rappel (réadmissions détectées)", fmt_pct(m["recall"]))
    k[1].metric("Précision (signalements justifiés)", fmt_pct(m["precision"]))
    k[2].metric("F1", f"{m['f1']:.3f}")
    k[3].metric("Patients signalés", fmt_pct(m["flagged_rate"]))
    st.info(f"Au seuil {SS['thr']:.2f}, le modèle détecte **{fmt_pct(m['recall'])}** des réadmissions à 30 jours en "
            f"signalant **{fmt_pct(m['flagged_rate'])}** des patients ; sur 100 patients signalés, environ "
            f"**{100 * m['precision']:.0f}** seront réellement réadmis (prévalence : {fmt_pct(yte.mean())}).")
    a, b = st.columns(2)
    a.plotly_chart(EV.confusion_fig(m), width="stretch")
    b.plotly_chart(EV.threshold_fig(yte, proba, SS["thr"]), width="stretch")


# =============================================================================== #
# Module 5 — IA explicable
# =============================================================================== #
def shap_for(name, n):
    cache = SS.setdefault("shap_cache", {})
    if (name, n) not in cache:
        P, m = SS["prep"], SS["results"][name]["model"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(P["B"]), min(n, len(P["B"])), replace=False)
        e = EX.make_shap_explainer(name, m, P["A"][rng.choice(len(P["A"]), min(100, len(P["A"])), replace=False)])
        Bt = P["B"][idx]
        cache[(name, n)] = (EX.shap_explain(name, e, Bt, P["fs"].to_display(Bt), P["fs"].names), idx)
    return cache[(name, n)]


def get_explainer(name):
    cache = SS.setdefault("explainers", {})
    if name not in cache:
        P = SS["prep"]
        b = SV.make_bundle(P["fs"], SS["results"][name]["model"], name, SS.get("thr", 0.15), P["Xtr"], P["A"], P["B"],
                           with_global=False)
        cache[name] = SV.Explainer(b)
    return cache[name]


PRIORITY = ["age", "gender", "race", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "admission_type",
            "discharge", "admission_source", "diag_1", "medical_specialty", "A1Cresult", "insulin", "change",
            "diabetesMed"]


def individual_form(P):
    Xtr = P["Xtr"]
    cols = [c for c in PRIORITY if c in Xtr.columns] or list(Xtr.columns[:12])
    vals, grid = {}, st.columns(3)
    for i, c in enumerate(cols):
        with grid[i % 3]:
            s = Xtr[c]
            if pd.api.types.is_numeric_dtype(s):
                lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                vals[c] = st.number_input(lab(c), lo, hi, med, step=10.0 if c == "age" else 1.0, key=f"f-{c}")
            else:
                opts = [str(v) for v in s.value_counts().index[:30]]
                vals[c] = st.selectbox(lab(c), opts, key=f"f-{c}")
    return vals


def page_xai():
    st.header("Module 5 — IA explicable")
    need("prep", "results")
    P, R = SS["prep"], SS["results"]
    fs = P["fs"]
    tg, tl, tf = st.tabs(["Explications globales", "Explication d'un individu", "Équité & variables sensibles"])

    # ---------------------------------------------------------------- global
    with tg:
        name = st.selectbox("Modèle à expliquer", list(R), key="xai_model")
        nat = EX.native_importance(name, R[name]["model"], fs.names)
        st.subheader("1. Importance propre au modèle")
        if nat is None:
            st.info("kNN n'a pas d'importance native : voir l'importance par permutation et SHAP.")
        else:
            top = nat.head(15).iloc[::-1]
            ttl = "Coefficients (variables normalisées)" if name == EX.LINEAR else "Importance par impureté"
            st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h", title=ttl, height=450),
                            width="stretch")
        st.subheader("2. Importance par permutation (baisse d'AUC) — comparable entre modèles")
        pm = st.multiselect("Modèles à comparer", list(R), default=[name])
        n_perm = st.slider("Individus de test utilisés", 300, 5000, 1500, 100)
        if st.button("Calculer l'importance par permutation"):
            cache = SS.setdefault("perm_cache", {})
            for m_ in pm:
                with st.spinner(f"Permutation : {m_}…"):
                    cache[(m_, n_perm)] = EX.permutation_by_variable(R[m_]["model"], fs, P["Xte"], P["yte"], n_perm)
        pc = SS.get("perm_cache", {})
        frames = [df.assign(modèle=m_) for (m_, n_), df in pc.items() if n_ == n_perm and m_ in pm]
        if frames:
            allp = pd.concat(frames)
            order = allp.groupby("variable")["importance"].mean().nlargest(12).index
            allp = allp[allp["variable"].isin(order)].assign(Variable=lambda d: d["variable"].map(lab))
            st.plotly_chart(px.bar(allp, y="Variable", x="importance", color="modèle", barmode="group",
                                   orientation="h", height=520, labels={"importance": "Baisse d'AUC"},
                                   category_orders={"Variable": [lab(v) for v in order]}), width="stretch")
        st.subheader("3. SHAP — graphique d'ensemble (beeswarm)")
        cap = 100 if name == EX.KNN else 2000
        n_shap = st.slider("Individus de test pour SHAP", 50, cap, min(500, cap), 50)
        if st.button("Calculer SHAP", type="primary"):
            with st.spinner("Calcul des valeurs SHAP…"):
                shap_for(name, n_shap)
        if (name, n_shap) in SS.get("shap_cache", {}):
            exp, _ = shap_for(name, n_shap)
            st.caption(f"Chaque point est un patient ; la position horizontale donne l'effet sur le risque "
                       f"({EX.shap_space(name)}), la couleur la valeur de la variable (rouge = élevée).")
            shap.plots.beeswarm(exp, max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            gv = EX.global_by_variable(exp.values, fs.meta).head(12).assign(Variable=lambda d: d["variable"].map(lab))
            st.plotly_chart(px.bar(gv.iloc[::-1], x="mean_abs_shap", y="Variable", orientation="h", color="sensible",
                                   color_discrete_map={True: "#DC2626", False: "#0F766E"}, height=450,
                                   title="Contribution moyenne |SHAP| par variable d'origine (rouge = sensible)",
                                   labels={"mean_abs_shap": "Contribution moyenne absolue"}), width="stretch")
            st.subheader("4. Graphiques de dépendance SHAP")
            order = np.argsort(-np.abs(exp.values).mean(0))
            names_sorted = [fs.names[i] for i in order]
            f1, f2 = st.columns(2)
            feat = f1.selectbox("Variable", names_sorted[:30])
            colb = f2.selectbox("Colorer par", ["(aucune)"] + names_sorted[:30])
            i = fs.names.index(feat)
            dfp = pd.DataFrame({"valeur": exp.data[:, i], "shap": exp.values[:, i]})
            if colb != "(aucune)":
                dfp["couleur"] = exp.data[:, fs.names.index(colb)]
            st.plotly_chart(px.scatter(dfp, x="valeur", y="shap", color="couleur" if colb != "(aucune)" else None,
                                       opacity=.6, labels={"valeur": feat, "shap": f"SHAP ({EX.shap_space(name)})"},
                                       title=f"Dépendance SHAP — {feat}"), width="stretch")

    # ----------------------------------------------------------------- local
    with tl:
        name = st.selectbox("Modèle", list(R), key="loc_model")
        thr = SS.get("thr", 0.15)
        st.caption(f"Seuil de décision courant : {thr:.2f} (réglable au Module 4).")
        how = st.radio("Individu", ["Sélectionner dans le jeu de test", "Saisir un individu fictif"], horizontal=True)
        if how.startswith("Sél"):
            c = st.columns([2, 1])
            if c[1].button("Tirer au hasard"):
                SS["loc_idx"] = int(np.random.default_rng().integers(len(P["Xte"])))
            i = c[0].number_input("Position dans le jeu de test", 0, len(P["Xte"]) - 1, SS.get("loc_idx", 0))
            row = P["Xte"].iloc[int(i)]
            st.caption(f"Issue réelle : **{'réadmission < 30 j' if P['yte'].iloc[int(i)] == 1 else 'pas de réadmission'}**")
            with st.expander("Valeurs de l'individu"):
                st.dataframe(row.rename("valeur").to_frame().T, width="stretch")
            values = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
            values = {k: v for k, v in values.items() if v is not None}
        else:
            values = individual_form(P)
        if st.button("Expliquer cet individu", type="primary"):
            with st.spinner("Calcul SHAP et LIME…"):
                SS["local"] = dict(name=name, thr=thr, out=get_explainer(name).explain(values, thr, top=12))
        L = SS.get("local")
        if L and L["name"] == name:
            out, exp = L["out"], L["out"]["_raw"]["exp"]
            c = st.columns(3)
            c[0].metric("Probabilité prédite", fmt_pct(out["probability"]))
            c[1].metric("Décision au seuil", "À signaler" if out["decision"] else "Pas de signalement")
            c[2].metric("Niveau de risque", out["risk_level"])
            st.success(out["sentence"])
            a, b = st.columns(2)
            with a:
                st.markdown("**SHAP — waterfall**")
                shap.plots.waterfall(exp[0], max_display=12, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
            with b:
                st.markdown("**LIME — même individu**")
                lm = pd.DataFrame(out["lime"]["rules"]).iloc[::-1]
                lm["sens"] = np.where(lm["weight"] > 0, "augmente le risque", "diminue le risque")
                st.plotly_chart(px.bar(lm, x="weight", y="rule", orientation="h", color="sens", height=430,
                                       color_discrete_map={"augmente le risque": "#DC2626", "diminue le risque": "#0F766E"},
                                       labels={"weight": "Poids LIME", "rule": ""}), width="stretch")
            st.markdown("**Comparaison SHAP vs LIME (par variable d'origine)**")
            cmp = EX.compare_shap_lime(exp.values[0], L["out"]["_raw"]["lime_w"], get_explainer(name).space.meta)
            k = st.columns(3)
            k[0].metric("Corrélation de rang (Spearman)", f"{cmp['spearman']:.2f}")
            k[1].metric(f"Variables communes dans le top {cmp['k']}", f"{int(cmp['top_k_overlap'] * cmp['k'])}/{cmp['k']}")
            k[2].metric("Même sens d'effet", fmt_pct(cmp["same_sign_share"]))
            cv_ = pd.DataFrame({"SHAP": cmp["shap_by_var"], "LIME": cmp["lime_by_var"]})
            cv_ = cv_.loc[(cv_["SHAP"].abs() + cv_["LIME"].abs()).nlargest(10).index]
            cv_.index = [lab(v) for v in cv_.index]
            st.plotly_chart(px.bar(cv_.iloc[::-1], barmode="group", orientation="h", height=420,
                                   labels={"value": "Contribution", "index": ""}), width="stretch")
            st.caption("SHAP et LIME sont sur des échelles différentes : on compare le classement et le sens des effets, "
                       "pas les valeurs absolues.")

    # ---------------------------------------------------------------- équité
    with tf:
        name = st.selectbox("Modèle", list(R), key="fair_model")
        thr = SS.get("thr", 0.15)
        audit = P["audit"].copy()
        if "age" in audit:
            audit["age"] = EV.age_band(audit["age"])
        attr = st.selectbox("Attribut sensible à auditer", list(audit.columns), format_func=lab)
        gm = EV.group_metrics(P["yte"], R[name]["proba"], thr, audit[attr].to_numpy())
        gaps = EV.fairness_gaps(gm)
        st.caption(f"Seuil courant : {thr:.2f}. Groupes de moins de 100 patients ignorés.")
        st.dataframe(gm.style.format({c: "{:.3f}" for c in gm.columns if c not in ("Groupe", "n")}), width="stretch")
        if gaps:
            k = st.columns(3)
            k[0].metric("Écart de taux de signalement (parité démographique)", f"{100 * gaps['demographic_parity_gap']:.1f} pts")
            k[1].metric("Écart de rappel (égalité des chances)", f"{100 * gaps['equal_opportunity_gap']:.1f} pts")
            k[2].metric("Écart de faux positifs", f"{100 * gaps['fpr_gap']:.1f} pts")
            st.plotly_chart(px.bar(gm, x="Groupe", y=["Taux réel de réadmission", "Taux signalé par le modèle", "Rappel"],
                                   barmode="group", title=f"Comparaison par {lab(attr).lower()}"), width="stretch")
        used = [c for c in D.SENSITIVE if c in P["Xtr"].columns]
        if used and (name, 500) in SS.get("shap_cache", {}):
            gv = EX.global_by_variable(SS["shap_cache"][(name, 500)][0].values, fs.meta)
            part = gv.loc[gv["sensible"], "part"].sum()
            st.info(f"Variables sensibles utilisées par le modèle ({', '.join(lab(c) for c in used)}) : "
                    f"{fmt_pct(part)} de la contribution SHAP totale.")
        elif used:
            st.info("Calculez SHAP avec 500 individus (onglet « globales ») pour mesurer le poids des variables sensibles.")
        else:
            st.success("Aucune variable sensible n'est utilisée par le modèle (exclues au Module 2). "
                       "L'audit ci-dessus reste utile : des proxys (diagnostics, destination de sortie) peuvent encoder les mêmes écarts.")
        st.caption("Un écart de taux ne prouve pas une discrimination (les groupes ont des cas cliniques différents) ; "
                   "il signale où investiguer avant tout usage décisionnel.")


# =============================================================================== #
# Module 6 — API d'interprétation (FastAPI)
# =============================================================================== #
def page_api():
    st.header("Module 6 — API d'interprétation (FastAPI)")
    need("prep", "results")
    P, R = SS["prep"], SS["results"]
    url, remote = api_url()
    kind = "API distante" if remote else "API locale lancée avec l'application (thread interne)"
    st.write(f"Service : `{url}` — {kind}")
    try:
        h = httpx.get(f"{url}/health", timeout=10).json()
        st.json(h)
    except Exception as e:
        st.error(f"API injoignable : {e}")
        st.stop()
    st.markdown(f"Documentation interactive (Swagger) : [{url}/docs]({url}/docs) · "
                "points d'accès : `/predict`, `/explain/shap`, `/explain/lime`, `/explain`, `/schema`, `/global/importance`.")
    st.subheader("1. Exporter un modèle vers l'API")
    name = st.selectbox("Modèle à publier", list(R), key="api_model")
    if st.button("Exporter vers l'API", type="primary"):
        with st.spinner("Construction du paquet modèle + importances SHAP…"):
            b = SV.make_bundle(P["fs"], R[name]["model"], name, SS.get("thr", 0.15), P["Xtr"], P["A"], P["B"],
                               metrics={k: v for k, v in EV.ranking_metrics(P["yte"], R[name]["proba"]).items()})
            path = SV.save_bundle(b)
        if remote:
            st.warning(f"Paquet écrit dans `{path}`. Pour une API distante, déployez ce fichier avec l'API puis appelez `POST /reload`.")
        else:
            httpx.post(f"{url}/reload", timeout=60)
            st.success("Modèle publié et rechargé par l'API.")
        st.rerun()
    st.subheader("2. Appeler l'API pour un individu du jeu de test")
    i = st.number_input("Position dans le jeu de test", 0, len(P["Xte"]) - 1, 0, key="api_idx")
    row = P["Xte"].iloc[int(i)]
    values = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if not pd.isna(v)}
    top = st.slider("Contributions renvoyées", 3, 20, 8)
    payload = dict(values=values, threshold=SS.get("thr", 0.15), top=top, lime_samples=1500)
    with st.expander("Requête JSON envoyée à POST /explain"):
        st.json(payload)
        st.code(f"curl -X POST {url}/explain -H 'Content-Type: application/json' -d '<json ci-dessus>'", language="bash")
    if st.button("Appeler POST /explain"):
        try:
            r = httpx.post(f"{url}/explain", json=payload, timeout=180)
            r.raise_for_status()
            j = r.json()
            st.success(j["sentence"])
            c = st.columns(3)
            c[0].metric("Probabilité", fmt_pct(j["probability"]))
            c[1].metric("Décision", "À signaler" if j["decision"] else "Pas de signalement")
            c[2].metric("Cohérence SHAP/LIME (Spearman)", f"{j['comparison']['spearman']:.2f}")
            sv = pd.DataFrame(j["shap"]["by_variable"]).iloc[::-1]
            st.plotly_chart(px.bar(sv, x="shap", y="label", orientation="h", title="Contributions SHAP renvoyées par l'API",
                                   labels={"shap": "SHAP", "label": ""}), width="stretch")
            with st.expander("Réponse JSON complète"):
                st.json(j)
        except Exception as e:
            st.error(f"Échec de l'appel : {e}")
            if isinstance(e, httpx.HTTPStatusError):
                st.code(e.response.text)


# =============================================================================== #
# Navigation
# =============================================================================== #
PAGES = {"0 · Vue d'ensemble": page_overview, "1 · Données": page_data, "2 · Préparation": page_prep,
         "3 · Modélisation": page_models, "4 · Évaluation": page_eval, "5 · IA explicable": page_xai,
         "6 · API (FastAPI)": page_api}

with st.sidebar:
    st.title("🏥 Réadmission à 30 jours")
    st.caption("Patients diabétiques · 130 hôpitaux US (UCI) · classification explicable")
    if os.getenv("USE_SYNTHETIC_DEMO") == "1":
        st.error("MODE DÉMO : données SYNTHÉTIQUES — aucun résultat réel.")
    page = st.radio("Modules", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.write(("✅" if "raw" in SS else "⬜") + " données · " + ("✅" if "prep" in SS else "⬜") + " préparation")
    st.write(("✅" if "results" in SS else "⬜") + f" modèles ({len(SS.get('results', {}))})")
    st.caption("Aide à la décision : ne remplace pas le jugement clinique.")

if page != "0 · Vue d'ensemble" and page != "1 · Données" and "raw" not in SS:
    SS.update(raw=load_uci(), mode="diabetes", source="uci")
PAGES[page]()
