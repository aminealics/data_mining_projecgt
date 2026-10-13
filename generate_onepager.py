"""Génère le one-pager (PDF, 1 page A4) à partir de artifacts/results.json produit par train.py.

    python generate_onepager.py --author "Prénom Nom" --app-url https://... --repo-url https://github.com/...

Tous les chiffres viennent de results.json. Les paragraphes d'analyse critique sont rédigés par
règles à partir de ces chiffres : RELISEZ-LES et ajustez-les avec votre jugement avant le rendu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data import label  # noqa: E402
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

TEAL, RED, GREY = colors.HexColor("#0F766E"), colors.HexColor("#B91C1C"), colors.HexColor("#475569")

ACTIONS = {
    "number_inpatient": "suivi post-sortie renforcé (appel à 48-72 h, consultation sous 7 jours) pour les patients déjà hospitalisés dans l'année",
    "number_emergency": "parcours de soins ambulatoires pour les patients qui passent souvent par les urgences",
    "prior_visits": "repérer dès l'admission les patients à forte utilisation antérieure des soins",
    "discharge": "coordination avec l'établissement ou le service d'accueil à la sortie (transmission du dossier, rendez-vous fixé)",
    "number_diagnoses": "revue de situation pluridisciplinaire quand le nombre de diagnostics est élevé",
    "num_medications": "revue pharmaceutique de sortie (simplification, éducation du patient)",
    "n_med_changes": "revue pharmaceutique quand le traitement antidiabétique a été ajusté pendant le séjour",
    "insulin": "éducation à l'insulinothérapie et suivi des ajustements de dose",
    "A1Cresult": "dosage HbA1c systématique pendant le séjour",
    "time_in_hospital": "anticiper la sortie des séjours longs (plan de retour à domicile)",
    "medical_specialty": "adapter le suivi à la spécialité de prise en charge",
    "diag_1": "protocoles de suivi par pathologie principale",
    "diag_2": "protocoles de suivi par pathologie principale",
    "num_lab_procedures": "intensité des examens = marqueur de gravité : coupler au dépistage des cas complexes",
}


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


def nb(n):
    return f"{n:,}".replace(",", " ")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def build_text(R):
    ds, best, rk, th = R["dataset"], R["best"], R["ranking"], R["threshold"]
    t, t60 = th["at_test"], th["recall60_at_test"]
    models = {m["Modèle"]: m for m in R["models"]}
    lr = models.get("Régression logistique")
    tops = R["top_variables"]
    ag, se = R["agreement"], R["sensitive"]
    txt = {}
    txt["probleme"] = ("<b>Problème métier.</b> À la sortie d'un patient diabétique, l'hôpital veut repérer ceux qui seront "
                       f"réadmis sous 30 jours ({pct(ds['positive_rate'])} des séjours) pour leur offrir un suivi renforcé, "
                       "sans saturer les équipes : il faut donc détecter beaucoup de cas <i>et</i> justifier chaque alerte.")
    txt["donnees"] = (f"<b>Données.</b> UCI « Diabetes 130-US hospitals » (1999-2008) : {nb(ds['n_raw'])} séjours → {nb(ds['n_final'])} après "
                      "nettoyage (décès et soins palliatifs retirés, une rencontre par patient), "
                      f"{ds['n_features']} variables (diagnostics regroupés CIM-9, traitements, historique de soins). "
                      f"Découpage par patient ({nb(ds['n_train'])} / {nb(ds['n_test'])}) ; classes déséquilibrées → pondération et PR-AUC.")
    txt["modele"] = (f"<b>Modèle retenu : {best}</b> (AUC-ROC {num(rk['auc'])}, PR-AUC {num(rk['pr_auc'])} pour une prévalence de "
                     f"{pct(R['dataset']['test_positive_rate'])}). Seuil {num(th['value'], 2)}, fixé par validation croisée sur "
                     f"l'entraînement : rappel {pct(t['recall'])}, précision {pct(t['precision'])}, {pct(t['flagged_rate'])} des "
                     f"patients signalés. En mode « rappel ≥ 60 % » (seuil {num(th['recall60_value'], 2)}) : précision "
                     f"{pct(t60['precision'])}, {pct(t60['flagged_rate'])} signalés.")
    items = []
    for v in tops[:5]:
        items.append(f"<b>{v['label']}</b> ({pct(v['part'], 0)} du poids SHAP) : {v['direction']}")
    txt["cles"] = items
    ex = R.get("example")
    txt["exemple"] = f"Exemple (patient du jeu de test) : « {ex['sentence']} »" if ex else ""
    perm = ", ".join(label(p["variable"]) for p in R["permutation_top"][:3])
    txt["perm"] = f"L'importance par permutation (baisse d'AUC) place en tête : {perm}."

    rho = ag["spearman_mean"]
    qual = ("bonne cohérence" if rho >= 0.7 else "cohérence partielle : LIME fragmente les variables catégorielles et varie d'un tirage à l'autre, "
            "SHAP est plus stable et sert de référence" if rho >= 0.4 else "faible cohérence : n'utiliser LIME que comme second avis")
    txt["coherence"] = (f"<b>Cohérence SHAP / LIME.</b> Sur {ag['n']} patients, corrélation de rang moyenne {num(rho, 2)} (min {num(ag['spearman_min'], 2)}), "
                        f"{num(ag['top5_overlap_mean'] * 5, 1)}/5 variables communes dans le top 5, même sens d'effet dans {pct(ag['same_sign_mean'], 0)} "
                        f"des cas : {qual}. Cohérence avec l'intuition clinique à faire valider par un médecin (gravité, "
                        "recours antérieur aux soins, modalités de sortie).")
    d_auc = se["auc_with"] - se["auc_without"]
    fair = R["fairness"]
    worst = max(((k, v["gaps"]) for k, v in fair.items() if v["gaps"]), key=lambda kv: kv[1]["equal_opportunity_gap"], default=None)
    names = {"age": "âge", "race": "origine", "gender": "sexe"}
    fair_txt = ""
    if worst:
        k, g = worst
        fair_txt = (f" Le plus grand écart concerne {names.get(k, k)} : {num(100 * g['demographic_parity_gap'], 1)} pts de taux de signalement, "
                    f"{num(100 * g['equal_opportunity_gap'], 1)} pts de rappel entre groupes.")
    verdict = ("leur retrait coûte peu : à recommander" if d_auc < 0.005 else "leur retrait a un coût mesurable en performance : arbitrage éthique et juridique à faire par le décideur")
    txt["equite"] = (f"<b>Variables sensibles.</b> Âge, origine et sexe portent {pct(se['shap_share'])} du poids SHAP ; sans elles l'AUC passe de "
                     f"{num(se['auc_with'])} à {num(se['auc_without'])} : {verdict}.{fair_txt} Des proxys (diagnostics, destination de sortie) "
                     "peuvent reproduire ces écarts : auditer par groupe avant tout usage.")
    if lr and best != "Régression logistique":
        gap = rk["auc"] - lr["AUC-ROC"]
        reco = ("l'écart est faible ou nul : <b>recommandation : régression logistique</b> (lisible, auditable par le régulateur et le clinicien), "
                f"{best} gardé comme modèle de contrôle." if gap < 0.01 else
                f"l'écart est net : <b>recommandation : {best} accompagné de SHAP</b> pour justifier chaque alerte, "
                "avec la régression logistique comme référence documentée.")
        txt["compromis"] = (f"<b>Performance vs interprétabilité.</b> {best} : AUC {num(rk['auc'])} contre {num(lr['AUC-ROC'])} pour la régression "
                            f"logistique (écart {num(gap, 3)}) ; {reco}")
    else:
        txt["compromis"] = ("<b>Performance vs interprétabilité.</b> La régression logistique est ici le modèle retenu : performante <i>et</i> directement "
                            "lisible (coefficients) — aucun arbitrage à faire.")
    acts = []
    for v in tops:
        a = ACTIONS.get(v["variable"])
        if a and a not in acts:
            acts.append(a)
    txt["actions"] = ("<b>Actions possibles</b> (hypothèses à valider cliniquement, les explications décrivent des associations, pas des causes) : "
                      + " ; ".join(acts[:4]) + ".") if acts else ""
    txt["limites"] = ("<b>Limites.</b> Données 1999-2008 de 130 hôpitaux américains (pratiques et codage datés) ; réadmission toutes causes confondues, "
                      "sans variables sociales (logement, soutien familial) ; seuil non calibré sur des coûts réels ; association ≠ causalité ; "
                      "performance modeste attendue sur ce problème : le score est une aide à la priorisation, pas un verdict individuel.")
    return txt


def make_pdf(R, out, author, app_url, repo_url, base=8.0, watermark=None):
    figs = Path(out).parent / "figs"
    txt = build_text(R)
    fs = base
    body = ParagraphStyle("b", fontName="DV", fontSize=fs, leading=fs * 1.32, alignment=TA_JUSTIFY, spaceAfter=fs * .35)
    bul = ParagraphStyle("u", parent=body, leftIndent=8, bulletIndent=0, spaceAfter=1)
    h = ParagraphStyle("h", fontName="DV-B", fontSize=fs + 1.6, leading=(fs + 1.6) * 1.2, textColor=TEAL, spaceBefore=3, spaceAfter=2)
    title = ParagraphStyle("t", fontName="DV-B", fontSize=14.5, leading=17, textColor=colors.HexColor("#0B3B36"))
    sub = ParagraphStyle("s", fontName="DV", fontSize=7.5, leading=9, textColor=GREY)
    W = A4[0] - 2 * 1.3 * cm
    syn = R["data_source"] == "synthetic"

    def onpage(c, d):
        if syn or watermark:
            c.saveState(); c.setFont("DV-B", 44); c.setFillColor(colors.Color(.85, .1, .1, alpha=.13))
            c.translate(A4[0] / 2, A4[1] / 2); c.rotate(35)
            c.drawCentredString(0, 20, watermark or "DONNÉES SYNTHÉTIQUES"); c.setFont("DV-B", 22)
            c.drawCentredString(0, -20, "aperçu de mise en page — résultats non réels"); c.restoreState()

    st = []
    st.append(Paragraph("Réadmission à 30 jours (diabète) : un score explicable", title))
    meta = f"Cas n°12 · Santé · Data Mining, ENSI · {R['created'][:10]}" + (f" · {author}" if author else "")
    links = " · ".join(x for x in [f"Application : {app_url}" if app_url else "", f"Code : {repo_url}" if repo_url else ""] if x)
    st.append(Paragraph(meta + (f"<br/>{links}" if links else ""), sub))
    st.append(Spacer(1, 4))

    # tableau comparatif
    rows = [["Modèle", "AUC-ROC", "PR-AUC", "Gini", "KS"]]
    for m in R["models"]:
        rows.append([m["Modèle"] + (" ★" if m["Modèle"] == R["best"] else ""), num(m["AUC-ROC"]), num(m["PR-AUC"]), num(m["Gini"]), num(m["KS"])])
    tb = Table(rows, colWidths=[3.6 * cm, 1.8 * cm, 1.4 * cm, 1.2 * cm, 1.1 * cm])
    tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "DV", fs - .8), ("FONT", (0, 0), (-1, 0), "DV-B", fs - .8),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#CCFBF1")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#94A3B8")),
                            ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    left = [Paragraph(txt["probleme"], body), Paragraph(txt["donnees"], body), Paragraph(txt["modele"], body), tb]
    right_w = 6.9 * cm
    right = [Image(str(figs / "roc.png"), width=right_w, height=right_w * 3.4 / 4.2),
             Image(str(figs / "shap_bar.png"), width=right_w, height=right_w * 3.2 / 4.4)]
    top = Table([[KeepInFrame(W - right_w - .4 * cm, 9.2 * cm, left, mode="shrink"), right]], colWidths=[W - right_w - .3 * cm, right_w + .3 * cm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 6)]))
    st.append(top)

    st.append(Paragraph("Explications clés", h))
    for it in txt["cles"]:
        st.append(Paragraph(it, bul, bulletText="•"))
    st.append(Paragraph(txt["perm"] + " " + txt["exemple"], body))
    st.append(Paragraph("Analyse critique et recommandations", h))
    for k in ("coherence", "equite", "compromis", "actions"):
        if txt.get(k):
            st.append(Paragraph(txt[k], body))
    st.append(Paragraph(txt["limites"], body))

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.3 * cm, rightMargin=1.3 * cm, topMargin=1.1 * cm, bottomMargin=1.0 * cm,
                            title="One-pager — Réadmission à 30 jours", author=author or "")
    doc.build(st, onFirstPage=onpage, onLaterPages=onpage)
    return doc.page


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/results.json")
    ap.add_argument("--out", default="one_pager.pdf")
    ap.add_argument("--author", default="")
    ap.add_argument("--app-url", default="")
    ap.add_argument("--repo-url", default="")
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    for size in (9.0, 8.8, 8.6, 8.4, 8.2, 7.9, 7.6, 7.3, 7.0, 6.7, 6.4):  # plus grande taille qui tient sur une page
        pages = make_pdf(R, a.out, a.author, a.app_url, a.repo_url, size)
        if pages == 1:
            print(f"✔ {a.out} (1 page, corps {size} pt)")
            break
    else:
        print("⚠ Le contenu dépasse une page même en petit corps : raccourcissez les textes.")
