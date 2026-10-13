# 🏥 Réadmission à 30 jours des patients diabétiques — classification explicable

Projet *Challenge 1* (Data Mining, ENSI) · **cas n°12 : Santé – réadmission hospitalière**
Jeu : [Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/dataset/296) (UCI, téléchargé automatiquement via `ucimlrepo`).
Cible : réadmission sous 30 jours (`readmitted == "<30"`) contre tous les autres cas.

| | Lien |
|---|---|
| Application (Streamlit) | _à renseigner après déploiement_ |
| API FastAPI (Swagger `/docs`) | _à renseigner après déploiement_ |
| One-pager | `one_pager.pdf` |

## Les six modules de l'application

| # | Module | Contenu |
|---|---|---|
| 0 | Vue d'ensemble | tableau de bord exécutif avec résumé de performance, comparaison rapide des modèles, seuil d'alerte et export CSV du bilan |
| 1 | Données | téléchargement automatique ou import CSV (+ cible), aperçu, statistiques, valeurs manquantes, classes, distributions / corrélations / taux de la classe positive par variable |
| 2 | Préparation | choix des variables, imputation, encodage one-hot, normalisation, découpage paramétrable (par patient), exclusion des variables sensibles |
| 3 | Modélisation | régression logistique, arbre, kNN, Random Forest, XGBoost ; hyperparamètres dans l'interface ; `RandomizedSearchCV` ; tableau comparatif |
| 4 | Évaluation | AUC-ROC, PR-AUC, précision / rappel / F1, Gini, KS ; ROC et PR superposées ; matrice de confusion ; **curseur de seuil** en direct |
| 5 | IA explicable | importance native, permutation, SHAP beeswarm + dépendance ; individu (test ou fictif) : probabilité, SHAP waterfall, LIME, comparaison, phrase de synthèse ; **audit d'équité** par âge / origine / sexe |
| 6 | API | export du modèle vers l'API FastAPI et appel de `POST /explain` depuis l'application |

### Améliorations ajoutées
- Dashboard de synthèse moderne avec remise en page plus claire et plus lisible.
- Choix automatique d'un port local libre pour l'API FastAPI afin d'éviter les conflits de port.
- Export CSV du résumé comparatif des modèles depuis la page d'accueil.
- Vue d'ensemble plus orientée décision / pilotage, utilisable avant la modélisation détaillée.

### Choix de conception à connaître
- **Fuite de données** : plusieurs séjours d'un même patient → par défaut une seule rencontre par patient ; sinon découpage par patient (`GroupShuffleSplit`).
- **Décès / soins palliatifs** retirés (ces patients ne peuvent pas être réadmis).
- **Déséquilibre** (~9-11 % de positifs) : pondération des classes, PR-AUC, seuil réglable. Au pipeline `train.py`, le seuil est choisi sur des prédictions *hors-pli de l'entraînement*, jamais sur le test.
- **SHAP / LIME** : explications calculées en unités d'origine (lisibles) ; les colonnes one-hot d'une même variable sont sommées pour les phrases et la comparaison SHAP–LIME.
- **Équité** : l'audit utilise âge, origine et sexe même s'ils sont exclus du modèle (détection d'effets par proxy).

## Structure
```
app.py                  application Streamlit (6 modules)
api/main.py             API FastAPI (/predict, /explain, /explain/shap, /explain/lime, /schema, /global/importance, /reload)
core/                   data, prep, models, evaluation, explain, service (partagé app + API), synthetic (tests)
train.py                pipeline complet hors interface -> artifacts/{results.json,bundle.joblib,figs/}
generate_onepager.py    one-pager PDF alimenté par results.json
notebook.ipynb          la même démarche pas à pas
tests/smoke_test.py     test de bout en bout de l'application (données synthétiques)
Dockerfile              API seule (Hugging Face Spaces Docker / Render)
```

## Installation et lancement
```bash
python -m venv .venv && source .venv/bin/activate       # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                     # l'API est lancée automatiquement en arrière-plan (port 8008)
```
API seule : `uvicorn api.main:app --port 8008` puis http://localhost:8008/docs

### Reproduire les résultats et le one-pager
```bash
python train.py                 # ~10-25 min selon la machine ; --n-iter 4 --max-train 20000 pour aller vite
python generate_onepager.py --author "Prénom Nom" --app-url <url> --repo-url <url>
```
**Relisez le PDF** : les chiffres viennent de `results.json`, mais les paragraphes d'analyse critique sont rédigés par règles et doivent être validés (cohérence clinique, recommandation) par vous.

## Déploiement
**Application — Streamlit Community Cloud** : poussez le dépôt sur GitHub, créez une app pointant sur `app.py`. Sans configuration, l'application lance l'API FastAPI dans un thread du même conteneur : le Module 6 fonctionne tel quel (le bouton « Exporter vers l'API » publie le modèle courant). Mémoire limitée (~1 Go) : gardez la taille d'entraînement ≤ 30 000 lignes.

**API séparée (optionnel) — Hugging Face Spaces (Docker)** : lancez `python train.py`, committez `artifacts/bundle.joblib` (`git add -f`), déployez le `Dockerfile`, puis ajoutez le secret `API_URL=https://<votre-space>.hf.space` à l'application Streamlit. Privilégiez XGBoost ou la régression logistique comme modèle retenu (un Random Forest de 200 arbres donne un fichier trop lourd pour GitHub).

## Captures d'écran
Ajoutez ici 4-6 captures de l'application déployée (données, comparaison des modèles, courbes ROC + seuil, SHAP beeswarm, explication individuelle, API Swagger) dans `docs/` : `![ROC](docs/roc.png)`.

## Tests
```bash
USE_SYNTHETIC_DEMO=1 python tests/smoke_test.py
```
`USE_SYNTHETIC_DEMO=1` charge un jeu **synthétique** de même schéma (hors-ligne). Il ne sert qu'à tester le code : aucun résultat obtenu avec lui ne doit figurer dans le rapport. L'application affiche alors une bannière rouge, et `train.py` marque `results.json` comme synthétique (le one-pager reçoit un filigrane).

## Limites
Données 1999-2008 (hôpitaux américains) ; réadmission toutes causes ; pas de variables sociales ; explications = associations, pas causalité ; outil d'aide à la priorisation, pas un dispositif médical.
