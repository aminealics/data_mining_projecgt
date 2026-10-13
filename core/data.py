"""Chargement et nettoyage des données — cas n°12 : réadmission hospitalière.

Jeu : « Diabetes 130-US hospitals for years 1999-2008 » (UCI, id 296).
Cible binaire : réadmission sous 30 jours (readmitted == "<30") vs. autres.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "diabetes_130_raw.pkl"
UCI_ID = 296
TARGET = "readmission_30j"
SENSITIVE = ["age", "race", "gender"]  # variables sensibles (équité)

# discharge_disposition_id : décès / soins palliatifs -> pas de réadmission possible
EXPIRED_HOSPICE = {11, 13, 14, 19, 20, 21}

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
NUM_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]

FR_LABELS = {
    "time_in_hospital": "Durée du séjour (jours)",
    "num_lab_procedures": "Nb d'examens de laboratoire",
    "num_procedures": "Nb de procédures",
    "num_medications": "Nb de médicaments",
    "number_outpatient": "Consultations externes (année précédente)",
    "number_emergency": "Passages aux urgences (année précédente)",
    "number_inpatient": "Hospitalisations (année précédente)",
    "number_diagnoses": "Nb de diagnostics",
    "age": "Âge", "gender": "Sexe", "race": "Origine",
    "admission_type": "Type d'admission", "discharge": "Destination de sortie",
    "admission_source": "Provenance du patient",
    "diag_1": "Diagnostic principal", "diag_2": "Diagnostic secondaire",
    "diag_3": "Diagnostic tertiaire", "medical_specialty": "Spécialité médicale",
    "A1Cresult": "Résultat HbA1c", "max_glu_serum": "Glycémie sérique",
    "change": "Changement de traitement", "diabetesMed": "Traitement antidiabétique",
    "insulin": "Insuline", "metformin": "Metformine", "glipizide": "Glipizide",
    "glyburide": "Glyburide", "glimepiride": "Glimépiride", "pioglitazone": "Pioglitazone",
    "rosiglitazone": "Rosiglitazone",
    "n_meds_active": "Nb d'antidiabétiques actifs", "n_med_changes": "Nb d'ajustements de dose",
    "prior_visits": "Visites antérieures (total)",
}


def label(col: str) -> str:
    return FR_LABELS.get(col, col)


# --------------------------------------------------------------------------- #
# Téléchargement automatique
# --------------------------------------------------------------------------- #
def fetch_diabetes(use_cache: bool = True) -> pd.DataFrame:
    """Télécharge le jeu UCI #296 via ucimlrepo (sans manipulation manuelle).

    Variable d'environnement USE_SYNTHETIC_DEMO=1 : jeu SYNTHÉTIQUE de même schéma,
    uniquement pour tester l'application hors-ligne (jamais pour les résultats).
    """
    if os.getenv("USE_SYNTHETIC_DEMO") == "1":
        from core.synthetic import make_synthetic
        return make_synthetic()
    if use_cache and CACHE.exists():
        return pd.read_pickle(CACHE)
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_ID)
    parts = [p for p in (ds.data.ids, ds.data.features, ds.data.targets) if p is not None]
    df = pd.concat(parts, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    try:
        CACHE.parent.mkdir(exist_ok=True)
        df.to_pickle(CACHE)
    except OSError:
        pass
    return df


# --------------------------------------------------------------------------- #
# Regroupements métier
# --------------------------------------------------------------------------- #
def icd9_group(code) -> str:
    """Regroupe un code diagnostic CIM-9 en grandes familles (Strack et al., 2014)."""
    if pd.isna(code):
        return "Manquant"
    s = str(code)
    if s[:1] in ("E", "V"):
        return "Autre"
    try:
        v = float(s)
    except ValueError:
        return "Autre"
    i = int(v)
    if 250 <= v < 251:
        return "Diabète"
    if 390 <= i <= 459 or i == 785:
        return "Circulatoire"
    if 460 <= i <= 519 or i == 786:
        return "Respiratoire"
    if 520 <= i <= 579 or i == 787:
        return "Digestif"
    if 800 <= i <= 999:
        return "Traumatisme"
    if 710 <= i <= 739:
        return "Musculo-squelettique"
    if 580 <= i <= 629 or i == 788:
        return "Génito-urinaire"
    if 140 <= i <= 239:
        return "Tumeurs"
    return "Autre"


def _admission_type(i):
    return {1: "Urgence", 2: "Urgent", 3: "Programmée", 7: "Traumatologie"}.get(i, "Autre/inconnu")


def _discharge(i):
    if i == 1:
        return "Domicile"
    if i in (6, 8):
        return "Domicile avec soins"
    if i in (2, 3, 4, 5, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30):
        return "Établissement de soins"
    if i == 7:
        return "Sortie contre avis médical"
    return "Autre/inconnu"


def _admission_source(i):
    if i == 7:
        return "Urgences"
    if i in (1, 2, 3):
        return "Adressé par médecin/clinique"
    if i in (4, 5, 6, 10, 18, 19, 22, 25, 26):
        return "Transfert"
    return "Autre/inconnu"


def _age_mid(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    if s.startswith("["):
        return int(s[1:s.index("-")]) + 5
    try:
        return float(s)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# Nettoyage spécifique au cas n°12
# --------------------------------------------------------------------------- #
def prepare_diabetes(df: pd.DataFrame, drop_expired: bool = True, first_encounter: bool = True,
                     top_specialties: int = 12, min_med_rate: float = 0.01):
    """Retourne (X, y, groups, info). y = 1 si réadmission < 30 jours."""
    d = df.replace("?", np.nan).copy()
    n_raw = len(d)

    if drop_expired and "discharge_disposition_id" in d:
        dd = pd.to_numeric(d["discharge_disposition_id"], errors="coerce")
        d = d[~dd.isin(EXPIRED_HOSPICE)]
    n_alive = len(d)

    if "patient_nbr" in d and first_encounter and "encounter_id" in d:
        d = d.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    if "gender" in d:
        d = d[d["gender"].astype(str) != "Unknown/Invalid"]
    d = d.reset_index(drop=True)
    groups = d["patient_nbr"].copy() if "patient_nbr" in d else None

    y = (d["readmitted"].astype(str) == "<30").astype(int).rename(TARGET)

    X = pd.DataFrame(index=d.index)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(d[c], errors="coerce")
    X["prior_visits"] = X["number_outpatient"] + X["number_emergency"] + X["number_inpatient"]
    X["age"] = d["age"].map(_age_mid)
    X["gender"] = d["gender"].astype(str)
    X["race"] = d["race"].fillna("Inconnue").astype(str)
    X["admission_type"] = pd.to_numeric(d["admission_type_id"], errors="coerce").map(_admission_type)
    X["discharge"] = pd.to_numeric(d["discharge_disposition_id"], errors="coerce").map(_discharge)
    X["admission_source"] = pd.to_numeric(d["admission_source_id"], errors="coerce").map(_admission_source)
    for k in ("diag_1", "diag_2", "diag_3"):
        X[k] = d[k].map(icd9_group)
    spec = d["medical_specialty"].fillna("Inconnue").astype(str)
    top = spec.value_counts().index[:top_specialties]
    X["medical_specialty"] = np.where(spec.isin(top), spec, "Autre")
    for c in ("A1Cresult", "max_glu_serum"):
        X[c] = d[c].fillna("Non mesuré").astype(str)

    meds = d[[c for c in MED_COLS if c in d]].fillna("No")
    X["n_meds_active"] = (meds != "No").sum(axis=1)
    X["n_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1)
    kept = [c for c in meds if (meds[c] != "No").mean() >= min_med_rate]
    for c in kept:
        X[c] = meds[c].astype(str)
    X["change"] = d["change"].astype(str)
    X["diabetesMed"] = d["diabetesMed"].astype(str)

    info = dict(
        n_raw=n_raw, n_after_expired=n_alive, n_final=len(X), positive_rate=float(y.mean()),
        meds_kept=kept, first_encounter=first_encounter, drop_expired=drop_expired,
        dropped_cols=["encounter_id", "patient_nbr", "weight (≈97 % manquant)",
                      "payer_code (≈40 % manquant)", "examide/citoglipton (constantes)"]
                     + [m for m in MED_COLS if m not in kept and m not in ("examide", "citoglipton")],
    )
    return X, y, groups, info


def prepare_generic(df: pd.DataFrame, target: str, positive) -> tuple:
    """Import CSV : cible binaire = (target == positive)."""
    d = df.copy()
    y = (d[target].astype(str) == str(positive)).astype(int).rename(TARGET)
    X = d.drop(columns=[target])
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.5 * len(X):
            X = X.drop(columns=[c])  # identifiants quasi-uniques
    info = dict(n_raw=len(df), n_final=len(X), positive_rate=float(y.mean()))
    return X, y, None, info
