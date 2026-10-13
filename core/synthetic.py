"""Jeu SYNTHÉTIQUE au schéma du jeu UCI #296 — pour tester l'appli hors-ligne UNIQUEMENT.

Aucun résultat obtenu sur ces données ne doit figurer dans le rapport.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n: int = 9000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ages = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]
    df = pd.DataFrame({
        "encounter_id": np.arange(1, n + 1) * 7,
        "patient_nbr": rng.integers(1, int(n * 0.75), n),
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"],
                           n, p=[.74, .19, .02, .01, .01, .03]),
        "gender": rng.choice(["Female", "Male", "Unknown/Invalid"], n, p=[.538, .461, .001]),
        "age": rng.choice(ages, n, p=[.002, .005, .02, .04, .1, .17, .22, .26, .17, .013]),
        "weight": np.where(rng.random(n) < .97, "?", "[75-100)"),
        "admission_type_id": rng.choice([1, 2, 3, 5, 6], n, p=[.53, .18, .19, .05, .05]),
        "discharge_disposition_id": rng.choice([1, 2, 3, 6, 7, 11, 22], n, p=[.58, .02, .14, .13, .01, .02, .10]),
        "admission_source_id": rng.choice([1, 4, 7, 17], n, p=[.29, .06, .56, .09]),
        "time_in_hospital": rng.integers(1, 15, n),
        "payer_code": np.where(rng.random(n) < .4, "?", "MC"),
        "medical_specialty": rng.choice(["?", "InternalMedicine", "Cardiology", "Surgery-General",
                                         "Family/GeneralPractice", "Emergency/Trauma", "Orthopedics"],
                                        n, p=[.49, .14, .05, .03, .07, .08, .14]),
        "num_lab_procedures": np.clip(rng.normal(43, 19, n), 1, 130).astype(int),
        "num_procedures": rng.choice(range(7), n, p=[.46, .2, .12, .09, .04, .05, .04]),
        "num_medications": np.clip(rng.normal(16, 8, n), 1, 80).astype(int),
        "number_outpatient": rng.poisson(.4, n),
        "number_emergency": rng.poisson(.2, n),
        "number_inpatient": rng.poisson(.65, n),
        "diag_1": rng.choice(["414", "428", "250.8", "486", "571", "820", "715", "599", "V57", "153", "786"], n),
        "diag_2": rng.choice(["250", "427", "276", "496", "599", "?"], n),
        "diag_3": rng.choice(["250", "401", "272", "V45", "?"], n),
        "number_diagnoses": np.clip(rng.normal(7.4, 1.9, n), 1, 16).astype(int),
        "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"], n, p=[.95, .015, .015, .02]),
        "A1Cresult": rng.choice(["None", ">7", ">8", "Norm"], n, p=[.83, .08, .08, .01]),
    })
    for m, p_no in {"metformin": .8, "repaglinide": .985, "nateglinide": .993, "chlorpropamide": .999,
                    "glimepiride": .95, "acetohexamide": 1, "glipizide": .88, "glyburide": .9,
                    "tolbutamide": 1, "pioglitazone": .93, "rosiglitazone": .94, "acarbose": .997,
                    "miglitol": 1, "troglitazone": 1, "tolazamide": 1, "examide": 1, "citoglipton": 1,
                    "insulin": .46, "glyburide-metformin": .993, "glipizide-metformin": 1,
                    "glimepiride-pioglitazone": 1, "metformin-rosiglitazone": 1,
                    "metformin-pioglitazone": 1}.items():
        rest = (1 - p_no)
        df[m] = rng.choice(["No", "Steady", "Up", "Down"], n,
                           p=[p_no, rest * .7, rest * .15, rest * .15])
    df["change"] = np.where((df[["insulin", "metformin"]] != "No").any(axis=1) & (rng.random(n) < .5), "Ch", "No")
    df["diabetesMed"] = np.where((df[["insulin", "metformin", "glipizide"]] != "No").any(axis=1), "Yes", "No")

    age_mid = df["age"].str[1:].str.split("-").str[0].astype(int) + 5
    logit = (-3.3 + .38 * df["number_inpatient"] + .18 * df["number_emergency"]
             + .035 * df["time_in_hospital"] + .09 * (df["number_diagnoses"] - 7)
             + .012 * (age_mid - 65) + .02 * (df["num_medications"] - 16)
             + .45 * (df["discharge_disposition_id"].isin([3, 22]))
             + .25 * (df["insulin"] == "Down") + .12 * (df["diag_1"] == "428")
             + rng.normal(0, .6, n))
    p = 1 / (1 + np.exp(-logit))
    r = rng.random(n)
    df["readmitted"] = np.where(r < p, "<30", np.where(r < p + .3, ">30", "NO"))
    return df
