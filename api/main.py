"""API d'interprétation (FastAPI) — prédiction et explication SHAP / LIME d'un individu.

Lancement :  uvicorn api.main:app --port 8008      Documentation : http://localhost:8008/docs
Le modèle est lu dans artifacts/bundle.joblib (ou la variable d'environnement MODEL_BUNDLE).
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import service

BUNDLE = Path(os.getenv("MODEL_BUNDLE", service.BUNDLE_PATH))
_state: dict = {"explainer": None, "loaded_at": None, "error": None}
_lock = threading.Lock()


def _load() -> None:
    with _lock:
        try:
            b = service.load_bundle(BUNDLE)
            _state.update(explainer=service.Explainer(b), loaded_at=b["created"], error=None)
        except FileNotFoundError:
            _state.update(explainer=None, error=f"Aucun modèle trouvé ({BUNDLE}). Entraînez et exportez un modèle.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(
    title="API d'interprétation — réadmission à 30 jours (patients diabétiques)",
    version="1.0.0",
    description=("Prédit le risque de réadmission d'un patient et explique la décision (SHAP et LIME) "
                 "dans un langage compréhensible. Les variables non fournies sont remplies par la "
                 "médiane / le mode du jeu d'entraînement (voir `GET /schema`). "
                 "**Outil d'aide à la décision, non un dispositif médical.**"),
    lifespan=lifespan,
)


class Patient(BaseModel):
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Valeurs brutes des variables (clés = colonnes de `GET /schema`).",
        examples=[{"number_inpatient": 3, "age": 75, "insulin": "Up", "discharge": "Établissement de soins"}])
    threshold: float | None = Field(None, ge=0, le=1, description="Seuil de décision (défaut : celui du modèle).")


class ExplainRequest(Patient):
    top: int = Field(10, ge=3, le=40, description="Nombre de contributions renvoyées.")
    lime_samples: int = Field(1500, ge=200, le=10000, description="Échantillons de perturbation LIME.")


def _ex():
    if _state["explainer"] is None:
        raise HTTPException(503, _state["error"] or "Modèle non chargé")
    return _state["explainer"]


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


@app.get("/health", tags=["service"], summary="État du service et du modèle chargé")
def health():
    e = _state["explainer"]
    return {"status": "ok" if e else "no_model", "model": e.name if e else None,
            "threshold": e.b["threshold"] if e else None, "loaded_at": _state["loaded_at"],
            "detail": _state["error"]}


@app.post("/reload", tags=["service"], summary="Recharge le modèle exporté depuis le disque")
def reload():
    _load()
    return health()


@app.get("/schema", tags=["modèle"], summary="Variables attendues, types, bornes et valeurs par défaut")
def schema():
    return _ex().b["schema"]


@app.get("/global/importance", tags=["modèle"], summary="Importance globale SHAP par variable")
def global_importance(top: int = 15):
    e = _ex()
    return {"model": e.name, "metrics": e.b["metrics"], "importance": e.b["global_importance"][:top]}


@app.post("/predict", tags=["prédiction"], summary="Probabilité de réadmission et décision au seuil")
def predict(req: Patient):
    try:
        return _ex().predict(req.values, req.threshold)
    except ValueError as err:
        raise HTTPException(422, str(err))


@app.post("/explain/shap", tags=["explication"], summary="Contributions SHAP + phrase de synthèse")
def explain_shap(req: ExplainRequest):
    try:
        r = _ex().explain(req.values, req.threshold, req.top, with_lime=False)
    except ValueError as err:
        raise HTTPException(422, str(err))
    return _clean(r)


@app.post("/explain/lime", tags=["explication"], summary="Explication LIME locale")
def explain_lime(req: ExplainRequest):
    try:
        r = _ex().explain(req.values, req.threshold, req.top, req.lime_samples, with_lime=True)
    except ValueError as err:
        raise HTTPException(422, str(err))
    return {"probability": r["probability"], "lime": r["lime"]}


@app.post("/explain", tags=["explication"], summary="Prédiction + SHAP + LIME + cohérence + phrase")
def explain(req: ExplainRequest):
    try:
        return _clean(_ex().explain(req.values, req.threshold, req.top, req.lime_samples))
    except ValueError as err:
        raise HTTPException(422, str(err))
