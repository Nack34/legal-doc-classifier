"""HTTP API: classify a filing and extract its key facts."""

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from classifier.extract import extract
from classifier.model import explain
from classifier.preprocess import normalize

ARTIFACTS = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "artifacts"))

LABEL_NAMES = {
    "demanda": "Demanda",
    "contestacion_demanda": "Contestación de demanda",
    "recurso_apelacion": "Recurso de apelación",
    "medida_cautelar": "Solicitud de medida cautelar",
    "ofrecimiento_prueba": "Ofrecimiento de prueba",
    "pronto_despacho": "Pedido de pronto despacho",
}

app = FastAPI(
    title="Legal Filing Classifier",
    version="1.0.0",
    description="Classifies Argentine court filings by type and extracts case number, parties, court, dates and amounts.",
)


@lru_cache
def get_model() -> Pipeline:
    path = ARTIFACTS / "model.joblib"
    if not path.exists():
        raise RuntimeError(f"Model not found at {path}. Run `python -m classifier.train` first.")
    return joblib.load(path)


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=20, max_length=50_000, description="Full text of the filing")


class Prediction(BaseModel):
    label: str
    name: str
    confidence: float


class ClassifyResponse(BaseModel):
    prediction: Prediction
    alternatives: list[Prediction]
    evidence: list[str] = Field(description="Terms in the text that supported the prediction")
    key_facts: dict
    low_confidence: bool


@app.get("/health")
def health() -> dict:
    get_model()
    return {"status": "ok"}


@app.get("/labels")
def labels() -> dict[str, str]:
    return LABEL_NAMES


@app.get("/model")
def model_info() -> dict:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    return {k: metrics[k] for k in ("best_model", "cv_scores", "test_macro_f1", "labels")}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    model = get_model()
    normalized = normalize(request.text)
    if not normalized:
        raise HTTPException(422, detail="The text has no usable words after normalization")

    probabilities = model.predict_proba([normalized])[0]
    ranked = sorted(zip(model.classes_, probabilities, strict=True), key=lambda kv: -kv[1])
    predictions = [
        Prediction(label=label, name=LABEL_NAMES.get(label, label), confidence=round(float(p), 4))
        for label, p in ranked
    ]
    best = predictions[0]
    return ClassifyResponse(
        prediction=best,
        alternatives=predictions[1:3],
        evidence=explain(model, normalized, best.label),
        key_facts=extract(request.text).to_dict(),
        # Below this, a human should review the filing before acting on the label.
        low_confidence=best.confidence < 0.6,
    )
