from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import os
import pandas as pd
import json

app = FastAPI(title="Final API Service")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
METRICS_FILE = Path(os.getenv("METRICS_FILE", "/data/metrics.json"))
MODEL_FILE = MODEL_DIR / "youtube_model.joblib"

class PredictionRequest(BaseModel):
    features: dict

@app.on_event("startup")
def load_model():
    if MODEL_FILE.exists():
        app.state.model = joblib.load(MODEL_FILE)
    else:
        app.state.model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}

@app.get("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/predict")
def predict(request: PredictionRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    df = pd.DataFrame([request.features])
    try:
        prediction = app.state.model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"prediction": int(prediction[0])}
