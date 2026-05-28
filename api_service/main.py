from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
import re
from urllib.parse import urlparse, parse_qs
from typing import Optional

app = FastAPI(title="Final API Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
METRICS_FILE = Path(os.getenv("METRICS_FILE", "/data/metrics.json"))
MODEL_FILE = MODEL_DIR / "youtube_model.joblib"
FEATURE_COLUMNS_FILE = MODEL_DIR / "feature_columns.json"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")


class PredictionRequest(BaseModel):
    features: Optional[dict] = None
    youtube_url: Optional[str] = None
    youtube_api_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(self):
        has_features = bool(self.features)
        has_url = bool(self.youtube_url)
        if has_features == has_url:
            raise ValueError("Enviar exactamente uno de los campos: features o youtube_url.")
        return self


def _parse_iso8601_duration_to_seconds(duration: str) -> int:
    if not duration:
        return 0
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    match = pattern.match(duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url.strip())

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/")

    if parsed.netloc in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2]

    return ""


def _fetch_youtube_raw(video_id: str, api_key: str) -> dict:
    videos_endpoint = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": api_key,
    }
    response = requests.get(videos_endpoint, params=params, timeout=20)
    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=f"YouTube API error: {response.status_code} {response.text[:200]}",
        )

    payload = response.json()
    items = payload.get("items", [])
    if not items:
        raise HTTPException(
            status_code=404,
            detail="YouTube API no devolvió datos para ese video_id.",
        )

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})

    # Obtener nombre de categoría
    category_id = snippet.get("categoryId", "")
    category_name = "Unknown"
    if category_id:
        cat_response = requests.get(
            "https://www.googleapis.com/youtube/v3/videoCategories",
            params={"part": "snippet", "id": category_id, "regionCode": "US", "key": api_key},
            timeout=20,
        )
        if cat_response.ok:
            cat_items = cat_response.json().get("items", [])
            if cat_items:
                category_name = cat_items[0].get("snippet", {}).get("title", "Unknown")

    duration_sec = float(_parse_iso8601_duration_to_seconds(content.get("duration", "")))
    views = int(stats.get("viewCount", 0) or 0)
    likes = int(stats.get("likeCount", 0) or 0)
    comments = int(stats.get("commentCount", 0) or 0)
    shares = 0  # La API de YouTube no expone shares públicamente
    language = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or "Unknown"
    region = "Unknown"

    # Calcular engagement_rate antes de log-transform
    engagement_rate = (likes + comments + shares) / (views + 1)

    # ── Feature engineering igual que data_service ───────────────────────────
    return {
        "duration_sec": duration_sec,
        "sentiment_score": 0.0,
        "views_log": float(np.log1p(views)),
        "likes_log": float(np.log1p(likes)),
        "comments_log": float(np.log1p(comments)),
        "shares_log": float(np.log1p(shares)),
        "engagement_rate_log": float(np.log1p(engagement_rate)),
        "category": str(category_name),
        "language": str(language),
        "region": str(region),
    }


def _build_feature_row(raw: dict) -> pd.DataFrame:
    """Construye un DataFrame con exactamente las columnas que espera el modelo."""
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, drop_first=True)
    feature_columns = app.state.feature_columns
    # Agregar columnas faltantes con 0 (categorías no vistas en entrenamiento)
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0
    return df.reindex(columns=feature_columns, fill_value=0)


def load_model():
    if MODEL_FILE.exists() and FEATURE_COLUMNS_FILE.exists():
        app.state.model = joblib.load(MODEL_FILE)
        with FEATURE_COLUMNS_FILE.open("r", encoding="utf-8") as f:
            app.state.feature_columns = json.load(f)
    else:
        app.state.model = None
        app.state.feature_columns = []


@app.on_event("startup")
def startup_event():
    load_model()


def ensure_model_loaded():
    if app.state.model is None:
        load_model()
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no cargado. Ejecuta primero: "
                "1) POST /ingest en data-service (puerto 8001), "
                "2) POST /train en trainer-service (puerto 8002)."
            ),
        )
    return app.state.model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.get("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Train the model first.")
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/reload")
def reload_model():
    """Recarga el modelo desde disco (útil después de entrenar sin reiniciar el contenedor)."""
    load_model()
    return {"status": "reloaded", "model_loaded": app.state.model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    model = ensure_model_loaded()
    source = None

    if request.youtube_url is not None:
        api_key = request.youtube_api_key or YOUTUBE_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Falta YOUTUBE_API_KEY para procesar la URL de YouTube.",
            )
        video_id = _extract_video_id(request.youtube_url)
        if not video_id:
            raise HTTPException(
                status_code=400,
                detail="No se pudo extraer video_id desde la URL proporcionada.",
            )
        raw = _fetch_youtube_raw(video_id, api_key)
        feature_row = _build_feature_row(raw)
        source = "youtube_url"
    else:
        feature_row = pd.DataFrame([request.features or {}])
        feature_row = pd.get_dummies(feature_row, drop_first=True)
        for column in app.state.feature_columns:
            if column not in feature_row.columns:
                feature_row[column] = 0
        feature_row = feature_row.reindex(columns=app.state.feature_columns, fill_value=0)
        source = "features"

    try:
        prediction = model.predict(feature_row)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {exc}")

    response = {
        "prediction": int(prediction[0]),
        "prediction_label": "Anuncios habilitados" if int(prediction[0]) == 1 else "Anuncios deshabilitados",
        "source": source,
        "feature_row": feature_row.iloc[0].to_dict(),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_row)
        response["probability_ads_enabled"] = float(proba[:, 1][0])

    return response
