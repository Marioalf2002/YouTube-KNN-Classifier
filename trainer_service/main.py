from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import json

app = FastAPI(title="Final Trainer Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
PREPROCESSED_FILE = Path(os.getenv("PREPROCESSED_FILE", "/data/preprocessed.parquet"))
METRICS_FILE = DATA_DIR / "metrics.json"
MODEL_FILE = MODEL_DIR / "youtube_model.joblib"
FEATURE_COLUMNS_FILE = MODEL_DIR / "feature_columns.json"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train():
    if not PREPROCESSED_FILE.exists():
        return {"status": "missing_data", "path": str(PREPROCESSED_FILE)}

    df = pd.read_parquet(PREPROCESSED_FILE)

    if "ads_enabled" not in df.columns:
        return {
            "status": "missing_target",
            "detail": "La columna ads_enabled no existe en los datos preprocesados.",
        }

    # Submuestreo estratificado si el dataset es muy grande
    if len(df) > 400000:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=400000, random_state=42)
        sample_index, _ = next(splitter.split(df, df["ads_enabled"]))
        df = df.iloc[sample_index].reset_index(drop=True)

    # Separar target
    y = df["ads_enabled"]

    # Excluir columnas que no son features: target y video_id si existe
    cols_to_drop = ["ads_enabled"]
    if "video_id" in df.columns:
        cols_to_drop.append("video_id")
    X = df.drop(columns=cols_to_drop)

    # Codificación de variables categóricas
    X_transformed = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "feature_columns": X_transformed.columns.tolist(),
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_type": "KNeighborsClassifier",
        "n_neighbors": 5,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with FEATURE_COLUMNS_FILE.open("w", encoding="utf-8") as f:
        json.dump(X_transformed.columns.tolist(), f, indent=2)

    return {
        "status": "trained",
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "model": str(MODEL_FILE),
        "metrics_file": str(METRICS_FILE),
        "feature_columns_file": str(FEATURE_COLUMNS_FILE),
        "sample_size": metrics["sample_size"],
        "train_size": metrics["train_size"],
        "test_size": metrics["test_size"],
    }
