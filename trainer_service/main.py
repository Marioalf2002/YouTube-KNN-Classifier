from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

app = FastAPI(title="Final Trainer Service")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
PREPROCESSED_FILE = Path(os.getenv("PREPROCESSED_FILE", "/data/preprocessed.parquet"))
METRICS_FILE = DATA_DIR / "metrics.json"
MODEL_FILE = MODEL_DIR / "youtube_model.joblib"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train():
    if not PREPROCESSED_FILE.exists():
        return {"status": "missing_data", "path": str(PREPROCESSED_FILE)}

    df = pd.read_parquet(PREPROCESSED_FILE)
    X = df.drop(columns=["ads_enabled"])
    y = df["ads_enabled"]

    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"status": "trained", "accuracy": metrics["accuracy"], "model": str(MODEL_FILE), "metrics_file": str(METRICS_FILE)}
