from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Final Data Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
RAW_FILENAME = os.getenv("RAW_FILENAME", "global_youtube_creator_data_large.csv")
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "")
PREPROCESSED_FILE = DATA_DIR / "preprocessed.parquet"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / RAW_FILENAME

    if not raw_path.exists():
        if not KAGGLE_DATASET:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Archivo fuente no encontrado: {raw_path}. "
                    "Coloca el CSV en Final/data/ o define KAGGLE_DATASET para descarga automática."
                ),
            )

        try:
            import kagglehub
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"kagglehub no está instalado: {exc}")

        try:
            dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error al descargar dataset desde Kaggle: {exc}")

        csv_candidates = list(dataset_path.glob("*.csv"))
        if not csv_candidates:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"No se encontró archivo CSV en la descarga de Kaggle: {dataset_path}. "
                    "Asegúrate de que el dataset contiene un CSV válido."
                ),
            )

        source_csv = csv_candidates[0]
        import shutil
        shutil.copy(source_csv, raw_path)

    df_raw = pd.read_csv(raw_path)

    # ── Limpieza básica ─────────────────────────────────────────────────────
    df = df_raw.drop_duplicates()
    df = df.dropna(subset=["ads_enabled"])
    df = df[df["ads_enabled"].isin([0, 1])]
    df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])

    # ── Feature engineering: log-transforms sobre métricas de engagement ───
    # Estas columnas existen en el dataset raw; las transformamos para que el
    # modelo trabaje en escala logarítmica (reduce sesgo de distribuciones Pareto)
    log_cols = ["views", "likes", "comments", "shares"]
    for col in log_cols:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            df.drop(columns=[col], inplace=True)

    # Calcular engagement_rate a partir de las columnas log si existen
    if all(c in df.columns for c in ["views_log", "likes_log", "comments_log", "shares_log"]):
        # Reconstruir valores aproximados para calcular tasa, luego re-loguear
        views = np.expm1(df["views_log"])
        likes = np.expm1(df["likes_log"])
        comments = np.expm1(df["comments_log"])
        shares = np.expm1(df["shares_log"])
        engagement_rate = (likes + comments + shares) / (views + 1)
        df["engagement_rate_log"] = np.log1p(engagement_rate)

    # Escalar duration_sec también
    if "duration_sec" in df.columns:
        df["duration_sec"] = df["duration_sec"].clip(lower=0)

    df.to_parquet(PREPROCESSED_FILE, index=False)

    return {
        "status": "ingested",
        "rows": len(df),
        "columns": df.columns.tolist(),
        "file": str(PREPROCESSED_FILE),
    }
