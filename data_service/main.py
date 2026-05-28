from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import os

app = FastAPI(title="Final Data Service")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
RAW_FILENAME = os.getenv("RAW_FILENAME", "global_youtube_creator_data_large.csv")
PREPROCESSED_FILE = DATA_DIR / "preprocessed.parquet"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest():
    raw_path = DATA_DIR / RAW_FILENAME
    if not raw_path.exists():
        return {"status": "missing_source", "path": str(raw_path)}

    df = pd.read_csv(raw_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["ads_enabled"])
    df = df[df["ads_enabled"].isin([0, 1])]
    df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])
    df.to_parquet(PREPROCESSED_FILE, index=False)

    return {"status": "ingested", "rows": len(df), "file": str(PREPROCESSED_FILE)}
