#!/usr/bin/env python3
import kagglehub
import os
import shutil
import pandas as pd

print("Descargando dataset de Kaggle...")
dataset_path = kagglehub.dataset_download('ehsanzx/youtube-1m-global-creator-analytics')
print(f"Dataset descargado en: {dataset_path}")

csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
if not csv_files:
    print("ERROR: No se encontró CSV en el dataset")
    exit(1)

source = os.path.join(dataset_path, csv_files[0])
dest = './data/global_youtube_creator_data_large.csv'
os.makedirs('./data', exist_ok=True)

print(f"Copiando {csv_files[0]}...")
shutil.copy(source, dest)
print(f"✓ CSV guardado en: {dest}")

print("Examinando dataset...")
df = pd.read_csv(dest, nrows=50000)
print(f"Shape (primeras 50k): {df.shape}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nValores únicos por columna:")
for col in df.columns:
    nunique = df[col].nunique()
    dtype = df[col].dtype
    print(f"  {col:<20} {nunique:>10} únicos | dtype: {dtype}")
