#!/usr/bin/env python3
"""
Crea un dataset de ejemplo con estructura compatible con el modelo.
Ejecutar: python3 create_sample_dataset.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Crear directorio
os.makedirs('./data', exist_ok=True)

print("Creando dataset de ejemplo...")

# Parámetros
n_samples = 400000
categories = ['Entertainment', 'Music', 'Gaming', 'Education', 'Sports']
languages = ['English', 'Spanish', 'Hindi', 'Portuguese', 'French']
regions = ['US', 'IN', 'BR', 'MX', 'GB']

# Crear datos
data = {
    'video_id': [f'vid_{i:08d}' for i in range(n_samples)],
    'duration_sec': np.random.uniform(60, 3600, n_samples),
    'views': np.random.exponential(5000, n_samples).astype(int),
    'likes': np.random.exponential(500, n_samples).astype(int),
    'comments': np.random.exponential(100, n_samples).astype(int),
    'shares': np.random.exponential(50, n_samples).astype(int),
    'sentiment_score': np.random.uniform(-1, 1, n_samples),
    'category': np.random.choice(categories, n_samples),
    'language': np.random.choice(languages, n_samples),
    'region': np.random.choice(regions, n_samples),
    'ads_enabled': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
}

df = pd.DataFrame(data)

# Guardar
output_file = './data/global_youtube_creator_data_large.csv'
df.to_csv(output_file, index=False)

print(f"✓ Dataset de ejemplo creado: {output_file}")
print(f"  Shape: {df.shape}")
print(f"  Columnas: {list(df.columns)}")
print(f"\nDistribución de ads_enabled:")
print(df['ads_enabled'].value_counts())
print(f"\nVerificación:")
print(f"  - Sin valores NaN: {df.isnull().sum().sum() == 0}")
print(f"  - Tamaño del archivo: {os.path.getsize(output_file) / (1024**2):.1f} MB")
