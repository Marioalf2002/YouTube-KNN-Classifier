# 🎬 YouTube KNN Classifier — Entrega Final

Sistema de clustering/clasificación de videos de YouTube con K-Nearest Neighbors, desplegado en Docker como microservicios.

## Arquitectura

```
┌─────────────┐   POST /ingest   ┌─────────────────┐
│   Frontend  │ ──────────────── │  data-service   │ :8001
│  (nginx)    │   POST /train    │  trainer-service│ :8002
│   :3000     │ ──────────────── │  api-service    │ :8003
└─────────────┘   POST /predict  └─────────────────┘
```

## Requisitos

- Docker >= 24
- Docker Compose >= 2.20
- (Opcional) Clave API de YouTube para predicción por URL

## Inicio Rápido

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Final
```

### 2. Colocar el dataset

Opción A — Dataset incluido en `data/`:
```bash
ls data/global_youtube_creator_data_large.csv  # ya está
```

Opción B — Descargar automáticamente desde Kaggle (requiere cuenta Kaggle):
```bash
# El servicio lo descarga automáticamente al hacer /ingest
# Dataset: ehsanzx/youtube-1m-global-creator-analytics
```

### 3. Levantar los servicios

```bash
# Con clave YouTube API (para predicción por URL):
export YOUTUBE_API_KEY="tu_clave_aqui"
docker-compose up --build

# Sin clave (sólo predicción por features JSON):
docker-compose up --build
```

### 4. Acceder a la interfaz

Abrir: **http://localhost:3000**

## Flujo de Uso (Interfaz Web)

1. **Paso 1 — Ingestión**: Clic en "Ejecutar Ingesta"
   - Carga el CSV, aplica limpieza y feature engineering (log-transforms)
   - Guarda `data/preprocessed.parquet`

2. **Paso 2 — Entrenamiento**: Clic en "Entrenar Modelo"
   - Entrena KNeighborsClassifier (k=5) con 400,000 muestras
   - Guarda `models/youtube_model.joblib` y `models/feature_columns.json`
   - Recarga el modelo en el api-service automáticamente

3. **Paso 3 — Predicción**: Introduce una URL de YouTube o features JSON
   - Predice si el video tiene anuncios habilitados (1) o no (0)

## API Endpoints

| Servicio | Puerto | Endpoint | Método | Descripción |
|----------|--------|----------|--------|-------------|
| data-service | 8001 | `/ingest` | POST | Carga y preprocesa el dataset |
| data-service | 8001 | `/health` | GET | Estado del servicio |
| trainer-service | 8002 | `/train` | POST | Entrena el modelo KNN |
| trainer-service | 8002 | `/health` | GET | Estado del servicio |
| api-service | 8003 | `/predict` | POST | Predicción (URL o JSON) |
| api-service | 8003 | `/reload` | POST | Recarga el modelo desde disco |
| api-service | 8003 | `/metrics` | GET | Métricas del último entrenamiento |
| api-service | 8003 | `/health` | GET | Estado + si modelo está cargado |

### Ejemplo de predicción por features JSON

```bash
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "duration_sec": 300.0,
      "sentiment_score": 0.2,
      "views_log": 10.5,
      "likes_log": 8.3,
      "comments_log": 5.1,
      "shares_log": 4.0,
      "engagement_rate_log": 0.05,
      "category": "Gaming",
      "language": "English",
      "region": "US"
    }
  }'
```

### Ejemplo de predicción por URL de YouTube

```bash
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://youtu.be/dQw4w9WgXcQ"}'
```

## Feature Engineering

El pipeline aplica las siguientes transformaciones al dataset:

| Feature original | Transformación | Feature final |
|-----------------|----------------|---------------|
| views | log1p | views_log |
| likes | log1p | likes_log |
| comments | log1p | comments_log |
| shares | log1p | shares_log |
| (calculado) | log1p(engagement_rate) | engagement_rate_log |
| duration_sec | sin transformar | duration_sec |
| sentiment_score | sin transformar | sentiment_score |
| category | one-hot encoding | category_Gaming, etc. |
| language | one-hot encoding | language_Spanish, etc. |
| region | one-hot encoding | region_US, etc. |

## Modelo

- **Algoritmo**: K-Nearest Neighbors (scikit-learn)
- **K = 5** vecinos
- **Target**: `ads_enabled` (0 = sin anuncios, 1 = con anuncios)
- **Train/Test split**: 80% / 20%

## Notas para Sistemas con SELinux (Fedora, RHEL)

El `docker-compose.yml` usa `:z` en los bind mounts para relaquetar el contexto SELinux automáticamente. No requiere configuración adicional.
