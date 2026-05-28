# Entrega Final: Plataforma de ML basada en Microservicios

Este proyecto implementa una plataforma con arquitectura de microservicios para un pipeline de Machine Learning basado en el dataset de YouTube 1M.

## Componentes incluidos

- `data_service`: ingesta y limpieza del dataset original
- `trainer_service`: entrenamiento del modelo y guardado de métricas
- `api_service`: endpoint de inferencia REST para consumir el modelo
- `frontend`: interfaz gráfica para interactuar con el pipeline

## Flujo general

1. `data_service` carga el dataset, limpia columnas y persiste datos preprocesados en `./data`
2. `trainer_service` lee los datos preprocesados, entrena un pipeline de ML y guarda un modelo serializado en `./models`
3. `api_service` expone el modelo guardado para predicción y consulta de métricas
4. `frontend` permite al usuario realizar predicciones, ver métricas y disparar el reentrenamiento

## Requisitos previos

- Docker
- Docker Compose
- Dataset original de YouTube 1M en `Final/data/global_youtube_creator_data_large.csv`, o credenciales Kaggle configuradas para descarga con `kagglehub`

## Ejecutar la plataforma

Desde `Final/`:

```bash
cd /home/marioher/Documentos/Programacion/Metodos-de-Clustering/Final
docker-compose up --build
```

Servicios expuestos:

- Frontend: `http://localhost:3000`
- API de inferencia: `http://localhost:8003`
- Trainer: `http://localhost:8002`
- Data service: `http://localhost:8001`

## Endpoints clave

- `POST http://localhost:8003/predict`
- `GET http://localhost:8003/model-info`
- `POST http://localhost:8002/train`
- `GET http://localhost:8001/status`

## Volúmenes y persistencia

- `./data`: dataset limpio y métricas
- `./models`: modelo entrenado serializado

## Notas importantes

- Si no tienes el CSV local, el `data_service` intenta descargarlo de Kaggle usando `kagglehub`.
- Si prefieres no usar Kaggle, copia manualmente el archivo CSV a `Final/data/global_youtube_creator_data_large.csv`.
