# 🎬 Análisis y Predicción de YouTube con Machine Learning: Arquitectura y Documentación Técnica

Este documento presenta una explicación completa del **Proyecto Final**, detallando las tecnologías utilizadas, la arquitectura del sistema, el flujo de comunicación, el pipeline de Machine Learning y el rol de cada contenedor dentro del ecosistema MLOps implementado.

El sistema se compone de los siguientes **4 contenedores**:

### 1. `final_frontend` (Puerto 3000)

- **¿Qué es?**: Un contenedor web ligero basado en `nginx:stable-alpine`.
- **¿Qué hace?**: Sirve la interfaz gráfica de usuario (UI) creada en HTML5, CSS3 y Vanilla Javascript. Actúa como el punto de interacción principal, permitiendo a los usuarios gatillar la limpieza de datos, entrenar el modelo y realizar predicciones sin tocar código.
- **Comunicación**: Se comunica con los servicios backend mediante peticiones HTTP (AJAX/Fetch) asíncronas hacia las APIs expuestas en los puertos `8001`, `8002` y `8003`.

### 2. `final_data_service` (Puerto 8001)

- **¿Qué es?**: Un microservicio desarrollado en Python utilizando FastAPI.
- **¿Qué hace?**: Se encarga del primer paso del MLOps: la **ingesta y el preprocesamiento de datos**. Lee el dataset bruto en CSV (datos analíticos de creadores de YouTube), aplica limpieza, descarta anomalías y transforma variables numéricas.
- **Salida**: Escribe los datos limpios en un formato binario optimizado llamado `Parquet` dentro del volumen compartido `/data`.

### 3. `final_trainer_service` (Puerto 8002)

- **¿Qué es?**: Microservicio de entrenamiento también desarrollado en Python (FastAPI + Scikit-Learn).
- **¿Qué hace?**: Consume el archivo `.parquet` generado por el servicio de datos, extrae las características (features), divide la data y entrena el modelo de clasificación.
- **Diferencia con el API Service**: Este contenedor **solo existe para aprender**. Su trabajo es "estudiar" los datos masivos para crear el algoritmo matemático. No atiende a los usuarios finales ni predice videos individuales, solo fabrica el modelo y lo guarda en disco.
- **Salida**: Genera y guarda el modelo serializado (`.joblib`), el mapeo de columnas y las métricas de entrenamiento (`metrics.json`) en los volúmenes `/models` y `/data`.

### 4. `final_api_service` (Puerto 8003)

- **¿Qué es?**: El servicio de inferencia en tiempo real (FastAPI).
- **¿Qué hace?**: Es el "cerebro" productivo en acción. Carga en memoria el modelo entrenado `.joblib` y expone el endpoint `/predict`. Al recibir una URL de YouTube (o un JSON con estadísticas), procesa la información para que coincida con el formato del modelo y devuelve la predicción (si el video es apto para monetización o tiene anuncios: `1` o `0`).
- **Diferencia con el Trainer Service**: Este contenedor **no entrena ni aprende nada nuevo**. Su único trabajo es "aplicar" lo que el Trainer ya aprendió. Actúa como el cajero o vendedor final: toma el `.joblib` fabricado previamente y lo utiliza para responder a las consultas de predicción en fracciones de segundo.
- **Salida**: Responde al frontend y persiste un historial de predicciones en disco.

---

## 💻 Tecnologías Utilizadas

### Backend y Lenguajes

- **Python 3**: Lenguaje central de todo el MLOps y lógica de servidor. Elegido por su vasto ecosistema en ciencias de datos.
- **FastAPI / Uvicorn**: Framework web de Python de altísimo rendimiento para construir las APIs REST.
- **Pandas y NumPy**: Librerías de manipulación matricial y estructuras de datos para transformar el dataset.
- **Scikit-Learn**: La librería principal de Machine Learning utilizada para entrenar el algoritmo y extraer métricas.

### Frontend

- **Nginx**: Servidor web estático rápido.
- **HTML5, CSS3, Vanilla JS**: Desarrollo nativo sin frameworks pesados (como React) para mantener el frontend rápido e interactivo, con un diseño moderno.

### Base de Datos / Almacenamiento Persistente

En lugar de una base de datos relacional tradicional (como PostgreSQL), el sistema utiliza un enfoque de **Data Lake local (Almacenamiento por archivos en Volúmenes de Docker compartidos)**:

- **CSV (`global_youtube_creator_data_large.csv`)**: 
  - *El origen bruto de los datos*. Es un archivo de texto plano separado por comas masivo (ej. de más de 1 millón de registros).
  - *¿Cómo se lee?*: El contenedor `data_service` lo carga en la memoria RAM utilizando la librería `pandas`. Los archivos CSV son universales pero computacionalmente "lentos" y pesados de procesar, ya que el sistema debe leer el texto línea por línea e inferir si una columna es texto, número, etc. Por ello, solo se lee de forma obligatoria en la fase inicial de Ingesta.
- **Parquet (`preprocessed.parquet`)**: 
  - *El puente optimizado*. Formato columnar binario altamente comprimido usado para pasar datos entre el _Data Service_ y el _Trainer Service_.
  - *¿Por qué se usa en lugar del CSV?*: Al ser columnar, es muchísimo más veloz de leer y ocupa mucha menos memoria. Este archivo contiene la data ya "limpia" (sin nulos, sin duplicados) y con las fórmulas aplicadas. Funciona como un "atajo" de altísimo rendimiento para que el `trainer_service` reciba los datos directamente estructurados, ahorrando la necesidad de repetir el pesado proceso de parsear el CSV cada vez que se requiere reentrenar el modelo.
- **JSON / Joblib**: Persistencia ligera. Se usa para guardar registros de auditoría (historial de predicciones), las métricas del modelo (`metrics.json`) y el modelo matemático congelado en sí (`youtube_model.joblib`), compartidos a través de los volúmenes de Docker.

---

## 🔄 Flujo de Comunicación (El Ciclo de Vida del Dato)

Para que el sistema funcione de manera automatizada y robusta, los datos viajan a través de una tubería o _Pipeline_ muy bien definido. Este flujo es unidireccional (los datos entran crudos y salen como una predicción) y se divide en tres grandes etapas operativas:

1. **La Ingesta de Datos (Data Ingestion)**
   - **¿Qué significa?**: En el mundo del Machine Learning, "ingestar" es el acto de absorber datos crudos y caóticos desde una fuente externa (un archivo CSV, una base de datos o una API) hacia el sistema interno para comenzar a trabajarlos. Es como recolectar la materia prima antes de cocinar.
   - **El flujo**: Cuando el usuario hace clic en "Ejecutar Ingesta" en el Frontend, se envía una señal HTTP (`POST /ingest`) al `data-service`. Este servicio toma el gigantesco archivo `global_youtube_creator_data_large.csv`, lo limpia, le aplica fórmulas matemáticas complejas (feature engineering) y lo convierte en el archivo `preprocessed.parquet`, dejándolo "masticado" y listo en el disco duro virtual.

2. **El Entrenamiento (Model Training)**
   - **¿Qué significa?**: Es la fase donde la "Inteligencia Artificial" realmente aprende. Toma la materia prima ya preparada y busca patrones matemáticos para saber diferenciar un video monetizado de uno que no lo está.
   - **El flujo**: Al hacer clic en "Entrenar Modelo", el Frontend hace un `POST /train` al `trainer-service`. Este contenedor despierta, lee el archivo `preprocessed.parquet`, entrena el modelo usando el algoritmo KNN, y finalmente guarda el "cerebro" resultante en un archivo congelado llamado `youtube_model.joblib`. Responde al frontend con la precisión y el F1-Score alcanzados.

3. **La Inferencia o Predicción (Model Serving)**
   - **¿Qué significa?**: Es la puesta en producción del modelo. Ya aprendió, ahora toca ponerlo a prueba en el mundo real con videos que nunca ha visto.
   - **El flujo**: El usuario ingresa una URL de YouTube en el Frontend. Se envía un `POST /predict` al `api-service`. Este servicio descarga la información actual de ese video desde los servidores de Google (visitas, likes, categoría), _le aplica exactamente las mismas transformaciones matemáticas que en la Ingesta_, y se lo pasa al cerebro (`youtube_model.joblib`) para que dicte su veredicto. Finalmente, el resultado se muestra en pantalla y se guarda en un historial de auditoría.

---

## 🧠 Pipeline de Machine Learning a Detalle

El objetivo principal de nuestro algoritmo es resolver un problema de **Clasificación Binaria**: decidir si un video tiene los anuncios habilitados (target: `ads_enabled = 1`) o deshabilitados (`ads_enabled = 0`).

Para lograrlo, la data pasa por un cuidadoso proceso de orfebrería:

### 1. Ingesta y Preprocesamiento de Datos (Feature Engineering)

La Inteligencia Artificial es tan buena como los datos que se le entregan ("Garbage in, garbage out"). Por ello, antes de entregar los datos a la IA, el _data-service_ aplica estas rigurosas reglas:

- **Limpieza Básica**: Se descartan las filas duplicadas y se eliminan todos aquellos registros donde la variable objetivo (`ads_enabled`) viene vacía o corrupta. El modelo no puede aprender de ejemplos inciertos.
- **Transformación Logarítmica (`log1p`)**:
  - _El Problema_: En YouTube rige la "Ley de Pareto". Un puñado de videos tienen decenas de millones de visualizaciones, mientras que la inmensa mayoría apenas supera las miles. Esto crea una gráfica asimétrica que confunde a la IA porque las distancias matemáticas se vuelven inmensas.
  - _La Solución_: A variables como Vistas (`views`), Likes (`likes`), Comentarios (`comments`) y Compartidos (`shares`) se les aplica una **función matemática logarítmica (`np.log1p`)**. Esto "suaviza" o "comprime" los números astronómicos, permitiendo al modelo comparar un video viral con uno pequeño bajo la misma lente.
- **Creación de Tasa de Engagement**: En lugar de solo mirar los "likes", se creó una nueva métrica avanzada llamada `engagement_rate = (likes + comments + shares) / (views + 1)`. Esto le permite al modelo saber qué tan "involucrada" está la audiencia, independientemente de si el video tiene mil o un millón de vistas.

### 2. Entrenamiento del Modelo y Algoritmo

El _trainer-service_ recibe los datos procesados y realiza lo siguiente:

- **One-Hot Encoding**: Las variables de texto como "Categoría" (Gaming, Vlogs) y "Región" se convierten en múltiples columnas de 0s y 1s (`get_dummies`), ya que la IA solo entiende números.
- **Data Split (División de Datos)**: Se dividen los datos; el **80%** se usa para _entrenar_ a la IA y el **20%** restante se oculta para _evaluar_ qué tan bien aprendió. (Se utiliza submuestreo estratificado hasta 400,000 muestras si el dataset es inmenso).
- **El Algoritmo: KNN (K-Nearest Neighbors)**:
  - Se utilizó el algoritmo **Clasificador K-Vecinos Más Cercanos**.
  - **¿Cómo funciona?**: Al intentar predecir un nuevo video, el algoritmo ubica las estadísticas del video en un espacio multidimensional y busca los **K=5** videos más cercanos a él ("sus vecinos"). Si la mayoría de sus 5 vecinos más parecidos tienen anuncios activados, asume que este nuevo video también los tiene.

### 3. Métricas de Evaluación

Para comprobar que el modelo no adivina al azar, se utilizan dos métricas clave de evaluación de Machine Learning que se devuelven al frontend:

1. **Accuracy (Exactitud)**:
   - _¿Qué es?_: El porcentaje total de aciertos. (Ej. "De 100 predicciones, acerté 90, mi Accuracy es del 90%").
   - Es útil para tener una visión rápida del rendimiento general.

2. **F1-Score**:
   - _¿Qué es?_: Es la media armónica entre la _Precisión_ (qué tan confiables son sus aciertos) y el _Recall_ (cuántos positivos reales encontró).
   - _¿Por qué se usa?_: En datasets de YouTube, a veces hay un fuerte desbalance (ej. 90% de los videos sí tienen anuncios). Si la IA dice siempre "SÍ", tendría un 90% de Accuracy mintiendo. El **F1-Score** penaliza esto y nos da la métrica _real_ de la calidad del modelo balanceando los falsos positivos y falsos negativos.

---

## 📡 Pruebas a la API (Comandos cURL)

Puedes probar todos los endpoints del sistema directamente desde tu terminal utilizando `curl`. Esto es especialmente útil para automatizaciones o verificar que los servicios están funcionando correctamente sin usar el frontend.

### 1. Iniciar la Ingesta de Datos (`POST /ingest`)

Este comando ordena al Data Service que lea el archivo CSV, aplique la limpieza y cree el `.parquet`.

```bash
curl -X POST http://localhost:8001/ingest
```

### 2. Entrenar el Modelo (`POST /train`)

Le indica al Trainer Service que lea el `.parquet` procesado y entrene el modelo KNN.

```bash
curl -X POST http://localhost:8002/train
```

### 3. Obtener las Métricas (`GET /metrics`)

Devuelve la información del entrenamiento (Accuracy, F1-Score, tamaño del dataset).

```bash
curl -X GET http://localhost:8003/metrics
```

### 4. Hacer una Predicción con Enlace de YouTube (`POST /predict`)

```bash
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://youtu.be/dQw4w9WgXcQ"}'
```

### 5. Hacer una Predicción con Variables Manuales JSON (`POST /predict`)

```bash
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "duration_sec": 300.0,
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

### 6. Ver Historial de Predicciones (`GET /predictions`)

Devuelve el JSON con todas las últimas predicciones persistidas en el disco duro.

```bash
curl -X GET http://localhost:8003/predictions
```

### 7. Comprobación de Salud / Health Checks (`GET /health`)

Verifica que cada uno de los microservicios esté encendido y recibiendo peticiones.

```bash
# Revisar Data Service
curl -X GET http://localhost:8001/health

# Revisar Trainer Service
curl -X GET http://localhost:8002/health

# Revisar API Service
curl -X GET http://localhost:8003/health
```
