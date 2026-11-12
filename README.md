# AireBot: agente de calidad del aire del Valle de Aburrá

Proyecto final de SI3003 Inteligencia Artificial, Universidad EAFIT, 2026.

Los ciudadanos ven a diario el PM2.5 o el color del ICA, pero no siempre saben qué significa ni qué hacer. Este proyecto combina un modelo clásico que predice el PM2.5 del día siguiente por estación con un agente LLM que consulta los datos, usa la predicción y responde con base en la normativa colombiana y en recomendaciones de salud, citando sus fuentes. Todo se presenta en un dashboard interactivo.

| Componente | Qué se usó |
|---|---|
| Datos reales | PM2.5 horario del SIATA (17 estaciones con datos, jul 2023 a jul 2024) y clima diario de Open-Meteo |
| Parte 1 | Regresión Ridge y Naive Bayes categórico implementados desde cero, con validación temporal contra persistencia y clase mayoritaria |
| Parte 2 | Agente con loop manual de tool calling (Groq), RAG híbrido (embeddings multilingües + BM25 con RRF) y 4 herramientas |
| Dashboard | Streamlit con mapa, serie de tiempo, perfil horario, métricas del modelo y chat |

## Cómo ejecutarlo

Requisitos: Python 3.10 o superior y una clave gratuita de Groq (https://console.groq.com/keys).

```bash
git clone https://github.com/JDavidCE2022/ProyectoFinalIA.git
cd ProyectoFinalIA
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux o Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Copie `.env.example` como `.env` y escriba su clave en `GROQ_API_KEY`. El archivo `.env` está en `.gitignore` y nunca se sube. También puede pegar la clave en la barra lateral de la app.

```bash
streamlit run app.py
```

La app abre en http://localhost:8501. El modelo entrenado y los datos procesados ya están en el repositorio, así que la app funciona sin pasos previos. La primera vez que el agente busca en los documentos descarga el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` (unos 470 MB). Si no hay conexión o falla la descarga, la búsqueda sigue funcionando solo con BM25.

### Reproducir los resultados

```bash
python src/datos.py        # construye data/processed/dataset_modelo.csv
python src/modelo.py       # validación temporal, métricas y modelo final
python evaluar_agente.py   # 13 preguntas de prueba al agente (requiere la clave)
```

El notebook `notebooks/01_datos_y_modelo.ipynb` contiene el análisis exploratorio, la verificación de la implementación y la evaluación del modelo, ya ejecutados.

## Estructura

```
app.py                     dashboard en Streamlit
evaluar_agente.py          evaluación del agente con preguntas fijas
src/datos.py               carga y limpieza de SIATA y Open-Meteo
src/modelo.py              Ridge y Naive Bayes desde cero, validación temporal
src/rag.py                 chunking y búsqueda híbrida con RRF
src/herramientas.py        herramientas del agente
src/agente.py              loop agéntico con tool calling
data/raw/                  datos descargados (SIATA y Open-Meteo)
data/processed/            dataset, predicciones de validación, métricas y modelo
data/corpus/               11 documentos de normativa y salud con su fuente
notebooks/                 análisis del modelo de la Parte 1
resultados/                evaluación del agente
INFORME.md / INFORME.pdf   informe del proyecto
```

## Datos y fuentes

PM2.5: archivo público del SIATA `https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json`, descargado el 11 de noviembre de 2026. Contiene datos horarios del 31 de julio de 2023 al 31 de julio de 2024. Se conservan los datos con indicador de calidad menor a 2.6 y el promedio diario se calcula con al menos 18 horas válidas. Copia en `data/raw/siata_pm25_horario.json`.

Clima: Open-Meteo Historical Weather API (reanálisis ERA5) para Medellín, `https://archive-api.open-meteo.com/v1/archive?latitude=6.2518&longitude=-75.5636&start_date=2023-07-31&end_date=2024-07-31&daily=precipitation_sum,wind_speed_10m_mean,wind_speed_10m_max,temperature_2m_mean,relative_humidity_2m_mean,shortwave_radiation_sum&timezone=America%2FBogota&format=csv`.

Corpus RAG: textos redactados por el equipo a partir de la Resolución 2254 de 2017 del Ministerio de Ambiente, la página de calidad del aire de la Alcaldía de Medellín, el POECA del Área Metropolitana, un boletín de Cornare y las guías de la OMS 2021. Cada documento indica su fuente y URL.

Solo se usan datos ambientales agregados, sin información de personas.

## Resultados principales

Validación temporal mes a mes de noviembre de 2023 a julio de 2024 (4198 predicciones estación y día):

| Modelo | MAE (µg/m³) | Exactitud | F1 macro | Recall Dañina |
|---|---|---|---|---|
| Ridge + umbrales ajustados | 3.41 | 0.804 | 0.676 | 0.692 |
| Naive Bayes | | 0.801 | 0.550 | 0.297 |
| Persistencia (línea base) | 3.61 | 0.818 | 0.685 | 0.604 |
| Clase mayoritaria (línea base) | | 0.786 | 0.293 | 0.000 |

La evaluación del agente está en `resultados/evaluacion_agente.md`.

## Limitaciones

Es una herramienta académica. No es el pronóstico oficial del SIATA ni consejo médico. Hay un solo año de datos y pocos días en categoría Dañina. El clima es de un punto de Medellín para todo el valle. El modelo no usa pronóstico meteorológico del día siguiente, por lo que en la temporada crítica no supera a la persistencia. El agente depende de la disponibilidad de la API de Groq y de su límite gratuito de solicitudes.

## Declaración de uso de IA generativa

Se usó Claude (Anthropic) como asistente de programación y redacción para: buscar y descargar las fuentes de datos, proponer la estructura del proyecto, escribir y depurar el código de `src/`, `app.py` y los scripts, redactar los documentos del corpus a partir de las fuentes citadas, y redactar este README y el informe. Las decisiones del proyecto, la revisión de los resultados y la verificación de que el código corre son responsabilidad del autor. Durante el desarrollo se descartaron propuestas que no funcionaron: el primer modelo Naive Bayes quedó por debajo de la persistencia, lo que llevó a implementar la regresión Ridge, y una primera versión del análisis afirmaba que el modelo mejoraba más en temporada crítica, lo que se corrigió al revisar la tabla por mes.

El agente del proyecto usa el modelo `llama-3.3-70b-versatile` a través de la API de Groq, y la búsqueda semántica usa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
