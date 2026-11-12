"""Evalúa el agente con preguntas fijas: si llamó la herramienta esperada y si citó la fuente.
Uso: python evaluar_agente.py  (requiere GROQ_API_KEY en .env)"""
import re
import sys
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()
import agente  # noqa: E402

PREGUNTAS = [
    ("¿Cuál fue el PM2.5 en la estación de Itagüí Casa de Justicia el 10 de marzo de 2024?", "get_pm25"),
    ("¿Cómo estuvo el aire en Bello la última semana de datos?", "get_pm25"),
    ("¿Qué categoría del ICA tuvo la estación de Belén el 5 de marzo de 2024?", "get_pm25"),
    ("¿Cómo estará la calidad del aire en Itagüí el 11 de marzo de 2024 según el modelo?", "predict_tomorrow"),
    ("Predice el PM2.5 del día siguiente al 20 de febrero de 2024 en Envigado.", "predict_tomorrow"),
    ("¿Qué es el POECA?", "search_docs"),
    ("¿Cuál es el nivel máximo permisible de PM2.5 en 24 horas en Colombia?", "search_docs"),
    ("¿Qué recomienda la OMS como límite diario de PM2.5?", "search_docs"),
    ("Si el ICA está en naranja y tengo asma, ¿qué precauciones debo tomar?", "search_docs"),
    ("¿Por qué en marzo empeora la calidad del aire en el Valle de Aburrá?", "search_docs"),
    ("¿Qué estaciones de calidad del aire están disponibles?", "listar_estaciones"),
    ("¿Mañana puedo salir a trotar en Sabaneta? Usa la predicción del 3 de marzo de 2024 y las recomendaciones de salud.", "predict_tomorrow+search_docs"),
    ("¿Cuál es el costo del parqueadero del Metro de Medellín?", "ninguna"),
]

CITA = re.compile(r"\[(\d{2}_[a-z0-9_]+#\d+|SIATA|Modelo del proyecto)[^\]]*\]", re.I)


def main():
    cliente = agente.crear_cliente()
    filas = []
    for i, (q, esperada) in enumerate(PREGUNTAS, 1):
        t0 = time.time()
        resp, traza = agente.responder(q, cliente)
        usadas = [t["herramienta"] for t in traza]
        if esperada == "ninguna":
            correcta = "search_docs" in usadas or not usadas
            cito = "no" in resp.lower() and ("información" in resp.lower() or "tengo" in resp.lower())
        else:
            correcta = all(h in usadas for h in esperada.split("+"))
            cito = bool(CITA.search(resp))
        filas.append({"n": i, "pregunta": q, "herramienta_esperada": esperada, "herramientas_usadas": ", ".join(usadas) or "-",
                      "herramienta_correcta": correcta, "cito_fuente": cito,
                      "errores_herramienta": sum(1 for t in traza if t["error"]),
                      "segundos": round(time.time() - t0, 1), "respuesta": resp.replace("\n", " ")})
        print(f"{i:2d} {'OK ' if correcta else 'MAL'} cita={cito} {usadas}")
        time.sleep(2)  # margen para el límite de solicitudes de la capa gratuita
    df = pd.DataFrame(filas)
    Path("resultados").mkdir(exist_ok=True)
    df.to_csv("resultados/evaluacion_agente.csv", index=False)
    resumen = (f"Herramienta correcta: {df.herramienta_correcta.sum()}/{len(df)}  \n"
               f"Cita de fuente (o abstención en la pregunta fuera del corpus): {df.cito_fuente.sum()}/{len(df)}  \n"
               f"Modelo: {agente.MODELO}")
    tabla = df.drop(columns=["respuesta"]).to_markdown(index=False)
    respuestas = "\n\n".join(f"**{r.n}. {r.pregunta}**\n\n{r.respuesta}" for r in df.itertuples())
    Path("resultados/evaluacion_agente.md").write_text(
        f"# Evaluación del agente\n\n{resumen}\n\n{tabla}\n\n## Respuestas completas\n\n{respuestas}\n", encoding="utf-8")
    print(resumen)


if __name__ == "__main__":
    main()
