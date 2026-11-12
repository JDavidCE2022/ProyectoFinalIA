"""Agente con loop manual de tool calling sobre un endpoint compatible con OpenAI (Groq por defecto)."""
import json
import os
from openai import OpenAI
from herramientas import HERRAMIENTAS, ESQUEMAS, FECHA_MAX

T_MAX = 5
MODELO = os.getenv("MODELO_CHAT", "llama-3.3-70b-versatile")
BASE_URL = os.getenv("BASE_URL_CHAT", "https://api.groq.com/openai/v1")

SISTEMA = f"""Eres AireBot, un asistente sobre la calidad del aire del Valle de Aburrá.
Responde siempre en español, de forma breve y clara para cualquier ciudadano.
Reglas:
1. Para datos medidos de PM2.5 usa get_pm25. Para el día siguiente usa predict_tomorrow. Para normativa, salud, POECA o conceptos usa search_docs. Si no sabes qué estaciones hay, usa listar_estaciones.
2. No inventes cifras. Toda cifra debe venir de una herramienta.
3. Cita las fuentes al final con el formato [id del fragmento] para documentos y [SIATA] o [Modelo del proyecto] para datos.
4. Si las herramientas no tienen la información, dilo explícitamente.
5. No des consejo médico. Ante síntomas recomienda consultar un servicio de salud.
6. Los datos disponibles van hasta {FECHA_MAX.date()}. Si el usuario dice hoy, usa esa fecha y acláralo."""


def crear_cliente(api_key=None):
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Falta GROQ_API_KEY. Defínala en el archivo .env o en la barra lateral de la app.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def ejecutar_herramienta(nombre, argumentos):
    if nombre not in HERRAMIENTAS:
        return {"error": f"La herramienta {nombre} no existe."}
    try:
        args = json.loads(argumentos or "{}")
    except json.JSONDecodeError:
        return {"error": "Los argumentos no son un JSON válido."}
    try:
        return HERRAMIENTAS[nombre](**args)
    except TypeError as e:
        return {"error": f"Argumentos inválidos para {nombre}: {e}"}


def responder(pregunta, cliente, historial=None, contexto=None):
    """Devuelve (respuesta, traza). La traza registra cada llamada a herramienta con su resultado."""
    mensajes = [{"role": "system", "content": SISTEMA}]
    if contexto:
        mensajes.append({"role": "system", "content": f"Contexto de la interfaz: {contexto}"})
    mensajes += list(historial or [])
    mensajes.append({"role": "user", "content": pregunta})
    traza = []
    for paso in range(T_MAX):
        try:
            r = cliente.chat.completions.create(model=MODELO, messages=mensajes, tools=ESQUEMAS,
                                                tool_choice="auto", temperature=0.1, max_tokens=700)
        except Exception as e:
            return f"No se pudo contactar al modelo de lenguaje: {e}", traza
        msg = r.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), traza
        mensajes.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            obs = ejecutar_herramienta(tc.function.name, tc.function.arguments)
            traza.append({"paso": paso + 1, "herramienta": tc.function.name, "argumentos": tc.function.arguments,
                          "error": obs.get("error") if isinstance(obs, dict) else None})
            # Los errores vuelven al modelo como observación para que pueda corregir
            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs, ensure_ascii=False, default=str)[:6000]})
    return "Se alcanzó el número máximo de pasos sin una respuesta final. Intente con una pregunta más concreta.", traza


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    c = crear_cliente()
    resp, tr = responder("¿Cómo estuvo el aire en Itagüí el 10 de marzo de 2024 y qué recomienda la norma?", c)
    print(resp)
    print(tr)
