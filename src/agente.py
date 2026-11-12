"""Agente con loop manual de tool calling sobre un endpoint compatible con OpenAI (Groq por defecto)."""
import json
import os
import re
import time
import requests
from herramientas import HERRAMIENTAS, ESQUEMAS, FECHA_MAX

T_MAX = 5
MODELO = os.getenv("MODELO_CHAT", "").strip()
# Orden de preferencia si no se fija MODELO_CHAT. Se usa el primero que la clave tenga disponible.
PREFERIDOS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
              "qwen/qwen3.8-27b", "llama-3.1-8b-instant"]
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


class ClienteChat:
    """Cliente mínimo para un endpoint /chat/completions compatible con OpenAI, usando solo requests."""

    def __init__(self, api_key, base_url=BASE_URL, timeout=60):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def modelos(self):
        r = requests.get(self.url.replace("/chat/completions", "/models"), headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def completar(self, reintentos=4, **cuerpo):
        for intento in range(reintentos + 1):
            r = requests.post(self.url, headers=self.headers, json=cuerpo, timeout=self.timeout)
            if r.status_code == 429 and intento < reintentos:
                # Límite de la capa gratuita: se espera lo que indica el servidor y se reintenta
                m = re.search(r"try again in ([\d.]+)s", r.text)
                time.sleep(min(float(m.group(1)) + 1 if m else 5 * (intento + 1), 30))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]


def crear_cliente(api_key=None):
    key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Falta GROQ_API_KEY. Defínala en el archivo .env o en la barra lateral de la app.")
    return ClienteChat(key)


def elegir_modelo(cliente):
    """Usa MODELO_CHAT si está definido. Si no, el primer modelo preferido disponible para la clave."""
    if MODELO:
        return MODELO
    try:
        disponibles = cliente.modelos()
    except Exception:
        return PREFERIDOS[0]
    for m in PREFERIDOS:
        if m in disponibles:
            return m
    chat = [m for m in disponibles if not any(x in m for x in ("whisper", "tts", "orpheus", "guard"))]
    return chat[0] if chat else PREFERIDOS[0]


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
    modelo = getattr(cliente, "modelo", None) or elegir_modelo(cliente)
    try:
        cliente.modelo = modelo
    except AttributeError:
        pass
    for paso in range(T_MAX):
        try:
            msg = cliente.completar(model=modelo, messages=mensajes, tools=ESQUEMAS,
                                    tool_choice="auto", temperature=0.1, max_tokens=700)
        except Exception as e:
            return f"No se pudo contactar al modelo de lenguaje ({type(e).__name__}): {e}", traza
        llamadas = msg.get("tool_calls") or []
        if not llamadas:
            texto = (msg.get("content") or "").strip()
            # Algunos modelos citan con corchetes 【 】; se normalizan a [ ]
            return texto.replace("【", "[").replace("】", "]"), traza
        mensajes.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                         "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                                        for tc in llamadas]})
        for tc in llamadas:
            nombre, args = tc["function"]["name"], tc["function"]["arguments"]
            obs = ejecutar_herramienta(nombre, args)
            traza.append({"paso": paso + 1, "herramienta": nombre, "argumentos": args,
                          "error": obs.get("error") if isinstance(obs, dict) else None})
            # Los errores vuelven al modelo como observación para que pueda corregir
            mensajes.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(obs, ensure_ascii=False, default=str)[:6000]})
    return "Se alcanzó el número máximo de pasos sin una respuesta final. Intente con una pregunta más concreta.", traza


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    c = crear_cliente()
    resp, tr = responder("¿Cómo estuvo el aire en Itagüí el 10 de marzo de 2024 y qué recomienda la norma?", c)
    print(resp)
    print(tr)
