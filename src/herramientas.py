"""Herramientas que el agente puede llamar. Cada una devuelve un dict serializable y nunca lanza excepciones al agente."""
import unicodedata
import numpy as np
import pandas as pd
from datos import cargar_diario, cargar_clima, cargar_estaciones, categoria_ica
from modelo import cargar_modelo, variables_regresion, a_clase

_DIARIO = cargar_diario()
_CLIMA = cargar_clima()
_EST = cargar_estaciones()
_REG, _META = cargar_modelo()
FECHA_MIN, FECHA_MAX = _DIARIO["fecha"].min(), _DIARIO["fecha"].max()
_BUSCADOR = None


def _norm(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return t


def resolver_estacion(texto):
    """Acepta serial, nombre corto o parte del nombre (municipio o barrio)."""
    activas = _EST[_EST["serial"].isin(_DIARIO["serial"].unique())]
    t = _norm(texto).strip()
    if t.isdigit() and int(t) in set(activas["serial"]):
        return activas[activas["serial"] == int(t)].iloc[0]
    for col in ["nombre_corto", "nombre"]:
        m = activas[activas[col].map(_norm).str.contains(t, regex=False)]
        if len(m):
            return m.iloc[0]
    palabras = [w for w in t.replace(",", " ").split() if len(w) > 3]
    for w in palabras:
        m = activas[activas["nombre"].map(_norm).str.contains(w, regex=False)]
        if len(m):
            return m.iloc[0]
    return None


def _fecha(fecha):
    if fecha in (None, "", "hoy", "ultima"):
        return FECHA_MAX
    return pd.Timestamp(fecha).normalize()


def listar_estaciones():
    activas = _EST[_EST["serial"].isin(_DIARIO["serial"].unique())]
    return {"estaciones": activas[["serial", "nombre_corto", "nombre"]].to_dict("records"),
            "rango_datos": [str(FECHA_MIN.date()), str(FECHA_MAX.date())],
            "fuente": "SIATA, archivo público de PM2.5 horario"}


def get_pm25(estacion, fecha=None, dias=7):
    """PM2.5 promedio diario de una estación en una fecha y los días previos."""
    try:
        e = resolver_estacion(estacion)
        if e is None:
            return {"error": f"No se encontró la estación '{estacion}'. Use listar_estaciones para ver las opciones."}
        f = _fecha(fecha)
        if not (FECHA_MIN <= f <= FECHA_MAX):
            return {"error": f"Solo hay datos entre {FECHA_MIN.date()} y {FECHA_MAX.date()}."}
        g = _DIARIO[(_DIARIO["serial"] == e["serial"]) & (_DIARIO["fecha"] <= f) & (_DIARIO["fecha"] > f - pd.Timedelta(days=int(dias)))]
        if g.empty:
            return {"error": f"La estación {e['nombre_corto']} no tiene datos válidos en esas fechas."}
        serie = [{"fecha": str(r.fecha.date()), "pm25": round(r.pm25_media, 1), "categoria_ica": r.categoria_ica} for r in g.itertuples()]
        ult = serie[-1]
        return {"estacion": e["nombre"], "serial": int(e["serial"]), "fecha_consultada": str(f.date()),
                "ultimo_dato": ult, "serie": serie,
                "aviso": None if ult["fecha"] == str(f.date()) else "No hay dato válido en la fecha pedida, se muestra el último disponible.",
                "fuente": "SIATA, PM2.5 promedio de 24 h con al menos 18 horas válidas"}
    except Exception as ex:
        return {"error": f"Fallo en get_pm25: {ex}"}


def predict_tomorrow(estacion, fecha=None):
    """Predice el PM2.5 y la categoría del día siguiente a 'fecha' con la regresión Ridge del proyecto."""
    try:
        e = resolver_estacion(estacion)
        if e is None:
            return {"error": f"No se encontró la estación '{estacion}'."}
        f = _fecha(fecha)
        s = int(e["serial"])
        hoy = _DIARIO[(_DIARIO["serial"] == s) & (_DIARIO["fecha"] == f)]
        ayer = _DIARIO[(_DIARIO["serial"] == s) & (_DIARIO["fecha"] == f - pd.Timedelta(days=1))]
        clima = _CLIMA[_CLIMA["fecha"] == f]
        if hoy.empty or ayer.empty or clima.empty:
            return {"error": f"Faltan datos de {e['nombre_corto']} para {f.date()} o el día anterior, no se puede predecir."}
        fila = pd.DataFrame({"fecha": [f], "fecha_manana": [f + pd.Timedelta(days=1)], "serial": [s],
                             "pm25_media": hoy["pm25_media"].values, "pm25_ayer": ayer["pm25_media"].values,
                             "ciudad_hoy": [_DIARIO[_DIARIO["fecha"] == f]["pm25_media"].mean()],
                             "lluvia_mm": clima["lluvia_mm"].values, "viento_media_kmh": clima["viento_media_kmh"].values,
                             "radiacion_mj": clima["radiacion_mj"].values})
        X = variables_regresion(fila, _META["seriales"])[_META["columnas"]]
        valor = float(_REG.predecir(X)[0])
        real = _DIARIO[(_DIARIO["serial"] == s) & (_DIARIO["fecha"] == f + pd.Timedelta(days=1))]
        return {"estacion": e["nombre"], "fecha_base": str(f.date()), "fecha_predicha": str((f + pd.Timedelta(days=1)).date()),
                "pm25_hoy": round(float(hoy["pm25_media"].iloc[0]), 1),
                "pm25_predicho": round(valor, 1), "error_tipico": 3.4,
                "clase_modelo": str(a_clase([valor], _META["t1"], _META["t2"])[0]),
                "categoria_ica_del_valor": categoria_ica(valor),
                "clima_hoy": {"lluvia_mm": float(clima["lluvia_mm"].iloc[0]), "viento_kmh": float(clima["viento_media_kmh"].iloc[0])},
                "valor_real_observado": None if real.empty else round(float(real["pm25_media"].iloc[0]), 1),
                "nota": "Modelo académico (Ridge implementado en el proyecto). No es el pronóstico oficial del SIATA.",
                "fuente": "Modelo del proyecto entrenado con datos SIATA y Open-Meteo"}
    except Exception as ex:
        return {"error": f"Fallo en predict_tomorrow: {ex}"}


def search_docs(consulta, k=3):
    """Busca en el corpus de normativa y salud y devuelve los fragmentos con su fuente."""
    global _BUSCADOR
    try:
        if _BUSCADOR is None:
            from rag import Buscador
            _BUSCADOR = Buscador()
        res = _BUSCADOR.buscar(consulta, int(k))
        return {"resultados": [{"id": r["id"], "titulo": r["titulo"], "texto": r["texto"], "fuente": r["fuente"], "url": r["url"]} for r in res],
                "modo_busqueda": _BUSCADOR.modo}
    except Exception as ex:
        return {"error": f"Fallo en search_docs: {ex}"}


HERRAMIENTAS = {"get_pm25": get_pm25, "predict_tomorrow": predict_tomorrow,
                "search_docs": search_docs, "listar_estaciones": listar_estaciones}

ESQUEMAS = [
    {"type": "function", "function": {"name": "get_pm25",
     "description": "Consulta el PM2.5 promedio diario medido por el SIATA en una estación y los días previos.",
     "parameters": {"type": "object", "properties": {
         "estacion": {"type": "string", "description": "Nombre, municipio, barrio o serial de la estación, por ejemplo 'Itagüí' o 'Belén'."},
         "fecha": {"type": "string", "description": "Fecha AAAA-MM-DD. Si se omite se usa la última fecha disponible."},
         "dias": {"type": "integer", "description": "Número de días hacia atrás, por defecto 7."}},
         "required": ["estacion"]}}},
    {"type": "function", "function": {"name": "predict_tomorrow",
     "description": "Predice el PM2.5 y la categoría del día siguiente a una fecha para una estación, con el modelo del proyecto.",
     "parameters": {"type": "object", "properties": {
         "estacion": {"type": "string", "description": "Nombre, municipio, barrio o serial de la estación."},
         "fecha": {"type": "string", "description": "Fecha base AAAA-MM-DD. Se predice el día siguiente."}},
         "required": ["estacion"]}}},
    {"type": "function", "function": {"name": "search_docs",
     "description": "Busca en documentos de normativa colombiana, POECA, guías de la OMS y recomendaciones de salud sobre calidad del aire.",
     "parameters": {"type": "object", "properties": {
         "consulta": {"type": "string", "description": "Pregunta o términos de búsqueda en español."},
         "k": {"type": "integer", "description": "Número de fragmentos, por defecto 3."}},
         "required": ["consulta"]}}},
    {"type": "function", "function": {"name": "listar_estaciones",
     "description": "Lista las estaciones disponibles y el rango de fechas de los datos.",
     "parameters": {"type": "object", "properties": {}}}},
]

if __name__ == "__main__":
    import json
    print(json.dumps(get_pm25("Itagüí", "2024-03-10", 3), ensure_ascii=False)[:400])
    print(json.dumps(predict_tomorrow("belén", "2024-03-10"), ensure_ascii=False))
    print(get_pm25("Marte"))
    print(json.dumps(search_docs("¿qué es el POECA?", 2), ensure_ascii=False)[:300])
