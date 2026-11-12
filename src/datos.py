"""Carga y preparación de los datos de PM2.5 (SIATA) y clima (Open-Meteo)."""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
PROC = RAIZ / "data" / "processed"

# Puntos de corte del ICA para PM2.5 en 24 h (Res. 2254 de 2017)
ICA_CORTES = [(0, 12, "Buena", "#00e400"), (12, 37, "Aceptable", "#ffff00"),
              (37, 55, "Dañina grupos sensibles", "#ff7e00"), (55, 150, "Dañina", "#ff0000"),
              (150, 250, "Muy dañina", "#8f3f97"), (250, 500, "Peligrosa", "#7e0023")]

# Clases del modelo: se unen las categorías desde naranja porque rojo casi no ocurre
CLASES = ["Buena", "Aceptable", "Dañina"]


def categoria_ica(pm25):
    for lo, hi, nombre, _ in ICA_CORTES:
        if pm25 <= hi:
            return nombre
    return "Peligrosa"


def clase_modelo(pm25):
    if pm25 <= 12:
        return "Buena"
    if pm25 <= 37:
        return "Aceptable"
    return "Dañina"


def cargar_estaciones():
    est = pd.read_csv(RAW / "siata_estaciones.csv")
    est["activa"] = ~est["nombre"].str.startswith("_OFF")
    est["nombre"] = est["nombre"].str.replace("_OFF-", "", regex=False)
    return est


def cargar_clima():
    c = pd.read_csv(RAW / "openmeteo_medellin_diario.csv", skiprows=3)
    c.columns = ["fecha", "lluvia_mm", "viento_media_kmh", "viento_max_kmh", "temp_media_c", "humedad_media", "radiacion_mj"]
    c["fecha"] = pd.to_datetime(c["fecha"])
    return c


def cargar_diario(min_horas=18):
    d = pd.read_csv(RAW / "siata_pm25_diario.csv", parse_dates=["fecha"])
    d = d[d["horas_validas"] >= min_horas].copy()
    est = cargar_estaciones()
    d = d.merge(est[["serial", "nombre", "nombre_corto"]], on="serial", how="left")
    d["categoria_ica"] = d["pm25_media"].apply(categoria_ica)
    return d.sort_values(["serial", "fecha"]).reset_index(drop=True)


def construir_dataset():
    """Una fila por estación y día t, con variables de t y la clase del día t+1."""
    d = cargar_diario()
    clima = cargar_clima()
    filas = []
    for serial, g in d.groupby("serial"):
        g = g.set_index("fecha").asfreq("D")
        g["serial"] = serial
        g["pm25_ayer"] = g["pm25_media"].shift(1)
        g["pm25_manana"] = g["pm25_media"].shift(-1)
        filas.append(g.reset_index())
    x = pd.concat(filas)
    x = x.dropna(subset=["pm25_media", "pm25_ayer", "pm25_manana"])
    x = x.merge(clima, on="fecha", how="left")
    x["clase_hoy"] = x["pm25_media"].apply(clase_modelo)
    x["clase_manana"] = x["pm25_manana"].apply(clase_modelo)
    x["fecha_manana"] = x["fecha"] + pd.Timedelta(days=1)
    return x.sort_values(["fecha", "serial"]).reset_index(drop=True)


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    ds = construir_dataset()
    ds.to_csv(PROC / "dataset_modelo.csv", index=False)
    print(ds.shape)
    print(ds["clase_manana"].value_counts())
