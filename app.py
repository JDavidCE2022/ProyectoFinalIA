"""Dashboard de calidad del aire del Valle de Aburrá con agente LLM. Ejecutar con: streamlit run app.py"""
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()

from datos import ICA_CORTES, RAW, PROC, categoria_ica  # noqa: E402
import herramientas as H  # noqa: E402
import agente  # noqa: E402

st.set_page_config(page_title="AireBot Valle de Aburrá", layout="wide")
COLORES = {n: c for _, _, n, c in ICA_CORTES}


@st.cache_data
def datos_base():
    perfil = pd.read_csv(RAW / "siata_perfil_horario.csv")
    val = pd.read_csv(PROC / "predicciones_validacion.csv", parse_dates=["fecha", "fecha_manana"])
    met = pd.read_csv(PROC / "metricas_modelo.csv", index_col=0)
    return perfil, val, met


diario = H._DIARIO
estaciones = H._EST[H._EST["serial"].isin(diario["serial"].unique())].copy()
perfil, validacion, metricas = datos_base()

# Barra lateral
st.sidebar.title("Parámetros")
opciones = {f"{r.nombre_corto} · {r.nombre}": int(r.serial) for r in estaciones.itertuples()}
etiqueta = st.sidebar.selectbox("Estación", list(opciones), index=list(opciones.values()).index(28) if 28 in opciones.values() else 0)
serial = opciones[etiqueta]
fecha = pd.Timestamp(st.sidebar.date_input("Fecha", value=pd.Timestamp("2024-03-10").date(),
                                           min_value=H.FECHA_MIN.date(), max_value=H.FECHA_MAX.date()))
st.sidebar.divider()
st.sidebar.subheader("Agente")
clave = st.sidebar.text_input("GROQ_API_KEY", type="password", value="",
                              help="Opcional si ya está en el archivo .env. No se guarda.")
st.sidebar.caption(f"Modelo de chat: {agente.MODELO}")
st.sidebar.caption("Datos: SIATA (PM2.5) y Open-Meteo (clima). Herramienta académica, no es pronóstico oficial.")

nombre_est = estaciones.set_index("serial").loc[serial, "nombre"]
st.title("Calidad del aire en el Valle de Aburrá")
st.caption(f"Estación seleccionada: {nombre_est} · Fecha: {fecha.date()}")

# Indicadores
dato = H.get_pm25(str(serial), str(fecha.date()), dias=1)
pred = H.predict_tomorrow(str(serial), str(fecha.date()))
c1, c2, c3 = st.columns(3)
if "error" in dato or dato.get("aviso"):
    c1.metric("PM2.5 del día", "sin dato válido")
else:
    v = dato["ultimo_dato"]["pm25"]
    c1.metric("PM2.5 del día (µg/m³)", f"{v:.1f}")
    cat = dato["ultimo_dato"]["categoria_ica"]
    c2.markdown(f"**Categoría ICA**<br><span style='background:{COLORES[cat]};padding:4px 10px;border-radius:6px;color:black'>{cat}</span>",
                unsafe_allow_html=True)
if "error" in pred:
    c3.metric("Predicción para mañana", "no disponible")
    c3.caption(pred["error"])
else:
    real = pred["valor_real_observado"]
    c3.metric(f"Predicción {pred['fecha_predicha']} (µg/m³)", f"{pred['pm25_predicho']:.1f} ± {pred['error_tipico']}",
              delta=None if real is None else f"real {real:.1f}", delta_color="off")
    c3.caption(f"Clase del modelo: {pred['clase_modelo']}")

tab_mapa, tab_serie, tab_perfil, tab_modelo, tab_chat = st.tabs(
    ["Mapa", "Serie de tiempo", "Perfil horario", "Modelo", "Chat con el agente"])

with tab_mapa:
    dia = diario[diario["fecha"] == fecha].merge(estaciones[["serial", "lat", "lon"]], on="serial")
    if dia.empty:
        st.info("No hay datos válidos para esa fecha.")
    else:
        fig = px.scatter_map(dia, lat="lat", lon="lon", color="categoria_ica", size="pm25_media", size_max=22,
                             color_discrete_map=COLORES, hover_name="nombre",
                             hover_data={"pm25_media": ":.1f", "lat": False, "lon": False},
                             zoom=10, center={"lat": 6.25, "lon": -75.58}, map_style="open-street-map", height=520)
        fig.update_layout(legend_title_text="Categoría ICA", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"PM2.5 promedio de 24 h en {len(dia)} estaciones el {fecha.date()}. Tamaño del punto proporcional a la concentración.")

with tab_serie:
    g = diario[diario["serial"] == serial]
    fig = go.Figure()
    for lo, hi, n, col in ICA_CORTES[:4]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.12, line_width=0)
    red = diario.groupby("fecha")["pm25_media"].mean()
    fig.add_trace(go.Scatter(x=red.index, y=red.values, name="Promedio de la red", line=dict(color="gray", width=1)))
    fig.add_trace(go.Scatter(x=g["fecha"], y=g["pm25_media"], name=g["nombre_corto"].iloc[0], line=dict(width=2, color="#1f4e9c")))
    fig.add_vline(x=fecha, line_dash="dash", line_color="black")
    fig.add_hline(y=15, line_dash="dot", annotation_text="Guía OMS 24 h (15)")
    fig.add_hline(y=37, line_dash="dot", annotation_text="Norma Colombia 24 h (37)")
    fig.update_layout(yaxis_title="PM2.5 (µg/m³)", height=480, yaxis_range=[0, max(60, g["pm25_media"].max() + 5)])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Las bandas de color corresponden a las categorías del ICA. Los picos de febrero y marzo de 2024 corresponden a la temporada crítica de transición a lluvias.")

with tab_perfil:
    p = perfil[perfil["serial"] == serial]
    pr = perfil.groupby("hora")["pm25_media"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p["hora"], y=p["pm25_media"], name="Estación"))
    fig.add_trace(go.Scatter(x=pr["hora"], y=pr["pm25_media"], name="Promedio de la red", line=dict(color="black")))
    fig.update_layout(xaxis_title="Hora del día", yaxis_title="PM2.5 medio (µg/m³)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Promedio por hora en todo el año. El pico de la mañana coincide con la hora pico de tráfico y la atmósfera estable.")

with tab_modelo:
    st.subheader("Validación temporal (noviembre 2023 a julio 2024)")
    st.dataframe(metricas.astype(float).round(3).astype(object).where(metricas.notna(), "-"), use_container_width=True)
    v = validacion[validacion["serial"] == serial]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v["fecha_manana"], y=v["pm25_manana"], name="Real", mode="lines"))
    fig.add_trace(go.Scatter(x=v["fecha_manana"], y=v["valor_ridge"], name="Predicho (Ridge)", mode="lines"))
    fig.update_layout(yaxis_title="PM2.5 (µg/m³)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada mes se predice con un modelo entrenado solo con los meses anteriores.")

with tab_chat:
    st.caption("Pregunte por datos de una estación, la predicción de mañana, la norma o recomendaciones de salud.")
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    pregunta = st.chat_input("Ejemplo: ¿cómo estará el aire mañana en esta estación?")
    if pregunta:
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            try:
                cliente = agente.crear_cliente(clave or None)
                ctx = f"estación seleccionada {nombre_est} (serial {serial}), fecha seleccionada {fecha.date()}"
                hist = [m for m in st.session_state.mensajes[:-1]][-6:]
                with st.spinner("Consultando herramientas..."):
                    resp, traza = agente.responder(pregunta, cliente, hist, ctx)
            except Exception as e:
                resp, traza = f"No se pudo usar el agente: {e}", []
            st.markdown(resp)
            if traza:
                with st.expander("Herramientas usadas"):
                    st.dataframe(pd.DataFrame(traza), use_container_width=True)
        st.session_state.mensajes.append({"role": "assistant", "content": resp})
