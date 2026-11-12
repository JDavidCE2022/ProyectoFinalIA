---
titulo: Cómo funciona la predicción de PM2.5 del día siguiente en este proyecto
fuente: Documentación del proyecto (repositorio ProyectoFinalIA)
url: https://github.com/JDavidCE2022/ProyectoFinalIA
---
La herramienta de predicción usa una regresión lineal Ridge implementada desde cero que estima el PM2.5 promedio del día siguiente para una estación. Usa el PM2.5 de hoy y de ayer en la estación, el promedio de toda la red hoy, la lluvia, el viento y la radiación solar del día según Open-Meteo, si el día siguiente cae en temporada crítica (febrero, marzo, octubre o noviembre), si es fin de semana o lunes, y un efecto propio de cada estación.

El valor estimado se convierte en categoría con cortes ajustados con validación cruzada para equilibrar el F1 de las tres clases: Buena, Aceptable y Dañina (que agrupa naranja o peor). Por eso el corte entre Aceptable y Dañina del modelo puede ser menor que 37 µg/m³, lo que lo hace más sensible a los días malos.

En validación temporal de noviembre de 2023 a julio de 2024 el error absoluto medio fue de unos 3.4 µg/m³, menor que el de repetir el valor de hoy (3.6), aunque en los meses de temporada crítica la ventaja desaparece. El modelo detecta cerca del 69 % de los días que terminan en categoría Dañina, pero cerca de la mitad de sus alertas son falsas alarmas. Es una herramienta académica y no un pronóstico oficial. El pronóstico oficial es el del SIATA.
