# Evaluación del agente

Herramienta correcta: 13/13  
Cita de fuente (o abstención en la pregunta fuera del corpus): 13/13  
Modelo: openai/gpt-oss-120b

|   n | pregunta                                                                                                           | herramienta_esperada         | herramientas_usadas           | herramienta_correcta   | cito_fuente   |   errores_herramienta |   segundos |
|----:|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------|:------------------------------|:-----------------------|:--------------|----------------------:|-----------:|
|   1 | ¿Cuál fue el PM2.5 en la estación de Itagüí Casa de Justicia el 10 de marzo de 2024?                               | get_pm25                     | get_pm25                      | True                   | True          |                     0 |        1.6 |
|   2 | ¿Cómo estuvo el aire en Bello la última semana de datos?                                                           | get_pm25                     | get_pm25                      | True                   | True          |                     0 |        1.9 |
|   3 | ¿Qué categoría del ICA tuvo la estación de Belén el 5 de marzo de 2024?                                            | get_pm25                     | get_pm25                      | True                   | True          |                     0 |        1.9 |
|   4 | ¿Cómo estará la calidad del aire en Itagüí el 11 de marzo de 2024 según el modelo?                                 | predict_tomorrow             | predict_tomorrow              | True                   | True          |                     0 |        1.9 |
|   5 | Predice el PM2.5 del día siguiente al 20 de febrero de 2024 en Envigado.                                           | predict_tomorrow             | predict_tomorrow              | True                   | True          |                     0 |        3.1 |
|   6 | ¿Qué es el POECA?                                                                                                  | search_docs                  | search_docs                   | True                   | True          |                     0 |       55.6 |
|   7 | ¿Cuál es el nivel máximo permisible de PM2.5 en 24 horas en Colombia?                                              | search_docs                  | search_docs                   | True                   | True          |                     0 |        2.1 |
|   8 | ¿Qué recomienda la OMS como límite diario de PM2.5?                                                                | search_docs                  | search_docs                   | True                   | True          |                     0 |        1.6 |
|   9 | Si el ICA está en naranja y tengo asma, ¿qué precauciones debo tomar?                                              | search_docs                  | search_docs                   | True                   | True          |                     0 |        2.2 |
|  10 | ¿Por qué en marzo empeora la calidad del aire en el Valle de Aburrá?                                               | search_docs                  | search_docs                   | True                   | True          |                     0 |        2.2 |
|  11 | ¿Qué estaciones de calidad del aire están disponibles?                                                             | listar_estaciones            | listar_estaciones             | True                   | True          |                     0 |        2.3 |
|  12 | ¿Mañana puedo salir a trotar en Sabaneta? Usa la predicción del 3 de marzo de 2024 y las recomendaciones de salud. | predict_tomorrow+search_docs | predict_tomorrow, search_docs | True                   | True          |                     0 |        3.4 |
|  13 | ¿Cuál es el costo del parqueadero del Metro de Medellín?                                                           | ninguna                      | -                             | True                   | True          |                     0 |        0.8 |

## Respuestas completas

**1. ¿Cuál fue el PM2.5 en la estación de Itagüí Casa de Justicia el 10 de marzo de 2024?**

El 10 de marzo de 2024 la estación **Itagüí – Casa de Justicia** registró un promedio de **PM2.5 de 25.3 µg/m³**, categoría “Aceptable”. [SIATA]

**2. ¿Cómo estuvo el aire en Bello la última semana de datos?**

En la estación **Bello – I.E. Fernando Vélez** los valores promedio de PM2.5 de los últimos 7 días fueron:  | Fecha | PM2.5 (µg/m³) | Categoría ICA | |-------|--------------|---------------| | 2024‑07‑24 | 14.8 | Aceptable | | 2024‑07‑27 | 10.0 | Buena | | 2024‑07‑28 | 10.4 | Buena | | 2024‑07‑29 | 9.4  | Buena | | 2024‑07‑30 | 12.6 | Aceptable |  Los valores se mantuvieron en rangos **buenos a aceptables**, sin superar los límites de alerta.    [SIATA]

**3. ¿Qué categoría del ICA tuvo la estación de Belén el 5 de marzo de 2024?**

El 5 de marzo de 2024 la estación Belén registró una categoría del ICA **“Dañina grupos sensibles”**. [SIATA]

**4. ¿Cómo estará la calidad del aire en Itagüí el 11 de marzo de 2024 según el modelo?**

Según el modelo académico, el 11 de marzo de 2024 la estación **Itagüí – Casa de Justicia** tendrá un PM2.5 predicho de **27.2 µg/m³**, lo que corresponde a la categoría **“Aceptable”** del ICA.    [Modelo del proyecto]

**5. Predice el PM2.5 del día siguiente al 20 de febrero de 2024 en Envigado.**

El modelo predice que el 21 de febrero de 2024 el PM2.5 en la estación **Envigado – E.S.E. Santa Gertrudis** será de **≈ 22.8 µg/m³**, lo que corresponde a la categoría **Aceptable** según el ICA.    *Fuente: [Modelo del proyecto]*

**6. ¿Qué es el POECA?**

El POECA (Protocolo Operacional para Enfrentar Episodios de Contaminación Atmosférica) es el plan de acción de corto plazo del Área Metropolitana del Valle de Aburrá que se activa cuando la calidad del aire supera los umbrales establecidos. Su objetivo es proteger la salud de la población mediante medidas extraordinarias de reducción de emisiones y gestión del riesgo, coordinadas por las autoridades ambientales [05_poeca#1][05_poeca#2].

**7. ¿Cuál es el nivel máximo permisible de PM2.5 en 24 horas en Colombia?**

El nivel máximo permisible de PM2.5 en un promedio de 24 horas en Colombia es **37 µg/m³**[02_limites_normativos#1].

**8. ¿Qué recomienda la OMS como límite diario de PM2.5?**

La Organización Mundial de la Salud (OMS) recomienda que el valor guía de PM2.5 para un promedio de 24 horas no supere los **15 µg/m³** (valor actualizado en 2021)[03_guias_oms_2021#1].

**9. Si el ICA está en naranja y tengo asma, ¿qué precauciones debo tomar?**

Cuando el ICA está en **naranja** (dañina para grupos sensibles) y tienes asma, debes tomar las siguientes precauciones :  - Evita hacer ejercicio intenso al aire libre, sobre todo en la mañana, cuando la concentración de contaminantes suele ser mayor.   - Prefiere actividades en espacios cerrados y, si es posible, en lugares con buena ventilación o con purificador de aire.   - Mantén a mano tu medicación para el asma (inhalador de rescate) y úsala según las indicaciones de tu médico si aparecen síntomas.   - Limita el tiempo que pasas cerca de fuentes de emisión (tráfico pesado, obras).   - Mantén puertas y ventanas cerradas si el aire exterior está muy contaminado; abrelas solo brevemente para ventilar.    Si notas dificultad para respirar, tos persistente o cualquier empeoramiento de tus síntomas, **consulta inmediatamente a un servicio de salud**.    Fuentes: [07_recomendaciones_salud#3] y [08_grupos_sensibles#1].

**10. ¿Por qué en marzo empeora la calidad del aire en el Valle de Aburrá?**

En marzo (y también en febrero) el Valle de Aburrá atraviesa una **temporada de transición entre la época seca y la lluviosa**. En esos meses se forma una capa de nubes bajas que reduce la turbulencia atmosférica; al haber menos mezcla vertical, los contaminantes emitidos por fuentes fijas y móviles quedan atrapados cerca del suelo. Además, el valle está rodeado de montañas que actúan como barrera al viento, dificultando la dispersión del material particulado PM2.5 y favoreciendo su acumulación [06_episodios_criticos#2][06_episodios_criticos#1].    En contraste, durante la época seca la radiación solar calienta la superficie, genera mayor turbulencia y ayuda a “levantar” la contaminación, mientras que la lluvia limpia la atmósfera arrastrando partículas [06_episodios_criticos#3].    Por eso, en marzo la calidad del aire tiende a empeorar.

**11. ¿Qué estaciones de calidad del aire están disponibles?**

Las estaciones de calidad del aire disponibles son:  - CEN-TRAF – Estación Tráfico Centro   - ITA-CJUS – Itagüí, Casa de Justicia   - ITA-CONC – Itagüí, I.E. Concejo Municipal   - MED-LAYE – Medellín, El Poblado – Tanques La Ye (EPM)   - CAL-JOAR – Caldas – E U Joaquín Aristizábal   - EST-HOSP – La Estrella – Hospital   - MED-ALTA – Medellín, Altavista – I.E. Pedro Octavio Amado   - MED-VILL – Medellín, Villahermosa – Planta de agua potable (EPM)   - BAR-TORR – Barbosa – Torre Social   - COP-CVID – Copacabana – Ciudadela Educativa La Vida   - MED-BEME – Medellín, Belén – I.E. Pedro Justo Berrío   - MED-TESO – Medellín, El Poblado – I.E. INEM sede Santa Catalina   - MED-SCRI – Medellín, San Cristóbal – Parque Biblioteca Fernando Botero   - MED-ARAN – Medellín, Aranjuez – I.E. Ciro Mendia   - BEL-FEVE – Bello – I.E. Fernando Vélez   - ENV-HOSP – Envigado – E.S.E. Santa Gertrudis   - SAB-RAME – Sabaneta – I.E. Rafael J. Mejía    Los datos de PM2.5 están disponibles desde el 31 julio 2023 hasta el 30 julio 2024. [SIATA]

**12. ¿Mañana puedo salir a trotar en Sabaneta? Usa la predicción del 3 de marzo de 2024 y las recomendaciones de salud.**

Según la predicción del modelo para el 4 de marzo de 2024 (día siguiente al 3 de marzo) en la estación de Sabaneta – I.E. Rafael J. Mejía, el PM2.5 esperado es **27.7 µg/m³**, lo que corresponde a la categoría **Aceptable (ICA amarillo)**[Modelo del proyecto].    **Recomendaciones de salud**: con ICA amarillo la población general puede seguir su rutina al aire libre, pero los grupos sensibles (niños, adultos mayores, gestantes y personas con enfermedades cardíacas o respiratorias) deben reducir el esfuerzo físico prolongado si presentan síntomas como tos o dificultad para respirar[07_recomendaciones_salud#2].    **Conclusión**: Puedes salir a trotar en Sabaneta, siempre que no pertenezcas a un grupo sensible o, en caso de presentar síntomas, reduzcas la intensidad del ejercicio. Si tienes alguna condición de salud, consulta a un profesional.

**13. ¿Cuál es el costo del parqueadero del Metro de Medellín?**

Lo siento, pero mi función es brindar información sobre la calidad del aire en el Valle de Aburrá. No dispongo de datos sobre el costo del parqueadero del Metro de Medellín. Le recomiendo consultar la página oficial del Metro o comunicarse directamente con ellos para obtener esa información.
