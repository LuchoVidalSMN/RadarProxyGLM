# GLM-FED Radar Proxy 🌩️✈️

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Meteorology](https://img.shields.io/badge/Meteorology-Aviation-orange)
![GOES](https://img.shields.io/badge/Satellite-GOES--R-success)

Aplicativo para la generación de un **proxy de reflectividad radar** a partir del producto **FED (Flash Extent Density)** del sensor **GLM (Geostationary Lightning Mapper)** a bordo de los satélites de la serie GOES-R. 

Este desarrollo está orientado a la identificación y monitoreo de núcleos convectivos de alta peligrosidad para vuelos comerciales, sirviendo como herramienta central para la **automatización de la emisión de mensajes SIGMET por tormentas**.

---

## 📖 Contexto Teórico

### El sensor GLM y el producto FED
El **GLM (Geostationary Lightning Mapper)**, a bordo de satélites como el GOES-16 y -19, es el primer mapeador de rayos óptico en órbita geoestacionaria. Este instrumento detecta continuamente la actividad eléctrica total (relámpagos intra-nube y nube-tierra) con una resolución espacial de 8-14 km.

Dentro de sus variables derivadas, el **FED (Flash Extent Density)** contabiliza el número de destellos que ocurren o atraviesan un píxel específico en un intervalo de tiempo. La literatura científica demuestra que la densidad y extensión de estos destellos están fuertemente correlacionadas con:
- La intensidad de las corrientes ascendentes (*updrafts*).
- El volumen de la región de fase mixta de la nube (donde coexiste agua sobreenfriada y hielo).
- La presencia de corrientes descendentes severas, granizo y turbulencia asociada a la convección profunda.

### Proxy de Reflectividad Radar
Las redes de radares meteorológicos terrestres son fundamentales, pero sufren de bloqueos orográficos, pérdida de resolución a la distancia y, sobre todo, falta de cobertura en áreas oceánicas y remotas.

Dado que la actividad eléctrica (FED) marca el pulso termodinámico de las tormentas más intensas, este aplicativo convierte dichos datos en un **Proxy de Reflectividad Radar** (o "pseudo-reflectividad"). Al integrar los datos del GLM (que tienen actualizaciones continuas y cobertura hemisférica), el sistema logra "simular" cómo se vería la reflectividad del radar, cerrando las brechas de información y generando un mapa continuo de peligrosidad convectiva.

---

## ✈️ Aplicación Aeronáutica: Automatización de SIGMETs

### Mitigación de Riesgos en Vuelo
Para la aviación comercial, las tormentas convectivas representan múltiples peligros: **turbulencia severa, engelamiento, granizo y cortantes de viento (*windshear*)**. Evitar los núcleos convectivos (especialmente aquellos en fase de rápido desarrollo) es imperativo para la seguridad estructural de las aeronaves y el confort de los pasajeros.

### Delimitación Automática de Zonas SIGMET
Los **SIGMET** (Significant Meteorological Information) son mensajes alfanuméricos de advertencia crítica para las aeronaves en ruta. Cuando ocurren tormentas severas, frecuentemente oscurecidas, embebidas o agrupadas en líneas de turbonada (*FRQ TS*, *EMBD TS*, *SQL TS*), las oficinas de vigilancia meteorológica deben emitir polígonos de alerta sobre las Regiones de Información de Vuelo (FIR).

El **RadarProxyGLM** aborda el desafío de la vigilancia en tiempo real de la siguiente manera:
1. **Detección ininterrumpida:** Aprovecha el GOES-16 para rastrear la actividad severa sobre cualquier punto del continente y los océanos adyacentes.
2. **Umbrales Objetivos de Peligro:** Al traducir el FED a "reflectividad radar equivalente", permite fijar umbrales matemáticos (ej. reflectividad equivalente > 40 dBZ) para identificar las celdas directamente incompatibles con el vuelo.
3. **Soporte a la Automatización:** Genera campos matriciales georreferenciados ideales para que algoritmos geométricos tracen automáticamente los polígonos de las tormentas. Esto reduce drásticamente el tiempo de análisis del pronosticador, permitiendo emitir SIGMETs automáticos, más precisos espacialmente y con actualizaciones casi en tiempo real.

---

# Aplicación Interactiva de Polígonos de Advertencia Meteorológica

Esta aplicación Streamlit proporciona un dashboard interactivo para visualizar y analizar polígonos de advertencia generados a partir de datos de satélites GOES-19 (GLM y ABI Canal 13 IR).

## Características

*   **Visualización en tiempo casi real:** Muestra datos de reflectividad proxy de GLM y temperaturas de brillo infrarrojas (IR) del ABI Canal 13 de GOES-19.
*   **Detección de Polígonos de Advertencia:** Identifica automáticamente áreas con potencial de tiempo severo (convección profunda) basándose en un umbral de reflectividad configurable.
*   **Métricas detalladas:** Calcula y presenta métricas clave para cada polígono de advertencia, incluyendo centroide, cantidad de píxeles afectados, reflectividad máxima, densidad de flashes máxima (FED) y temperatura IR mínima.
*   **Interactividad:** Permite a los usuarios:
    *   Seleccionar la fecha y hora de los datos a visualizar.
    *   Ajustar dinámicamente el umbral de reflectividad (en dBZ) para la generación de polígonos.
    *   Resaltar polígonos específicos en el mapa al seleccionarlos de una tabla de métricas.
    *   Descargar las métricas de los polígonos en formato CSV.
*   **Capas geográficas:** Incluye fronteras de países, FIRs (Regiones de Información de Vuelo) y ubicaciones de aeropuertos para una mejor referencia contextual.

## Datos Utilizados

La aplicación utiliza datos públicos de AWS S3 de los satélites GOES-19:

*   **GLM (Geostationary Lightning Mapper) Nivel 2 LCFA:** Datos de flashes de rayos, utilizados para generar un proxy de reflectividad.
*   **ABI (Advanced Baseline Imager) Nivel 2 CMIPF Banda 13:** Datos de temperatura de brillo infrarroja, utilizados como capa de fondo para contextualizar la nubosidad.

## Estructura del Proyecto

## Cómo Ejecutar la Aplicación Localmente

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DE_TU_REPOSITORIO>
    cd <TU_REPOSITORIO>
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\Scripts\activate   # En Windows
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación Streamlit:**
    ```bash
    streamlit run app.py
    ```

    Esto abrirá la aplicación en tu navegador web por defecto.

## Despliegue en Streamlit Cloud

La aplicación está diseñada para ser fácilmente desplegable en [Streamlit Cloud](https://streamlit.io/cloud). Asegúrate de que tu repositorio de GitHub contiene `app.py`, `requirements.txt` y la estructura de `data/` con todos los archivos necesarios. Luego, sigue las instrucciones de Streamlit Cloud para conectar tu repositorio y desplegar la aplicación.

## Contacto

Para preguntas o sugerencias, por favor abre un 'issue' en este repositorio de GitHub.
