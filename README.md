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
