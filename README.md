# GLM-FED Radar Proxy 🌩️✈️

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Meteorology](https://img.shields.io/badge/Meteorology-Aviation-orange)
![GOES](https://img.shields.io/badge/Satellite-GOES--R-success)

Aplicativo para la generación de un **proxy de reflectividad radar** a partir del producto **FED (Flash Extent Density)** del sensor **GLM (Geostationary Lightning Mapper)** a bordo de los satélites de la serie GOES-R. 

Este desarrollo está orientado a la identificación y monitoreo de núcleos convectivos de alta peligrosidad para vuelos comerciales, sirviendo como herramienta central para la **automatización de la emisión de mensajes SIGMET por tormentas**.

---

## 1. Fundamento Microfísico y Electrodinámico

El vínculo entre la reflectividad de radar ($Z$) y la actividad de rayos observada desde satélites geoestacionarios (como el sensor GLM de la serie GOES-R o el LMI de FengYun-4A) radica en la **microfísica de fase mixta** dentro de las corrientes ascendentes (*updrafts*):

1. **Carga No Inductiva:** La electrificación principal de las nubes convectivas se produce por colisiones entre granizo blando/graupel y cristales de hielo en presencia de agua líquida sobreenfriada en la zona de temperaturas entre $0^\circ\text{C}$ y $-40^\circ\text{C}$ (particularmente activa entre $-10^\circ\text{C}$ y $-25^\circ\text{C}$).
2. **Volumen de Graupel vs. Reflectividad:** El radar meteorológico en microondas (bandas S, C, X) es sensible a la sexta potencia del diámetro de los hidrometeoros ($Z \propto \sum D_i^6$). Las partículas grandes de fase mixta (graupel y granizo) generan los ecos más elevados de reflectividad ($> 40\text{ dBZ}$).
3. **Escalamiento Físico:** Como la tasa de descargas eléctricas escala aproximadamente con la masa o flujo de colisión de graupel dentro del volumen de la corriente ascendente ($f \propto w \cdot M_{graupel}$ o $f \propto w^4 - w^6$), existe una estrecha correlación no lineal entre la densidad de flashes (ej. *Flash Extent Density*, FED) y el valor máximo de reflectividad en la columna vertical.

---

## 2. El sensor GLM y el producto FED
El **GLM (Geostationary Lightning Mapper)**, a bordo de satélites como el GOES-16 y -19, es el primer mapeador de rayos óptico en órbita geoestacionaria. Este instrumento detecta continuamente la actividad eléctrica total (relámpagos intra-nube y nube-tierra) con una resolución espacial de 8-14 km.

Dentro de sus variables derivadas, el **FED (Flash Extent Density)** contabiliza el número de destellos que ocurren o atraviesan un píxel específico en un intervalo de tiempo. La literatura científica demuestra que la densidad y extensión de estos destellos están fuertemente correlacionadas con:
- La intensidad de las corrientes ascendentes (*updrafts*).
- El volumen de la región de fase mixta de la nube (donde coexiste agua sobreenfriada y hielo).
- La presencia de corrientes descendentes severas, granizo y turbulencia asociada a la convección profunda.

---

## 3. Formulaciones Clásicas de Radar Proxy (GSI / WRF-DA)

En sistemas operativos de asimilación de datos (como el *Gridpoint Statistical Interpolation*, GSI, y el modelo RAP/HRRR del NCEP/NOAA), la densidad de descargas en una celda se convierte a una reflectividad bidimensional máxima ($Z_{max, proxy}$) mediante relaciones empíricas continuas o segmentadas.

### 3.1 Modelo Logarítmico Clásico (GSI-1)
$$Z_{\text{max, proxy}} = A \cdot \log_{10}(\text{FED} + 1) + B$$

Donde:
* $\text{FED}$ es la densidad de extensión de flashes acumulada en una ventana temporal típica (ej. 5 a 15 minutos).
* $A$ y $B$ son parámetros de ajuste empírico calibrados contra redes de radar terrestres ($A \approx 10.0 - 15.0$, $B \approx 30.0 - 35.0\text{ dBZ}$).
* El valor resultante suele saturarse operativamente entre $35\text{ dBZ}$ (umbral de inicio de convección profunda) y $60 - 65\text{ dBZ}$ (núcleos severos con granizo).

### 3.2 Modelo Lineal por Tramos / Umbrales Discretos (GSI-2)
Para evitar sobredimensionar la reflectividad en celdas con actividad eléctrica esporádica:

$$Z_{\text{max, proxy}} = \begin{cases} 0 & \text{si } \text{FED} = 0 \\
35.0 + C_1 \cdot \text{FED} & \text{si } 0 < \text{FED} \le \text{FED}_{th} \\
55.0 + C_2 \cdot \log_{10}(\text{FED}) & \text{si } \text{FED} > \text{FED}_{th}
\end{cases}$$

### 3.3 **Visualización Aeronáutica (ARINC 708):**
La reflectividad se categoriza bajo el estándar operacional de radar de cabina:
   - **Level 1 (Verde, 20–30 dBZ):** Retornos débiles / Lluvia ligera
   - **Level 2 (Amarillo, 30–40 dBZ):** Retornos moderados / Lluvia moderada
   - **Level 3 (Rojo, 40–50 dBZ):** Convección fuerte
   - **Level 4 (Magenta, > 50 dBZ):** Convección severa / Granizo / Turbulencia extrema

---

## 4. Revisión de Trabajos Destacados

### 4.1 Chen et al. (2020) — Recuperación 3D y Asimilación de LMI / FY-4A
* **Referencia:** Chen, Y., Yu, Z., Han, W., He, J., & Chen, M. (2020). *Case Study of a Retrieval Method of 3D Proxy Reflectivity from FY-4A Lightning Data and Its Impact on the Assimilation and Forecasting for Severe Rainfall Storms*. *Remote Sensing*, 12(7), 1165. [DOI: 10.3390/rs12071165](https://doi.org/10.3390/rs12071165).

#### Aportes Principales:
1. **Relación Empírica Mejorada:** Propusieron una función de transferencia logarítmica adaptada al sensor geoestacionario asiático LMI (Lightning Mapping Imager) en una malla de 3 km y 13 km:
   $$Z_{\text{max}} = a \cdot \ln(\text{LD} + 1) + b$$
   demostrando que las relaciones en mallas finas de alta resolución reproducen mejor la variabilidad espectral del radar observado que las formulaciones estándar de GSI.
2. **Extensión Vertical 3D (Construcción del Perfil):** A partir del valor de reflectividad máxima $Z_{\text{max}}$, reconstruyen la estructura vertical $Z(x, y, z)$ aplicando factores de ponderación vertical $W(z)$ derivados de perfiles estadísticos y térmicos (nivel de congelamiento, altura de cima nubosa):
   $$Z(x, y, z) = Z_{\text{max}}(x, y) \cdot W(z)$$
3. **Impacto en Pronóstico Numérico (NWP):** La asimilación de la reflectividad proxy 3D en el sistema RMAPS-ST (*Rapid-refresh Multi-scale Analysis and Prediction System*) mejoró de forma sostenida los índices de precipitación (*Fractions Skill Score*, FSS) hasta 6 horas de pronóstico, particularmente en regiones montañosas y de costa donde los radares terrestres sufren bloqueo de haz o falta de cobertura.

---

### 4.2 Gómez Mayol y otros (2020) — Contexto Regional en Sudamérica
* **Referencia:** Gómez Mayol, M., Vidal, L., Salio, P., & Sacco, M. (2020). *Sobre el uso de datos de rayos como proxy para la reflectividad radar en la región central de Argentina*. *Meteorologica*, 45.

#### Aportes Principales:
1. **Calibración en Convección Extrema Sudamericana:** La región central y norte de Argentina y la cuenca del Plata albergan algunas de las tormentas convectivas más profundas del planeta (con frecuencia asociadas a granizo gigante y cimas penetrantes, analizadas en proyectos como RELAMPAGO-CACTI).
2. **Comparación Radar (RMA/SMN) vs. Redes de Rayos:** El trabajo analizó la relación entre datos de descargas (redes terrestres y sensores satelitales) frente a volúmenes de radar polarimétrico, evidenciando que las tormentas locales presentan gradientes de reflectividad más pronunciados y mayor densidad de carga en niveles altos en comparación con los esquemas ajustados en EE. UU.
3. **Modelado Avanzado (Redes Neuronales / Machine Learning):** Evaluaron la transición desde curvas puramente estadísticas logarítmicas hacia técnicas de aprendizaje profundo (incluyendo arquitecturas convolucionales y generativas condicionales - cGANs) para predecir cortes CAPPI y reflectividad compuesta ($Z_{\text{comp}}$) a partir de descargas eléctricas y temperatura de brillo infrarrojo de GOES ABI (Canal 13).

---

## 5. Comparativa Metodológica

| Enfoque | Variables de Entrada | Resolución Típica | Ventajas | Limitaciones |
| :--- | :--- | :--- | :--- | :--- |
| **GSI / Clásico Empírico** | FED (GLM/LMA) | 3 – 13 km | Computacionalmente instantáneo; fácil integración en modelos 3D-Var. | Tiende a producir "parches" uniformes; ignora lluvia estratiforme. |
| **Chen et al. (2020)** | Densidad de Flashes LMI + Perfil térmico | 3 km / 1 h / 5 min | Genera perfiles 3D coherentes listos para operadores de asimilación radar. | Requiere conocimiento de la altura del nivel de fusión ($0^\circ\text{C}$). |
| **Gómez Mayol et al. (2020)** | Rayos + IR ABI GOES + Radares RMA | 1 – 4 km | Calibrado para la convección sudamericana; incorpora IA para capturar formas convectivas complejas. | Mayor costo computacional si se usan modelos neuronales en tiempo real. |

---

## 6. Pipeline Recomendado de Implementación en Python

Para generar un campo de $Z_{proxy}$ operacionalmente balanceado que combine la física de GLM con el contexto de nubes de ABI C13:

1. **Acumulación Temporal:** 5 minutos (15 archivos de GLM-L2-LCFA en AWS S3).
2. **Suavizado Espacial:** Filtro Gaussiano ($\sigma \approx 1.5 - 2.5$) para emular la dispersión lateral de hidrometeoros y la divergencia en el tope convectivo.
3. **Máscara Condicional:** Restringir la generación de $Z_{proxy}$ a zonas con $T_B(C13) \le -30^\circ\text{C}$ para evitar falsos ecos en descargas aisladas en bordes secos.
4. **Categorización Aeronáutica (ARINC 708):** Discretización en 4 niveles operativos (Verde: $20-30\text{ dBZ}$, Amarillo: $30-40\text{ dBZ}$, Rojo: $40-50\text{ dBZ}$, Magenta: $> 50\text{ dBZ}$).

---

## 7. Referencias Bibliográficas

1. **Chen, Y., Yu, Z., Han, W., He, J., & Chen, M. (2020).** *Case Study of a Retrieval Method of 3D Proxy Reflectivity from FY-4A Lightning Data and Its Impact on the Assimilation and Forecasting for Severe Rainfall Storms*. *Remote Sensing*, 12(7), 1165. [https://doi.org/10.3390/rs12071165](https://doi.org/10.3390/rs12071165)
2. **Gómez Mayol, M., Vidal, L., Salio, P., & Sacco, M. (2020).** *Sobre el uso de datos de rayos como proxy para la reflectividad radar en la región central de Argentina*. *Meteorologica*, 45. [http://www.meteorologica.org.ar/wp-content/uploads/2020/11/Gomez_y_otros_2020.pdf](http://www.meteorologica.org.ar/wp-content/uploads/2020/11/Gomez_y_otros_2020.pdf)
3. **Fierro, A. O., Mansell, E. R., Ziegler, C. L., & MacGorman, D. R. (2012).** *Application of a Lightning Data Assimilation Technique in the WRF-ARW Model at Cloud-Resolving Scales for the 8 May 2003 Oklahoma City Supercell*. *Monthly Weather Review*, 140(8), 2609–2627.
4. **Bruning, E. C., Tillier, C. E., Edgington, S. F., Rudlosky, S. D., Zajic, J., Gravelle, C., ... & Calhoun, K. (2019).** *Meteorological imagery for the Geostationary Lightning Mapper*. *Journal of Geophysical Research: Atmospheres*, 124(24), 14285-14309.

---

# 🚀 Requisitos e Instalación

Instala las dependencias necesarias mediante `pip`:

```bash
pip install s3fs netCDF4 numpy matplotlib cartopy scipy

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
