
# ============================================================================ #

import io
import s3fs
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

from shapely.geometry import Point # Para calcular métricas
from shapely.geometry import Polygon # Asegúrate de tener shapely instalado

from scipy.ndimage import gaussian_filter, label
from matplotlib.colors import ListedColormap, BoundaryNorm

from skimage.measure import find_contours

# ============================================================================ #

path_shape_depto = './data/shp_arg/operativo/departamentos_edit.shp'
path_shape_prov = './data/shp_arg/operativo/provincias_edit.shp'
path_shape_paises = './data/shp_arg/cartopy/10m_admin_0_countries.shp'
path_dir_FIR = './data/fir_txt/'
path_fir_ezeiza = './data/shp_arg/FIR/FIR_EZEIZA_backup.shp'
path_fir_cordoba = './data/shp_arg/FIR/FIR_CORDOBA.shp'
path_fir_resistencia = './data/shp_arg/FIR/FIR_RESISTENCIA.shp'
path_fir_mendoza = './data/shp_arg/FIR/FIR_MENDOZA.shp'
path_fir_comodoro = './data/shp_arg/FIR/FIR_COMODORO.shp'

# ============================================================================ #
# 0. Definiciones globales o constantes (fuera de funciones para Streamlit)
# ============================================================================ #

# Paleta de colores para radar real
aviation_colors = [
    "#00FF00",  # Verde  (Level 1: 20-30 dBZ) - Débil
    "#FFFF00",  # Amarillo (Level 2: 30-40 dBZ) - Moderado
    "#FF0000",  # Rojo   (Level 3: 40-50 dBZ) - Fuerte
    "#FF00FF",  # Magenta (Level 4: > 50 dBZ)  - Extremo / Granizo / Turbulencia
]
cmap_aviation = ListedColormap(aviation_colors)
levels_aviation = [20, 30, 40, 50, 65]
norm_aviation = BoundaryNorm(levels_aviation, cmap_aviation.N)

# Inicializar S3FileSystem una vez globalmente
fs_global = s3fs.S3FileSystem(anon=True)

# ============================================================================ #
# 1. Funciones auxiliares de carga y procesamiento (Cacheables con Streamlit)
# ============================================================================ #

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_glm_files_for_window(_fs, start_time, minutes=5):
    all_files = []
    num_steps = (minutes * 60) // 20
    for i in range(num_steps):
        current_time = start_time + timedelta(seconds=i * 20)
        time_prefix = current_time.strftime("s%Y%j%H%M%S")
        folder_path = f"noaa-goes19/GLM-L2-LCFA/{current_time.strftime('%Y/%j/%H/')}"
        try:
            matching_files = _fs.glob(f"{folder_path}*_{time_prefix}*")
            all_files.extend(matching_files)
        except:
            continue
    return all_files

@st.cache_data(ttl=3600) # Cachea los resultados por 1 hora
def get_abi_c13_file(_fs, target_time):
    time_prefix = target_time.strftime("s%Y%j%H%M")
    folder_path = f"noaa-goes19/ABI-L2-CMIPF/{target_time.strftime('%Y/%j/%H/')}"
    files = _fs.glob(f"{folder_path}*C13_*_{time_prefix}*")
    return files[0] if files else None

# La función cluster_and_get_polygons también debe estar definida en app.py
# Es una función intensiva, por lo que st.cache_data es muy útil aquí.
@st.cache_data
def cluster_and_get_polygons(reflectivity_data, threshold_dbz, lon_mesh, lat_mesh):
    thresholded_data = reflectivity_data >= threshold_dbz
    labeled_array, num_features = label(thresholded_data)

    polygons = []
    if num_features == 0:
        return polygons

    for i in range(1, num_features + 1):
        current_cluster_mask = labeled_array == i
        contours = find_contours(current_cluster_mask, level=0.5)

        for contour_idx, contour in enumerate(contours):
            r_indices = np.clip(np.round(contour[:, 0]).astype(int), 0, reflectivity_data.shape[0] - 1)
            c_indices = np.clip(np.round(contour[:, 1]).astype(int), 0, reflectivity_data.shape[1] - 1)

            lon_coords = lon_mesh[r_indices, c_indices]
            lat_coords = lat_mesh[r_indices, c_indices]

            if lon_coords.size > 0 and (lon_coords[0] != lon_coords[-1] or lat_coords[0] != lat_coords[-1]):
                lon_coords = np.append(lon_coords, lon_coords[0])
                lat_coords = np.append(lat_coords, lat_coords[0])

            if len(lon_coords) > 2:
                poly_coords = list(zip(lon_coords, lat_coords))
                try:
                    poly = Polygon(poly_coords)
                    if not poly.is_empty and poly.is_valid:
                        polygons.append(poly)
                except Exception as e:
                    st.warning(f"Error creando Polígono del contorno: {e}") # st.warning para Streamlit

    return polygons

@st.cache_resource # st.cache_resource es mejor para objetos no serializables como ShapelyFeature
def load_shape_features(path_shp):
    """Carga un shapefile y devuelve un objeto ShapelyFeature."""
    try:
        return ShapelyFeature(Reader(path_shp).geometries(), ccrs.PlateCarree())
    except Exception as e:
        st.error(f"Error cargando shapefile {path_shp}: {e}")
        return None


@st.cache_data # cache_data para DataFrames
def load_airport_data(path_csv):
    """Carga los datos de aeropuertos desde un CSV."""
    try:
        return pd.read_csv(path_csv,
                         sep=r'\s+',
                         header=None,
                         names=['Codigo ICAO','Lat','Lon'])
    except Exception as e:
        st.error(f"Error cargando datos de aeropuertos {path_csv}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600) # Cachea los resultados de todo el procesamiento de datos
def load_and_process_data(start_window_datetime, _fs_param):
    # fs = s3fs.S3FileSystem(anon=True) # Ya no se crea aquí, se recibe como _fs_param

    # --- RUTAS RELATIVAS A LOS ARCHIVOS DE DATOS EN TU REPOSITORIO GITHUB ---
    # Asegúrate de que estos archivos estén en la estructura 'data/' en tu repo.
    path_shape_depto_rel = './data/shp_arg/operativo/departamentos_edit.shp'
    path_shape_prov_rel = './data/shp_arg/operativo/provincias_edit.shp'
    path_shape_paises_rel = './data/shp_arg/cartopy/10m_admin_0_countries.shp'
    path_dir_FIR_rel = './data/fir_txt/FIR_aeropuertos.txt' # Ruta completa al archivo TXT
    path_fir_ezeiza_rel = './data/shp_arg/FIR/FIR_EZEIZA_backup.shp'
    path_fir_cordoba_rel = './data/shp_arg/FIR/FIR_CORDOBA.shp'
    path_fir_resistencia_rel = './data/shp_arg/FIR/FIR_RESISTENCIA.shp'
    path_fir_mendoza_rel = './data/shp_arg/FIR/FIR_MENDOZA.shp'
    path_fir_comodoro_rel = './data/shp_arg/FIR/FIR_COMODORO.shp'

    # Cargar shapefiles y datos de aeropuertos
    # municipios = load_shape_features(path_shape_depto_rel) # Si lo necesitas, descomenta
    # provincias = load_shape_features(path_shape_prov_rel) # Si lo necesitas, descomenta
    paises = load_shape_features(path_shape_paises_rel)
    df_airports = load_airport_data(path_dir_FIR_rel)
    fir_ezeiza = load_shape_features(path_fir_ezeiza_rel)
    fir_cordoba = load_shape_features(path_fir_cordoba_rel)
    fir_resistencia = load_shape_features(path_fir_resistencia_rel)
    fir_mendoza = load_shape_features(path_fir_mendoza_rel)
    fir_comodoro = load_shape_features(path_fir_comodoro_rel)


    # --- Carga y preprocesamiento de datos GLM y ABI ---
    glm_files = get_glm_files_for_window(_fs_param, start_window_datetime, minutes=10)
    abi_file = get_abi_c13_file(_fs_param, start_window_datetime)

    if not glm_files:
        st.warning(f"No se encontraron archivos GLM para {start_window_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        return None # O manejar el caso de no datos
    if abi_file is None:
        st.warning(f"No se encontró archivo ABI C13 para {start_window_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        return None # O manejar el caso de no datos

    accumulated_lats = []
    accumulated_lons = []
    for file_path in glm_files:
        with _fs_param.open(file_path, "rb") as f:
            with Dataset("dummy", mode="r", memory=f.read()) as nc:
                accumulated_lats.extend(nc.variables["flash_lat"][:])
                accumulated_lons.extend(nc.variables["flash_lon"][:])

    all_lats = np.array(accumulated_lats)
    all_lons = np.array(accumulated_lons)

    with _fs_param.open(abi_file, "rb") as f:
        with Dataset("dummy", mode="r", memory=f.read()) as nc:
            ir_data = nc.variables["CMI"][:] - 273.15
            proj_info = nc.variables["goes_imager_projection"]
            h = proj_info.perspective_point_height
            x = nc.variables["x"][:] * h
            y = nc.variables["y"] * h
            abi_crs = ccrs.Geostationary(central_longitude=proj_info.longitude_of_projection_origin, satellite_height=h)

    lat_min, lat_max = -45.0, -19.0
    lon_min, lon_max = -75.0, -50.0
    grid_res_high = 0.05
    lat_bins_high = np.arange(lat_min, lat_max + grid_res_high, grid_res_high)
    lon_bins_high = np.arange(lon_min, lon_max + grid_res_high, grid_res_high)

    fed_high, _, _ = np.histogram2d(all_lats, all_lons, bins=[lat_bins_high, lon_bins_high])
    fed_smoothed = gaussian_filter(fed_high, sigma=1.5)

    max_reflectivity_proxy = np.zeros_like(fed_smoothed)
    mask = fed_smoothed > 0.1
    max_reflectivity_proxy[mask] = 33.0 + 10.0 * np.log10(fed_smoothed[mask])

    lon_mesh_high, lat_mesh_high = np.meshgrid(
        (lon_bins_high[:-1] + lon_bins_high[1:]) / 2,
        (lat_bins_high[:-1] + lat_bins_high[1:]) / 2
    )

    # --- Detección de Polígonos de Advertencia y Métricas ---
    warning_threshold_dbz = 30
    warning_polygons = cluster_and_get_polygons(max_reflectivity_proxy, warning_threshold_dbz, lon_mesh_high, lat_mesh_high)

    metrics_list = []
    if warning_polygons:
        lon_flat = lon_mesh_high.flatten()
        lat_flat = lat_mesh_high.flatten()
        grid_points = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]

        for idx, poly in enumerate(warning_polygons):
            poly_id = idx + 1
            centroid_lon = poly.centroid.x
            centroid_lat = poly.centroid.y

            reflectivity_values_in_poly = []
            ir_temps_in_poly = []
            fed_values_in_poly = []
            num_pixels = 0

            for i_flat in range(len(grid_points)):
                current_point = grid_points[i_flat]
                if poly.contains(current_point):
                    num_pixels += 1
                    r, c = np.unravel_index(i_flat, max_reflectivity_proxy.shape)

                    reflectivity_values_in_poly.append(max_reflectivity_proxy[r, c])
                    fed_values_in_poly.append(fed_smoothed[r, c])

                    x_transformed, y_transformed = abi_crs.transform_point(lon_mesh_high[r, c], lat_mesh_high[r, c], ccrs.PlateCarree())
                    idx_x_abi = np.argmin(np.abs(x - x_transformed))
                    idx_y_abi = np.argmin(np.abs(y - y_transformed))
                    ir_temps_in_poly.append(ir_data[idx_y_abi, idx_x_abi])

            max_reflectivity = np.max(reflectivity_values_in_poly) if reflectivity_values_in_poly else np.nan
            max_fed = np.max(fed_values_in_poly) if fed_values_in_poly else np.nan
            min_ir_temp = np.min(ir_temps_in_poly) if ir_temps_in_poly else np.nan

            metrics_list.append({
                'ID': poly_id,
                'CenLon': centroid_lon,
                'CenLat': centroid_lat,
                'Pixels': num_pixels,
                'MaxRef': max_reflectivity,
                'MaxFED': max_fed,
                'MinIR': min_ir_temp
            })

    metrics_df = pd.DataFrame(metrics_list)

    return {
        "warning_polygons": warning_polygons,
        "metrics_df": metrics_df,
        "ir_data": ir_data,
        "x": x,
        "y": y,
        "abi_crs": abi_crs,
        "max_reflectivity_proxy": max_reflectivity_proxy,
        "lon_mesh_high": lon_mesh_high,
        "lat_mesh_high": lat_mesh_high,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "paises": paises,
        "fir_ezeiza": fir_ezeiza,
        "fir_cordoba": fir_cordoba,
        "fir_resistencia": fir_resistencia,
        "fir_mendoza": fir_mendoza,
        "fir_comodoro": fir_comodoro,
        "df_airports": df_airports,
        "start_window": start_window_datetime
    }

# ============================================================================ #
# 2. Función para dibujar el mapa (adaptada de tu cuaderno)                   #
# ============================================================================ #

def plot_interactive_map_streamlit(
    warning_polygons, metrics_df, ir_data, x, y, abi_crs,
    max_reflectivity_proxy, lon_mesh_high, lat_mesh_high,
    lon_min, lon_max, lat_min, lat_max,
    paises, fir_ezeiza, fir_cordoba, fir_resistencia, fir_mendoza, fir_comodoro,
    df_airports, start_window,
    highlight_poly_id=None
):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=ccrs.Mercator())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    im_ir = ax.imshow(
        ir_data, origin="upper",
        extent=[x.min(), x.max(), y.min(), y.max()],
        transform=abi_crs,
        cmap="Greys", vmin=-90, vmax=40, zorder=1
    )

    proxy_masked = np.ma.masked_where(max_reflectivity_proxy < 10, max_reflectivity_proxy)

    im_proxy = ax.pcolormesh(
        lon_mesh_high, lat_mesh_high, proxy_masked,
        cmap=cmap_aviation, norm=norm_aviation,
        alpha=0.75, transform=ccrs.PlateCarree(), zorder=2
    )

    if paises is not None: ax.add_feature(paises, facecolor='None', edgecolor='gray', linewidth=1.5)
    if fir_ezeiza is not None: ax.add_feature(fir_ezeiza, facecolor='None', edgecolor='blue', linewidth=1.5, zorder=2)
    if fir_cordoba is not None: ax.add_feature(fir_cordoba, facecolor='None', edgecolor='blue', linewidth=1.5, zorder=2)
    if fir_resistencia is not None: ax.add_feature(fir_resistencia, facecolor='None', edgecolor='blue', linewidth=1.5, zorder=2)
    if fir_mendoza is not None: ax.add_feature(fir_mendoza, facecolor='None', edgecolor='blue', linewidth=1.5, zorder=2)
    if fir_comodoro is not None: ax.add_feature(fir_comodoro, facecolor='None', edgecolor='blue', linewidth=1.5, zorder=2)

    if not df_airports.empty:
        for i, type_code in enumerate(df_airports['Codigo ICAO'].values):
            px = df_airports['Lon'].values[i]
            py = df_airports['Lat'].values[i]
            if (lon_min < px < lon_max) and (lat_min < py < lat_max):
                plt.scatter(px, py, marker='s', s=10, color='b', zorder=5, transform=ccrs.PlateCarree())
                plt.text(px + 0.15, py - 0.21, type_code, fontsize=8, c='b', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

    cbar_proxy = plt.colorbar(
        im_proxy, ax=ax, orientation="horizontal", pad=0.01, shrink=0.65,
        ticks=[25, 35, 45, 57.5]
    )
    cbar_proxy.ax.set_xticklabels(["Leve", "Moderado", "Fuerte", "Intenso / Extremo"], fontsize=11)
    cbar_proxy.ax.tick_params(axis='x', length=0)

    for poly_idx, poly in enumerate(warning_polygons):
        current_id = poly_idx + 1
        edge_color = 'cyan'
        line_width = 2

        if highlight_poly_id == current_id:
            edge_color = 'red'
            line_width = 4
            zorder = 7
        else:
            zorder = 6

        ax.add_geometries([poly], ccrs.PlateCarree(),
                          facecolor='none', edgecolor=edge_color, linewidth=line_width, linestyle='-', zorder=zorder)

    plt.title(
        f"GOES-19 Canal 13 IR + Proxy radar GLM 5-min\n"
        f"{start_window.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=12
    )
    plt.tight_layout()
    return fig

# ============================================================================ #
# 3. Estructura de la aplicación Streamlit (Main App Logic)                  #
# ============================================================================ #

st.set_page_config(layout="wide")
st.image("smn_horizontal_arg-01.jpg", width=250) # Puedes ajustar el número para cambiar el tamaño
st.title("Producto TS-SIGMET | Dashboard Interactivo ")

# Selector de fecha/hora para permitir al usuario elegir el momento a visualizar
# Puedes establecer un rango por defecto o fechas con datos conocidos.
initial_datetime = datetime(2025, 11, 4, 3, 0, 0) # Fecha de ejemplo con actividad

# Streamlit slider para seleccionar la hora (solo la hora, para simplificar)
# O puedes usar st.date_input y st.time_input para control total
selected_date = st.date_input("Selecciona la fecha", value=initial_datetime.date())
selected_time = st.time_input("Selecciona la hora (UTC)", value=initial_datetime.time(), step=300) # Paso de 5 minutos

start_window_user = datetime.combine(selected_date, selected_time)

# Carga y procesa los datos (se cacheará)
data = load_and_process_data(start_window_user, fs_global)

if data is None: # Si no se pudieron cargar los datos (ej. archivos GLM/ABI no encontrados)
    st.warning("No se pudieron cargar los datos para la fecha y hora seleccionadas. Intenta con otra fecha/hora.")
else:
    warning_polygons = data["warning_polygons"]
    metrics_df = data["metrics_df"]

    # Columna izquierda para el mapa, columna derecha para la tabla y selección
    col1, col2 = st.columns([1, 1])

    with col2:
        st.header("Métricas de Polígonos")
        options = [f"ID: {int(row.ID)}, MaxdBZ: {row.MaxRef:.2f}, MaxFED: {row.MaxFED:.2f}"
                   for idx, row in metrics_df.iterrows()]
        options.insert(0, "-- Seleccionar Polígono --")

        selected_option = st.selectbox(
            "Seleccionar un polígono para resaltar en el mapa:",
            options,
            index=0
        )

        highlight_poly_id = None
        if selected_option != "-- Seleccionar Polígono --":
            highlight_poly_id = int(float(selected_option.split(',')[0].replace('ID: ', '')))

        st.dataframe(metrics_df, height=600)

        # Botón de descarga para metrics_df como CSV
        if not metrics_df.empty:
            csv_buffer = io.StringIO()
            metrics_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Descargar Métricas como CSV",
                data=csv_buffer.getvalue(),
                file_name="warning_polygons_metrics.csv",
                mime="text/csv",
            )

    with col1:
        st.header("Mapa de Polígonos de Advertencia")
        fig = plot_interactive_map_streamlit(
            warning_polygons, metrics_df,
            data["ir_data"], data["x"], data["y"], data["abi_crs"],
            data["max_reflectivity_proxy"], data["lon_mesh_high"], data["lat_mesh_high"],
            data["lon_min"], data["lon_max"], data["lat_min"], data["lat_max"],
            data["paises"], data["fir_ezeiza"], data["fir_cordoba"], data["fir_resistencia"], data["fir_mendoza"], data["fir_comodoro"],
            data["df_airports"], data["start_window"],
            highlight_poly_id=highlight_poly_id
        )
        st.pyplot(fig)

# ============================================================================ #

st.markdown("""
	    **Cómo ejecutar esta aplicación Streamlit:**
		1.  Guarda el código anterior como un archivo `.py` (ej. `app.py`).
		2.  Asegúrate de que tienes un archivo `requirements.txt` con todas las librerías necesarias.
		3.  Asegúrate de tener tus archivos de datos organizados en la carpeta `data/` dentro de tu repositorio de GitHub.
		4.  Abre una terminal en el directorio donde guardaste `app.py`.
		5.  Ejecuta el comando: `streamlit run app.py`
		6.  Se abrirá una nueva pestaña en tu navegador con la aplicación Streamlit.
	    """)

# ============================================================================ #

