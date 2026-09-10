
# ============================================================================ #
# 1. Librerías estándar de Python
# ============================================================================ #
from datetime import datetime, timedelta
import io
from typing import Any, Dict, List, Optional

# ============================================================================ #
# 2. Computación científica, acceso a datos y entorno web
# ============================================================================ #
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import s3fs
import streamlit as st

# ============================================================================ #
# 3. Procesamiento geoespacial y análisis geométrico
# ============================================================================ #
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from shapely.geometry import Point, Polygon, box

# ============================================================================ #
# 4. Visión por computadora y procesamiento de imágenes
# ============================================================================ #
from scipy.ndimage import gaussian_filter, label
from skimage.measure import find_contours

# ============================================================================ #
# 5. Renderizado y visualización gráfica
# ============================================================================ #
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# ============================================================================ #
# 0. Constantes y Configuraciones Operativas
# ============================================================================ #

# Estilo de entramado para polígonos SIGMET en Cartopy
mpl.rcParams["hatch.linewidth"] = 0.8

# Escala de reflectividad aeronáutica y niveles de corte
AVIATION_COLORS = [
                   "#00FF00",  # Nivel 1: 20-30 dBZ (Leve)
                   "#FFFF00",  # Nivel 2: 30-40 dBZ (Moderado)
                   "#FF0000",  # Nivel 3: 40-50 dBZ (Fuerte)
                   "#FF00FF",  # Nivel 4: > 50 dBZ (Extremo)
                  ]
CMAP_AVIATION = ListedColormap(AVIATION_COLORS)
LEVELS_AVIATION = [20, 30, 40, 50, 65]
NORM_AVIATION = BoundaryNorm(LEVELS_AVIATION, CMAP_AVIATION.N)

# Parámetros físicos y límites espaciales (Argentina / Cono Sur)
SPATIAL_BOUNDS = {
                  "data_lat_min": -47.0,
                  "data_lat_max": -18.5,
                  "data_lon_min": -75.5,
                  "data_lon_max": -37.0,
                  "plot_lat_min": -45.0,
                  "plot_lat_max": -19.0,
                  "plot_lon_min": -75.0,
                  "plot_lon_max": -50.0,
                 }

# Rutas de capas vectoriales operativas
SHAPEFILE_PATHS = {
                   "paises": "./data/shp_arg/cartopy/10m_admin_0_countries.shp",
                   "airports": "./data/fir_txt/FIR_aeropuertos.txt",
                   "fir_ezeiza": "./data/shp_arg/FIR/FIR_EZEIZA_backup.shp",
                   "fir_cordoba": "./data/shp_arg/FIR/FIR_CORDOBA.shp",
                   "fir_resistencia": "./data/shp_arg/FIR/FIR_RESISTENCIA.shp",
                   "fir_mendoza": "./data/shp_arg/FIR/FIR_MENDOZA.shp",
                   "fir_comodoro": "./data/shp_arg/FIR/FIR_COMODORO.shp",
                  }

# Conexión persistente de solo lectura para AWS S3
FS_GLOBAL = s3fs.S3FileSystem(anon=True)

# ============================================================================ #
# 1. Funciones Físicas y Meteorológicas
# ============================================================================ #

def pressure_to_flight_level(p_hpa: float) -> Optional[int]:
    """Calcula el Nivel de Vuelo (FL) según la atmósfera estándar ISA.

    Parameters
    ----------
    p_hpa : float
        Presión atmosférica en hectopascales (hPa).

    Returns
    -------
    Optional[int]
        Nivel de vuelo redondeado a múltiplos de 10 (ej. FL380), o np.nan si inválido.
    """
    if p_hpa <= 0 or np.isnan(p_hpa) or np.ma.is_masked(p_hpa):
        return np.nan

    # Troposfera (hasta ~36,000 ft / 226.32 hPa)
    if p_hpa > 226.32:
        alt_ft = 145366.45 * (1 - (p_hpa / 1013.25) ** 0.190284)
    # Tropopausa / Baja Estratosfera (> 36,000 ft)
    else:
        alt_ft = 36089.24 - 20805.7 * np.log(p_hpa / 226.32)

    fl_exact = alt_ft / 100.0
    return int(round(fl_exact / 10.0) * 10)

def pressure_to_altitude_km(p_hpa: float) -> float:
    """Convierte la presión en hPa a altitud geopotencial en kilómetros (km).

    Parameters
    ----------
    p_hpa : float
        Presión en hectopascales.

    Returns
    -------
    float
        Altitud estimada en km redondeada a un decimal.
    """
    if p_hpa <= 0 or np.isnan(p_hpa) or np.ma.is_masked(p_hpa):
        return np.nan

    if p_hpa > 226.32:
        alt_ft = 145366.45 * (1 - (p_hpa / 1013.25) ** 0.190284)
    else:
        alt_ft = 36089.24 - 20805.7 * np.log(p_hpa / 226.32)

    return round((alt_ft * 0.3048) / 1000.0, 1)

def rumbo_to_arrow(angle_deg: float) -> str:
    """Genera una flecha y acrónimo azimutal a partir del ángulo con el Norte."""
    val = angle_deg % 180.0
    if val <= 22.5 or val > 157.5:
        return "↕ S-N"
    elif 22.5 < val <= 67.5:
        return "↗ SW-NE"
    elif 67.5 < val <= 112.5:
        return "↔ W-E"
    return "↘ NW-SE"

def compute_sigmet_convex_hull_properties(
    poly: Polygon, simplify_deg: float = 0.08) -> Dict[str, Any]:
    """Calcula la envolvente convexa simplificada y dimensiones físicas en km.

    Aplica el algoritmo Ramer-Douglas-Peucker y un rectángulo circunscrito
    mínimo orientado para derivar ejes principales y orientación azimutal.
    """
    hull = poly.convex_hull
    hull_simplified = hull.simplify(tolerance=simplify_deg, preserve_topology=True)
    if not hull_simplified.is_valid or hull_simplified.geom_type != "Polygon":
        hull_simplified = hull

    centroid_lat = hull_simplified.centroid.y
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(centroid_lat))

    min_rect = hull_simplified.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]

    lados_km = []
    vectores = []
    for k in range(4):
        p1, p2 = rect_coords[k], rect_coords[(k + 1) % 4]
        dx_km = (p2[0] - p1[0]) * km_per_deg_lon
        dy_km = (p2[1] - p1[1]) * km_per_deg_lat
        lados_km.append(np.hypot(dx_km, dy_km))
        vectores.append((dx_km, dy_km))

    idx_major = int(np.argmax(lados_km[:2]))
    major_axis_km = max(lados_km[0], lados_km[1])
    minor_axis_km = min(lados_km[0], lados_km[1])

    dx_maj, dy_maj = vectores[idx_major]
    angle_deg = np.degrees(np.arctan2(dx_maj, dy_maj)) % 180.0
    area_hull_km2 = hull_simplified.area * km_per_deg_lon * km_per_deg_lat

    return {
            "hull_polygon": hull_simplified,
            "vertices": len(list(hull_simplified.exterior.coords)) - 1,
            "major_axis_km": round(major_axis_km, 1),
            "minor_axis_km": round(minor_axis_km, 1),
            "orientation_deg": int(round(angle_deg)),
            "area_hull_km2": round(area_hull_km2, 1),
           }

def classify_convective_morphology(
    area_km2: float, major_axis_km: float, minor_axis_km: float, max_dbz: float) -> Dict[str, str]:
    """
    Clasifica el sistema convectivo siguiendo criterios morfológicos de radar:
    - IC  : Isolated Cell (Celda Individual / Celda Aislada)
    - CC  : Cluster of Cells (Clúster Convectivo Multicelular)
    - QLCS: Quasi-Linear Convective System / Squall Line (Línea Convectiva)
    - MCS : Mesoscale Convective System (Sistema Convectivo de Mesoescala)
    """
    minor_axis = max(minor_axis_km, 1.0)
    aspect_ratio = major_axis_km / minor_axis

    if major_axis_km >= 100.0 and aspect_ratio >= 3.0:
        return {
                "codigo": "QLCS",
                "tipo": "Quasi-Linear Convective System / Squall Line (Línea Convectiva)",
                "peligros": "Frentes de ráfagas violentos (*gust fronts*), cortante horizontal/vertical del viento (*low-level windshear*), turbulencia extrema a lo largo del frente y granizo que puede proyectarse varios kilómetros por delante del borde de ataque.",
                "impacto": "Bloqueo transversal total de aerovías. La penetración frontal está formalmente contraindicada. Se requieren desvíos de largo radio circunvalando los extremos de la línea o demoras en circuito de espera hasta el pasaje del sistema.",
               }
    elif area_km2 >= 1000.0 or (major_axis_km >= 100.0 and minor_axis_km >= 40.0):
        return {
                "codigo": "MCS",
                "tipo": "Mesoscale Convective System (Sistema Convectivo de Mesoescala)",
                "peligros": "Engelamiento severo generalizado en niveles de crucero, topes nubosos penetrantes (overshooting tops) que superan **FL400**, y actividad eléctrica intra-nube y nube-tierra continua.",
                "impacto": "Disrupción masiva del espacio aéreo (escala FIR). Colapso de rutas troncales y sectores de control. Exige reformulación de planes de vuelo, desvíos interprovinciales obligatorios y aplicación inmediata de procedimientos de contingencia y espaciamiento por flujo (ATFM).",
               }
    elif area_km2 >= 400.0 or major_axis_km >= 50.0:
        return {
                "codigo": "CC",
                "tipo": "Cluster of Cells (Clúster Convectivo Multicelular)",
                "peligros": "Turbulencia severa en aire claro (CAT), engelamiento fuerte en niveles medios y presencia de 'corredores engañosos' (*blind alleys*) entre núcleos activos.",
                "impacto": "Prohibida la penetración a través de brechas estrechas entre ecos con reflectividad >35 dBZ. Rutas de desvío estratégicas; requiere coordinación temprana con control de ruta para evitar atrapamiento entre celdas secundarias.",
               }
    return {
            "codigo": "IC",
            "tipo": "Isolated Cell (Celda Individual / Celda Aislada)",
            "peligros": "Microfrentes de ráfagas locales (*microbursts*), granizo localizado y turbulencia severa acotada al núcleo y su entorno inmediato (<5 NM)",
            "impacto": "Desvíos tácticos mínimos (5 a 10 NM a barlovento). Alta probabilidad de circunnavegación visual o con radar de a bordo (WXR) sin saturar los sectores terminales",
           }

# ============================================================================ #
# 2. Ingesta y Procesamiento de Datos (Caché Streamlit)
# ============================================================================ #

@st.cache_data(ttl=3600)
def detect_goes_bucket(_fs: s3fs.S3FileSystem, target_time: datetime) -> str:
    """Detecta la disponibilidad del bucket S3 de NOAA (conmutación GOES-16/19)."""
    year, doy, hour = target_time.strftime("%Y"), target_time.strftime("%j"), target_time.strftime("%H")
    folder_g16 = f"noaa-goes16/GLM-L2-LCFA/{year}/{doy}/{hour}/"
    try:
        if len(_fs.ls(folder_g16)) > 0:
            return "noaa-goes16"
    except Exception:
        pass
    return "noaa-goes19"

@st.cache_data(ttl=3600)
def get_glm_files_for_window(
    _fs: s3fs.S3FileSystem, start_time: datetime, bucket_name: str, minutes: int = 5) -> List[str]:
    """Obtiene la lista de archivos GLM de 20 segundos para la ventana dada."""
    all_files = []
    num_steps = (minutes * 60) // 20
    for i in range(num_steps):
        current_time = start_time + timedelta(seconds=i * 20)
        time_prefix = current_time.strftime("s%Y%j%H%M%S")
        folder_path = f"{bucket_name}/GLM-L2-LCFA/{current_time.strftime('%Y/%j/%H/')}"
        try:
            all_files.extend(_fs.glob(f"{folder_path}*_{time_prefix}*"))
        except Exception:
            continue
    return all_files

@st.cache_data(ttl=3600)
def get_abi_c13_file(_fs: s3fs.S3FileSystem, target_time: datetime, bucket_name: str) -> Optional[str]:
    """Localiza el archivo C13 (IR Onda Larga) en S3 más próximo al timestamp."""
    prefix = target_time.strftime("s%Y%j%H%M")
    folder = f"{bucket_name}/ABI-L2-CMIPF/{target_time.strftime('%Y/%j/%H/')}"
    files = _fs.glob(f"{folder}*C13_*_{prefix}*")
    return files[0] if files else None

@st.cache_data(ttl=3600)
def get_abi_ctp_file(_fs: s3fs.S3FileSystem, target_time: datetime, bucket_name: str) -> Optional[str]:
    """Localiza el producto CTPF (Cloud Top Pressure) correspondiente."""
    prefix = target_time.strftime("s%Y%j%H%M")
    folder = f"{bucket_name}/ABI-L2-CTPF/{target_time.strftime('%Y/%j/%H/')}"
    files = _fs.glob(f"{folder}*CTPF*_{prefix}*")
    return files[0] if files else None

@st.cache_data
def cluster_and_get_polygons(
    reflectivity_data: np.ndarray,
    threshold_dbz: float,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    min_area_km2: float = 100.0,) -> List[Polygon]:
    """Segmenta núcleos convectivos y los transforma en polígonos cerrados."""
    thresholded = reflectivity_data >= threshold_dbz
    labeled_arr, num_features = label(thresholded)
    polygons = []

    if num_features == 0:
        return polygons

    for i in range(1, num_features + 1):
        for contour in find_contours(labeled_arr == i, level=0.5):
            r = np.clip(np.round(contour[:, 0]).astype(int), 0, reflectivity_data.shape[0] - 1)
            c = np.clip(np.round(contour[:, 1]).astype(int), 0, reflectivity_data.shape[1] - 1)

            lons, lats = lon_mesh[r, c], lat_mesh[r, c]
            if len(lons) > 2:
                if lons[0] != lons[-1] or lats[0] != lats[-1]:
                    lons = np.append(lons, lons[0])
                    lats = np.append(lats, lats[0])

                poly = Polygon(zip(lons, lats))
                if not poly.is_valid:
                    poly = poly.buffer(0)

                if not poly.is_empty:
                    lat_rad = np.radians(poly.centroid.y)
                    area_km2 = poly.area * 111.32 * (111.32 * np.cos(lat_rad))
                    if area_km2 >= min_area_km2:
                        polygons.append(poly)
    return polygons

@st.cache_resource
def load_shape_features(path_shp: str) -> Optional[ShapelyFeature]:
    """Carga shapefiles geográficos como features vectoriales para Cartopy."""
    try:
        return ShapelyFeature(Reader(path_shp).geometries(), ccrs.PlateCarree())
    except Exception as e:
        st.error(f"Error cargando shapefile {path_shp}: {e}")
        return None

@st.cache_data
def load_airport_data(path_csv: str) -> pd.DataFrame:
    """Carga la base de puntos ICAO y coordenadas de aeropuertos."""
    try:
        return pd.read_csv(
            path_csv, sep=r"\s+", header=None, names=["Codigo ICAO", "Lat", "Lon"]
        )
    except Exception as e:
        st.error(f"Error cargando datos de aeropuertos: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_and_process_data(
    start_window_datetime: datetime, _fs_param: s3fs.S3FileSystem) -> Optional[Dict[str, Any]]:
    """Pipeline de ingesta, cálculo proxy y segmentación morfológica."""
    bounds = SPATIAL_BOUNDS

    # Detección del satélite activo
    bucket_name = detect_goes_bucket(_fs_param, start_window_datetime)
    sat_label = "GOES-16" if "16" in bucket_name else "GOES-19"

    glm_files = get_glm_files_for_window(_fs_param, start_window_datetime, bucket_name, minutes=5)
    abi_file = get_abi_c13_file(_fs_param, start_window_datetime, bucket_name)
    ctp_file = get_abi_ctp_file(_fs_param, start_window_datetime, bucket_name)

    if not glm_files or abi_file is None or ctp_file is None:
        st.warning(f"Datos satelitales incompletos para {start_window_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        return None

    # Recorte e ingesta GLM
    accum_lats, accum_lons = [], []
    for fp in glm_files:
        with _fs_param.open(fp, "rb") as f:
            with Dataset("dummy", mode="r", memory=f.read()) as nc:
                lats = nc.variables["flash_lat"][:]
                lons = nc.variables["flash_lon"][:]
                m = (
                    (lats >= bounds["data_lat_min"])
                    & (lats <= bounds["data_lat_max"])
                    & (lons >= bounds["data_lon_min"])
                    & (lons <= bounds["data_lon_max"])
                )
                accum_lats.extend(lats[m])
                accum_lons.extend(lons[m])

    all_lats, all_lons = np.array(accum_lats), np.array(accum_lons)

    # Recorte ABI C13
    with _fs_param.open(abi_file, "rb") as f:
        with Dataset("dummy", mode="r", memory=f.read()) as nc:
            proj_info = nc.variables["goes_imager_projection"]
            h = proj_info.perspective_point_height
            x_full = nc.variables["x"][:] * h
            y_full = nc.variables["y"][:] * h
            abi_crs = ccrs.Geostationary(
                central_longitude=proj_info.longitude_of_projection_origin,
                satellite_height=h,
            )

            pt_ul = abi_crs.transform_point(bounds["data_lon_min"], bounds["data_lat_max"], ccrs.PlateCarree())
            pt_lr = abi_crs.transform_point(bounds["data_lon_max"], bounds["data_lat_min"], ccrs.PlateCarree())
            x_min_proj, x_max_proj = pt_ul[0], pt_lr[0]
            y_min_proj, y_max_proj = pt_lr[1], pt_ul[1]

            idx_x = np.where((x_full >= x_min_proj) & (x_full <= x_max_proj))[0]
            idx_y = np.where((y_full >= y_min_proj) & (y_full <= y_max_proj))[0]

            xs, xe = idx_x[0], idx_x[-1] + 1
            ys, ye = idx_y[0], idx_y[-1] + 1
            ir_data = nc.variables["CMI"][ys:ye, xs:xe] - 273.15
            x, y = x_full[xs:xe], y_full[ys:ye]

    # Recorte ABI CTP (resolución 10 km)
    ctp_data, x_ctp, y_ctp = None, None, None
    try:
        with _fs_param.open(ctp_file, "rb") as f:
            with Dataset("dummy_ctp", mode="r", memory=f.read()) as nc_ctp:
                h_ctp = nc_ctp.variables["goes_imager_projection"].perspective_point_height
                x_full_ctp = nc_ctp.variables["x"][:] * h_ctp
                y_full_ctp = nc_ctp.variables["y"][:] * h_ctp
                idx_x_act = np.where((x_full_ctp >= x_min_proj) & (x_full_ctp <= x_max_proj))[0]
                idx_y_act = np.where((y_full_ctp >= y_min_proj) & (y_full_ctp <= y_max_proj))[0]
                xs_a, xe_a = idx_x_act[0], idx_x_act[-1] + 1
                ys_a, ye_a = idx_y_act[0], idx_y_act[-1] + 1
                ctp_data = nc_ctp.variables["PRES"][ys_a:ye_a, xs_a:xe_a]
                x_ctp, y_ctp = x_full_ctp[xs_a:xe_a], y_full_ctp[ys_a:ye_a]
    except Exception as e:
        st.warning(f"Error procesando CTP: {e}")

    # Grilla FED y estimación empírica de Reflectividad Proxy
    res = 0.05
    lat_bins = np.arange(bounds["data_lat_min"], bounds["data_lat_max"] + res, res)
    lon_bins = np.arange(bounds["data_lon_min"], bounds["data_lon_max"] + res, res)
    fed_raw, _, _ = np.histogram2d(all_lats, all_lons, bins=[lat_bins, lon_bins])
    fed_smoothed = gaussian_filter(fed_raw, sigma=1.5)

    max_reflectivity_proxy = np.zeros_like(fed_smoothed)
    mask_fed = fed_smoothed > 0.02
    max_reflectivity_proxy[mask_fed] = 42.2 + 12.4 * np.log10(fed_smoothed[mask_fed])

    lon_mesh, lat_mesh = np.meshgrid((lon_bins[:-1] + lon_bins[1:]) / 2, (lat_bins[:-1] + lat_bins[1:]) / 2)

    # Segmentación y filtrado espacial
    raw_polys = cluster_and_get_polygons(max_reflectivity_proxy, 25, lon_mesh, lat_mesh)
    plot_box = box(bounds["plot_lon_min"], bounds["plot_lat_min"], bounds["plot_lon_max"], bounds["plot_lat_max"])
    warning_polygons = [p for p in raw_polys if p.intersects(plot_box)]

    metrics_list, sigmet_hulls = [], []
    grid_points = [Point(lo, la) for lo, la in zip(lon_mesh.flatten(), lat_mesh.flatten())]

    for idx, poly in enumerate(warning_polygons):
        poly_id = idx + 1
        hull_info = compute_sigmet_convex_hull_properties(poly, simplify_deg=0.08)
        sigmet_hulls.append(hull_info["hull_polygon"])

        refl_vals, ctt_vals, fed_vals, ctp_vals = [], [], [], []
        for i_flat, pt in enumerate(grid_points):
            if poly.contains(pt):
                r_idx, c_idx = np.unravel_index(i_flat, max_reflectivity_proxy.shape)
                refl_vals.append(max_reflectivity_proxy[r_idx, c_idx])
                fed_vals.append(fed_smoothed[r_idx, c_idx])

                x_t, y_t = abi_crs.transform_point(
                    lon_mesh[r_idx, c_idx], lat_mesh[r_idx, c_idx], ccrs.PlateCarree()
                )
                ctt_vals.append(ir_data[np.argmin(np.abs(y - y_t)), np.argmin(np.abs(x - x_t))])

                if ctp_data is not None and x_ctp is not None and y_ctp is not None:
                    p_val = ctp_data[np.argmin(np.abs(y_ctp - y_t)), np.argmin(np.abs(x_ctp - x_t))]
                    if not np.ma.is_masked(p_val) and not np.isnan(p_val):
                        ctp_vals.append(p_val)

        max_refl = np.max(refl_vals) if refl_vals else np.nan
        min_pres = np.min(ctp_vals) if ctp_vals else np.nan
        morph = classify_convective_morphology(
            hull_info["area_hull_km2"], hull_info["major_axis_km"], hull_info["minor_axis_km"], max_refl
        )

        metrics_list.append({
                                "ID": poly_id,
                                "CenLon": poly.centroid.x,
                                "CenLat": poly.centroid.y,
                                "Area": hull_info["area_hull_km2"],
                                "Tipo": morph["codigo"],
                                "Descripcion": morph["tipo"],
                                "Peligros": morph["peligros"],
                                "Impacto": morph["impacto"],
                                "Aspect_Ratio": round(hull_info["major_axis_km"] / max(hull_info["minor_axis_km"], 1.0), 2),
                                "EjeMayor_km": hull_info["major_axis_km"],
                                "EjeMenor_km": hull_info["minor_axis_km"],
                                "Orientacion": f"{hull_info['orientation_deg']:03d}°",
                                "MaxRef": round(max_refl, 1),
                                "MaxFED": round(np.max(fed_vals), 1) if fed_vals else np.nan,
                                "MinCTT": round(np.min(ctt_vals), 1) if ctt_vals else np.nan,
                                "MaxFL": pressure_to_flight_level(min_pres),
                                "MaxH_km": pressure_to_altitude_km(min_pres),
                             })

    return {
            "sat_label": sat_label,
            "warning_polygons": sigmet_hulls,
            "metrics_df": pd.DataFrame(metrics_list),
            "ir_data": ir_data,
            "x": x,
            "y": y,
            "abi_crs": abi_crs,
            "max_reflectivity_proxy": max_reflectivity_proxy,
            "lon_mesh": lon_mesh,
            "lat_mesh": lat_mesh,
            "paises": load_shape_features(SHAPEFILE_PATHS["paises"]),
            "fir_ezeiza": load_shape_features(SHAPEFILE_PATHS["fir_ezeiza"]),
            "fir_cordoba": load_shape_features(SHAPEFILE_PATHS["fir_cordoba"]),
            "fir_resistencia": load_shape_features(SHAPEFILE_PATHS["fir_resistencia"]),
            "fir_mendoza": load_shape_features(SHAPEFILE_PATHS["fir_mendoza"]),
            "fir_comodoro": load_shape_features(SHAPEFILE_PATHS["fir_comodoro"]),
            "df_airports": load_airport_data(SHAPEFILE_PATHS["airports"]),
            "start_window": start_window_datetime,
           }

# ============================================================================ #
# 3. Funciones de Renderizado Gráfico
# ============================================================================ #

def plot_interactive_map_streamlit(
    warning_polygons: List[Polygon],
    metrics_df: pd.DataFrame,
    ir_data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    abi_crs: ccrs.Geostationary,
    max_reflectivity_proxy: np.ndarray,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    paises: Optional[ShapelyFeature],
    fir_ezeiza: Optional[ShapelyFeature],
    fir_cordoba: Optional[ShapelyFeature],
    fir_resistencia: Optional[ShapelyFeature],
    fir_mendoza: Optional[ShapelyFeature],
    fir_comodoro: Optional[ShapelyFeature],
    df_airports: pd.DataFrame,
    start_window: datetime,
    highlight_poly_id: Optional[int] = None,) -> plt.Figure:
    """Renderiza el mapa aeronáutico regional con Cartopy y Matplotlib."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=ccrs.Mercator())
    b = SPATIAL_BOUNDS
    ax.set_extent([b["plot_lon_min"], b["plot_lon_max"], b["plot_lat_min"], b["plot_lat_max"]], crs=ccrs.PlateCarree())

    if ir_data is not None:
        ax.imshow(
                  ir_data,
                  origin="upper",
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  transform=abi_crs,
                  cmap="Greys",
                  vmin=-90,
                  vmax=40,
                  zorder=1,
                 )

    proxy_masked = np.ma.masked_where(max_reflectivity_proxy < 20, max_reflectivity_proxy)
    im_proxy = ax.pcolormesh(
                             lon_mesh,
                             lat_mesh,
                             proxy_masked,
                             cmap=CMAP_AVIATION,
                             norm=NORM_AVIATION,
                             alpha=0.75,
                             transform=ccrs.PlateCarree(),
                             zorder=2,
                            )

    # Capas vectoriales FIR y división política
    if paises:
        ax.add_feature(paises, facecolor="none", edgecolor="#778da9", linewidth=1)
    for fir in [fir_ezeiza, fir_cordoba, fir_resistencia, fir_mendoza, fir_comodoro]:
        if fir:
            ax.add_feature(fir, facecolor="none", edgecolor="#072ac8", linewidth=1, zorder=2)

    # Aeródromos ICAO
    if not df_airports.empty:
        for _, ap in df_airports.iterrows():
            if (b["plot_lon_min"] < ap.Lon < b["plot_lon_max"]) and (b["plot_lat_min"] < ap.Lat < b["plot_lat_max"]):
                ax.scatter(ap.Lon, ap.Lat, marker="s", s=12, color="#072ac8", zorder=5, transform=ccrs.PlateCarree())
                ax.text(
                        ap.Lon + 0.15,
                        ap.Lat - 0.21,
                        ap["Codigo ICAO"],
                        fontsize=8,
                        c="#072ac8",
                        clip_on=True,
                        zorder=5,
                        transform=ccrs.PlateCarree(),
                       )

    cbar = plt.colorbar(im_proxy, ax=ax, orientation="horizontal", pad=0.01, shrink=0.65, ticks=[25, 35, 45, 57.5])
    cbar.ax.set_xticklabels(["Leve", "Moderado", "Fuerte", "Extremo"], fontsize=11)
    cbar.ax.tick_params(axis="x", length=0)

    # Polígonos de advertencia
    for idx, poly in enumerate(warning_polygons):
        current_id = idx + 1
        is_sel = highlight_poly_id == current_id

        ax.add_geometries(
                          [poly],
                          ccrs.PlateCarree(),
                          facecolor="none",
                          edgecolor="red" if is_sel else "#219ebc",
                          linewidth=2.5 if is_sel else 1.5,
                          hatch="///",
                          zorder=7 if is_sel else 6,
                         )

        if not metrics_df.empty and current_id in metrics_df["ID"].values:
            row = metrics_df[metrics_df["ID"] == current_id].iloc[0]
            if row.EjeMayor_km >= 100:
                rumbo = float(str(row.Orientacion).replace("°", ""))
                arrow_km = min(row.EjeMayor_km * 0.4, 60.0)
                dlat = (arrow_km / 111.32) * np.cos(np.radians(rumbo))
                dlon = (arrow_km / (111.32 * np.cos(np.radians(row.CenLat)))) * np.sin(np.radians(rumbo))

                ax.annotate(
                            "",
                            xy=(row.CenLon + dlon, row.CenLat + dlat),
                            xytext=(row.CenLon - dlon, row.CenLat - dlat),
                            arrowprops=dict(
                                arrowstyle="<->, head_width=0.2, head_length=0.3",
                                color="red" if is_sel else "#219ebc",
                                linewidth=2.0 if is_sel else 1.2,
                                mutation_scale=12,
                            ),
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            zorder=8,
                           )

    plt.tight_layout()
    return fig

def plot_parallel_coordinates(
    metrics_df: pd.DataFrame, highlight_poly_id: Optional[int] = None) -> Optional[plt.Figure]:
    """Genera coordenadas paralelas normalizadas con marcas reales y resaltado dinámico."""
    if metrics_df.empty or len(metrics_df) < 2:
        return None

    df = metrics_df.copy()
    df["Orientacion_Num"] = df["Orientacion"].str.replace("°", "").astype(float)

    cols = ["Area", "EjeMayor_km", "EjeMenor_km", "Orientacion_Num", "MaxFL", "MaxH_km", "MaxRef", "MinCTT"]
    titulos = ["Área\n(km²)", "Eje Mayor\n(km)", "Eje Menor\n(km)", "Rumbo\n(°)", "Tope\n(FL)", "Tope\n(km)", "Refl. Máx\n(dBZ)", "Min CTT\n(°C)"]

    mins, maxs = df[cols].min(), df[cols].max()
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    df_norm = (df[cols] - mins) / ranges
    df_norm["Tipo"] = df["Tipo"]
    df_norm["ID"] = df["ID"]

    color_dict = {"IC": "#2a9d8f", "CC": "#e9c46a", "QLCS": "#f4a261", "MCS": "#e76f51"}
    fig, ax = plt.subplots(figsize=(14, 5.2))
    has_sel = highlight_poly_id is not None and highlight_poly_id in df_norm["ID"].values

    # Líneas de fondo no seleccionadas
    for _, r in df_norm.iterrows():
        if has_sel and (r["ID"] == highlight_poly_id):
            continue
        ax.plot(
            range(len(cols)),
            [r[c] for c in cols],
            color="#ced4da" if has_sel else color_dict.get(r["Tipo"], "gray"),
            linewidth=1.0 if has_sel else 1.5,
            alpha=0.20 if has_sel else 0.55,
            zorder=2,
        )

    # Línea seleccionada en primer plano
    if has_sel:
        sel_n = df_norm[df_norm["ID"] == highlight_poly_id].iloc[0]
        sel_r = df[df["ID"] == highlight_poly_id].iloc[0]
        y_sel = [sel_n[c] for c in cols]

        ax.plot(
            range(len(cols)),
            y_sel,
            color="red",
            linewidth=3.5,
            alpha=1.0,
            zorder=10,
            marker="o",
            markersize=7,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        for i, col in enumerate(cols):
            val = sel_r[col]
            txt = f"{val:.0f}" if col in ["Area", "EjeMayor_km", "MaxFL", "Orientacion_Num"] else f"{val:.1f}"
            ax.text(
                i,
                y_sel[i] + 0.04,
                txt,
                fontsize=9,
                color="red",
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=0.85, linewidth=0.8),
            )

    # Ejes verticales y rótulos cuantitativos
    for i, col in enumerate(cols):
        ax.axvline(i, color="#adb5bd", linestyle="-", linewidth=1.2, zorder=1)
        for y_n in [0.0, 0.25, 0.5, 0.75, 1.0]:
            v = mins[col] + y_n * (maxs[col] - mins[col])
            s = f"{v:.0f}" if col in ["Area", "EjeMayor_km", "MaxFL", "Orientacion_Num"] else f"{v:.1f}"
            ax.plot([i - 0.03, i + 0.03], [y_n, y_n], color="#6c757d", linewidth=0.8, zorder=2)
            ax.text(i - 0.05, y_n, s, fontsize=8, color="#6c757d", ha="right", va="center", zorder=4)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(titulos, fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(-0.35, len(cols) - 0.65)
    ax.set_ylim(-0.06, 1.14)
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)

    legend_items = [
        Line2D([0], [0], color=c, lw=2.5, label=t) for t, c in color_dict.items() if t in df_norm["Tipo"].values
    ]
    if has_sel:
        legend_items.append(Line2D([0], [0], color="red", lw=3.0, marker="o", label=f"Selección (ID: {highlight_poly_id})"))
    ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.0, 1.15), ncol=len(legend_items), frameon=True, framealpha=0.9)

    plt.tight_layout()
    return fig

# ============================================================================ #
# 4. Interfaz de Usuario (Streamlit Presentation Layer)
# ============================================================================ #

def main() -> None:
    """Punto de entrada principal de la aplicación Streamlit."""
    st.set_page_config(layout="wide")
    st.image("smn_horizontal_arg-01.jpg", width=250)
    st.title(":blue[Producto TS-SIGMET | Dashboard Interactivo (EXPERIMENTAL)]")
    
    with st.expander("⚠️ **Manual Operativo: Clasificación convectiva, morfología radar y seguridad operacional**"):
        st.markdown("""
        ### Criterios de Clasificación Morfológica y Toma de Decisiones

        El sistema procesa la geometría de la envolvente convexa simplificada (*Convex Hull*) y el campo de reflectividad proxy para categorizar las tormentas según taxonomías estandarizadas (*Parker & Johnson; Gallus et al.*) adaptadas al monitoreo y emisión de mensajes **TS-SIGMET**.        """)

        # Tabla comparativa de umbrales cuantitativos
        st.markdown("""
                    | Categoría | Acrónimo | Eje Mayor (L) | Relación de Aspecto (L/W) | Cobertura (A) | Reflectividad Típica |
                    | :--- | :---: | :---: | :---: | :---: | :---: |
                    | **Celda Aislada** | `IC` | <50 km | <2.5 | < 400 km² | 35 – 50 dBZ |
                    | **Clúster Multicelular** | `CC` | >50 km | <3 | >400 km² | 40 – 55 dBZ |
                    | **Línea Convectiva** | `QLCS` | >100 km | >3 | Variable | 45 – >60 dBZ |
                    | **Sistema Mesoescalar** | `MCS` | >100 km | Variable | >1000 km² | 40 – >55 dBZ |
                    """)

        st.markdown("---")

        # Tarjetas desglosadas por tipología
        t1, t2 = st.columns(2)

        with t1:
            st.markdown("""
            #### 🟢 **Celda Aislada (IC - Isolated Cell)**
            * **Estructura Meteorológica:** Celdas convectivas pulsantes u ordinarias de escala local con corrientes ascendentes y descendentes bien delimitadas.
            * **Peligros Principales:** Microfrentes de ráfagas locales (*microbursts*), granizo localizado y turbulencia severa acotada al núcleo y su entorno inmediato (< 5 NM).
            * **Gestión de Tránsito Aéreo (ATC) & Pilotos:**
                * Desvíos tácticos mínimos (5 a 10 NM a barlovento).
                * Alta probabilidad de circunnavegación visual o con radar de a bordo (WXR) sin saturar los sectores terminales.
            """)

            st.markdown("""
            #### 🟡 **Clúster Multicelular (CC - Cluster of Cells)**
            * **Estructura Meteorológica:** Agrupación desorganizada o semiorganizada de celdas en diferentes etapas de ciclo de vida (iniciación, madurez, disipación).
            * **Peligros Principales:** Turbulencia severa en aire claro (CAT), engelamiento fuerte en niveles medios y presencia de "corredores engañosos" (*blind alleys*) entre núcleos activos.
            * **Gestión de Tránsito Aéreo (ATC) & Pilotos:**
                * Prohibida la penetración a través de brechas estrechas entre ecos con reflectividad > 35 dBZ.
                * Rutas de desvío estratégicas; requiere coordinación temprana con control de ruta para evitar atrapamiento entre celdas secundarias.
            """)

        with t2:
            st.markdown("""
            #### 🟠 **Sistema Cuasi-Lineal / Línea de Inestabilidad (QLCS)**
            * **Estructura Meteorológica:** Banda convectiva alargada y continua (*Squall Line* / frente frío activo) con fuerte forzamiento dinámico lineal.
            * **Peligros Principales:** Frentes de ráfagas violentos (*gust fronts*), cizalladura horizontal/vertical del viento (*low-level windshear*), turbulencia extrema a lo largo del frente y granizo que puede proyectarse varios kilómetros por delante del borde de ataque.
            * **Gestión de Tránsito Aéreo (ATC) & Pilotos:**
                * **Bloqueo transversal total de aerovías:** La penetración frontal está formalmente contraindicada.
                * Se requieren desvíos de largo radio circunvalando los extremos de la línea o demoras en circuito de espera hasta el pasaje del sistema.
            """)

            st.markdown("""
            #### 🔴 **Sistema Convectivo de Mesoescala (MCS)**
            * **Estructura Meteorológica:** Complejo convectivo de gran escala con extensas regiones de lluvia estratiforme electrificada que engloba múltiples núcleos de tormenta severa y topes que habitualmente sobrepasan la tropopausa.
            * **Peligros Principales:** Engelamiento severo generalizado en niveles de crucero, cimas nubosas penetrantes (overshooting tops) que superan **FL400**, y actividad eléctrica intra-nube y nube-tierra continua.
            * **Gestión de Tránsito Aéreo (ATC) & Pilotos:**
                * **Disrupción masiva del espacio aéreo (escala FIR):** Colapso de rutas troncales y sectores de control.
                * Exige reformulación de planes de vuelo, desvíos interprovinciales obligatorios y aplicación inmediata de procedimientos de contingencia y espaciamiento por flujo (ATFM).
            """)

        st.info(
            "💡 **Pauta Operativa Anexo 3 OACI:** Todo eco con reflectividad mayor que 40 dBZ o topes por encima de FL350 debe ser considerado zona de exclusión de vuelo con margen de seguridad horizontal mínimo de 20 NM a barlovento."        )

    initial_dt = datetime(2023, 12, 17, 6, 0, 0)
    sel_date = st.date_input(":blue[Selecciona la fecha]", value=initial_dt.date())
    sel_time = st.time_input(":blue[Selecciona la hora (UTC)]", value=initial_dt.time(), step=300)
    start_window = datetime.combine(sel_date, sel_time)

    data = load_and_process_data(start_window, FS_GLOBAL)
    if data is None:
        st.warning(":blue[No se pudieron cargar los datos para la fecha y hora seleccionadas.]")
        return

    warning_polygons = data["warning_polygons"]
    metrics_df = data["metrics_df"]
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by="Area", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 1])

    with col2:
        st.header(":blue[Tabla de Advertencias]")
        options = [
                   f"ID: {int(r.ID)}, Tipo: {r.Tipo}, Tope: FL{int(r.MaxFL):03d}, Area: {int(r.Area)} km²"
                   for _, r in metrics_df.iterrows()
                  ]
        options.insert(0, "-- Seleccionar Polígono --")

        selected_option = st.selectbox(":blue[Seleccionar una advertencia para resaltar en el mapa:]", options, index=0)

        highlight_poly_id = None
        if selected_option != "-- Seleccionar Polígono --":
            highlight_poly_id = int(float(selected_option.split(",")[0].replace("ID: ", "")))
            poly_data = metrics_df[metrics_df["ID"] == highlight_poly_id].iloc[0]

            st.markdown(f"### Detalles de la Tormenta (ID: {highlight_poly_id})")
            mc1, mc2 = st.columns(2)
            mc1.metric("Clasificación", f"{poly_data.Tipo}", help=poly_data.Descripcion)
            mc2.metric("Tope Nuboso", f"FL{int(poly_data.MaxFL):03d}", delta=f"{poly_data.MaxH_km:.1f} km", delta_color="off")
            
            mc3, mc4 = st.columns(2)
            mc3.metric("Eje Mayor", f"{poly_data.EjeMayor_km:.0f} km")
            mc4.metric("Eje Menor", f"{poly_data.EjeMenor_km:.0f} km")

            mc3, mc4 = st.columns(2)
            mc3.metric("Reflectividad", f"{poly_data.MaxRef:.1f} dBZ")
            mc4.metric(
                       "Orientación",
                       f"{poly_data.Orientacion}",
                       delta=rumbo_to_arrow(int(str(poly_data.Orientacion).replace("°", ""))),
                      )

            st.error(f"⚠️ **Peligros Principales:** {poly_data.Peligros}")
            st.error(f"🚨 **Impacto Operacional Estimado:** {poly_data.Impacto}")

        st.dataframe(
                     metrics_df,
                     column_order=["ID", "Tipo", "Area", "MaxFL", "MaxH_km", "MaxRef", "MinCTT", "MaxFED"],
                     height=450,
                     hide_index=True,
                    )

        if not metrics_df.empty:
            csv_buffer = io.StringIO()
            metrics_df.to_csv(csv_buffer, index=False)
            st.download_button(
                               label="Descargar métricas como CSV",
                               data=csv_buffer.getvalue(),
                               file_name=f"sigmet_metrics_{start_window.strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv",
                              )

    with col1:
        st.header(":blue[Mapa de Advertencias]")
        fig_map = plot_interactive_map_streamlit(
                                                 warning_polygons,
                                                 metrics_df,
                                                 data["ir_data"],
                                                 data["x"],
                                                 data["y"],
                                                 data["abi_crs"],
                                                 data["max_reflectivity_proxy"],
                                                 data["lon_mesh"],
                                                 data["lat_mesh"],
                                                 data["paises"],
                                                 data["fir_ezeiza"],
                                                 data["fir_cordoba"],
                                                 data["fir_resistencia"],
                                                 data["fir_mendoza"],
                                                 data["fir_comodoro"],
                                                 data["df_airports"],
                                                 data["start_window"],
                                                 highlight_poly_id=highlight_poly_id,
                                                )
        st.pyplot(fig_map, use_container_width=True)

    # Análisis Multivariado
    st.markdown("---")
    st.subheader(":blue[Análisis Multivariado de Propiedades de las Tormentas]")

    with st.expander("ℹ️ ¿Cómo interpretar este gráfico?", expanded=False):
        st.markdown("""
        * **Cada línea representa un polígono SIGMET detectado.**
        * **Color de la línea:** Clasificación morfológica (**IC:** Verde azulado, **CC:** Amarillo, **QLCS:** Naranja, **MCS:** Rojo coral).
        * **Sincronización:** Al seleccionar un polígono en la lista superior, su trayectoria se resalta en **rojo** con etiquetas de valor puntual sobre cada eje.
        """)

    if not metrics_df.empty and len(metrics_df) >= 2:
        fig_parallel = plot_parallel_coordinates(metrics_df, highlight_poly_id=highlight_poly_id)
        if fig_parallel is not None:
            st.pyplot(fig_parallel, use_container_width=True)
    else:
        st.info("Se requieren al menos 2 advertencias detectadas para trazar el gráfico de coordenadas paralelas.")


if __name__ == "__main__":
    main()

# ============================================================================ #
