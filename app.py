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
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

from shapely.geometry import Point, Polygon, box

from scipy.ndimage import gaussian_filter, label
from matplotlib.colors import ListedColormap, BoundaryNorm

from skimage.measure import find_contours

# ============================================================================ #
# 0. Definiciones globales o constantes (fuera de funciones para Streamlit)
# ============================================================================ #

# Paleta de colores para radar real
aviation_colors = [
		   "#00FF00",  # Verde    (Level 1: 20-30 dBZ) | Débil
		   "#FFFF00",  # Amarillo (Level 2: 30-40 dBZ) | Moderado
		   "#FF0000",  # Rojo     (Level 3: 40-50 dBZ) | Fuerte
		   "#FF00FF",  # Magenta  (Level 4: > 50 dBZ)  | Extremo
                  ]
cmap_aviation = ListedColormap(aviation_colors)
levels_aviation = [20, 30, 40, 50, 65]
norm_aviation = BoundaryNorm(levels_aviation, cmap_aviation.N)

# Inicializar S3FileSystem una vez globalmente
fs_global = s3fs.S3FileSystem(anon=True)

# Calcula nivel de vuelo (FL) a partir del valor minimo de presión del tope nuboso
def pressure_to_flight_level(p_hpa):
    """Convierte presión (hPa) a Nivel de Vuelo (FL) según la atmósfera estándar ISA"""
    if p_hpa <= 0 or np.isnan(p_hpa) or np.ma.is_masked(p_hpa):
        return np.nan
    
    # Troposfera (hasta ~36,000 pies / 226.32 hPa)
    if p_hpa > 226.32:
        alt_ft = 145366.45 * (1 - (p_hpa / 1013.25)**0.190284)
    # Tropopausa / Baja Estratosfera (por encima de ~36,000 pies)
    else:
        alt_ft = 36089.24 - 20805.7 * np.log(p_hpa / 226.32)
        
    # El Nivel de Vuelo (FL) exacto (ej. 382.4)
    fl_exact = alt_ft / 100.0
    
    # Convención aeronáutica: Redondear a la decena más cercana (múltiplos de 10)
    # Ej: 382.4 -> 38.24 -> round(38) -> 38 * 10 -> 380
    fl_rounded = int(round(fl_exact / 10.0) * 10)
    
    return fl_rounded

def rumbo_to_arrow(angle_deg):
    """Retorna una flecha tipográfica según el ángulo de orientación respecto al Norte."""
    val = angle_deg % 180.0
    if val <= 22.5 or val > 157.5:
        return "↕ N-S"
    elif 22.5 < val <= 67.5:
        return "↗ NE-SW"
    elif 67.5 < val <= 112.5:
        return "↔ E-W"
    else:
        return "↘ SE-NW"

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

@st.cache_data(ttl=3600)
def get_abi_ctp_file(_fs, target_time):
    time_prefix = target_time.strftime("s%Y%j%H%M")
    folder_path = f"noaa-goes19/ABI-L2-CTPF/{target_time.strftime('%Y/%j/%H/')}"
    files = _fs.glob(f"{folder_path}*CTPF*_{time_prefix}*")
    return files[0] if files else None

@st.cache_data
def cluster_and_get_polygons(reflectivity_data, threshold_dbz, lon_mesh, lat_mesh, min_area_km2=100):

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
                    
                    # Reparar polígonos inválidos
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                        
                    if not poly.is_empty:
                        # Calcular el área directamente con la geometría de Shapely
                        # poly.area devuelve grados cuadrados. Lo pasamos a km2 usando el centroide:
                        lat_rad = np.radians(poly.centroid.y)
                        area_geom_km2 = poly.area * 111.32 * (111.32 * np.cos(lat_rad))
                        
                        # Guardar solo si supera el umbral de área
                        if area_geom_km2 >= min_area_km2:
                            polygons.append(poly)
                            
                except Exception as e:
                    st.warning(f"Error creando Polígono del contorno: {e}")

    return polygons

@st.cache_resource 
def load_shape_features(path_shp):
    try:
        return ShapelyFeature(Reader(path_shp).geometries(), ccrs.PlateCarree())
    except Exception as e:
        st.error(f"Error cargando shapefile {path_shp}: {e}")
        return None

@st.cache_data 
def load_airport_data(path_csv):
    try:
        return pd.read_csv(path_csv,
                         sep=r'\s+',
                         header=None,
                         names=['Codigo ICAO','Lat','Lon'])
    except Exception as e:
        st.error(f"Error cargando datos de aeropuertos {path_csv}: {e}")
        return pd.DataFrame()

def compute_sigmet_convex_hull_properties(poly, simplify_deg=0.08):
    """
    Calcula el Convex Hull simplificado y sus propiedades morfológicas
    (eje mayor, eje menor, orientación y área) en unidades físicas (km).
    """
    # 1. Envoltura Convexa
    hull = poly.convex_hull

    # 2. Reducir cantidad de vértices para formato SIGMET (Ramer-Douglas-Peucker)
    # tolerance en grados: ~0.05° a 0.1° (~6 a 11 km de tolerancia)
    hull_simplified = hull.simplify(tolerance=simplify_deg, preserve_topology=True)
    if not hull_simplified.is_valid or hull_simplified.geom_type != "Polygon":
        hull_simplified = hull

    # 3. Factor de conversión métrica local (geodésico aproximado)
    centroid_lat = hull_simplified.centroid.y
    lat_rad = np.radians(centroid_lat)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(lat_rad)

    # 4. Rectángulo Mínimo Orientado (Oriented Bounding Box)
    # Da las dimensiones principales exactas de la envoltura
    min_rect = hull_simplified.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]

    # Distancias entre lados consecutivos en km
    lados_km = []
    vectores = []
    for k in range(4):
        p1 = rect_coords[k]
        p2 = rect_coords[(k + 1) % 4]
        dx_km = (p2[0] - p1[0]) * km_per_deg_lon
        dy_km = (p2[1] - p1[1]) * km_per_deg_lat
        dist_km = np.hypot(dx_km, dy_km)
        lados_km.append(dist_km)
        vectores.append((dx_km, dy_km))

    # Identificar eje mayor y eje menor
    idx_major = int(np.argmax(lados_km[:2]))
    major_axis_km = max(lados_km[0], lados_km[1])
    minor_axis_km = min(lados_km[0], lados_km[1])

    # 5. Orientación respecto al Norte (Ángulo azimutal de 0° a 180°)
    dx_maj, dy_maj = vectores[idx_major]
    # np.arctan2(dx, dy) mide el ángulo partiendo del Norte (+Y) hacia el Este (+X)
    angle_deg = np.degrees(np.arctan2(dx_maj, dy_maj)) % 180.0

    # 6. Área del Convex Hull en km²
    area_hull_km2 = hull_simplified.area * km_per_deg_lon * km_per_deg_lat

    return {
        "hull_polygon": hull_simplified,
        "vertices": len(list(hull_simplified.exterior.coords)) - 1,
        "major_axis_km": round(major_axis_km, 1),
        "minor_axis_km": round(minor_axis_km, 1),
        "orientation_deg": int(round(angle_deg)),
        "area_hull_km2": round(area_hull_km2, 1),
    }

@st.cache_data(ttl=3600) 
def load_and_process_data(start_window_datetime, _fs_param):
    
    # --- Límites espaciales para lectura de datos (NOTA: no son los mismo limites que para la visualizacion) ---
    lat_min, lat_max = -47.0, -18.5
    lon_min, lon_max = -75.5, -37.0
    
    # --- Limites para filtrar polígonos fuera del área de visualización ---
    lat_min_plot, lat_max_plot = -45.0, -19.0
    lon_min_plot, lon_max_plot = -75.0, -50.0

    # --- RUTAS RELATIVAS A LOS ARCHIVOS DE DATOS EN TU REPOSITORIO GITHUB ---
    #path_shape_depto_rel     = './data/shp_arg/operativo/departamentos_edit.shp'
    #path_shape_prov_rel      = './data/shp_arg/operativo/provincias_edit.shp'
    path_shape_paises_rel    = './data/shp_arg/cartopy/10m_admin_0_countries.shp'
    path_dir_FIR_rel         = './data/fir_txt/FIR_aeropuertos.txt' 
    path_fir_ezeiza_rel      = './data/shp_arg/FIR/FIR_EZEIZA_backup.shp'
    path_fir_cordoba_rel     = './data/shp_arg/FIR/FIR_CORDOBA.shp'
    path_fir_resistencia_rel = './data/shp_arg/FIR/FIR_RESISTENCIA.shp'
    path_fir_mendoza_rel     = './data/shp_arg/FIR/FIR_MENDOZA.shp'
    path_fir_comodoro_rel    = './data/shp_arg/FIR/FIR_COMODORO.shp'

    paises          = load_shape_features(path_shape_paises_rel)
    df_airports     = load_airport_data(path_dir_FIR_rel)
    fir_ezeiza      = load_shape_features(path_fir_ezeiza_rel)
    fir_cordoba     = load_shape_features(path_fir_cordoba_rel)
    fir_resistencia = load_shape_features(path_fir_resistencia_rel)
    fir_mendoza     = load_shape_features(path_fir_mendoza_rel)
    fir_comodoro    = load_shape_features(path_fir_comodoro_rel)

    # --- Carga y preprocesamiento de datos GLM y ABI ---
    glm_files = get_glm_files_for_window(_fs_param, start_window_datetime, minutes=5)
    abi_file  = get_abi_c13_file(_fs_param, start_window_datetime)
    ctp_file  = get_abi_ctp_file(_fs_param, start_window_datetime)

    if not glm_files:
        st.warning(f"No se encontraron archivos GLM para {start_window_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        return None 
    if abi_file is None:
        st.warning(f"No se encontró archivo ABI-C13 para {start_window_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        return None
    if ctp_file is None:
        st.warning(f"No se encontró archivo ABI-CTP para {start_window_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        return None 

    # --- Leer y recortar GLM al vuelo ---
    accumulated_lats = []
    accumulated_lons = []
    for file_path in glm_files:
        with _fs_param.open(file_path, "rb") as f:
            with Dataset("dummy", mode="r", memory=f.read()) as nc:
                lats = nc.variables["flash_lat"][:]
                lons = nc.variables["flash_lon"][:]
                # Aplicar máscara espacial
                mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
                accumulated_lats.extend(lats[mask])
                accumulated_lons.extend(lons[mask])

    all_lats = np.array(accumulated_lats)
    all_lons = np.array(accumulated_lons)

    # --- Leer y recortar ABI al vuelo (resolucione espacial de 2 km) ---
    with _fs_param.open(abi_file, "rb") as f:
        with Dataset("dummy", mode="r", memory=f.read()) as nc:
            proj_info = nc.variables["goes_imager_projection"]
            h = proj_info.perspective_point_height
            x_rad = nc.variables["x"][:]
            y_rad = nc.variables["y"][:]
            x_full = x_rad * h
            y_full = y_rad * h
            abi_crs = ccrs.Geostationary(central_longitude=proj_info.longitude_of_projection_origin, satellite_height=h)
            
            # Transformar límites a proyección geoestacionaria
            point_ul = abi_crs.transform_point(lon_min, lat_max, ccrs.PlateCarree())
            point_lr = abi_crs.transform_point(lon_max, lat_min, ccrs.PlateCarree())
            
            x_min_proj, x_max_proj = point_ul[0], point_lr[0]
            y_min_proj, y_max_proj = point_lr[1], point_ul[1] # Eje Y invertido en GOES

            idx_x = np.where((x_full >= x_min_proj) & (x_full <= x_max_proj))[0]
            idx_y = np.where((y_full >= y_min_proj) & (y_full <= y_max_proj))[0]
            
            if len(idx_x) > 0 and len(idx_y) > 0:
                x_start, x_end = idx_x[0], idx_x[-1] + 1
                y_start, y_end = idx_y[0], idx_y[-1] + 1
                
                # Cargar solo el subconjunto de la matriz CMI
                ir_data = nc.variables["CMI"][y_start:y_end, x_start:x_end] - 273.15
                x = x_full[x_start:x_end]
                y = y_full[y_start:y_end]
            else:
                ir_data, x, y = None, None, None
          
    # --- Leer y recortar ACTPF usando sus propias coordenadas pues la resolucion espacial es de 10 km ---
    ctp_data, x_ctp, y_ctp = None, None, None
    if ctp_file is not None and len(idx_x) > 0 and len(idx_y) > 0:
        try:
            with _fs_param.open(ctp_file, "rb") as f:
                with Dataset("dummy_ctp", mode="r", memory=f.read()) as nc_ctp:
                    # Extraer resolución de proyección específica del CTPF
                    h_ctp = nc_ctp.variables["goes_imager_projection"].perspective_point_height
                    x_full_ctp = nc_ctp.variables["x"][:] * h_ctp
                    y_full_ctp = nc_ctp.variables["y"][:] * h_ctp
                    
                    # Buscar índices en la grilla del CTPF
                    idx_x_act = np.where((x_full_ctp >= x_min_proj) & (x_full_ctp <= x_max_proj))[0]
                    idx_y_act = np.where((y_full_ctp >= y_min_proj) & (y_full_ctp <= y_max_proj))[0]
                    
                    if len(idx_x_act) > 0 and len(idx_y_act) > 0:
                        xs_a, xe_a = idx_x_act[0], idx_x_act[-1] + 1
                        ys_a, ye_a = idx_y_act[0], idx_y_act[-1] + 1
                        
                        ctp_data = nc_ctp.variables["PRES"][ys_a:ye_a, xs_a:xe_a]
                        x_ctp = x_full_ctp[xs_a:xe_a]
                        y_ctp = y_full_ctp[ys_a:ye_a]
        except Exception as e:
            st.warning(f"No se pudo procesar CTP: {e}")

    grid_res_high = 0.05
    lat_bins_high = np.arange(lat_min, lat_max + grid_res_high, grid_res_high)
    lon_bins_high = np.arange(lon_min, lon_max + grid_res_high, grid_res_high)

    fed_high, _, _ = np.histogram2d(all_lats, all_lons, bins=[lat_bins_high, lon_bins_high])
    fed_smoothed   = gaussian_filter(fed_high, sigma=1.5)

    max_reflectivity_proxy = np.zeros_like(fed_smoothed)
    mask = fed_smoothed > 0.1
    
    # Modelos REF-FED
    # max_reflectivity_proxy[mask] = 33.0 + 10.0 * np.log10(fed_smoothed[mask])
    max_reflectivity_proxy[mask] = 42.2 + 12.4 * np.log10(fed_smoothed[mask])
    # max_reflectivity_proxy[mask] = 44.6 + 23.5 * np.log10(fed_smoothed[mask]) + 8.6 * np.power(np.log10(fed_smoothed[mask]),2)

    lon_mesh_high, lat_mesh_high = np.meshgrid(
                            					   (lon_bins_high[:-1] + lon_bins_high[1:]) / 2,
                            					   (lat_bins_high[:-1] + lat_bins_high[1:]) / 2
                            					  )

    # --- Detección de Polígonos de Advertencia y Métricas ---
    warning_threshold_dbz = 25
    warning_polygons = cluster_and_get_polygons(
                                                max_reflectivity_proxy,
                                                warning_threshold_dbz,
                                                lon_mesh_high,
                                                lat_mesh_high
                                               )

    # --- Crear un bounding box con los límites del mapa ---
    plot_bbox = box(lon_min_plot, lat_min_plot, lon_max_plot, lat_max_plot)
    
    # --- Conservar solo los polígonos que intersectan con el área visible ---
    warning_polygons = [poly for poly in warning_polygons if poly.intersects(plot_bbox)]

    metrics_list = []
    
    if warning_polygons and ir_data is not None:
        
        lon_flat = lon_mesh_high.flatten()
        lat_flat = lat_mesh_high.flatten()
        grid_points = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]
        
        sigmet_hulls = []

        for idx, poly in enumerate(warning_polygons):
            
            poly_id = idx + 1
            
            centroid_lon = poly.centroid.x
            centroid_lat = poly.centroid.y

            # --- Calcular Convex Hull simplificado y métricas morfológicas ---
            hull_info = compute_sigmet_convex_hull_properties(poly, simplify_deg=0.08)
            sigmet_poly = hull_info["hull_polygon"]
            sigmet_hulls.append(sigmet_poly)

            reflectivity_values_in_poly = []
            ir_temps_in_poly = []
            fed_values_in_poly = []
            ctp_values_in_poly = []
            num_pixels = 0

            for i_flat in range(len(grid_points)):
                
                current_point = grid_points[i_flat]
                
                # Evaluamos contenido con el polígono original de reflectividad
                if poly.contains(current_point):
                    
                    num_pixels += 1
                    r, c = np.unravel_index(i_flat, max_reflectivity_proxy.shape)

                    reflectivity_values_in_poly.append(max_reflectivity_proxy[r, c])
                    fed_values_in_poly.append(fed_smoothed[r, c])

                    x_transformed, y_transformed = abi_crs.transform_point(
                                                                           lon_mesh_high[r, c],
                                                                           lat_mesh_high[r, c],
                                                                           ccrs.PlateCarree(),
                                                                          )
                    idx_x_abi = np.argmin(np.abs(x - x_transformed))
                    idx_y_abi = np.argmin(np.abs(y - y_transformed))
                    ir_temps_in_poly.append(ir_data[idx_y_abi, idx_x_abi])

                    if (ctp_data is not None and x_ctp is not None and y_ctp is not None):
                        idx_x_act = np.argmin(np.abs(x_ctp - x_transformed))
                        idx_y_act = np.argmin(np.abs(y_ctp - y_transformed))
                        val_pres = ctp_data[idx_y_act, idx_x_act]
                        if not np.ma.is_masked(val_pres) and not np.isnan(val_pres):
                            ctp_values_in_poly.append(val_pres)

            max_reflectivity = (
                                np.max(reflectivity_values_in_poly)
                                if reflectivity_values_in_poly
                                else np.nan
                               )
            max_fed = (np.max(fed_values_in_poly) if fed_values_in_poly else np.nan)
            min_ir_temp = (np.min(ir_temps_in_poly) if ir_temps_in_poly else np.nan)

            # Clasificación por área del Hull
            area_sigmet_km2 = hull_info["area_hull_km2"]
            if area_sigmet_km2 < 500:
                categoria = "CO"
            elif area_sigmet_km2 < 1000:
                categoria = "MC"
            else:
                categoria = "SC"

            min_ctp = np.min(ctp_values_in_poly) if ctp_values_in_poly else np.nan
            max_fl = pressure_to_flight_level(min_ctp)

            metrics_list.append(
                                {
                                    "ID": poly_id,
                                    "CenLon": centroid_lon,
                                    "CenLat": centroid_lat,
                                    "Area": area_sigmet_km2,
                                    "Escala": categoria,
                                    "EjeMayor_km": hull_info["major_axis_km"],
                                    "EjeMenor_km": hull_info["minor_axis_km"],
                                    "Rumbo": f"{hull_info['orientation_deg']:03d}°",
                                    "MaxRef": round(max_reflectivity, 1),
                                    "MaxFED": round(max_fed, 1),
                                    "MinCTT": round(min_ir_temp, 1),
                                    "MaxFL": max_fl,
                                }
                               )

        # Reemplazamos los polígonos originales por las envolturas SIGMET
        warning_polygons = sigmet_hulls

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
# 2. Función para dibujar el mapa                                             #
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

    # Límites espaciales para visualización
    lat_min_plot, lat_max_plot = -45.0, -19.0
    lon_min_plot, lon_max_plot = -75.0, -50.0
    ax.set_extent([lon_min_plot, lon_max_plot, lat_min_plot, lat_max_plot], crs=ccrs.PlateCarree())
    
    if ir_data is not None:
        im_ir = ax.imshow(
            ir_data, origin="upper",
            extent=[x.min(), x.max(), y.min(), y.max()],
            transform=abi_crs,
            cmap="Greys", vmin=-90, vmax=40, zorder=1
        )

    proxy_masked = np.ma.masked_where(max_reflectivity_proxy < 5, max_reflectivity_proxy)

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
            if (lon_min_plot < px < lon_max_plot) and (lat_min_plot < py < lat_max_plot):
                plt.scatter(px, py, marker='s', s=10, color='b', zorder=5, transform=ccrs.PlateCarree())
                plt.text(px + 0.15, py - 0.21, type_code, fontsize=8, c='b', clip_on=True, zorder=5, transform=ccrs.PlateCarree())

    cbar_proxy = plt.colorbar(
                			      im_proxy, ax=ax, orientation="horizontal", pad=0.01, shrink=0.65,
                			      ticks=[25, 35, 45, 57.5]
                		         )
    cbar_proxy.ax.set_xticklabels(["Leve", "Moderado", "Fuerte", "Extremo"], fontsize=11)
    cbar_proxy.ax.tick_params(axis='x', length=0)
    
    for poly_idx, poly in enumerate(warning_polygons):
        
        current_id = poly_idx + 1

        if highlight_poly_id == current_id:
            edge_color = "red"
            line_width = 2.5
            zorder = 7
        else:
            edge_color = "orange"  # Contorno tipo SIGMET
            line_width = 1
            zorder = 6

        # Se dibuja la envoltura convexa simplificada
        ax.add_geometries(
                           [poly],
                           ccrs.PlateCarree(),
                           facecolor="none",
                           edgecolor=edge_color,
                           linewidth=line_width,
                           linestyle="-",
                           zorder=zorder,
                         )

        # Opcional: Dibujar el ID en el centroide del polígono
        centroid = poly.centroid
        ax.text(
                centroid.x,
                centroid.y,
                str(current_id),
                color=edge_color,
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                transform=ccrs.PlateCarree(),
                zorder=zorder + 1,
               )
        
        # --- DIBUJAR FLECHA DE ORIENTACIÓN DEL EJE MAYOR ---
        if not metrics_df.empty and current_id in metrics_df["ID"].values:
            
            row_poly = metrics_df[metrics_df["ID"] == current_id].iloc[0]

            # Solo dibujamos flecha en sistemas con alargamiento perceptible (ej. eje mayor > 40 km)
            if row_poly.EjeMayor_km >= 40:
                cen_x = row_poly.CenLon
                cen_y = row_poly.CenLat
                rumbo_deg = float(str(row_poly.Rumbo).replace("°", ""))

                # Semilongitud en grados aproximados para dibujar la flecha
                # Limitamos la longitud visual para que no tape todo el polígono
                arrow_len_km = min(row_poly.EjeMayor_km * 0.4, 60.0)
                lat_rad = np.radians(cen_y)
                dlat = (arrow_len_km / 111.32) * np.cos(np.radians(rumbo_deg))
                dlon = ((arrow_len_km / (111.32 * np.cos(lat_rad))) * np.sin(np.radians(rumbo_deg)))

                # Flecha bidireccional alineada con el eje mayor del Convex Hull
                ax.annotate(
                    "",
                    xy=(cen_x + dlon, cen_y + dlat),
                    xytext=(cen_x - dlon, cen_y - dlat),
                    arrowprops=dict(
                        arrowstyle="<->, head_width=0.4, head_length=0.6",
                        color="red" if highlight_poly_id == current_id else edge_color,
                        linewidth=2.0 if highlight_poly_id == current_id else 1.2,
                        mutation_scale=12,
                    ),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    zorder=zorder + 1,
                )

    #plt.title(
    #    f"GOES-19 Canal 13 IR + Proxy radar GLM 5-min\n"
    #    f"{start_window.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=12
    #)
    
    plt.tight_layout()
    return fig

# ============================================================================ #
# 3. Estructura de la aplicación Streamlit (Main App Logic)
# ============================================================================ #

st.set_page_config(layout="wide")
st.image("smn_horizontal_arg-01.jpg", width=250) 
st.title("Producto TS-SIGMET | Dashboard Interactivo ")

# --- Glosario expansible de referencias ---
with st.expander("⚠️ **Referencia de tipo de tormentas y seguridad operacional**"):
    st.markdown("""
    Esta clasificación categoriza las tormentas según su extensión espacial y su impacto esperado en la gestión del tránsito aéreo:

    *   **CELDA ORDINARIA (CO): 100 a 500 km²**
        *   **Estructura:** Tormentas individuales o pulsos convectivos de escala local.
        *   **Impacto operacional:** Obligan a desvíos tácticos cortos. Generalmente, la tripulación puede evadirlas utilizando el radar meteorológico de a bordo y el contacto visual, solicitando alteraciones menores de rumbo al ATC.
    
    *   **MULTICELULAR (MC): 500 a 1000 km²**
        *   **Estructura:** Clústeres o agrupaciones de celdas convectivas que interactúan entre sí.
        *   **Impacto operacional:** Bloquean tramos considerables de aerovías. Requieren ruteos alternativos planificados que aumentan el consumo de combustible y exigen coordinación anticipada con el Control de Tráfico Aéreo.
    
    *   **SISTEMA CONVECTIVO (SC): > 1000 km²**
        *   **Estructura:** Sistemas Convectivos de Mesoescala (MCS), Complejos Convectivos (MCC) o Líneas de Inestabilidad (Squall Lines).
        *   **Impacto operacional:** Disrupción severa del espacio aéreo. Tienen el potencial de bloquear Regiones de Información de Vuelo (FIR) completas, forzando desvíos estratégicos masivos, demoras generalizadas o el cierre temporal de rutas.
    """)
# -------------------------------------------------

initial_datetime = datetime(2025, 11, 4, 3, 0, 0) 

selected_date = st.date_input("Selecciona la fecha", value=initial_datetime.date())
selected_time = st.time_input("Selecciona la hora (UTC)", value=initial_datetime.time(), step=300) 

start_window_user = datetime.combine(selected_date, selected_time)

data = load_and_process_data(start_window_user, fs_global)

if data is None: 
    st.warning("No se pudieron cargar los datos para la fecha y hora seleccionadas. Intenta con otra fecha/hora.")
else:
    warning_polygons = data["warning_polygons"]
    metrics_df = data["metrics_df"]

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by='Area', ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 1])

    with col2:
        st.header("Tabla de Advertencias por Tormenta")

        options = [f"ID: {int(row.ID)}, Tipo: {row.Escala}, Tope: FL{int(row.MaxFL):03d}, Area: {int(row.Area)} km²"
                   for idx, row in metrics_df.iterrows()]  
        options.insert(0, "-- Seleccionar Polígono --")

        selected_option = st.selectbox(
            "Seleccionar una advertencia para resaltar en el mapa:",
            options,
            index=0
        )
           
        highlight_poly_id = None
        if selected_option != "-- Seleccionar Polígono --":
            highlight_poly_id = int(float(selected_option.split(',')[0].replace('ID: ', '')))
            
            st.markdown(f"### Detalles de la Tormenta (ID: {highlight_poly_id})")
            poly_data = metrics_df[metrics_df["ID"] == highlight_poly_id].iloc[
                0
            ]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Tope (FL)", f"FL{int(poly_data.MaxFL):03d}")
            mc2.metric("Reflectividad", f"{poly_data.MaxRef:.1f} dBZ")
            mc3.metric("Área Envolvente", f"{poly_data.Area:.0f} km²")
            mc4.metric("Tipo", f"{poly_data.Escala}")

            mc5, mc6, mc7 = st.columns(3)
            mc5.metric("Eje Mayor", f"{poly_data.EjeMayor_km:.0f} km")
            mc6.metric("Eje Menor", f"{poly_data.EjeMenor_km:.0f} km")         
            arrow_symbol = rumbo_to_arrow(int(poly_data.Rumbo.replace("°", "")))
            mc7.metric("Orientación / Rumbo", f"{poly_data.Rumbo}", delta=arrow_symbol)  # delta muestra la flecha y dirección

        st.dataframe(metrics_df, height=600, hide_index=True)

        if not metrics_df.empty:
            csv_buffer = io.StringIO()
            metrics_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Descargar métricas como CSV",
                data=csv_buffer.getvalue(),
                file_name="warning_polygons_metrics.csv",
                mime="text/csv",
            )

    with col1:
        st.header("Mapa de Advertencias por Tormenta")
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
        