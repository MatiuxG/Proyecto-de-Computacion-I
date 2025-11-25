import pandas as pd
import re
import sys
import os

# ==================== CONFIGURACIÓN DE RUTAS ====================
RUTA_UBICACIONES = r"Trafico_Scripts\DocumentacionNecesaria\UbicacionEstacionesPermanentesSentidos.csv"
# NOTA: Asegúrate de que este archivo contenga datos de Julio-Septiembre 2025
RUTA_DATOS = r"Trafico_Scripts\DocumentacionNecesaria\DATOS_ESTACIONES_MARZO_2025.csv" 
RUTA_VIALES = r"Trafico_Scripts\DocumentacionNecesaria\VialesVigentesDistritos_20251119.csv"
RUTA_RESULTADO = r"Trafico_Scripts\Trafico_Scripts\Resultados\resultado.csv"  
# ================================================================

DISTRITOS_MADRID = {
    1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín",
    6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca",
    10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas",
    14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde",
    18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas"
}

def limpiar_nombre_calle(texto):
    if pd.isna(texto): return ""
    t = str(texto).upper().strip()
    t = t.replace('FRANCISO', 'FRANCISCO') 
    t = re.sub(r'\(.*?\)', '', t)
    t = t.replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    particulas = ['DE LA', 'DEL', 'DE', 'LAS', 'LOS', 'LA', 'EL', 'AL']
    particulas.sort(key=len, reverse=True)
    for p in particulas:
        t = re.sub(r'\b' + p + r'\b', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def cargar_csv_seguro(ruta, sep=';'):
    if not os.path.exists(ruta):
        return None, "Archivo no encontrado"
    try:
        return pd.read_csv(ruta, sep=sep, encoding='utf-8'), "utf-8"
    except:
        try:
            return pd.read_csv(ruta, sep=sep, encoding='latin-1'), "latin-1"
        except Exception as e:
            return None, str(e)

print("1. Cargando archivos...")

viales, _ = cargar_csv_seguro(RUTA_VIALES)
if viales is None:
    print(f"ERROR: No se encuentra el archivo de viales: {RUTA_VIALES}")
    sys.exit()

ubicaciones, _ = cargar_csv_seguro(RUTA_UBICACIONES)
if ubicaciones is None:
    print(f"ERROR: No se encuentra el archivo de ubicaciones: {RUTA_UBICACIONES}")
    sys.exit()

datos, _ = cargar_csv_seguro(RUTA_DATOS)
if datos is None:
    print(f"ERROR: No se encuentra el archivo de datos: {RUTA_DATOS}")
    sys.exit()

viales.columns = viales.columns.str.strip()
ubicaciones.columns = ubicaciones.columns.str.strip()
datos.columns = datos.columns.str.strip()

print("2. Validando contenido...")

if 'Nombre' not in ubicaciones.columns and 'Estación' not in ubicaciones.columns:
    print("ERROR: El archivo cargado en RUTA_UBICACIONES no parece correcto.")
    sys.exit()

if 'HOR1' not in datos.columns and 'FEST' not in datos.columns:
    print("ERROR: El archivo cargado en RUTA_DATOS no parece correcto.")
    sys.exit()

print("   -> Archivos validados correctamente.")

print("3. Cruzando información...")

viales['key_match'] = (viales['VIA_CLASE'].astype(str) + ' ' + viales['VIA_NOMBRE'].astype(str)).apply(limpiar_nombre_calle)
viales['CO_DISTRITO'] = pd.to_numeric(viales['CO_DISTRITO'], errors='coerce')
viales = viales.dropna(subset=['CO_DISTRITO'])
viales['district_id'] = viales['CO_DISTRITO'].astype(int)
viales['district_name'] = viales['district_id'].map(DISTRITOS_MADRID)
viales_unicos = viales[['key_match', 'district_id', 'district_name']].drop_duplicates(subset='key_match')

col_estacion = 'Estación' if 'Estación' in ubicaciones.columns else 'Estacion'
ubicaciones = ubicaciones.rename(columns={col_estacion: 'station_id'})
ubicaciones['key_match'] = ubicaciones['Nombre'].apply(limpiar_nombre_calle)

estaciones_map = ubicaciones[['station_id', 'key_match']].drop_duplicates(subset='station_id')
estaciones_con_distrito = estaciones_map.merge(viales_unicos, on='key_match', how='left')

encontrados = estaciones_con_distrito['district_id'].notnull().sum()
print(f"   -> Estaciones ubicadas: {encontrados} de {len(estaciones_con_distrito)}")

print("4. Calculando resultados...")
datos = datos[datos['FEST'].notnull()].copy()
datos['station_id'] = datos['FEST'].astype(str).str.replace('ES', '', regex=False).astype(int)

datos_full = datos.merge(estaciones_con_distrito[['station_id', 'district_id', 'district_name']], on='station_id', how='left')

cols_horas = [c for c in datos_full.columns if c.startswith('HOR')]
datos_full['volumen_total'] = datos_full[cols_horas].sum(axis=1)

datos_full[['Dia', 'Mes', 'Año']] = datos_full['FDIA'].astype(str).str.split('/', expand=True)

# --- MODIFICADO: FILTRO FECHAS (Julio-Septiembre 2025) ---
print("Filtrando datos para Julio-Septiembre 2025...")
mask = (datos_full["Año"].astype(str) == "2025") & (datos_full["Mes"].astype(str).str.zfill(2).isin(["07", "08", "09"]))
datos_full = datos_full[mask]

resultado = datos_full.groupby(
    ['FDIA', 'Dia', 'Mes', 'Año', 'district_id', 'district_name']
)['volumen_total'].sum().reset_index(name='volumen_total_distrito')

resultado['intensidad_media_diaria'] = resultado['volumen_total_distrito'] / 24

resultado['district_id'] = resultado['district_id'].astype(int)
resultado['intensidad_media_diaria'] = resultado['intensidad_media_diaria'].astype(int)

resultado = resultado[['Dia', 'Mes', 'Año', 'district_id', 'district_name', 'intensidad_media_diaria']]
resultado.to_csv(RUTA_RESULTADO, index=False, sep=';', decimal=',')

print(f"¡LISTO! Archivo guardado en: {RUTA_RESULTADO}")
print(resultado.head())