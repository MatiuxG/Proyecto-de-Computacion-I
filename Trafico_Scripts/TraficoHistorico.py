import pandas as pd
import re
import sys
import os

# ==================== CONFIGURACIÓN DE RUTAS ====================
RUTA_UBICACIONES = r"Trafico_Scripts\DocumentacionNecesaria\UbicacionEstacionesPermanentesSentidos.csv"
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

# ==================== FUNCIONES DE LIMPIEZA ====================
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
    """Intenta cargar con utf-8, si falla prueba latin-1"""
    if not os.path.exists(ruta):
        return None, "Archivo no encontrado"
    try:
        return pd.read_csv(ruta, sep=sep, encoding='utf-8'), "utf-8"
    except:
        try:
            return pd.read_csv(ruta, sep=sep, encoding='latin-1'), "latin-1"
        except Exception as e:
            return None, str(e)

# ==================== 1. CARGA DE ARCHIVOS ====================
print("1. Cargando archivos...")

# Cargar VIALES
viales, _ = cargar_csv_seguro(RUTA_VIALES)
if viales is None:
    print(f"ERROR: No se encuentra el archivo de viales: {RUTA_VIALES}")
    sys.exit()

# Cargar UBICACIONES
ubicaciones, _ = cargar_csv_seguro(RUTA_UBICACIONES)
if ubicaciones is None:
    print(f"ERROR: No se encuentra el archivo de ubicaciones: {RUTA_UBICACIONES}")
    sys.exit()

# Cargar DATOS
datos, _ = cargar_csv_seguro(RUTA_DATOS)
if datos is None:
    print(f"ERROR: No se encuentra el archivo de datos: {RUTA_DATOS}")
    sys.exit()

# Limpieza básica de nombres de columna
viales.columns = viales.columns.str.strip()
ubicaciones.columns = ubicaciones.columns.str.strip()
datos.columns = datos.columns.str.strip()

# ==================== 2. VALIDACIÓN DE ARCHIVOS ====================
print("2. Validando contenido...")

# Verificar si Ubicaciones es realmente Ubicaciones
if 'Nombre' not in ubicaciones.columns and 'Estación' not in ubicaciones.columns:
    print("ERROR: El archivo cargado en RUTA_UBICACIONES no parece correcto.")
    print(f"Columnas encontradas: {ubicaciones.columns.tolist()}")
    print("Asegúrate de que RUTA_UBICACIONES apunta a 'UbicacionEstacionesPermanentesSentidos.csv'")
    sys.exit()

# Verificar si Datos es realmente Datos
if 'HOR1' not in datos.columns and 'FEST' not in datos.columns:
    print("ERROR: El archivo cargado en RUTA_DATOS no parece correcto (faltan columnas HOR1 o FEST).")
    sys.exit()

print("   -> Archivos validados correctamente.")

# ==================== 3. PROCESAMIENTO ====================
print("3. Cruzando información...")

# --- A. VIALES ---
viales['key_match'] = (viales['VIA_CLASE'].astype(str) + ' ' + viales['VIA_NOMBRE'].astype(str)).apply(limpiar_nombre_calle)
viales['CO_DISTRITO'] = pd.to_numeric(viales['CO_DISTRITO'], errors='coerce')
viales = viales.dropna(subset=['CO_DISTRITO'])
viales['district_id'] = viales['CO_DISTRITO'].astype(int)
viales['district_name'] = viales['district_id'].map(DISTRITOS_MADRID)
viales_unicos = viales[['key_match', 'district_id', 'district_name']].drop_duplicates(subset='key_match')

# --- B. UBICACIONES ---
col_estacion = 'Estación' if 'Estación' in ubicaciones.columns else 'Estacion'
ubicaciones = ubicaciones.rename(columns={col_estacion: 'station_id'})
ubicaciones['key_match'] = ubicaciones['Nombre'].apply(limpiar_nombre_calle)

estaciones_map = ubicaciones[['station_id', 'key_match']].drop_duplicates(subset='station_id')
estaciones_con_distrito = estaciones_map.merge(viales_unicos, on='key_match', how='left')

encontrados = estaciones_con_distrito['district_id'].notnull().sum()
print(f"   -> Estaciones ubicadas: {encontrados} de {len(estaciones_con_distrito)}")

# --- C. DATOS Y CÁLCULO ---
print("4. Calculando resultados...")
datos = datos[datos['FEST'].notnull()].copy()
datos['station_id'] = datos['FEST'].astype(str).str.replace('ES', '', regex=False).astype(int)

# Unir todo
datos_full = datos.merge(estaciones_con_distrito[['station_id', 'district_id', 'district_name']], on='station_id', how='left')

# Calcular volumen diario
cols_horas = [c for c in datos_full.columns if c.startswith('HOR')]
datos_full['volumen_total'] = datos_full[cols_horas].sum(axis=1)

# Extraer fecha
datos_full[['Dia', 'Mes', 'Año']] = datos_full['FDIA'].astype(str).str.split('/', expand=True)

# Agrupar por distrito
resultado = datos_full.groupby(
    ['FDIA', 'Dia', 'Mes', 'Año', 'district_id', 'district_name']
)['volumen_total'].sum().reset_index(name='volumen_total_distrito')

# IMD (dividir por 24h)
resultado['intensidad_media_diaria'] = resultado['volumen_total_distrito'] / 24

# === CORRECCIÓN FINAL DE FORMATO ===
# Forzamos que el district_id sea entero para quitar el '.0'
resultado['district_id'] = resultado['district_id'].astype(int)
# 2. Quitamos decimales de intensidad_media_diaria
resultado['intensidad_media_diaria'] = resultado['intensidad_media_diaria'].astype(int)


resultado = resultado[['Dia', 'Mes', 'Año', 'district_id', 'district_name', 'intensidad_media_diaria']]
resultado.to_csv(RUTA_RESULTADO, index=False, sep=';', decimal=',')

print(f"¡LISTO! Archivo guardado en: {RUTA_RESULTADO}")
print(resultado.head())