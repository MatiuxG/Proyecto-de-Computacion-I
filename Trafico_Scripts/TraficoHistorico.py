import pandas as pd
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIR_DATOS = os.path.join(BASE_DIR, "DatosHistoricos")
DIR_DOCS = os.path.join(BASE_DIR, "DocumentacionNecesaria")
ARCHIVOS_TRAFICO = [
    f for f in os.listdir(DIR_DATOS)
    if f.lower().endswith(".csv")
]
ARCHIVO_UBICACION = "pmed_ubicacion_09-2025.csv"
DIR_RESULTADOS = os.path.join(BASE_DIR, "Resultados")
RUTA_SALIDA = os.path.join(DIR_RESULTADOS, "resultado.csv")


DISTRITOS = {
    1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín",
    6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca",
    10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas",
    14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde",
    18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas"
}

def crear_directorios():
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

def cargar_csv_flexible(ruta):
    df_res = None
    if os.path.exists(ruta):
        formatos = [(';', 'utf-8'), (';', 'latin-1'), (',', 'utf-8'), (',', 'latin-1')]
        encontrado = False
        for sep, enc in formatos:
            if not encontrado:
                try:
                    df_temp = pd.read_csv(ruta, sep=sep, encoding=enc, nrows=10, on_bad_lines='skip')
                    if len(df_temp.columns) > 1:
                        # Si es válido, cargamos todo y marcamos como encontrado
                        df_res = pd.read_csv(ruta, sep=sep, encoding=enc, on_bad_lines='skip')
                        encontrado = True
                except:
                    pass
    return df_res


print("--- INICIANDO PROCESAMIENTO (CORRECCIÓN FECHAS) ---")
crear_directorios()

ruta_ubic = os.path.join(DIR_DOCS, ARCHIVO_UBICACION)
print(f"\n1. Cargando mapa de sensores: {ruta_ubic}")
df_ubic = cargar_csv_flexible(ruta_ubic)

if df_ubic is None:
    print(f"ERROR: No se pudo leer {ruta_ubic}")
    sys.exit()

df_ubic.columns = df_ubic.columns.str.strip().str.upper()

col_id_sensor = next((c for c in df_ubic.columns if c in ['ID', 'CODIGO', 'COD_CENT', 'COD_UBIC']), None)
col_distrito = next((c for c in df_ubic.columns if 'DISTRIT' in c), None)

if not col_id_sensor:
    print("ERROR: No se encuentra ID en archivo de ubicación.")
    sys.exit()

maestro_sensores = df_ubic[[col_id_sensor]].copy()
maestro_sensores.rename(columns={col_id_sensor: 'id_sensor'}, inplace=True)

if col_distrito:
    maestro_sensores['id_distrito'] = pd.to_numeric(df_ubic[col_distrito].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)
else:
    maestro_sensores['id_distrito'] = 0

maestro_sensores['nombre_distrito'] = maestro_sensores['id_distrito'].map(DISTRITOS).fillna("Desconocido")
maestro_sensores = maestro_sensores[maestro_sensores['id_distrito'] > 0].copy()

print("\n2. Procesando archivos de tráfico...")
dfs_trafico = []

for archivo in ARCHIVOS_TRAFICO:
    ruta = os.path.join(DIR_DATOS, archivo)
    print(f"   -> Leyendo: {archivo}")
    df = cargar_csv_flexible(ruta)
    
    if df is not None:
        df.columns = df.columns.str.strip().str.upper()
        
        col_id_traf = next((c for c in df.columns if c in ['ID', 'PUNTO_MEDIDA', 'COD_CENT', 'IDELEM']), None)
        col_fecha = next((c for c in df.columns if c in ['FECHA', 'FDIA']), None)
        
        if col_id_traf and col_fecha:
            df.rename(columns={col_id_traf: 'id_sensor'}, inplace=True)
            
            # Intensidad
            cols_horas = [c for c in df.columns if c.startswith('HOR')]
            if cols_horas:
                df['trafico_dia'] = df[cols_horas].sum(axis=1, numeric_only=True)
            elif 'INTENSIDAD' in df.columns:
                df['trafico_dia'] = df['INTENSIDAD']
            else:
                df['trafico_dia'] = 0
            
            try:
                fechas_dt = pd.to_datetime(df[col_fecha], dayfirst=False, errors='coerce')
                df['dia'] = fechas_dt.dt.day
                df['mes'] = fechas_dt.dt.month
                df['año'] = fechas_dt.dt.year
            except:
                split_fecha = df[col_fecha].astype(str).str.split('/', expand=True)
                df['mes'] = split_fecha[0].astype(int) 
                df['dia'] = split_fecha[1].astype(int) 
                df['año'] = split_fecha[2].astype(int)

            dfs_trafico.append(df[['id_sensor', 'dia', 'mes', 'año', 'trafico_dia']])
        else:
            print(f"      AVISO: Columnas faltantes en {archivo}")

if not dfs_trafico:
    print("ERROR: No hay datos.")
    sys.exit()

df_trafico_total = pd.concat(dfs_trafico, ignore_index=True)

print("\n3. Generando dataset final...")
df_final = df_trafico_total.merge(maestro_sensores, on='id_sensor', how='inner')
resultado = df_final.groupby(['dia', 'mes', 'año', 'id_distrito', 'nombre_distrito'])['trafico_dia'].mean().reset_index()

resultado.rename(columns={'trafico_dia': 'trafico_medio'}, inplace=True)
resultado['trafico_medio'] = resultado['trafico_medio'].round(0).astype(int)
resultado = resultado[['dia', 'mes', 'año', 'id_distrito', 'nombre_distrito', 'trafico_medio']]

print(f"\n4. Guardando en: {RUTA_SALIDA}")
resultado.to_csv(RUTA_SALIDA, index=False, sep=';', decimal=',')
print("¡HECHO!")
print(resultado.head())
