import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(BASE_DIR, "DatosHistoricos")
DIR_DOCS = os.path.join(BASE_DIR, "DocumentacionNecesaria")
DIR_RESULTADOS = os.path.join(BASE_DIR, "Resultados")
ARCHIVO_UBICACION = "pmed_ubicacion_09-2025.csv"
RUTA_SALIDA = os.path.join(DIR_RESULTADOS, "resultado.csv")
DISTRITOS = {
    1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín",
    6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca",
    10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas",
    14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde",
    18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas",
}

def crear_directorios():
    os.makedirs(DIR_RESULTADOS, exist_ok=True)


def listar_csv_trafico(): # lista todos los csv históricos (si no hay carpeta, devuelve lista vacía)
    archivos = []
    if os.path.isdir(DIR_DATOS):
        archivos = [
            f for f in os.listdir(DIR_DATOS)
            if f.lower().endswith(".csv")
        ]
    return archivos


def cargar_csv_flexible(ruta): # intenta leer un csv con varios separadores/encodings
    df_res = None
    existe = os.path.exists(ruta)
    if existe:
        formatos = [(';', 'utf-8'), (';', 'latin-1'), (',', 'utf-8'), (',', 'latin-1')]
        encontrado = False
        for sep, enc in formatos:
            if not encontrado:
                try:
                    df_temp = pd.read_csv(ruta, sep=sep, encoding=enc, nrows=10, on_bad_lines='skip')
                    if len(df_temp.columns) > 1:
                        # si es válido, cargamos todo y marcamos como encontrado
                        df_res = pd.read_csv(ruta, sep=sep, encoding=enc, on_bad_lines='skip')
                        encontrado = True
                except:
                    pass
    return df_res


def detectar_columnas(df, candidatos): # busca la primera columna que encaje en la lista de candidatos
    col = None
    if df is not None:
        col = next((c for c in df.columns if c in candidatos), None)
    return col


def preparar_maestro_sensores(df_ubic): # limpia columnas y construye el maestro con distrito
    df_ubic.columns = df_ubic.columns.str.strip().str.upper()
    col_id_sensor = detectar_columnas(df_ubic, ['ID', 'CODIGO', 'COD_CENT', 'COD_UBIC'])
    col_distrito = next((c for c in df_ubic.columns if 'DISTRIT' in c), None)

    maestro = pd.DataFrame()
    if col_id_sensor:
        maestro = df_ubic[[col_id_sensor]].copy()
        maestro.rename(columns={col_id_sensor: 'id_sensor'}, inplace=True)

        if col_distrito:
            maestro['id_distrito'] = pd.to_numeric(
                df_ubic[col_distrito].astype(str).str.extract(r'(\d+)', expand=False),
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            maestro['id_distrito'] = 0

        maestro['nombre_distrito'] = maestro['id_distrito'].map(DISTRITOS).fillna("Desconocido")
        maestro = maestro[maestro['id_distrito'] > 0].copy()

    return maestro, col_id_sensor


def calcular_trafico_dia(df):#calcula el total diario de tráfico desde columnas por hora o intensidad directa
    cols_horas = [c for c in df.columns if c.startswith('HOR')]
    if cols_horas:
        df['trafico_dia'] = df[cols_horas].sum(axis=1, numeric_only=True)
    elif 'INTENSIDAD' in df.columns:
        df['trafico_dia'] = df['INTENSIDAD']
    else:
        df['trafico_dia'] = 0
    return df


def parsear_fecha(df, col_fecha):# intenta parsear fechas 
    fechas_dt = pd.to_datetime(df[col_fecha], dayfirst=False, errors='coerce')
    df['dia'] = fechas_dt.dt.day
    df['mes'] = fechas_dt.dt.month
    df['año'] = fechas_dt.dt.year

    # si todo queda vacío, probamos el split por '/'
    if df['dia'].isna().all() and df['mes'].isna().all() and df['año'].isna().all():
        split_fecha = df[col_fecha].astype(str).str.split('/', expand=True)
        if split_fecha.shape[1] >= 3:
            df['mes'] = split_fecha[0].astype(int)
            df['dia'] = split_fecha[1].astype(int)
            df['año'] = split_fecha[2].astype(int)

    return df

def procesar_archivo_trafico(ruta, nombre_archivo): #lee un archivo de tráfico y devuelve un dataframe limpio
    df = cargar_csv_flexible(ruta)
    if df is not None:
        df.columns = df.columns.str.strip().str.upper()
        col_id_traf = detectar_columnas(df, ['ID', 'PUNTO_MEDIDA', 'COD_CENT', 'IDELEM'])
        col_fecha = detectar_columnas(df, ['FECHA', 'FDIA'])
        if col_id_traf and col_fecha:
            df.rename(columns={col_id_traf: 'id_sensor'}, inplace=True)
            df = calcular_trafico_dia(df)
            df = parsear_fecha(df, col_fecha)
            df = df[['id_sensor', 'dia', 'mes', 'año', 'trafico_dia']]
        else:
            print(f"      aviso: columnas faltantes en {nombre_archivo}")
            df = None

    return df


def construir_resultado(dfs_trafico, maestro_sensores): # junta todo y calcula el tráfico medio por distrito y día
    df_trafico_total = pd.concat(dfs_trafico, ignore_index=True)
    df_final = df_trafico_total.merge(maestro_sensores, on='id_sensor', how='inner')

    resultado = df_final.groupby(
        ['dia', 'mes', 'año', 'id_distrito', 'nombre_distrito']
    )['trafico_dia'].mean().reset_index()
    resultado.rename(columns={'trafico_dia': 'trafico_medio'}, inplace=True)
    resultado['trafico_medio'] = resultado['trafico_medio'].round(0).astype(int)
    resultado = resultado[['dia', 'mes', 'año', 'id_distrito', 'nombre_distrito', 'trafico_medio']]
    return resultado


def main():
    crear_directorios()
    ruta_ubic = os.path.join(DIR_DOCS, ARCHIVO_UBICACION)
    print(f"\n1. cargando mapa de sensores: {ruta_ubic}")
    df_ubic = cargar_csv_flexible(ruta_ubic)
    maestro_sensores = pd.DataFrame()
    col_id_sensor = None
    if df_ubic is None:
        print(f"error: no se pudo leer {ruta_ubic}")
    else:
        maestro_sensores, col_id_sensor = preparar_maestro_sensores(df_ubic)
        if not col_id_sensor:
            print("error: no se encuentra ID en archivo de ubicación.")

    ok_maestro = (df_ubic is not None) and bool(col_id_sensor) and not maestro_sensores.empty
    dfs_trafico = []
    if ok_maestro:
        print("\n2. procesando archivos de tráfico...")
        for archivo in listar_csv_trafico():
            ruta = os.path.join(DIR_DATOS, archivo)
            print(f"   -> leyendo: {archivo}")
            df_archivo = procesar_archivo_trafico(ruta, archivo)
            if df_archivo is not None:
                dfs_trafico.append(df_archivo)

    if not ok_maestro:
        print("error: no se pudo preparar el maestro de sensores.")
    if ok_maestro and not dfs_trafico:
        print("error: no hay datos de tráfico válidos.")

    if ok_maestro and dfs_trafico:
        print("\n3. generando dataset final...")
        resultado = construir_resultado(dfs_trafico, maestro_sensores)
        print(f"\n4. guardando en: {RUTA_SALIDA}")
        resultado.to_csv(RUTA_SALIDA, index=False, sep=';', decimal=',')
        print("¡hecho!")
        print(resultado.head())
        estado = 0
    else:
        estado = 1

    return estado


if __name__ == "__main__":
    sys.exit(main())
