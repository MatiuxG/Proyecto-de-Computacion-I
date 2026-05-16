"""
Scraper de trafico historico de Madrid.

Descarga los ZIPs mensuales del dataset 208627 (Trafico - Historico desde 2013)
y el CSV de ubicaciones de los puntos de medida (202468) para mapear cada
sensor a su distrito.

Por defecto baja los N ZIPs mas recientes (variable MESES_RECIENTES) para
no tirarse horas: cada ZIP pesa unos 90MB y dentro hay un CSV con una
medicion por sensor y cada 15 minutos, asi que se procesa por chunks
para no comernos toda la RAM.

Los zips se cachean en .cache_zips/ para que si reprocesas no vuelvas
a descargar 1GB. La primera ejecucion tarda mucho, las siguientes son
rapidas.

Salida: Trafico_Scripts/Resultados/resultado.csv con columnas
    dia;mes;año;hora;id_distrito;nombre_distrito;trafico_medio
"""

import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {"User-Agent": "EcoTrafficScraper/1.0"}
TIMEOUT_CONEXION = 30
TIMEOUT_DESCARGA = 600  # los zips son grandes
CHUNK_FILAS = 200_000   # filas por trozo cuando leemos el CSV gigante

# Dataset 208627: trafico historico (zips mensuales)
URL_HISTORICO = "https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/downloads"
# Dataset 202468: ubicaciones de los puntos de medida (sensor -> distrito)
URL_UBICACIONES = "https://datos.madrid.es/dataset/202468-0-intensidad-trafico/downloads"

# Cuantos meses descargar. Los meses se sirven en orden inverso, asi que
# "1" baja el ultimo, "3" baja los tres mas recientes. Subelo si quieres
# mas historia (cada mes son unos 90MB descomprimido bastante mas).
MESES_RECIENTES = 12

DISTRITOS = {
    1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín",
    6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca",
    10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas",
    14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde",
    18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas",
    21: "Barajas",
}

BASE_DIR = Path(__file__).resolve().parent
DIR_RESULTADOS = BASE_DIR / "Resultados"
DIR_CACHE = BASE_DIR / ".cache_zips"
RUTA_SALIDA = DIR_RESULTADOS / "resultado.csv"


def listar_enlaces(url_pagina, extension):
    """Devuelve todos los enlaces de una pagina que terminan en la extension dada."""
    enlaces = []
    respuesta = requests.get(url_pagina, headers=HEADERS, timeout=TIMEOUT_CONEXION)
    sopa = BeautifulSoup(respuesta.text, "html.parser")
    for tag in sopa.find_all("a", href=True):
        href = tag["href"]
        url_completa = href if href.startswith("http") else urljoin(url_pagina, href)
        if url_completa.lower().endswith(extension):
            enlaces.append(url_completa)
    return enlaces


def descargar_ubicaciones():
    """Descarga el CSV mas reciente de ubicaciones de puntos de medida
    y devuelve un dict {id_sensor: id_distrito} con solo los distritos
    validos (1-21)."""
    print(f"\n[1/3] Descargando ubicaciones de sensores...")
    enlaces = listar_enlaces(URL_UBICACIONES, ".csv")
    if not enlaces:
        print("  [Error] No se encontraron CSVs de ubicaciones.")
        return {}

    # El primero suele ser el mas reciente
    url_csv = enlaces[0]
    print(f"  bajando: {url_csv[:90]}...")
    respuesta = requests.get(url_csv, headers=HEADERS, timeout=TIMEOUT_CONEXION)
    respuesta.raise_for_status()

    # El CSV de ubicaciones suele venir en latin-1 (nombres con tildes)
    df = None
    for codificacion in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            df = pd.read_csv(io.BytesIO(respuesta.content), sep=";", dtype=str, encoding=codificacion)
            break
        except UnicodeDecodeError:
            df = None
    if df is None:
        print("  [Error] No se pudo decodificar el CSV de ubicaciones.")
        return {}
    df.columns = df.columns.str.strip().str.lower()

    # Las columnas esperadas son "id" (sensor) y "distrito" (numero)
    col_id = next((c for c in df.columns if c == "id"), None)
    col_dist = next((c for c in df.columns if "distrito" in c), None)

    mapa = {}
    if col_id and col_dist:
        for _, fila in df.iterrows():
            sensor = str(fila[col_id]).strip()
            distrito_txt = str(fila[col_dist]).strip()
            if sensor and distrito_txt:
                # nos quedamos con los digitos
                digitos = "".join(c for c in distrito_txt if c.isdigit())
                if digitos:
                    num = int(digitos)
                    if 1 <= num <= 21:
                        mapa[sensor] = num
    print(f"  [OK] {len(mapa)} sensores mapeados a distrito")
    return mapa


def listar_zips_historicos():
    """Lista todos los zips disponibles del dataset historico, en el orden
    que devuelve la pagina (los mas recientes primero)."""
    enlaces = listar_enlaces(URL_HISTORICO, ".zip")
    return enlaces


def descargar_zip(url):
    """Descarga un zip a la cache de disco si no esta ya, y devuelve la ruta.
    Si el zip ya existe en cache no vuelve a bajarlo."""
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    #usamos el nombre tras el ultimo / como nombre de cache (por ejemplo 208627-156-...zip)
    nombre = url.rstrip("/").rsplit("/", 1)[-1]
    if not nombre.endswith(".zip"):
        nombre = nombre + ".zip"
    ruta_destino = DIR_CACHE / nombre

    if ruta_destino.exists() and ruta_destino.stat().st_size > 1024:
        print(f"  [cache] {ruta_destino.name} ({ruta_destino.stat().st_size/1024/1024:.1f} MB)")
        return str(ruta_destino)

    print(f"  descargando {url[:90]}...")
    respuesta = requests.get(url, headers=HEADERS, timeout=TIMEOUT_DESCARGA, stream=True)
    respuesta.raise_for_status()

    total = 0
    with open(ruta_destino, "wb") as fichero:
        for chunk in respuesta.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fichero.write(chunk)
                total += len(chunk)
    print(f"    {total/1024/1024:.1f} MB en disco -> {ruta_destino.name}")
    return str(ruta_destino)


def procesar_csv_por_chunks(zip_path, mapa_sensor_distrito):
    """Lee el CSV dentro del zip por trozos y va acumulando suma y conteo
    de intensidad por (sensor, dia, mes, año, hora).
    Devuelve un dataframe con la intensidad media por sensor-hora.
    """
    # Acumulador: clave -> [suma, conteo]
    acumulado = {}

    with zipfile.ZipFile(zip_path) as archivo_zip:
        # Cada zip contiene un unico CSV mensual
        nombre_csv = archivo_zip.namelist()[0]
        with archivo_zip.open(nombre_csv) as fichero_csv:
            lector = pd.read_csv(
                fichero_csv,
                sep=";",
                dtype=str,
                usecols=["id", "fecha", "intensidad"],
                chunksize=CHUNK_FILAS,
            )
            for trozo in lector:
                # Solo nos quedamos con sensores que sabemos mapear
                trozo = trozo[trozo["id"].isin(mapa_sensor_distrito)]
                if trozo.empty:
                    continue

                # Parseamos fecha e intensidad
                fechas = pd.to_datetime(trozo["fecha"], errors="coerce")
                intensidades = pd.to_numeric(trozo["intensidad"], errors="coerce").fillna(0)

                # Saltamos filas con fecha invalida
                validos = fechas.notna()
                if not validos.any():
                    continue
                fechas = fechas[validos]
                intensidades = intensidades[validos]
                sensores = trozo.loc[validos, "id"]

                # Construimos las claves agregadas (ahora con hora del dia)
                claves = list(zip(
                    sensores.values,
                    fechas.dt.day.astype(int).values,
                    fechas.dt.month.astype(int).values,
                    fechas.dt.year.astype(int).values,
                    fechas.dt.hour.astype(int).values,
                ))

                for clave, valor in zip(claves, intensidades.values):
                    if clave in acumulado:
                        sumas = acumulado[clave]
                        sumas[0] += valor
                        sumas[1] += 1
                    else:
                        acumulado[clave] = [valor, 1]

    # Convertimos a dataframe
    filas = []
    for (sensor, dia, mes, año, hora), (suma, conteo) in acumulado.items():
        if conteo > 0:
            filas.append({
                "id_sensor": sensor,
                "dia": dia,
                "mes": mes,
                "año": año,
                "hora": hora,
                "intensidad_media": suma / conteo,
            })
    return pd.DataFrame(filas)


def construir_resultado(dfs_mensuales, mapa_sensor_distrito):
    """Une todos los meses, mapea sensores a distritos y agrega por
    (dia, mes, año, hora, distrito) calculando la intensidad media."""
    df_total = pd.concat(dfs_mensuales, ignore_index=True)

    # Mapeo a distrito
    df_total["id_distrito"] = df_total["id_sensor"].map(mapa_sensor_distrito)
    df_total = df_total.dropna(subset=["id_distrito"])
    df_total["id_distrito"] = df_total["id_distrito"].astype(int)

    # Media por distrito y hora
    df_distrito = (
        df_total
        .groupby(["dia", "mes", "año", "hora", "id_distrito"])["intensidad_media"]
        .mean()
        .reset_index()
        .rename(columns={"intensidad_media": "trafico_medio"})
    )
    df_distrito["trafico_medio"] = df_distrito["trafico_medio"].round(1)
    df_distrito["nombre_distrito"] = df_distrito["id_distrito"].map(DISTRITOS)

    columnas = ["dia", "mes", "año", "hora", "id_distrito", "nombre_distrito", "trafico_medio"]
    return df_distrito[columnas]


def main():
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Scraper de trafico historico - meses a bajar: {MESES_RECIENTES}")

    mapa_sensor_distrito = descargar_ubicaciones()
    if not mapa_sensor_distrito:
        print("[Error] No se pudo construir el mapa de sensores. Abortando.")
        return 1

    print(f"\n[2/3] Listando zips historicos...")
    enlaces_zips = listar_zips_historicos()
    if not enlaces_zips:
        print("[Error] No hay zips historicos disponibles.")
        return 1
    print(f"  {len(enlaces_zips)} zips encontrados, bajamos los {MESES_RECIENTES} mas recientes")

    print(f"\n[3/3] Descargando y procesando meses...")
    dfs = []
    for url_zip in enlaces_zips[:MESES_RECIENTES]:
        try:
            ruta_zip = descargar_zip(url_zip)
            df_mes = procesar_csv_por_chunks(ruta_zip, mapa_sensor_distrito)
            #ojo: NO borramos el zip, queda en .cache_zips/ por si reprocesamos.
            #si quieres liberar espacio, borra esa carpeta a mano.
            if not df_mes.empty:
                print(f"    -> {len(df_mes)} filas sensor-hora agregadas")
                dfs.append(df_mes)
        except Exception as error:
            print(f"  [Aviso] zip fallido ({error}); seguimos con el siguiente")

    if not dfs:
        print("[Error] Ningun zip se proceso correctamente.")
        return 1

    resultado = construir_resultado(dfs, mapa_sensor_distrito)
    resultado.to_csv(RUTA_SALIDA, sep=";", index=False, encoding="utf-8-sig")

    print(f"\n[OK] {RUTA_SALIDA}")
    print(f"     filas: {len(resultado)}")
    print(f"     dias distintos: {resultado[['dia','mes','año']].drop_duplicates().shape[0]}")
    print(f"     distritos cubiertos: {resultado['id_distrito'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
