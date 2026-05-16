import io
import os
from datetime import datetime
from io import StringIO
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup


# Dataset de ubicaciones de los puntos de medida (sensor -> distrito).
# Si no encontramos el CSV en local, lo bajamos del portal.
URL_UBICACIONES = "https://datos.madrid.es/dataset/202468-0-intensidad-trafico/downloads"

DISTRITOS = {
    1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín",
    6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca",
    10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas",
    14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde",
    18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas",
}

def scrapper_informo_realtime():
    url_catalogo = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=202087-0-trafico-intensidad"
    df_final = pd.DataFrame()
    enlace_encontrado = ""

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res_catalogo = requests.get(url_catalogo, headers=headers, timeout=15)
        
        if res_catalogo.status_code == 200:
            sopa = BeautifulSoup(res_catalogo.text, 'html.parser')
            enlaces = sopa.find_all('a')
            for tag in enlaces:
                link = str(tag.get('href'))
                if "pm.xml" in link:
                    enlace_encontrado = link if link.startswith('http') else "https://informo.madrid.es" + link

        if enlace_encontrado == "":
            enlace_encontrado = "https://informo.madrid.es/informo/tmadrid/pm.xml"

        res_datos = requests.get(enlace_encontrado, headers=headers, timeout=15)
        if res_datos.status_code == 200:
            datos_xml = StringIO(res_datos.text)
            df_raw = pd.read_xml(datos_xml, parser='etree')
            df_raw.columns = df_raw.columns.str.strip().str.upper()
            
            if 'IDELEM' in df_raw.columns:
                ahora = datetime.now()
                df_final = pd.DataFrame({
                    'id_sensor': df_raw['IDELEM'],
                    'intensidad': df_raw['INTENSIDAD'].fillna(0).astype(int),
                    'dia': ahora.day,
                    'mes': ahora.month,
                    'año': ahora.year
                })
    except Exception as e:
        print(f"Error en el scraping: {e}")
    return df_final

def descargar_maestro_sensores():
    #descarga del portal el csv de ubicaciones de los puntos de medida
    #y devuelve un dataframe con las columnas id_sensor y id_distrito
    print("Descargando maestro de sensores desde datos.madrid.es...")
    headers = {"User-Agent": "Mozilla/5.0"}
    maestro = pd.DataFrame()
    try:
        respuesta_pagina = requests.get(URL_UBICACIONES, headers=headers, timeout=30)
        sopa = BeautifulSoup(respuesta_pagina.text, "html.parser")

        url_csv = ""
        for tag in sopa.find_all("a", href=True):
            href = tag["href"]
            full = href if href.startswith("http") else urljoin(URL_UBICACIONES, href)
            if full.lower().endswith(".csv"):
                url_csv = full
                break

        if url_csv:
            respuesta_csv = requests.get(url_csv, headers=headers, timeout=60)
            #el portal sirve este CSV en utf-8 pero por seguridad probamos los dos
            df_ubic = None
            for codificacion in ("utf-8", "latin-1"):
                try:
                    df_ubic = pd.read_csv(io.BytesIO(respuesta_csv.content), sep=";", dtype=str, encoding=codificacion)
                    break
                except UnicodeDecodeError:
                    df_ubic = None

            if df_ubic is not None:
                df_ubic.columns = df_ubic.columns.str.strip().str.lower()
                if "id" in df_ubic.columns and "distrito" in df_ubic.columns:
                    maestro = df_ubic[["id", "distrito"]].copy()
                    maestro.rename(columns={"id": "id_sensor", "distrito": "id_distrito"}, inplace=True)
                    maestro["id_distrito"] = pd.to_numeric(
                        maestro["id_distrito"].astype(str).str.extract(r"(\d+)")[0],
                        errors="coerce",
                    )
    except Exception as error:
        print(f"Error descargando el maestro: {error}")
    return maestro


def integrar_con_distritos(df_realtime):
    df_final = pd.DataFrame()

    maestro = descargar_maestro_sensores()
    if maestro.empty:
        print("ATENCION: no se pudo obtener el maestro de sensores. No se mapean distritos.")
    else:
        #el XML llega con id como flotante (idelem=9841.0) y el csv lo trae como entero ("9841").
        #si los castas directos a str salen "9841.0" vs "9841" y el merge no encuentra nada.
        #convertimos pasando antes por numerico para que ambos lados queden iguales.
        df_realtime = df_realtime.copy()
        df_realtime = df_realtime.dropna(subset=["id_sensor"])
        df_realtime["id_sensor"] = pd.to_numeric(df_realtime["id_sensor"], errors="coerce").astype("Int64").astype(str)
        maestro["id_sensor"] = pd.to_numeric(maestro["id_sensor"], errors="coerce").astype("Int64").astype(str)

        df_unificado = pd.merge(df_realtime, maestro, on="id_sensor", how="inner")
        df_final = df_unificado.groupby(["dia", "mes", "año", "id_distrito"])["intensidad"].mean().reset_index()
        df_final.rename(columns={"intensidad": "trafico_medio"}, inplace=True)
        df_final["id_distrito"] = df_final["id_distrito"].astype(int)
        df_final["nombre_distrito"] = df_final["id_distrito"].map(DISTRITOS)

    return df_final

if __name__ == "__main__":
    print("Iniciando Scraper de tráfico en tiempo real...")
    datos_crudos = scrapper_informo_realtime()
    
    if not datos_crudos.empty:
        print("Datos crudos obtenidos. Mapeando con distritos...")
        resultado_final = integrar_con_distritos(datos_crudos)
        
        if not resultado_final.empty:
          
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            DIR_RES = os.path.join(BASE_DIR, "Resultados")
            os.makedirs(DIR_RES, exist_ok=True)
            
            ruta_csv = os.path.join(DIR_RES, "resultado_realtime_distritos.csv")
            resultado_final.to_csv(ruta_csv, index=False, sep=";")
            
            print(f"\n--- PROCESO COMPLETADO ---")
            print(resultado_final.head(10))
            print(f"\nArchivo guardado en: {ruta_csv}")
    else:
        print("No se pudieron obtener datos del servidor de Madrid.")