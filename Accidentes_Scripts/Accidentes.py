import re
import io
import unicodedata
from pathlib import Path
from datetime import date as date_cls
import requests
import pandas as pd
from bs4 import BeautifulSoup

#configuracion general del script
#usamos la vista /downloads del portal CKAN porque ahi si que se ven todos los enlaces directos a .csv
DATASET_PAGES = [
    "https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/downloads",
]

#el output se guarda en una carpeta Resultados al lado del script,
OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_accidentes.csv"

#cabecera simple para que no se nos bloque por no identificarnos
HEADERS = {"User-Agent": "MateoScraperBot/1.1"}
FINAL_COLUMNS = ["Dia", "Mes", "Año", "Hora", "district_code", "district_name", "total_de_accidentes"]

MADRID_DISTRICTS = {
    "1": "Centro", "01": "Centro", "2": "Arganzuela", "02": "Arganzuela",
    "3": "Retiro", "03": "Retiro", "4": "Salamanca", "04": "Salamanca",
    "5": "Chamartín", "05": "Chamartín", "6": "Tetuán", "06": "Tetuán",
    "7": "Chamberí", "07": "Chamberí", "8": "Fuencarral-El Pardo", "08": "Fuencarral-El Pardo",
    "9": "Moncloa-Aravaca", "09": "Moncloa-Aravaca", "10": "Latina", "11": "Carabanchel",
    "12": "Usera", "13": "Puente de Vallecas", "14": "Moratalaz", "15": "Ciudad Lineal",
    "16": "Hortaleza", "17": "Villaverde", "18": "Villa de Vallecas", "19": "Vicálvaro",
    "20": "San Blas-Canillejas", "21": "Barajas",
}

#funciones para procesar los datos

def fetch_text(url):
    #pide una pagina y devuelve el html como texto 
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    #ajusta la codificacion si el servidor no la dice o no la da bien 
    if not response.encoding or response.encoding.lower() in ("ascii", "utf-8"):
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text

def find_download_links_html(soup, base_url):
    #saca posibles enlaces de descarga desde html y los guarda en una lista
    links = []
    for anchor in soup.find_all("a", href=True): #recorremos todos los enlaces con href
        #limpia el texto visible del enlace
        label = " ".join(anchor.get_text(" ", strip=True).split())
        href_value = anchor.get("href")
        #limpia el href y lo convierte a url absoluta si es relativo 
        href = str(href_value).strip() if href_value is not None else "" #si no hay href, lo dejamos como cadena vacia
        if href.startswith("/"): #si el enlace empieza con /, lo convertimos a url absoluta usando la base
            from urllib.parse import urljoin
            href = urljoin(base_url, href)

        #solo guardamos enlaces que parecen descargas y que sean urls de verdad
        #(nos saltamos anclas tipo "#" y enlaces vacios que confundirian a load_remote_table)
        if href.startswith("#") or not href.startswith("http"):
            continue
        if href.lower().endswith((".csv", ".json", ".geojson")):
            links.append((label, href))

    def score(item):
        #prioriza csv, luego json/geojson
        url_lower = item[1].lower()
        value = 5
        if url_lower.endswith(".csv"):
            value = 0
        elif url_lower.endswith((".json", ".geojson")):
            value = 1
        return value

    #ordena para intentar primero los mejores formatos
    links.sort(key=score)
    return links

def find_downloads(page_url, html_text):
    #decide como extraer urls de descarga
    res = []
    if "<rdf:RDF" in html_text or "http://www.w3.org/ns/dcat#" in html_text: #miramos si encontramso indicios de que es un documento RDF
        #rdf suele traer urls directas a los ficheros
        res = re.findall(r'rdf:resource="([^"]+)"', html_text) #[^"]+ busca cualquier cosa que no sea comillas dentro del valor de rdf:resource
    else:
        #html normal con enlaces visibles
        soup = BeautifulSoup(html_text, "html.parser")
        pairs = find_download_links_html(soup, page_url)
        res = [u for _, u in pairs] #solo nos quedamos con las urls, no con las etiquetas
    return res

def load_remote_table(url):
    #descarga el fichero y lo convierte a tabla
    df_res = None
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        url_lower = url.lower()
        if url_lower.endswith(".csv"):
            df_res = pd.read_csv(io.StringIO(response.text), sep=None, engine="python", dtype=str) #sep=None con engine python permite detectar el separador automaticamente, dtype=str para evitar problemas con tipos mixtos
        elif url_lower.endswith((".json", ".geojson")):
            data = response.json()
            if isinstance(data, dict) and "features" in data:
                #geojson, normalizamos la lista de features
                df_res = pd.json_normalize(data["features"]) #lo convierte en tabla
            else:
                df_res = pd.json_normalize(data) #tmb en tabla
    except Exception as e:
        print(f"Error en carga: {e}")
    return df_res

def filter_by_window(df, start, end):
    #filtra filas dentro del rango de fechas
    df_out = df.iloc[0:0] #df vacio pero tiene las mismas columnas que el original
    if not df.empty:
        keys = ["fecha", "date", "timestamp", "f_accidente"]
        dt_col = next((c for c in df.columns if any(k in c.lower() for k in keys)), None) #busca la columna de fecha con las keys de antes, si no la deja vacia
        if dt_col:
            ts = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
            ts_series = pd.Series(ts)
            #pasa cada valor a fecha simple para comparar con date (pndas)
            date_series = ts_series.map(lambda x: x.date() if pd.notna(x) else pd.NaT)
            #parte 1: fechas iguales o despues del inicio
            from_start = date_series >= start
            #parte 2: fechas iguales o antes del final
            until_end = date_series <= end
            #solo nos quedamos con las que cumplen ambas condiciones
            mask = from_start & until_end
            df_out = df[mask]
    return df_out

#procesamiento principal

def process_one(page_url, start, end):
    #procesa una ficha y devuelve su dataframe unido
    dfs = []
    try:
        #primero se mira la pagina de datos
        page_html = fetch_text(page_url)
        #luego buscamos los enlaces de descarga
        download_urls = find_downloads(page_url, page_html)
        for download_url in download_urls:
            #intentamos cargar cada fichero
            df = load_remote_table(download_url)
            if df is not None:
                #solo nos quedamos con fechas dentro del rango
                df = filter_by_window(df, start, end)
                if not df.empty:
                    dfs.append(df)
    except Exception as e:
        print(f"Error procesando ficha: {e}")
    
    res = pd.DataFrame()
    if dfs:
        res = pd.concat(dfs, ignore_index=True, sort=False)
    return res

def build_contract(df_raw):
    #deja el dataframe con columnas normales
    res = pd.DataFrame(columns=FINAL_COLUMNS)
    if not df_raw.empty:
        #todo en COPIA, NO TOCAR EL ORIGINAL
        df = df_raw.copy()
        #normaliza nombres de columnas
        df.columns = [
            "".join(ch for ch in unicodedata.normalize("NFD", c.lower()) if unicodedata.category(ch) != "Mn")
            .replace(" ", "_")
            for c in df.columns
        ]

        #busca la columna de fecha 
        dt_col = next((c for c in df.columns if "fecha" in c), None)
        if dt_col is None and len(df.columns) > 0:
            #si no hay columna clara, usamos la primera
            dt_col = df.columns[0]
        if dt_col is not None:
            dt_parsed = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
        else:
            dt_parsed = pd.to_datetime(pd.Series([], dtype="object"), errors="coerce", dayfirst=True)
        dt_series = pd.Series(dt_parsed)
        
        #descompone fecha en dia, mes y año
        res["Dia"] = dt_series.dt.day
        res["Mes"] = dt_series.dt.month
        res["Año"] = dt_series.dt.year

        #extrae la hora desde la columna 'hora' si la trae el CSV (formato HH:MM:SS),
        #o desde el propio datetime como respaldo. si no se puede, ponemos 0.
        hora_col = next((c for c in df.columns if c == "hora"), None)
        if hora_col is not None:
            #cogemos los dos primeros caracteres del string de hora ("09:30:00" -> 9)
            res["Hora"] = pd.to_numeric(
                df[hora_col].astype(str).str[:2],
                errors="coerce",
            ).fillna(0).astype(int)
        else:
            res["Hora"] = dt_series.dt.hour.fillna(0).astype(int)

        #extrae codigo de distrito si existe
        dcode_col = next((c for c in df.columns if "cod" in c and "distrito" in c), None)
        if dcode_col:
            #convierte a texto por si hay numeros mezclados con letras
            district_raw = df[dcode_col].astype(str)
            #saca solo la parte numerica (primer grupo de digitos)
            district_digits = district_raw.str.extract(r"(\d+)")[0]
            #rellena vacios con NA (asi nos evitamos los nulos)
            res["district_code"] = district_digits.fillna("NA")
        else:
            res["district_code"] = "NA"
        #mapea codigo a nombre de distrito
        res["district_name"] = res["district_code"].apply(
            #rellena con ceros a la izquierda y busca en el diccionario
            lambda x: MADRID_DISTRICTS.get(str(x).zfill(2), "NA") #si no lo encuentra sera NA
        )
    return res

def main():
    start_date = date_cls(2022, 1, 1) #se pone que fechas queremos
    end_date = date_cls(2025, 10, 31)
    all_data = []

    for url in DATASET_PAGES:
        #procesa cada pagina y guarda lo que encuentre
        df = process_one(url, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    if all_data:
        #unimos todo lo descargado
        raw = pd.concat(all_data, ignore_index=True)
        standardized = build_contract(raw)
        
        #agrega por dia, hora y distrito (un accidente cuenta una vez en su hora)
        group_cols = ["Dia", "Mes", "Año", "Hora", "district_code", "district_name"]
        final_df = standardized.groupby(group_cols).size().reset_index(name='total_de_accidentes')
        final_df.to_csv(OUT_FILE, index=False, sep=";", encoding="utf-8-sig")
        print(f"Archivo generado: {OUT_FILE}")
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()