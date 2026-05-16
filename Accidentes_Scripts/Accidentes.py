import io
import re
import unicodedata
from datetime import date as date_cls
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


#vista /downloads del portal CKAN; la antigua no expone enlaces directos
DATASET_PAGES = [
    "https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/downloads",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_accidentes.csv"

#user agent para que no nos bloqueen
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


def fetch_text(url):
    #baja el html resolviendo la codificacion
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() in ("ascii", "utf-8"):
        response.encoding = response.apparent_encoding or "utf-8"
    res = response.text
    return res


def find_download_links_html(soup, base_url):
    #saca los enlaces a csv/json/geojson ordenados por preferencia (csv primero)
    links = []
    for anchor in soup.find_all("a", href=True):
        label = " ".join(anchor.get_text(" ", strip=True).split())
        href_value = anchor.get("href")
        href = str(href_value).strip() if href_value is not None else ""
        if href.startswith("/"):
            href = urljoin(base_url, href)

        #descartamos anclas y rutas no descargables
        enlace_valido = href.startswith("http") and not href.startswith("#")
        if enlace_valido and href.lower().endswith((".csv", ".json", ".geojson")):
            links.append((label, href))

    def score(item):
        url_lower = item[1].lower()
        valor = 5
        if url_lower.endswith(".csv"):
            valor = 0
        elif url_lower.endswith((".json", ".geojson")):
            valor = 1
        return valor

    links.sort(key=score)
    return links


def find_downloads(page_url, html_text):
    #saca los enlaces segun si la pagina es RDF o HTML
    res = []
    if "<rdf:RDF" in html_text or "http://www.w3.org/ns/dcat#" in html_text:
        #en RDF las urls van dentro de rdf:resource
        res = re.findall(r'rdf:resource="([^"]+)"', html_text)
    else:
        soup = BeautifulSoup(html_text, "html.parser")
        pairs = find_download_links_html(soup, page_url)
        res = [u for _, u in pairs]
    return res


def load_remote_table(url):
    #descarga csv/json/geojson y lo pasa a dataframe
    df_res = None
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        url_lower = url.lower()
        if url_lower.endswith(".csv"):
            #sep=None deja que pandas detecte el separador
            df_res = pd.read_csv(io.StringIO(response.text), sep=None, engine="python", dtype=str)
        elif url_lower.endswith((".json", ".geojson")):
            data = response.json()
            if isinstance(data, dict) and "features" in data:
                df_res = pd.json_normalize(data["features"])
            else:
                df_res = pd.json_normalize(data)
    except Exception as e:
        print(f"Error en carga: {e}")
    return df_res


def filter_by_window(df, start, end):
    #se queda con las filas dentro del rango de fechas
    df_out = df.iloc[0:0]
    if not df.empty:
        keys = ["fecha", "date", "timestamp", "f_accidente"]
        dt_col = next((c for c in df.columns if any(k in c.lower() for k in keys)), None)
        if dt_col:
            ts = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
            ts_series = pd.Series(ts)
            date_series = ts_series.map(lambda x: x.date() if pd.notna(x) else pd.NaT)
            mask = (date_series >= start) & (date_series <= end)
            df_out = df[mask]
    return df_out


def process_one(page_url, start, end):
    #procesa una pagina-dataset entera y devuelve un unico dataframe
    dfs = []
    try:
        page_html = fetch_text(page_url)
        download_urls = find_downloads(page_url, page_html)
        for download_url in download_urls:
            df = load_remote_table(download_url)
            if df is not None:
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
    #normaliza columnas y deja dia/mes/año/hora/distrito, una fila por accidente
    res = pd.DataFrame(columns=FINAL_COLUMNS)
    if not df_raw.empty:
        df = df_raw.copy()
        #minusculas, sin tildes, espacios a guion bajo
        df.columns = [
            "".join(ch for ch in unicodedata.normalize("NFD", c.lower()) if unicodedata.category(ch) != "Mn")
            .replace(" ", "_")
            for c in df.columns
        ]

        #localizamos la columna de fecha; si no hay, tiramos de la primera
        dt_col = next((c for c in df.columns if "fecha" in c), None)
        if dt_col is None and len(df.columns) > 0:
            dt_col = df.columns[0]

        if dt_col is not None:
            dt_parsed = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
        else:
            dt_parsed = pd.to_datetime(pd.Series([], dtype="object"), errors="coerce", dayfirst=True)
        dt_series = pd.Series(dt_parsed)

        res["Dia"] = dt_series.dt.day
        res["Mes"] = dt_series.dt.month
        res["Año"] = dt_series.dt.year

        #si hay columna 'hora' cogemos los 2 primeros digitos; si no, del datetime
        hora_col = next((c for c in df.columns if c == "hora"), None)
        if hora_col is not None:
            res["Hora"] = pd.to_numeric(
                df[hora_col].astype(str).str[:2],
                errors="coerce",
            ).fillna(0).astype(int)
        else:
            res["Hora"] = dt_series.dt.hour.fillna(0).astype(int)

        dcode_col = next((c for c in df.columns if "cod" in c and "distrito" in c), None)
        if dcode_col:
            district_raw = df[dcode_col].astype(str)
            district_digits = district_raw.str.extract(r"(\d+)")[0]
            res["district_code"] = district_digits.fillna("NA")
        else:
            res["district_code"] = "NA"

        #zfill por si llega "1" en vez de "01"
        res["district_name"] = res["district_code"].apply(
            lambda x: MADRID_DISTRICTS.get(str(x).zfill(2), "NA")
        )
    return res


def main():
    #scrapea, unifica columnas, agrupa por dia/hora/distrito y escribe el csv
    start_date = date_cls(2022, 1, 1)
    end_date = date_cls(2025, 10, 31)
    all_data = []

    for url in DATASET_PAGES:
        df = process_one(url, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    if all_data:
        raw = pd.concat(all_data, ignore_index=True)
        standardized = build_contract(raw)

        group_cols = ["Dia", "Mes", "Año", "Hora", "district_code", "district_name"]
        final_df = standardized.groupby(group_cols).size().reset_index(name="total_de_accidentes")
        final_df.to_csv(OUT_FILE, index=False, sep=";", encoding="utf-8-sig")
        print(f"Archivo generado: {OUT_FILE}")
    else:
        print("No se encontraron datos.")


if __name__ == "__main__":
    main()
