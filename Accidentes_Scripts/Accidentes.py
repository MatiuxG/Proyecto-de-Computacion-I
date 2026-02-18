import re
import io
import unicodedata
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date as date_cls
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
DATASET_PAGES = [
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=40085fb0e70b7410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
]

OUTPUT_DIR = Path("Accidentes_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_accidentes.csv"

HEADERS = {"User-Agent": "MateoScraperBot/1.1"}
FINAL_COLUMNS = ["Dia", "Mes", "Año", "district_code", "district_name", "total_de_accidentes"]

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

# --- FUNCIONES DE SOPORTE ---

def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("ascii", "utf-8"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def find_download_links_html(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
    links = []
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).split())
        href = a["href"].strip()
        if href.startswith("/"):
            from urllib.parse import urljoin
            href = urljoin(base_url, href)
        
        low = (label + " " + href).lower()
        if ("descarg" in low) or href.lower().endswith((".csv", ".json", ".geojson")):
            links.append((label, href))
            
    def score(item):
        u = item[1].lower()
        val = 5
        if u.endswith(".csv"): val = 0
        elif u.endswith((".json", ".geojson")): val = 1
        return val
        
    links.sort(key=score)
    return links

def find_downloads(page_url: str, html_text: str) -> List[str]:
    res = []
    if "<rdf:RDF" in html_text or "http://www.w3.org/ns/dcat#" in html_text:
        # Simplificación de RDF
        res = [re.findall(r'rdf:resource="([^"]+)"', html_text)] # Ejemplo simplificado para flujo único
    else:
        soup = BeautifulSoup(html_text, "html.parser")
        pairs = find_download_links_html(soup, page_url)
        res = [u for _, u in pairs]
    return res

def load_remote_table(url: str) -> Optional[pd.DataFrame]:
    df_res = None
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        u = url.lower()
        if u.endswith(".csv"):
            df_res = pd.read_csv(io.StringIO(r.text), sep=None, engine="python", dtype=str)
        elif u.endswith((".json", ".geojson")):
            data = r.json()
            if isinstance(data, dict) and "features" in data:
                df_res = pd.json_normalize(data["features"])
            else:
                df_res = pd.json_normalize(data)
    except Exception as e:
        print(f"Error en carga: {e}")
    return df_res

def filter_by_window(df: pd.DataFrame, start: date_cls, end: date_cls) -> pd.DataFrame:
    df_out = df.iloc[0:0]
    if not df.empty:
        keys = ["fecha", "date", "timestamp", "f_accidente"]
        dt_col = next((c for c in df.columns if any(k in c.lower() for k in keys)), None)
        if dt_col:
            ts = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
            mask = (ts.dt.date >= start) & (ts.dt.date <= end)
            df_out = df[mask]
    return df_out

# --- PROCESAMIENTO ---

def process_one(page_url: str, start: date_cls, end: date_cls) -> pd.DataFrame:
    dfs = []
    try:
        html = fetch_text(page_url)
        urls = find_downloads(page_url, html)
        for dl in urls:
            df = load_remote_table(dl)
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

def build_contract(df_raw: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(columns=FINAL_COLUMNS)
    if not df_raw.empty:
        df = df_raw.copy()
        # Limpiar columnas
        df.columns = [unicodedata.normalize("NFD", c.lower()).replace(" ", "_") for c in df.columns]
        
        # Extraer Fecha
        dt_col = next((c for c in df.columns if "fecha" in c), df.columns[0])
        dt_parsed = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
        
        res["Dia"] = dt_parsed.dt.day
        res["Mes"] = dt_parsed.dt.month
        res["Año"] = dt_parsed.dt.year
        
        # Distrito
        dcode_col = next((c for c in df.columns if "cod" in c and "distrito" in c), None)
        if dcode_col:
            res["district_code"] = df[dcode_col].astype(str).str.extract(r"(\d+)")[0].fillna("NA")
        else:
            res["district_code"] = "NA"
            
        res["district_name"] = res["district_code"].apply(
            lambda x: MADRID_DISTRICTS.get(str(x).zfill(2), "NA")
        )
    return res

def main():
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
        
        # Agregación
        group_cols = ["Dia", "Mes", "Año", "district_code", "district_name"]
        final_df = standardized.groupby(group_cols).size().reset_index(name='total_de_accidentes')
        
        final_df.to_csv(OUT_FILE, index=False, sep=";", encoding="utf-8-sig")
        print(f"Archivo generado: {OUT_FILE}")
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()