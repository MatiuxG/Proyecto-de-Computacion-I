import re
import io
import sys
import math
import csv
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
import xml.etree.ElementTree as ET

# ================================
# Configuración
# ================================

DATASET_PAGES = [
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=40085fb0e70b7410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
]

HARDCODED_DOWNLOADS: Dict[str, List[str]] = {}

OUTPUT_DIR = Path("Accidentes_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_accidentes.csv"

HEADERS = {
    "User-Agent": "MateoScraperBot/1.1 (+contact: your-email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,application/rdf+xml,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

# --- RANGO DE FECHAS (2022 a 2024) ---
DEFAULT_START = date_cls(2022, 1, 1)
DEFAULT_END = date_cls(2025, 10, 31)

# --- Columnas finales solicitadas ---
FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name", "total_de_accidentes"
]

MADRID_DISTRICTS = {
    "1": "Centro", "01": "Centro",
    "2": "Arganzuela", "02": "Arganzuela",
    "3": "Retiro", "03": "Retiro",
    "4": "Salamanca", "04": "Salamanca",
    "5": "Chamartín", "05": "Chamartín",
    "6": "Tetuán", "06": "Tetuán",
    "7": "Chamberí", "07": "Chamberí",
    "8": "Fuencarral-El Pardo", "08": "Fuencarral-El Pardo",
    "9": "Moncloa-Aravaca", "09": "Moncloa-Aravaca",
    "10": "Latina",
    "11": "Carabanchel",
    "12": "Usera",
    "13": "Puente de Vallecas",
    "14": "Moratalaz",
    "15": "Ciudad Lineal",
    "16": "Hortaleza",
    "17": "Villaverde",
    "18": "Villa de Vallecas",
    "19": "Vicálvaro",
    "20": "San Blas-Canillejas",
    "21": "Barajas",
}

# ================================
# Red y parsing
# ================================

def fetch(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r

def fetch_text(url: str) -> str:
    r = fetch(url)
    if not r.encoding or r.encoding.lower() in ("ascii", "utf-8"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def parse_rdf_for_downloads(xml_text: str) -> List[Tuple[str, str, str]]:
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dcat": "http://www.w3.org/ns/dcat#",
        "dct": "http://purl.org/dc/terms/",
    }
    out: List[Tuple[str, str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for dist in root.findall(".//dcat:Distribution", ns):
        title = (dist.findtext("dct:title", default="", namespaces=ns) or "").strip()
        media = (dist.findtext("dcat:mediaType", default="", namespaces=ns) or "").strip()
        access = ""
        a = dist.find("dcat:accessURL", ns)
        if a is not None:
            access = a.attrib.get(f"{{{ns['rdf']}}}resource", "").strip()
        if access:
            out.append((title, media, access))
    return out

def find_download_links_html(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
    links: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).split())
        href = a["href"].strip()
        if href.startswith("/"):
            from urllib.parse import urljoin
            href = urljoin(base_url, href)
        low = (label + " " + href).lower()
        if ("descarg" in low) or href.lower().endswith(JSON_EXT) or href.lower().endswith(CSV_EXT):
            links.append((label, href))
    def score(item):
        _, u = item
        u = u.lower()
        if u.endswith(".csv"): return 0
        if u.endswith(".json") or u.endswith(".geojson"): return 1
        return 5
    links.sort(key=score)
    return links

def find_downloads(page_url: str, html_text: str) -> List[str]:
    for key, urls in HARDCODED_DOWNLOADS.items():
        if key in page_url:
            return urls[:]
    if "<rdf:RDF" in html_text or "http://www.w3.org/ns/dcat#" in html_text:
        dists = parse_rdf_for_downloads(html_text)
        dists.sort(key=lambda t: 0 if "csv" in t[1].lower() or t[2].lower().endswith(".csv") else 1)
        return [d[2] for d in dists if d[2]]
    soup = make_soup(html_text)
    pairs = find_download_links_html(soup, page_url)
    return [u for _, u in pairs]

def load_remote_table(url: str) -> Optional[pd.DataFrame]:
    print(f"  [Descarga] {url}")
    r = fetch(url)
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()

    if u.endswith(".csv") or "csv" in ctype:
        try:
            df = pd.read_csv(io.StringIO(r.text), sep=None, engine="python", dtype=str, low_memory=False)
        except Exception:
            df = None
            for sep in (";", ",", "\t", "|"):
                try:
                    df = pd.read_csv(io.StringIO(r.text), sep=sep, dtype=str, low_memory=False)
                    break
                except Exception:
                    pass
            if df is None:
                raise
        print(f"    [OK CSV] shape={df.shape}")
        return df

    if u.endswith(".json") or u.endswith(".geojson") or "json" in ctype:
        data = r.json()
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            df = pd.json_normalize(data["features"])
        else:
            df = pd.json_normalize(data)
        df = df.astype(str)
        print(f"    [OK JSON] shape={df.shape}")
        return df

    print("    [Aviso] Tipo no soportado (se espera CSV/JSON).")
    return None

def nfd_lower(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        base = nfd_lower(str(c))
        base = re.sub(r"\s+", "_", base)
        mapping[c] = base
    return df.rename(columns=mapping)

def guess_datetime_cols(df: pd.DataFrame) -> List[str]:
    keys = ["fecha_hora","fechahora","datetime","timestamp","fecha","date","f_suceso","f_accidente"]
    return [c for c in df.columns if any(k in c for k in keys)]

def guess_street_cols(df: pd.DataFrame) -> List[str]:
    keys = ["calle","via","vía","direccion","dirección","ubicacion","ubicación","lugar","punto","domicilio",
            "carretera","tramo","pk","interseccion","intersección","cruce","kilometro","kilómetro","street","road","address","addr",
            "lugar_accidente", "localizacion"]
    return [c for c in df.columns if any(k in c for k in keys)]

def guess_district_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    code_keys = ["codigo_distrito","cod_distrito","cod.distrito","coddistrito","c_distrito","district_code","codigo__distrito"]
    name_keys = ["distrito","distrito_nombre","nombre_distrito","district","district_name"]
    code = next((c for c in df.columns if any(k in c for k in code_keys)), None)
    name = next((c for c in df.columns if any(k in c for k in name_keys)), None)
    return code, name

def filter_by_window(df: pd.DataFrame, start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    if df.empty:
        return df
    mask = None
    for c in guess_datetime_cols(df):
        try:
            ts = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            m = (ts.dt.date >= start_date) & (ts.dt.date <= end_date)
            mask = m if mask is None else (mask | m)
        except Exception:
            continue
    if mask is None:
        return df.iloc[0:0]
    return df[mask]

def process_one(page_url: str, start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    print(f"\n[Ficha] {page_url}")
    try:
        html = fetch_text(page_url)
    except Exception as e:
        print(f"  [Error al abrir ficha] {e}")
        return pd.DataFrame()

    download_urls = find_downloads(page_url, html)
    if not download_urls:
        print("  [Aviso] No se hallaron URLs de descarga.")
        return pd.DataFrame()

    # --- MODIFICADO: Acumular TODOS los dataframes válidos de la página ---
    dfs = [] 
    for dl in download_urls:
        if not (dl.lower().endswith(CSV_EXT) or dl.lower().endswith(JSON_EXT)):
            continue
        try:
            df = load_remote_table(dl)
        except Exception as e:
            print(f"  [Error descarga] {e}")
            continue
        if df is None or df.empty:
            continue

        df = normalize_cols(df)
        df = filter_by_window(df, start_date, end_date)
        if df.empty:
            print(f"    [Info] Datos descargados fuera de ventana temporal (ignorado): {dl}")
            continue

        df["__source_page__"] = page_url
        df["__download__"] = dl
        print(f"  [Filtrado] {df.shape} filas en ventana.")
        dfs.append(df)

    if not dfs:
        print("  [Info] Ninguna descarga produjo filas en la ventana.")
        return pd.DataFrame()
    
    # Combinar todos los archivos encontrados (2022, 2023, 2024...)
    return pd.concat(dfs, ignore_index=True, sort=False)

def build_contract_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df = df_raw.copy()
    out = pd.DataFrame()

    # --- Fechas ---
    dt_cols = guess_datetime_cols(df)
    if dt_cols:
        full = None
        for c in dt_cols:
            if any(k in c for k in ["fecha_hora","fechahora","datetime","timestamp"]):
                full = c; break
        if full is None:
            full = dt_cols[0]

        dt_parsed = pd.to_datetime(df[full], errors="coerce", dayfirst=True)
        out["Dia"] = dt_parsed.dt.day
        out["Mes"] = dt_parsed.dt.month
        out["Año"] = dt_parsed.dt.year
    else:
        dcol = next((c for c in df.columns if c in ("fecha","date","dia","día")), None)
        if dcol:
            dt_parsed = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)
            out["Dia"] = dt_parsed.dt.day
            out["Mes"] = dt_parsed.dt.month
            out["Año"] = dt_parsed.dt.year
        else:
            out["Dia"] = pd.Series(dtype='Int64')
            out["Mes"] = pd.Series(dtype='Int64')
            out["Año"] = pd.Series(dtype='Int64')

    # --- Distritos: CORREGIDO ---
    dcode, dname = guess_district_cols(df)
    
    # Prioridad: Obtener el código limpio.
    # Si tenemos columna de código, la usamos. Si no, usamos la de nombre asumiendo que contiene el código (tu caso).
    raw_code = None
    if dcode:
        raw_code = df[dcode]
    elif dname:
        raw_code = df[dname]
    
    if raw_code is not None:
        # Extraer dígitos para limpiar el código
        out["district_code"] = raw_code.astype(str).str.extract(r"(\d+)")[0].fillna("NA")
    else:
        out["district_code"] = "NA"

    # Mapeo FORZADO: Usar el diccionario basándose estrictamente en el código
    def get_district_name(code):
        if pd.isna(code) or code == "NA":
            return "NA"
        c = str(code).strip()
        # Intentar match exacto (ej: "1") o con padding (ej: "01")
        return MADRID_DISTRICTS.get(c, MADRID_DISTRICTS.get(c.zfill(2), "NA"))

    out["district_name"] = out["district_code"].apply(get_district_name)
    
    return out

def main():
    start_date = DEFAULT_START
    end_date = DEFAULT_END
    print(f"[Ventana] {start_date.isoformat()} -> {end_date.isoformat()}")

    parts = []
    for url in DATASET_PAGES:
        try:
            df = process_one(url, start_date, end_date)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  [Error inesperado] {e}")

    if not parts:
        print("\n[Resultado] No se encontraron filas en la ventana seleccionada.")
        empty = pd.DataFrame(columns=FINAL_COLUMNS)
        empty.to_csv(
            OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        print(f"[OK] Datasheet vacío escrito: {OUT_FILE.resolve()}")
        return

    raw = pd.concat(parts, ignore_index=True, sort=False)
    standardized = build_contract_from_raw(raw)
    standardized = standardized.fillna("NA")
    standardized = standardized.replace(r"^\s*$", "NA", regex=True)
    
    # === Agregación de datos ===
    group_cols = ["Dia", "Mes", "Año", "district_code", "district_name"]
    valid_cols = [c for c in group_cols if c in standardized.columns]
    
    if valid_cols:
        standardized = standardized.groupby(valid_cols).size().reset_index(name='total_de_accidentes')
    else:
        standardized['total_de_accidentes'] = 1

    for col in FINAL_COLUMNS:
        if col not in standardized.columns:
            standardized[col] = "NA"
    
    standardized = standardized[FINAL_COLUMNS]

    standardized.to_csv(
        OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )

    print(f"\n[OK] Datasheet escrito: {OUT_FILE.resolve()}")
    print(f"[Filas Agregadas] {len(standardized)}")
    print("\n[Preview]")
    print(standardized.head(10))

if __name__ == "__main__":
    main()