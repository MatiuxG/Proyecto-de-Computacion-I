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
from dateutil.relativedelta import relativedelta
import xml.etree.ElementTree as ET

# ================================
# Configuración
# ================================

# --- MODIFICADO ---
# Se eliminan las fuentes de la Comunidad de Madrid para centrarnos solo
# en el Ayuntamiento, que tiene el detalle de 'lugar_accidente' (calle).
DATASET_PAGES = [
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=40085fb0e70b7410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
]

# --- MODIFICADO ---
# Se eliminan los overrides de la Comunidad de Madrid.
HARDCODED_DOWNLOADS: Dict[str, List[str]] = {}

# Carpeta de salida (cada script debe escribir su propio datasheet aquí)
OUTPUT_DIR = Path("Accidentes_Scripts\Accidentes_Scripts\Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_accidentes.csv"

# Cabeceras HTTP para scraping
HEADERS = {
    "User-Agent": "MateoScraperBot/1.1 (+contact: your-email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,application/rdf+xml,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

# Ventana por defecto: hoy y 2 meses atrás (Plantilla)
TODAY = datetime.today().date()
DEFAULT_START = TODAY - relativedelta(months=2)
DEFAULT_END = TODAY

# Columnas finales deseadas
FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name", "ubicacion"
]

# Mapeo códigos → nombres de distritos del Ayuntamiento de Madrid
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
    """Descarga una URL con headers estándar y timeout."""
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r

def fetch_text(url: str) -> str:
    """Descarga una página y devuelve texto con codificación detectada."""
    r = fetch(url)
    if not r.encoding or r.encoding.lower() in ("ascii", "utf-8"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def make_soup(html: str) -> BeautifulSoup:
    """Crea un BeautifulSoup robusto (lxml si está disponible)."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def parse_rdf_for_downloads(xml_text: str) -> List[Tuple[str, str, str]]:
    """Extrae distribuciones DCAT de un RDF y devuelve (title, mediaType, accessURL)."""
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
    """Encuentra enlaces de descarga (CSV/JSON) en HTML y los prioriza por CSV."""
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
    """Dada la ficha, devuelve una lista de URLs de datos (CSV/JSON), con overrides."""
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
    """Descarga y normaliza una tabla remota (CSV o JSON/GeoJSON) como DataFrame de strings."""
    print(f"  [Descarga] {url}")
    r = fetch(url)
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()

    # CSV: autodetección de separador; fallback
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

    # JSON / GeoJSON: normalizar estructura a tabla plana
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

# ================================
# Normalización de columnas y campos
# ================================

def nfd_lower(s: str) -> str:
    """Minúsculas + sin acentos."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columna: minúsculas, sin acentos, espacios->guiones bajos."""
    mapping = {}
    for c in df.columns:
        base = nfd_lower(str(c))
        base = re.sub(r"\s+", "_", base)
        mapping[c] = base
    return df.rename(columns=mapping)

def guess_datetime_cols(df: pd.DataFrame) -> List[str]:
    """Detecta columnas candidatas de fecha/hora por nombre."""
    keys = ["fecha_hora","fechahora","datetime","timestamp","fecha","date","f_suceso","f_accidente"]
    return [c for c in df.columns if any(k in c for k in keys)]

def guess_street_cols(df: pd.DataFrame) -> List[str]:
    """Detecta columnas relacionadas con vía/dirección/ubicación."""
    # --- MODIFICADO: Añadido 'localizacion' ---
    keys = ["calle","via","vía","direccion","dirección","ubicacion","ubicación","lugar","punto","domicilio",
            "carretera","tramo","pk","interseccion","intersección","cruce","kilometro","kilómetro","street","road","address","addr",
            "lugar_accidente", "localizacion"] # <--- AÑADIDO
    return [c for c in df.columns if any(k in c for k in keys)]

def guess_district_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detecta posibles columnas de código/nombre de distrito."""
    code_keys = ["codigo_distrito","cod_distrito","cod.distrito","coddistrito","c_distrito","district_code","codigo__distrito"]
    name_keys = ["distrito","distrito_nombre","nombre_distrito","district","district_name"]
    code = next((c for c in df.columns if any(k in c for k in code_keys)), None)
    name = next((c for c in df.columns if any(k in c for k in name_keys)), None)
    return code, name

def to_hour_minute(value) -> Optional[Tuple[int, int]]:
    """Convierte representaciones comunes de hora a (HH,MM); acepta decimales tipo 8.5 -> 08:30."""
    if pd.isna(value): return None
    if isinstance(value, pd.Timestamp): return value.hour, value.minute
    sv = str(value).strip().replace(",", ".")
    if re.match(r"^\d+(\.\d+)?$", sv):
        f = float(sv)
        hh = int(math.floor(f))
        mm = int(round((f - hh) * 60))
        return hh, (0 if mm == 60 else mm)
    m = re.match(r"^\s*(\d{1,2})\s*[:hH]?\s*(\d{1,2})?\s*$", sv.replace(".", ":"))
    if m:
        return int(m.group(1)), int(m.group(2) or 0)
    try:
        dt = dtparser.parse(sv, dayfirst=True, fuzzy=True)
        return dt.hour, dt.minute
    except Exception:
        return None

def as_date(s: Any) -> str:
    """Devuelve fecha ISO YYYY-MM-DD o '' si no válida (luego se rellena a 'NA')."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    try:
        dt = pd.to_datetime(str(s), errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def as_time(s: Any) -> str:
    """Devuelve hora HH:MM o '' si no válida (luego se rellena a 'NA')."""
    t = to_hour_minute(s)
    if not t:
        return ""
    hh, mm = t
    return f"{hh:02d}:{mm:02d}"

# ================================
# Filtrado por ventana temporal y procesamiento
# ================================

def filter_by_window(df: pd.DataFrame, start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    """Filtra filas cuya(s) columna(s) de fecha/hora caigan en [start_date, end_date]."""
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
    """Procesa una ficha: encuentra descargas, carga la primera que tenga filas en la ventana."""
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
            continue

        # Proveniencia para trazabilidad
        df["__source_page__"] = page_url
        df["__download__"] = dl
        print(f"  [Filtrado] {df.shape} filas")
        return df

    print("  [Info] Ninguna descarga produjo filas en la ventana.")
    return pd.DataFrame()

# ================================
# Mapeo al contrato común (accidentes)
# ================================

def build_contract_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Mapea el DataFrame bruto a las columnas finales solicitadas."""
    if df_raw.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df = df_raw.copy()

    # 1) Inicializa DataFrame de salida
    out = pd.DataFrame()

    # 2) Fecha/hora -> Dia, Mes, Año
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
        # Si no hay datetime, intentar parsear 'date'
        if dcol:
            dt_parsed = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)
            out["Dia"] = dt_parsed.dt.day
            out["Mes"] = dt_parsed.dt.month
            out["Año"] = dt_parsed.dt.year
        else:
            out["Dia"] = pd.Series(dtype='Int64')
            out["Mes"] = pd.Series(dtype='Int64')
            out["Año"] = pd.Series(dtype='Int64')


    # 3) Distrito (código/nombre) con mapeo a nombres
    dcode, dname = guess_district_cols(df)

    # Extrae serie de código (si existe explícita o si "nombre" es numérico)
    code_series = None
    if dcode:
        code_series = df[dcode].astype(str).str.extract(r"(\d+)")[0]
    elif dname and df[dname].astype(str).str.match(r"^\s*\d+\s*$").all():
        code_series = df[dname].astype(str).str.extract(r"(\d+)")[0]

    # Asignar district_code
    out["district_code"] = (
        code_series.fillna("") if code_series is not None
        else (df.get(dcode, "").astype(str) if dcode else "")
    )

    # Asignar district_name (si no hay nombre real, mapear por código)
    if dname and not df[dname].astype(str).str.match(r"^\s*\d+\s*$").all():
        out["district_name"] = df[dname].astype(str)
    else:
        if code_series is not None:
            out["district_name"] = code_series.map(
                lambda x: MADRID_DISTRICTS.get(x, MADRID_DISTRICTS.get(x.zfill(2), "NA"))
            )
        else:
            out["district_name"] = "NA"

    # 4) Ubicación (vía/calle)
    
    # --- MODIFICADO: Añadido 'localizacion' a la prioridad ---
    # Lista de columnas candidatas en orden de prioridad (de más específica a más genérica)
    priority_keys = [
        "lugar_accidente", # Suele ser la descripción completa (Ayto. Madrid)
        "calle",           
        "via",             
        "vía",
        "direccion",
        "dirección",
        "localizacion", # <--- AÑADIDO
        "interseccion",
        "intersección",
        "emplazamiento",
        "ubicacion",
        "ubicación",
        "lugar",
        "domicilio"
    ]

    street_col_found = None
    
    # Iterar por prioridad
    for key in priority_keys:
        # df.columns ya están normalizados (minúsculas, sin acentos)
        for col_name in df.columns:
            if key in col_name:
                # Evitar columnas que solo describen el 'tipo' o 'código'
                if "tipo" not in col_name and "cod" not in col_name and "codigo" not in col_name:
                     street_col_found = col_name
                     break # Encontramos la mejor coincidencia para esta clave
        if street_col_found:
            break # Salir del bucle de prioridad

    # Si después de todo no encontramos una columna prioritaria,
    # usar la lógica original (la primera que encuentre 'guess_street_cols')
    if not street_col_found:
        street_cols_generic = guess_street_cols(df)
        if street_cols_generic:
            street_col_found = next((c for c in street_cols_generic if "tipo" not in c and "cod" not in c and "codigo" not in c), None)
            if not street_col_found and street_cols_generic:
                 street_col_found = street_cols_generic[0]
    
    out["ubicacion"] = df.get(street_col_found, "").astype(str) if street_col_found else ""


    # 5) Severidad
    sev_candidates = ["gravedad","lesividad","severidad","resultado","resultado_accidente","severity"]
    sev_col = next((c for c in df.columns if c in sev_candidates), None)
    out["severidad"] = df.get(sev_col, "") if sev_col else ""

    # 6) ID origen
    id_candidates = ["id","codigo","cod_accidente","id_accidente","expediente","num_exp","codigo_accidente",
                     "num_expediente"]
    id_col = next((c for c in df.columns if c in id_candidates), None)
    out["ID_origen"] = df.get(id_col, "") if id_col else df.get("__download__", "")

    # 7) Extras de accidentes (opcionales)
    cause_candidates = ["causa","causa_accidente","motivo","concurrencia","tipo_accidente","descripcion"]
    nv_candidates = ["n_vehiculos","num_vehiculos","nvehiculos","vehiculos_implicados","vehiculos",
                     "no_vehiculos_implicados"]
    ni_candidates = ["n_heridos","num_heridos","heridos","lesionados","n_lesionados",
                     "no_victimas"]
    nf_candidates = ["n_fallecidos","num_fallecidos","fallecidos","muertos","n_muertos"]

    def pick_first(cols: List[str]) -> Optional[str]:
        return next((c for c in df.columns if c in cols), None)

    cause_col = pick_first(cause_candidates)
    nv = pick_first(nv_candidates)
    ni = pick_first(ni_candidates)
    nf = pick_first(nf_candidates)

    out["causa"] = df.get(cause_col, "") if cause_col else ""
    out["num_vehiculos"] = df.get(nv, "") if nv else ""
    out["num_heridos"] = df.get(ni, "") if ni else ""
    out["num_fallecidos"] = df.get(nf, "") if nf else ""

    # 8) Proveniencia (útil para depurar; no es parte del contrato pero lo mantenemos)
    if "__source_page__" in df.columns:
        out["__source_page__"] = df["__source_page__"]
    if "__download__" in df.columns:
        out["__download__"] = df["__download__"]

    return out

# ================================
# Main
# ================================

def main():
    # Ventana temporal (Plantilla: hoy-2m -> hoy)
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
        # CSV vacío con SOLO las columnas finales
        empty = pd.DataFrame(columns=FINAL_COLUMNS)
        empty.to_csv(
            OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        print(f"[OK] Datasheet vacío escrito: {OUT_FILE.resolve()}")
        return

    raw = pd.concat(parts, ignore_index=True, sort=False)
    standardized = build_contract_from_raw(raw)

    # === Relleno NA y selección final de columnas ===
    standardized = standardized.fillna("NA")
    standardized = standardized.replace(r"^\s*$", "NA", regex=True)
    
    # Asegurar que todas las columnas finales existan
    for col in FINAL_COLUMNS:
        if col not in standardized.columns:
            standardized[col] = "NA"
    
    # Recortar exactamente a las columnas solicitadas
    standardized = standardized[FINAL_COLUMNS]

    # Guardado final (utf-8-sig recomendado para Excel)
    standardized.to_csv(
        OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )

    print(f"\n[OK] Datasheet escrito: {OUT_FILE.resolve()}")
    print(f"[Filas] {len(standardized)}")
    print("\n[Preview]")
    print(standardized.head(10))

if __name__ == "__main__":
    main()