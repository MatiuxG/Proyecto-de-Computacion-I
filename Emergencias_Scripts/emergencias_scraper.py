# -*- coding: utf-8 -*-
"""
Emergencias (Madrid) — Datasheet normalizado (contrato común)
- Lee varias fichas del portal (SAMUR, Bomberos, Policía…)
- Descarta agregados sin fecha/hora por evento
- Ventana temporal: HOY y 2 meses hacia atrás
- Exporta: ./Emergencias_Scripts/Resultados/datasheet_emergencias.csv
- Contrato: dataset,event_type,date,time,datetime,district_code,district_name,lat,lon,location,severity,value,units,source_id
"""

import re
import io
import csv
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta

# ----------- Fichas (puedes ampliar/ajustar) -----------
PAGES = [
    # Bomberos (suele ser agregado mensual → será omitido si no hay fecha/hora por evento)
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=fa677996afc6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    # SAMUR activaciones (evento)
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=50d7d35982d6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    # Policía / Emergencias (ejemplo)
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
]

# ----------- Salida -----------
OUTPUT_DIR = Path("./Emergencias_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_emergencias.csv"

# ----------- Red -----------
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "MateoScraperBot/1.7 (+contact: your-email@example.com)",
}
REQ_TIMEOUT = 60
CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

# Enlaces a ignorar
EXCLUDE_PATTERNS = [
    "wms", "wmts", "ogc", "service", "arcgis", "esri",
    ".zip", ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".pdf", ".rdf", ".xml", "sparql", "mailto:", "javascript:", "#"
]
# Pistas para priorizar enlaces de datos
PREFERRED_HINTS = ["download", "descarg", "csv", "json"]

# Overrides (si conoces URL de datos exacta por ficha)
OVERRIDES: Dict[str, str] = {
    # "50d7d35982d6f510": "https://datos.madrid.es/egobfiles/.../activaciones_samur_2025.csv",
}

# ----------- Ventana temporal -----------
TODAY = datetime.today().date()
MONTHS_BACK_DEFAULT = 2
START_DATE = TODAY - relativedelta(months=MONTHS_BACK_DEFAULT)
END_DATE = TODAY

# ----------- Contrato común -----------
CONTRACT_COLS = [
    "dataset","event_type","date","time","datetime",
    "district_code","district_name","lat","lon","location",
    "severity","value","units","source_id"
]

# ----------- Mapa distrito nombre ↔ código (Madrid) -----------
# Nota: nombres normalizados (sin acentos, minúsculas)
DISTRICT_NAME_TO_CODE = {
    "centro": "01", "arganzuela": "02", "retiro": "03", "salamanca": "04",
    "chamartin": "05", "tetuan": "06", "chamberi": "07", "fuencarral-el pardo": "08",
    "moncloa-aravaca": "09", "latina": "10", "carabanchel": "11", "usera": "12",
    "puente de vallecas": "13", "moratalaz": "14", "ciudad lineal": "15", "hortaleza": "16",
    "villaverde": "17", "villa de vallecas": "18", "vicalvaro": "19", "san blas-canillejas": "20",
    "barajas": "21"
}
CODE_TO_DISTRICT_NAME = {v: k.title() for k, v in DISTRICT_NAME_TO_CODE.items()}

# ============= Utilidades texto / normalización =============
def nfd_lower(s: str) -> str:
    """Minúsculas + sin acentos."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def normalize_cols_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas (minúsculas, sin acentos, espacios->guión bajo)."""
    mapping = {}
    for c in df.columns:
        base = nfd_lower(str(c))
        base = re.sub(r"\s+", "_", base)
        mapping[c] = base
    return df.rename(columns=mapping)

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("ascii", "utf-8"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def absolutize(base_url: str, href: str) -> str:
    from urllib.parse import urljoin
    href = (href or "").strip()
    if href.startswith("//"): return "https:" + href
    if href.startswith("/"):  return urljoin(base_url, href)
    return href

def head_or_get_headers(url: str) -> Optional[requests.Response]:
    try:
        r = requests.head(url, headers=HEADERS, timeout=REQ_TIMEOUT, allow_redirects=True)
        if r.status_code < 400 and (r.headers.get("Content-Type") or r.headers.get("Content-Disposition")):
            return r
        r2 = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT, stream=True, allow_redirects=True)
        return r2
    except Exception:
        return None

def content_says_data(resp: requests.Response, url: str) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    cd = (resp.headers.get("Content-Disposition") or "")
    ul = (url or "").lower()
    if "text/csv" in ct or "application/json" in ct or "application/geo+json" in ct:
        return True
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', cd, flags=re.I)
    if m:
        fname = m.group(1).strip().lower()
        if fname.endswith((".csv",".json",".geojson")):
            return True
    if ul.endswith((".csv",".json",".geojson")):
        return True
    return False

def find_valid_data_url_from_page(page_url: str, soup: BeautifulSoup) -> Optional[str]:
    for key, forced in OVERRIDES.items():
        if key in page_url:
            return forced

    candidates = []
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).split())
        href = absolutize(page_url, a["href"])
        low_href, low_label = href.lower(), label.lower()
        if not (low_href.startswith("http://") or low_href.startswith("https://") or low_href.startswith("//")):
            continue
        if any(pat in low_href for pat in EXCLUDE_PATTERNS):
            continue
        looks_data = low_href.endswith(CSV_EXT) or low_href.endswith(JSON_EXT) \
                     or any(h in (low_href + " " + low_label) for h in PREFERRED_HINTS)
        if not looks_data:
            continue
        score = 1000
        if low_href.endswith(".csv"): score -= 300
        elif low_href.endswith(".json") or low_href.endswith(".geojson"): score -= 250
        if "download" in low_href or "descarg" in low_href or "descarga" in low_label: score -= 200
        if not (low_href.endswith(".csv") or low_href.endswith(".json") or low_href.endswith(".geojson")):
            score += 100
        candidates.append((score, href))

    for _, u in sorted(candidates, key=lambda t: t[0]):
        resp = head_or_get_headers(u)
        if resp is None:
            continue
        final_url = str(resp.url)
        if content_says_data(resp, final_url):
            return final_url
    return None

def load_remote_table(url: str) -> Optional[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()

    # CSV
    if u.endswith(".csv") or "csv" in ctype:
        data = r.content
        for enc in ("utf-8", "latin-1", None):
            for sep in (";", ",", "\t", "|"):
                try:
                    if enc is None:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False)
                    else:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False, encoding=enc)
                    if df.shape[1] == 1 and sep != ",":
                        continue
                    return df
                except Exception:
                    continue
        # Autodetección textual
        try:
            txt = r.content.decode("utf-8", errors="ignore")
            return pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str, low_memory=False)
        except Exception:
            return None

    # JSON / GeoJSON
    if u.endswith(".json") or u.endswith(".geojson") or "json" in ctype:
        data = r.json()
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            return pd.json_normalize(data["features"])
        return pd.json_normalize(data)

    return None

# ============= Detección de granularidad =============
def is_event_level(df: pd.DataFrame) -> bool:
    """True si hay fecha/hora por evento. False si es agregado (solo AÑO/MES...)."""
    cols = [str(c).lower() for c in df.columns]
    has_time = any("hora" in c or "time" in c for c in cols)
    has_date = any(x in c for c in cols for x in ["fecha_hora","fechahora","datetime","timestamp","fecha","date"])
    has_month = any(c == "mes" or c == "month" for c in cols)
    has_year  = any(c in ["año","ano","anio","year"] for c in cols)
    if (has_month or has_year) and not (has_time or has_date):
        return False
    return has_time or has_date

# ============= Ventana de fechas =============
def in_date_window(d: date_cls, end_date: date_cls, months_back: int) -> bool:
    start = end_date - relativedelta(months=months_back)
    return start <= d <= end_date

def extract_event_datetime_cols(row: pd.Series, df_cols: List[str]) -> Tuple[str, str, str]:
    """Construye date, time, datetime tomando columnas posibles (robusto)."""
    # Compuesto
    for c in df_cols:
        lc = str(c).lower()
        if any(k in lc for k in ["fecha_hora","fechahora","datetime","timestamp"]):
            ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
            if pd.notna(ts):
                return ts.date().isoformat(), ts.strftime("%H:%M:%S"), ts.strftime("%Y-%m-%d %H:%M:%S")
    # Separadas
    fecha_col = next((c for c in df_cols if "fecha" in str(c).lower() or "date" in str(c).lower()), None)
    hora_col  = next((c for c in df_cols if "hora"  in str(c).lower() or "time" in str(c).lower()), None)
    if fecha_col:
        f = pd.to_datetime(row[fecha_col], errors="coerce", dayfirst=True)
        if pd.notna(f):
            if hora_col:
                raw = str(row[hora_col] or "").strip().replace(",", ".")
                # 8.5 → 08:30:00 ; 8:5 → 08:05:00
                hhmmss = ""
                if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                    ff = float(raw); hh = int(ff); mm = int(round((ff - hh)*60)); 
                    if mm == 60: hh += 1; mm = 0
                    hhmmss = f"{hh:02d}:{mm:02d}:00"
                elif ":" in raw or "h" in raw:
                    raw2 = raw.replace("h", ":")
                    try:
                        hh, mm = raw2.split(":", 1)
                        mm = ''.join(ch for ch in mm if ch.isdigit()) or "0"
                        hhmmss = f"{int(hh):02d}:{int(mm):02d}:00"
                    except Exception:
                        hhmmss = ""
                if hhmmss:
                    return f.date().isoformat(), hhmmss, f"{f.date().isoformat()} {hhmmss}"
            return f.date().isoformat(), "", f.strftime("%Y-%m-%d")
    return "", "", ""

# ============= Geoparsing sencillo =============
def split_geopoint(val: str) -> Tuple[str, str]:
    """Acepta 'lat,lon', 'lon,lat', 'POINT (lon lat)' → devuelve (lat, lon) si se puede."""
    s = str(val or "").strip()
    if not s:
        return "", ""
    # POINT (lon lat)
    m = re.match(r"point\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)", s, flags=re.I)
    if m:
        lon, lat = m.group(1), m.group(2)
        return lat, lon
    # lat,lon o lon,lat (heurística: lat ~ 40.x en Madrid)
    if "," in s:
        a, b = s.split(",", 1)
        a, b = a.strip(), b.strip()
        try:
            fa, fb = float(a), float(b)
            if 39 <= fa <= 41:   # parece lat
                return a, b
            if 39 <= fb <= 41:   # b parece lat → swap
                return b, a
        except Exception:
            pass
    return "", ""

# ============= Mapeo → contrato común =============
def build_contract(df: pd.DataFrame, source_url: str) -> pd.DataFrame:
    """Mapea columnas de emergencias a contrato común con normalización y mapeo de distrito."""
    if df is None or df.empty:
        return pd.DataFrame(columns=CONTRACT_COLS)

    # Normaliza nombres de columnas
    df = normalize_cols_df(df)
    cols_low = {str(c).lower(): c for c in df.columns}

    # Candidatos semánticos
    type_keys = ["tipo","incidente","servicio","categoria","subcategoria","tipo_servicio","tiposervicio"]
    dcode_keys = ["cod_distrito","codigo_distrito","distrito_codigo","cdistrito","coddistrito","district_code"]
    dname_keys = ["distrito","nombre_distrito","distrito_nombre","district","district_name"]
    lat_keys = ["lat","latitud"]
    lon_keys = ["lon","longitud","lng","long"]
    geopoint_keys = ["geo_point_2d","geopoint","coordenadas","coordinates","geom","geometry"]
    location_keys = [
        "direccion","dirección","ubicacion","ubicación","lugar","punto","domicilio",
        "via_publica","via","vía","carretera","tramo","cruce","localizacion","localización",
        "street","road","address"
    ]
    severity_keys = ["prioridad","gravedad","nivel","rating","severidad"]

    def pick(keys):
        for k in keys:
            if k in cols_low:
                return cols_low[k]
        return None

    type_col = pick(type_keys)
    dcode_col = pick(dcode_keys)
    dname_col = pick(dname_keys)
    lat_col = pick(lat_keys)
    lon_col = pick(lon_keys)
    geop_col = pick(geopoint_keys)

    out_rows = []
    df_cols = list(df.columns)
    for _, row in df.iterrows():
        # Fecha/hora
        date_str, time_str, dt_str = extract_event_datetime_cols(row, df_cols)

        # Lat/Lon (usa geopoint si existe)
        lat_val, lon_val = "", ""
        if geop_col:
            l1, l2 = split_geopoint(row.get(geop_col, ""))
            lat_val, lon_val = l1, l2
        if not lat_val and lat_col:
            lat_val = str(row.get(lat_col, "")).strip()
        if not lon_val and lon_col:
            lon_val = str(row.get(lon_col, "")).strip()

        # Tipo de evento
        ev_type = str(row.get(type_col, "")).strip() if type_col else "emergencia"

        # Distrito (intenta completar nombre↔código)
        dcode = str(row.get(dcode_col, "")).strip() if dcode_col else ""
        dname = str(row.get(dname_col, "")).strip() if dname_col else ""

        dname_norm = nfd_lower(dname)
        if not dcode and dname_norm:
            dcode = DISTRICT_NAME_TO_CODE.get(dname_norm, "")
        if not dname and dcode:
            dname = CODE_TO_DISTRICT_NAME.get(dcode.zfill(2), dname)

        if dcode:
            dcode = re.sub(r"\D", "", dcode).zfill(2)

        # Location
        location_val = ""
        for lk in location_keys:
            c = cols_low.get(lk)
            if c:
                v = str(row.get(c, "")).strip()
                if v:
                    location_val = v
                    break

        # Severity
        sev_val = ""
        for sk in severity_keys:
            c = cols_low.get(sk)
            if c:
                v = str(row.get(c, "")).strip()
                if v:
                    sev_val = v
                    break

        out_rows.append({
            "dataset": "emergencias",
            "event_type": ev_type,
            "date": date_str,
            "time": time_str,
            "datetime": dt_str,
            "district_code": dcode,
            "district_name": dname,
            "lat": lat_val,
            "lon": lon_val,
            "location": location_val,
            "severity": sev_val,
            "value": "",      # normalmente vacío
            "units": "",      # normalmente vacío
            "source_id": source_url
        })

    out = pd.DataFrame(out_rows)
    for c in CONTRACT_COLS:
        if c not in out.columns:
            out[c] = ""
    return out[CONTRACT_COLS]

# ============= Procesado por ficha =============
def process_page(page_url: str) -> pd.DataFrame:
    """Descarga, filtra por ventana y mapea a contrato."""
    print(f"\n[Procesando] {page_url}")
    try:
        html = fetch_html(page_url)
    except Exception as e:
        print(f"  [Error] al abrir ficha: {e}")
        return pd.DataFrame()
    soup = make_soup(html)

    # URL de datos
    data_url = None
    for key, forced in OVERRIDES.items():
        if key in page_url:
            data_url = forced; break
    if not data_url:
        data_url = find_valid_data_url_from_page(page_url, soup)
    if not data_url:
        print("  [Aviso] sin URL CSV/JSON válida")
        return pd.DataFrame()

    # Carga
    df = load_remote_table(data_url)
    if df is None or df.empty:
        print("  [Aviso] descarga vacía")
        return pd.DataFrame()

    # Descarta agregados
    if not is_event_level(df):
        print("  [Skip] dataset agregado (sin fecha/hora por evento)")
        return pd.DataFrame()

    # Normaliza columnas para el filtro
    df = normalize_cols_df(df)

    # Filtro de ventana
    kept = []
    cols = list(df.columns)
    for _, row in df.iterrows():
        date_str, _, dt_str = extract_event_datetime_cols(row, cols)
        d = None
        if date_str:
            try:
                d = dtparser.parse(date_str, dayfirst=True).date()
            except Exception:
                d = None
        if d is None and dt_str:
            try:
                d = dtparser.parse(dt_str, dayfirst=True).date()
            except Exception:
                d = None
        if d and in_date_window(d, END_DATE, MONTHS_BACK_DEFAULT):
            kept.append(row.to_dict())

    if not kept:
        print("  [OK] 0 filas dentro de la ventana")
        return pd.DataFrame()

    df_kept = pd.DataFrame(kept)

    # Mapear a contrato
    out = build_contract(df_kept, source_url=data_url)

    # Orden por datetime si existe
    try:
        ts = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.iloc[ts.sort_values().index]
    except Exception:
        pass

    print(f"  [OK] {len(out)} filas mapeadas → contrato común")
    return out

# ============= Export con NA (consistente) =============
def export_contract_csv(df: pd.DataFrame, out_path: Path):
    """Exporta contrato con separador ';', sin comillas y 'NA' para vacíos."""
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].astype(str).str.strip()
        df2.loc[df2[c] == "", c] = "NA"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_path, index=False, sep=";", encoding="utf-8-sig", quoting=csv.QUOTE_NONE, escapechar="\\")

# ============= Main =============
def main():
    print(f"[Ventana] {START_DATE.isoformat()} -> {END_DATE.isoformat()} (meses atrás={MONTHS_BACK_DEFAULT})")
    parts: List[pd.DataFrame] = []
    for url in PAGES:
        try:
            dfp = process_page(url)
            if not dfp.empty:
                parts.append(dfp)
        except Exception as e:
            print(f"  [Error inesperado] {e}")

    if not parts:
        print("\n[Resultado] 0 filas en todas las fichas.")
        export_contract_csv(pd.DataFrame(columns=CONTRACT_COLS), OUT_FILE)
        print(f"[OK] Archivo vacío con cabecera: {OUT_FILE.resolve()}")
        return

    out = pd.concat(parts, ignore_index=True, sort=False)
    # Garantiza orden/columnas del contrato
    for c in CONTRACT_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[CONTRACT_COLS]

    export_contract_csv(out, OUT_FILE)
    print(f"\n[OK] Guardado: {OUT_FILE.resolve()}")
    print(f"[Filas] {len(out)}  [Columnas] {len(out.columns)}")
    try:
        print(out.head(10))
    except Exception:
        pass

if __name__ == "__main__":
    main()
