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

# Páginas de catálogo a escanear (puedes ampliar esta lista si lo necesitas)
DATASET_PAGES = [
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=40085fb0e70b7410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    # Catálogo CAM (forzamos URLs reales abajo)
    "https://datos.comunidad.madrid/catalogos/#/dataset/1908061?view=info",
]

# Overrides para el dataset de CAM (cuando el catálogo expone RDF o visor)
HARDCODED_DOWNLOADS: Dict[str, List[str]] = {
    "comunidad.madrid/catalogos/#/dataset/1908061": [
        "https://datos.comunidad.madrid/dataset/fb9c5a17-afb0-4e95-a7b1-186e7cacc901/resource/58e39362-fbd1-45f6-865b-91505f6bd199/download/accidentes-de-circulacion-con-victimas-por-ubicacion-y-resultado-del-accidente.csv",
        "https://datos.comunidad.madrid/dataset/fb9c5a17-afb0-4e95-a7b1-186e7cacc901/resource/69a6b3e0-f711-47c5-aa2d-a87b0f82fd31/download/accidentes-de-circulacion-con-victimas-por-ubicacion-y-resultado-del-accidente.json",
    ],
}

# Carpeta de salida (cada script debe escribir su propio datasheet aquí)
OUTPUT_DIR = Path("./Accidentes_Scripts/Resultados")
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

# Columnas base del contrato común (todas los datasheets deben incluirlas)
CONTRACT_BASE = [
    "dataset", "event_type", "date", "time", "datetime",
    "district_code", "district_name", "lat", "lon", "location",
    "severity", "value", "units", "source_id",
]

# Columnas extra (específicas de accidentes) que intentaremos poblar si existen
EXTRA_ACCIDENTS = [
    "road", "cause", "num_vehicles", "num_injured", "num_fatalities"
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
    keys = ["calle","via","vía","direccion","dirección","ubicacion","ubicación","lugar","punto","domicilio",
            "carretera","tramo","pk","interseccion","intersección","cruce","kilometro","kilómetro","street","road","address","addr"]
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
    """Mapea el DataFrame bruto al contrato común + campos útiles de accidentes."""
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    # 1) Inicializa contrato base
    out = pd.DataFrame()
    out["dataset"] = "accidentes"
    out["event_type"] = "accidente"

    # 2) Fecha/hora
    dt_cols = guess_datetime_cols(df)
    if dt_cols:
        full = None
        for c in dt_cols:
            if any(k in c for k in ["fecha_hora","fechahora","datetime","timestamp"]):
                full = c; break
        if full is None:
            full = dt_cols[0]

        dt_parsed = pd.to_datetime(df[full], errors="coerce", dayfirst=True)
        out["datetime"] = dt_parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        out["date"] = dt_parsed.dt.strftime("%Y-%m-%d")
        out["time"] = dt_parsed.dt.strftime("%H:%M")

        out.loc[out["time"] == "NaT", "time"] = ""
        out.loc[out["date"] == "NaT", "date"] = ""
        out.loc[out["datetime"] == "NaT", "datetime"] = ""
    else:
        dcol = next((c for c in df.columns if c in ("fecha","date","dia","día")), None)
        tcol = next((c for c in df.columns if c in ("hora","time","hr","h_acc","acc_hora")), None)
        out["date"] = df[dcol].map(as_date) if dcol else ""
        out["time"] = df[tcol].map(as_time) if tcol else ""
        out["datetime"] = ""

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

    # 4) Lat/Lon
    lat_col = next((c for c in df.columns if c in ("lat","latitude","y","latitud")), None)
    lon_col = next((c for c in df.columns if c in ("lon","longitud","long","x","longitude")), None)
    out["lat"] = df.get(lat_col, "") if lat_col else ""
    out["lon"] = df.get(lon_col, "") if lon_col else ""

    # 5) Ubicación (vía/calle)
    street_cols = guess_street_cols(df)
    out["location"] = df[street_cols[0]].astype(str) if street_cols else ""

    # 6) Severidad / valor / unidades
    sev_candidates = ["gravedad","lesividad","severidad","resultado","resultado_accidente","severity"]
    sev_col = next((c for c in df.columns if c in sev_candidates), None)
    out["severity"] = df.get(sev_col, "") if sev_col else ""
    out["value"] = "NA"   # Plantilla: 'NA' cuando no hay métrica
    out["units"] = "NA"

    # 7) ID origen
    id_candidates = ["id","codigo","cod_accidente","id_accidente","expediente","num_exp","codigo_accidente"]
    id_col = next((c for c in df.columns if c in id_candidates), None)
    out["source_id"] = df.get(id_col, "") if id_col else df.get("__download__", "")

    # 8) Extras de accidentes (opcionales)
    road_candidates = ["carretera","via","tramo","pk","road","calle","vía"]
    cause_candidates = ["causa","causa_accidente","motivo","concurrencia","tipo_accidente","descripcion"]
    nv_candidates = ["n_vehiculos","num_vehiculos","nvehiculos","vehiculos_implicados","vehiculos"]
    ni_candidates = ["n_heridos","num_heridos","heridos","lesionados","n_lesionados"]
    nf_candidates = ["n_fallecidos","num_fallecidos","fallecidos","muertos","n_muertos"]

    def pick_first(cols: List[str]) -> Optional[str]:
        return next((c for c in df.columns if c in cols), None)

    road_col = pick_first(road_candidates)
    cause_col = pick_first(cause_candidates)
    nv = pick_first(nv_candidates)
    ni = pick_first(ni_candidates)
    nf = pick_first(nf_candidates)

    out["road"] = df.get(road_col, "") if road_col else out.get("location", "")
    out["cause"] = df.get(cause_col, "") if cause_col else ""
    out["num_vehicles"] = df.get(nv, "") if nv else ""
    out["num_injured"] = df.get(ni, "") if ni else ""
    out["num_fatalities"] = df.get(nf, "") if nf else ""

    # 9) Proveniencia (útil para depurar; no es parte del contrato pero lo mantenemos)
    if "__source_page__" in df.columns:
        out["__source_page__"] = df["__source_page__"]
    if "__download__" in df.columns:
        out["__download__"] = df["__download__"]

    # 10) Garantiza columnas del contrato y extras
    for col in CONTRACT_BASE:
        if col not in out.columns:
            out[col] = ""
    for col in EXTRA_ACCIDENTS:
        if col not in out.columns:
            out[col] = ""

    # (Opcional) Si no hay hora real y no hay datetime, dejar vacío para que luego sea 'NA'
    out.loc[out["time"].isin(["NaT", "00:00"]) & out["datetime"].eq(""), "time"] = ""

    # Orden tentativo (recortaremos al final)
    ordered = CONTRACT_BASE + EXTRA_ACCIDENTS + [c for c in out.columns if c not in CONTRACT_BASE + EXTRA_ACCIDENTS]
    out = out[ordered]
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
        # CSV vacío con SOLO las columnas del contrato
        empty = pd.DataFrame(columns=CONTRACT_BASE)
        empty.to_csv(
            OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
        )
        print(f"[OK] Datasheet vacío escrito: {OUT_FILE.resolve()}")
        return

    raw = pd.concat(parts, ignore_index=True, sort=False)
    standardized = build_contract_from_raw(raw)

    # === Plantilla compliance: NA también para strings vacíos/espacios y NaN ===
    standardized = standardized.fillna("NA")
    standardized = standardized.replace(r"^\s*$", "NA", regex=True)
    for c in CONTRACT_BASE:
        if c not in standardized.columns:
            standardized[c] = "NA"

    # Recortar exactamente a las 14 columnas del contrato
    standardized = standardized[CONTRACT_BASE]

    # Guardado final (utf-8-sig recomendado para Excel)
    standardized.to_csv(
        OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
    )

    print(f"\n[OK] Datasheet escrito: {OUT_FILE.resolve()}")
    print(f"[Filas] {len(standardized)}")
    print("\n[Preview]")
    print(standardized.head(10))

    # ===== OPCIONAL: Guardar extras en CSV aparte =====
    # Si quieres conservar columnas extra para análisis internos:
    full_std = build_contract_from_raw(raw).fillna("NA").replace(r"^\s*$", "NA", regex=True)
    extras_cols = [c for c in EXTRA_ACCIDENTS if c in full_std.columns]
    keep_cols = []
    if extras_cols:
        keep_cols.extend(extras_cols)
        for meta in ["__source_page__", "__download__"]:
            if meta in full_std.columns and meta not in keep_cols:
                keep_cols.append(meta)
    if keep_cols:
        extra_df = full_std[keep_cols].copy()
        extra_path = OUT_FILE.with_name(OUT_FILE.stem.replace("datasheet_", "extras_") + OUT_FILE.suffix)
        extra_df.to_csv(
            extra_path, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
        )
        print(f"[OK] Extras guardados en: {extra_path.resolve()}")

if __name__ == "__main__":
    main()
