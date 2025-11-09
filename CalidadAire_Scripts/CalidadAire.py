# -*- coding: utf-8 -*-
# Code in English, comments in Spanish

import re
import io
import csv
import unicodedata
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ================================
# Config
# ================================

PAGES = [
    # Ficha "Calidad del aire: datos diarios desde 2001"
    "https://datos.madrid.es/sites/v/index.jsp?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD",
]

# Override directo al CSV estable de diarios
OVERRIDES: Dict[str, str] = {
    "aecb88a7e2b73410": "https://datos.madrid.es/egob/catalogo/201410-10306624-calidad-aire-diario.csv"
}

# Catálogos de estaciones candidatos (el script probará en orden)
STATIONS_CATALOG_CANDIDATES = [
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.json",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.json",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.json",
]

OUTPUT_DIR = Path("./CalidadAire_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_calidad_aire.csv"
EXTRAS_FILE = OUTPUT_DIR / "extras_calidad_aire.csv"
DEBUG_HEAD = OUTPUT_DIR / "debug_calidad_aire_head.csv"

# CSV local opcional (si falla internet / formato raro)
LOCAL_STATIONS_CSV = Path("./CalidadAire_Scripts/estaciones_custom.csv")

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "MateoScraperBot/1.7 (+contact: your-email@example.com)"
}
REQ_TIMEOUT = 60

TODAY = datetime.today().date()
DEFAULT_START = TODAY - relativedelta(months=2)
DEFAULT_END = TODAY

# 14 columnas del contrato
CONTRACT_BASE = [
    "dataset","event_type","date","time","datetime",
    "district_code","district_name","lat","lon","location",
    "severity","value","units","source_id",
]

# Extras (guardamos aparte)
AIR_FIELDS = ["pollutant","aq_value","aq_unit","station_code","station_name","__source_url"]

# MAGNITUD → contaminante conocido
MAGNITUD_MAP = {1:"so2", 6:"co", 7:"no", 8:"no2", 9:"o3", 10:"pm10", 12:"pm25"}
OPEN_DATA_UNIT = {
    "so2":"µg/m³","co":"mg/m³","no":"µg/m³","no2":"µg/m³","o3":"µg/m³","pm10":"µg/m³","pm25":"µg/m³"
}

# Fallback embebido (puedes ampliarlo fácilmente)
# code: (district_code, district_name, lat, lon, name)
STATION_FALLBACK: Dict[str, Tuple[str,str,str,str,str]] = {
    # Las tres de tu muestra
    "011": ("01", "Centro",               "", "", "Plaza del Carmen"),
    "016": ("15", "Ciudad Lineal",        "", "", "Arturo Soria"),
    "017": ("08", "Fuencarral-El Pardo",  "", "", "Barrio del Pilar"),
    # Algunas habituales (útiles si aparecen)
    "012": ("03", "Retiro",               "", "", "Retiro"),
    "013": ("05", "Chamartín",            "", "", "Castellana"),
    "014": ("07", "Chamberí",             "", "", "Escuelas Aguirre"),
    "018": ("21", "Barajas",              "", "", "Barajas"),
    "019": ("20", "San Blas-Canillejas",  "", "", "San Blas"),
    "020": ("10", "Latina",               "", "", "Casa de Campo"),
    "022": ("16", "Hortaleza",            "", "", "Hortaleza"),
}

# ================================
# Helpers
# ================================

def nfd_lower(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        name = str(c).replace("\ufeff", "")
        base = nfd_lower(name)
        base = re.sub(r"\s+", "_", base)
        cols.append(base)
    df2 = df.copy()
    df2.columns = cols
    return df2

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("ascii","utf-8"):
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

def find_valid_data_url_from_page(page_url: str, soup: BeautifulSoup) -> Optional[str]:
    for key, forced in OVERRIDES.items():
        if key in page_url:
            return forced
    candidates = []
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).split())
        href = absolutize(page_url, a["href"])
        low = (href + " " + label).lower()
        if href.lower().endswith(".csv") or "descarg" in low or "download" in low:
            score = 1000
            if href.lower().endswith(".csv"): score -= 300
            if "download" in low or "descarg" in low: score -= 200
            candidates.append((score, href))
    return sorted(candidates, key=lambda t: t[0])[0][1] if candidates else None

def load_table(url: str) -> Optional[pd.DataFrame]:
    """Carga CSV/JSON con detección robusta de encoding/separador."""
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()

    if u.endswith(".csv") or "csv" in ctype:
        data = r.content
        for enc in ("utf-8-sig","utf-8","latin-1",None):
            for sep in (";","\t",",","|"):
                try:
                    if enc is None:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False)
                    else:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False, encoding=enc)
                    if df.shape[1] == 1:  # separador erróneo
                        continue
                    return df
                except Exception:
                    continue
        return None

    if u.endswith(".json") or u.endswith(".geojson") or "json" in ctype:
        data = r.json()
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            return pd.json_normalize(data["features"])
        return pd.json_normalize(data)

    return None

# ================================
# Estaciones → distrito/coords
# ================================

def build_station_lookup() -> Dict[str, Dict[str,str]]:
    """
    Devuelve { '011': {'district_code':'01','district_name':'Centro','lat':'..','lon':'..','name':'..'}, ... }
    Prioridad: catálogo online → CSV local 'estaciones_custom.csv' → fallback embebido.
    """
    # 1) Intento catálogo online
    for url in STATIONS_CATALOG_CANDIDATES:
        try:
            df = load_table(url)
            if df is None or df.empty:
                print(f"[Lookup] Vacío/no válido: {url}")
                continue
            df = normalize_cols(df)
            print(f"[Lookup] Cargado: {url} - cols: {list(df.columns)[:10]}...")

            stc = next((c for c in ["codigo_estacion","cod_estacion","estacion","estación","codigo","code","id_estacion"] if c in df.columns), None)
            name = next((c for c in ["nombre","nombre_estacion","estacion_nombre","estación_nombre","station_name"] if c in df.columns), None)
            dist_name = next((c for c in ["distrito","nombre_distrito","district","district_name"] if c in df.columns), None)
            dist_code = next((c for c in ["cod_distrito","codigo_distrito","district_code","codigo__distrito"] if c in df.columns), None)
            latc = next((c for c in ["lat","latitud","latitude","y"] if c in df.columns), None)
            lonc = next((c for c in ["lon","longitud","longitude","x"] if c in df.columns), None)

            if stc is None:
                print(f"[Lookup] No encuentro columna de código estación en {url}")
                continue

            lookup: Dict[str, Dict[str,str]] = {}
            for _, row in df.iterrows():
                m = re.search(r"(\d{1,3})", str(row.get(stc, "")))
                if not m:
                    continue
                sid = m.group(1).zfill(3)

                dcode = str(row.get(dist_code, "") or "").strip()
                if dcode != "":
                    dcode = re.sub(r"\D", "", dcode).zfill(2)

                lookup[sid] = {
                    "district_code": dcode,
                    "district_name": str(row.get(dist_name, "") or "").strip(),
                    "lat": str(row.get(latc, "") or "").strip(),
                    "lon": str(row.get(lonc, "") or "").strip(),
                    "name": str(row.get(name, "") or "").strip(),
                }
            if lookup:
                print(f"[Lookup] Catálogo estaciones OK: {url} ({len(lookup)} estaciones)")
                return lookup
        except Exception as e:
            print(f"[Lookup] Error con {url}: {e}")
            continue

    # 2) Intento CSV local opcional
    if LOCAL_STATIONS_CSV.exists():
        try:
            df = pd.read_csv(LOCAL_STATIONS_CSV, dtype=str)
            df = normalize_cols(df)
            req = {"station_code","district_code","district_name","lat","lon","name"}
            if not req.issubset(df.columns):
                print(f"[Lookup] CSV local sin columnas requeridas: {LOCAL_STATIONS_CSV}")
            else:
                lookup = {}
                for _, row in df.iterrows():
                    sid = re.sub(r"\D","", str(row["station_code"])).zfill(3)
                    dcode = re.sub(r"\D","", str(row["district_code"])).zfill(2)
                    lookup[sid] = {
                        "district_code": dcode,
                        "district_name": str(row["district_name"]),
                        "lat": str(row["lat"]),
                        "lon": str(row["lon"]),
                        "name": str(row["name"]),
                    }
                if lookup:
                    print(f"[Lookup] Usando CSV local: {LOCAL_STATIONS_CSV} ({len(lookup)} estaciones)")
                    return lookup
        except Exception as e:
            print(f"[Lookup] Error leyendo CSV local: {e}")

    # 3) Fallback embebido
    if STATION_FALLBACK:
        print("[Lookup] Usando fallback estático de estaciones.")
        return {k: {"district_code":v[0],"district_name":v[1],"lat":v[2],"lon":v[3],"name":v[4]}
                for k,v in STATION_FALLBACK.items()}

    print("[Lookup] *Sin* catálogo ni fallback — distritos quedarán 'NA'.")
    return {}

def extract_station_code(punto_muestreo: str) -> Optional[str]:
    """
    PUNTO_MUESTREO típico: '28079NNN_XX_Y' → sacamos NNN (3 dígitos).
    """
    if not punto_muestreo:
        return None
    m = re.search(r"28079(\d{3})", str(punto_muestreo))
    return m.group(1) if m else None

# ================================
# Limpieza y expansión diaria
# ================================

def clean_value_str(s):
    if s is None:
        return None
    s = str(s).strip().strip('"').strip("'")
    if s == "" or s.upper() == "NA" or s == "-":
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    return m.group(0).replace(",", ".")

def expand_month_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Usa D01..D31 como valor; V01..V31 (validación) no es obligatoria."""
    if df.empty:
        return df

    df = normalize_cols(df)

    ycol = next((c for c in ["ano","año","anio","year"] if c in df.columns), None)
    mcol = next((c for c in ["mes","month"] if c in df.columns), None)
    mag  = next((c for c in ["magnitud","cod_magnitud","codigo_magnitud"] if c in df.columns), None)
    pm   = next((c for c in ["punto_muestreo","punto_de_muestreo"] if c in df.columns), None)
    st_name = next((c for c in ["nombre","estacion_nombre","estación_nombre","station_name"] if c in df.columns), None)

    if ycol is None or mcol is None:
        return pd.DataFrame()

    dcols = [f"d{d:02d}" for d in range(1, 32) if f"d{d:02d}" in df.columns]
    if not dcols:
        return pd.DataFrame()

    blocks = []
    base_cols = [c for c in [ycol, mcol, mag, pm, st_name] if c is not None]
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = pd.to_numeric(df[mcol], errors="coerce")

    for d in range(1, 32):
        dcol = f"d{d:02d}"
        if dcol not in df.columns:
            continue
        val_series = df[dcol].map(clean_value_str)
        tmp = df[base_cols].copy()
        tmp["year__"] = y
        tmp["month__"] = m
        tmp["day__"] = d
        tmp["value_txt__"] = val_series
        blocks.append(tmp)

    if not blocks:
        return pd.DataFrame()

    long_df = pd.concat(blocks, ignore_index=True)
    long_df = long_df[long_df["value_txt__"].notna()]
    long_df["fecha"] = pd.to_datetime(dict(
        year=long_df["year__"].astype("Int64"),
        month=long_df["month__"].astype("Int64"),
        day=long_df["day__"].astype("Int64")
    ), errors="coerce").dt.date
    long_df = long_df[pd.notna(long_df["fecha"])]

    long_df["aq_value"] = pd.to_numeric(long_df["value_txt__"], errors="coerce")
    long_df = long_df[long_df["aq_value"].notna()]

    out = pd.DataFrame()
    out["fecha"] = long_df["fecha"]

    # pollutant + unit (fallbacks)
    def _map_pol_and_unit(v):
        try:
            code = int(float(str(v).replace(",", ".")))
        except Exception:
            return ("mag_unknown", "NA")
        pol = MAGNITUD_MAP.get(code, f"mag_{code}")
        unit = OPEN_DATA_UNIT.get(pol, "NA")
        return (pol, unit)

    pol_unit = long_df[mag].map(_map_pol_and_unit) if mag in long_df.columns else [("mag_unknown","NA")] * len(long_df)
    out["pollutant"] = [t[0] for t in pol_unit]
    out["aq_unit"]   = [t[1] for t in pol_unit]

    # Station fields
    out["station_code"] = long_df[pm].map(extract_station_code) if pm in long_df.columns else None
    out["station_name"] = long_df[st_name].astype(str) if st_name in long_df.columns else ""

    out["aq_value"] = long_df["aq_value"].astype(float)
    return out.reset_index(drop=True)

# ================================
# Pipeline
# ================================

def process_one(page_url: str) -> pd.DataFrame:
    print(f"\n[Ficha] {page_url}")
    try:
        html = fetch_html(page_url)
    except Exception as e:
        print(f"  [Error al abrir ficha] {e}")
        return pd.DataFrame()
    soup = make_soup(html)

    data_url = None
    for key, forced in OVERRIDES.items():
        if key in page_url:
            data_url = forced; break
    if not data_url:
        data_url = find_valid_data_url_from_page(page_url, soup)
    if not data_url:
        print("  [Aviso] No se encontró URL de datos.")
        return pd.DataFrame()

    print(f"  [Descarga] {data_url}")
    df_raw = load_table(data_url)
    if df_raw is None or df_raw.empty:
        print("  [Aviso] Descarga vacía o no válida.")
        return pd.DataFrame()

    try:
        df_raw.head(200).to_csv(
            DEBUG_HEAD, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
        )
        print(f"  [Debug] Primeras 200 filas → {DEBUG_HEAD.resolve()}")
    except Exception:
        pass

    daily = expand_month_to_daily(df_raw)
    print(f"  [Expandido] {daily.shape} filas diarias")
    daily["__source_url"] = data_url
    return daily.reset_index(drop=True)

def filter_window_daily(daily: pd.DataFrame, start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    if daily.empty or "fecha" not in daily.columns:
        return daily.iloc[0:0]
    mask = daily["fecha"].between(start_date, end_date)
    return daily[mask].copy()

def build_contract(df_daily: pd.DataFrame, station_lookup: Dict[str, Dict[str,str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_daily.empty:
        return (pd.DataFrame(columns=CONTRACT_BASE),
                pd.DataFrame(columns=AIR_FIELDS))

    # Enriquecer con distrito/coords usando station_code
    def enrich(row):
        sid = (row.get("station_code") or "").strip()
        meta = station_lookup.get(sid, {})
        return pd.Series({
            "district_code": meta.get("district_code",""),
            "district_name": meta.get("district_name",""),
            "lat": meta.get("lat",""),
            "lon": meta.get("lon",""),
            "location": meta.get("name","") or row.get("station_name","")
        })

    enrich_df = df_daily.apply(enrich, axis=1)

    out = pd.DataFrame()
    out["dataset"] = "calidad_aire"
    out["event_type"] = df_daily.get("pollutant", "medicion").fillna("medicion")
    out["date"] = pd.to_datetime(df_daily["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["time"] = ""
    out["datetime"] = ""
    out["district_code"] = enrich_df["district_code"]
    out["district_name"] = enrich_df["district_name"]
    out["lat"] = enrich_df["lat"]
    out["lon"] = enrich_df["lon"]
    out["location"] = enrich_df["location"].where(enrich_df["location"].str.len() > 0, df_daily.get("station_name",""))
    out["severity"] = ""
    out["value"] = pd.to_numeric(df_daily.get("aq_value", pd.Series(dtype=float)), errors="coerce")
    out["units"] = df_daily.get("aq_unit", "NA")
    out["source_id"] = df_daily.get("__source_url", "")

    # Garantiza contrato y relleno NA
    for col in CONTRACT_BASE:
        if col not in out.columns:
            out[col] = ""
    out = out[CONTRACT_BASE]
    out = out.fillna("NA").replace(r"^\s*$", "NA", regex=True)

    # Extras
    extras = pd.DataFrame()
    for c in AIR_FIELDS:
        extras[c] = df_daily.get(c, "NA")
    extras = extras.fillna("NA").replace(r"^\s*$", "NA", regex=True)
    return out, extras

def main():
    start_date = DEFAULT_START
    end_date = DEFAULT_END
    print(f"[Ventana] {start_date.isoformat()} -> {end_date.isoformat()}")

    # 1) Catálogo de estaciones → lookup
    station_lookup = build_station_lookup()
    if not station_lookup:
        print("[Aviso] No se pudo cargar catálogo de estaciones ni fallback útil. Distritos 'NA'.")

    # 2) Carga y filtro ventana
    parts = []
    for url in PAGES:
        daily_all = process_one(url)
        daily_win = filter_window_daily(daily_all, start_date, end_date)
        print(f"  [Filtrado 2m] {daily_win.shape} filas")
        if not daily_win.empty:
            parts.append(daily_win)

    if not parts:
        print("\n[Resultado] Sin filas en la ventana.")
        pd.DataFrame(columns=CONTRACT_BASE).to_csv(
            OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
        )
        print(f"[OK] Datasheet vacío: {OUT_FILE.resolve()}")
        return

    daily = pd.concat(parts, ignore_index=True, sort=False)

    # 3) Construir contrato + extras
    datasheet, extras = build_contract(daily, station_lookup)

    # 4) Guardado
    datasheet.to_csv(
        OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
    )
    if not extras.empty:
        extras.to_csv(
            EXTRAS_FILE, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
        )

    # Diagnóstico: estaciones sin distrito
    if "district_code" in datasheet.columns:
        missing = datasheet["district_code"].isin(["NA",""])
        if missing.any():
            problematic = (daily.loc[missing, ["station_code","station_name"]]
                           .drop_duplicates()
                           .sort_values(by="station_code"))
            print("\n[Diag] Estaciones sin mapeo de distrito (añade al fallback o CSV local):")
            try:
                print(problematic.to_string(index=False))
            except Exception:
                print(problematic.head(20))

    print(f"\n[OK] Datasheet escrito: {OUT_FILE.resolve()}")
    if not extras.empty:
        print(f"[OK] Extras escritos: {EXTRAS_FILE.resolve()}")
    print(f"[Filas] {len(datasheet)}")

if __name__ == "__main__":
    main()
