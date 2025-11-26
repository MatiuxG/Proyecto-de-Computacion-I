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
    "https://datos.madrid.es/sites/v/index.jsp?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD",
]

OVERRIDES: Dict[str, str] = {
    "aecb88a7e2b73410": "https://datos.madrid.es/egob/catalogo/201410-10306624-calidad-aire-diario.csv"
}

STATIONS_CATALOG_CANDIDATES = [
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.json",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.json",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.json",
]

OUTPUT_DIR = Path("CalidadAire_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_calidad_aire_agregado.csv"
DEBUG_HEAD = OUTPUT_DIR / "debug_calidad_aire_head.csv"

LOCAL_STATIONS_CSV = Path("./CalidadAire_Scripts/estaciones_custom.csv")

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "MateoScraperBot/1.7 (+contact: your-email@example.com)"
}
REQ_TIMEOUT = 60

# --- RANGO DE FECHAS FIJO (Julio - Septiembre 2025) ---
DEFAULT_START = date_cls(2025, 7, 1)
DEFAULT_END = date_cls(2025, 9, 30)

NEW_COLUMNS = [
    "dia", "mes", "año", "numero de distrito", "nombre del distrito",
    "Oxidosde nitrogeno", "Particulas"
]

MAGNITUD_MAP = {
    7:"no",
    8:"no2",
    10:"pm10",
    12:"pm25",
    1:"so2",
    9:"o3",
}
OPEN_DATA_UNIT = {
    "so2":"µg/m³","co":"mg/m³","no":"µg/m³","no2":"µg/m³","o3":"µg/m³","pm10":"µg/m³","pm25":"µg/m³"
}

# --- MODIFICADO: LISTADO COMPLETO DE ESTACIONES ACTIVAS ---
# Formato: ID: (DistritoID, NombreDistrito, Lat, Lon, NombreEstacion)
STATION_FALLBACK: Dict[str, Tuple[str,str,str,str,str]] = {
    "004": ("09", "Moncloa-Aravaca", "40.423853", "-3.712247", "Pza. de España"),
    "008": ("04", "Salamanca", "40.421564", "-3.682319", "Escuelas Aguirre"),
    "011": ("05", "Chamartín", "40.451475", "-3.677356", "Avda. Ramón y Cajal"),
    "016": ("15", "Ciudad Lineal", "40.440047", "-3.639233", "Arturo Soria"),
    "017": ("17", "Villaverde", "40.347138", "-3.713322", "Villaverde Alto"),
    "018": ("11", "Carabanchel", "40.394782", "-3.731853", "Farolillo"),
    "024": ("09", "Moncloa-Aravaca", "40.419356", "-3.747347", "Casa de Campo"),
    "027": ("21", "Barajas", "40.476928", "-3.580031", "Barajas Pueblo"),
    "035": ("01", "Centro", "40.419208", "-3.703170", "Pza. del Carmen"),
    "036": ("14", "Moratalaz", "40.407948", "-3.645306", "Moratalaz"),
    "038": ("06", "Tetuán", "40.445544", "-3.707128", "Cuatro Caminos"),
    "039": ("08", "Fuencarral-El Pardo", "40.478228", "-3.711542", "Barrio del Pilar"),
    "040": ("13", "Puente de Vallecas", "40.388153", "-3.651522", "Vallecas"),
    "047": ("02", "Arganzuela", "40.398114", "-3.686825", "Mendez Alvaro"),
    "048": ("05", "Chamartín", "40.439897", "-3.690372", "Castellana"),
    "049": ("03", "Retiro", "40.414437", "-3.682562", "Parque del Retiro"),
    "050": ("05", "Chamartín", "40.465572", "-3.688769", "Plaza Castilla"),
    "054": ("18", "Villa de Vallecas", "40.372933", "-3.616344", "Ensanche de Vallecas"),
    "055": ("21", "Barajas", "40.462531", "-3.580747", "Urb. Embajada"),
    "056": ("11", "Carabanchel", "40.385033", "-3.718728", "Pza. Elíptica"),
    "057": ("16", "Hortaleza", "40.494208", "-3.660503", "Sanchinarro"),
    "058": ("08", "Fuencarral-El Pardo", "40.518058", "-3.774611", "El Pardo"),
    "059": ("21", "Barajas", "40.460725", "-3.616344", "Juan Carlos I"),
    "060": ("08", "Fuencarral-El Pardo", "40.500547", "-3.689731", "Tres Olivos"),
}

# ================================
# Helpers y lógica
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
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()

    if u.endswith(".csv") or "csv" in ctype:
        data = r.content
        for enc in ("utf-8-sig","utf-8","latin-1",None):
            for sep in (";", "\t", ",", "|"):
                try:
                    if enc is None:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False)
                    else:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False, encoding=enc)
                    if df.shape[1] == 1:
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

def build_station_lookup() -> Dict[str, Dict[str,str]]:
    # Intentar cargar catálogo online
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

    # Si falla, intentar CSV local
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

    # Si todo falla, usar fallback completo
    if STATION_FALLBACK:
        print("[Lookup] Usando fallback estático de estaciones (LISTA COMPLETA).")
        return {k: {"district_code":v[0],"district_name":v[1],"lat":v[2],"lon":v[3],"name":v[4]}
                for k,v in STATION_FALLBACK.items()}

    print("[Lookup] *Sin* catálogo ni fallback — distritos quedarán 'NA'.")
    return {}

def extract_station_code(punto_muestreo: str) -> Optional[str]:
    if not punto_muestreo:
        return None
    m = re.search(r"28079(\d{3})", str(punto_muestreo))
    return m.group(1) if m else None

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
    out["station_code"] = long_df[pm].map(extract_station_code) if pm in long_df.columns else None
    out["station_name"] = long_df[st_name].astype(str) if st_name in long_df.columns else ""
    out["aq_value"] = long_df["aq_value"].astype(float)
    return out.reset_index(drop=True)

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

def build_custom_datasheet(df_daily: pd.DataFrame, station_lookup: Dict[str, Dict[str,str]]) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame(columns=NEW_COLUMNS)

    def get_district_info(station_code):
        meta = station_lookup.get(str(station_code).strip(), {})
        return pd.Series({
            "district_code": meta.get("district_code", ""),
            "district_name": meta.get("district_name", "")
        })

    district_info = df_daily["station_code"].apply(get_district_info)
    df_enriched = pd.concat([df_daily, district_info], axis=1)

    contaminantes_necesarios = ['no', 'no2', 'pm10', 'pm25']
    # Asegurar que no se filtren datos si el distrito existe
    df_filtered = df_enriched[
        df_enriched['pollutant'].isin(contaminantes_necesarios) &
        (df_enriched['district_code'] != "") &
        (df_enriched['district_code'] != "NA")
    ].copy()

    if df_filtered.empty:
        print("  [Aviso] No se encontraron datos para los contaminantes o distritos solicitados.")
        return pd.DataFrame(columns=NEW_COLUMNS)

    df_pivot_station = df_filtered.pivot_table(
        index=['fecha', 'district_code', 'district_name', 'station_code'],
        columns='pollutant',
        values='aq_value'
    ).reset_index()

    df_agg_district = df_pivot_station.groupby(
        ['fecha', 'district_code', 'district_name']
    ).mean(numeric_only=True).reset_index()

    df_final = pd.DataFrame()
    df_final['fecha_dt'] = pd.to_datetime(df_agg_district['fecha'])
    df_final['dia'] = df_final['fecha_dt'].dt.day
    df_final['mes'] = df_final['fecha_dt'].dt.month
    df_final['año'] = df_final['fecha_dt'].dt.year
    df_final['numero de distrito'] = df_agg_district['district_code']
    df_final['nombre del distrito'] = df_agg_district['district_name']

    cols_no_nox = [c for c in ['no', 'no2'] if c in df_agg_district.columns]
    if not cols_no_nox:
        df_final['Oxidosde nitrogeno'] = pd.NA
    else:
        df_final['Oxidosde nitrogeno'] = df_agg_district[cols_no_nox].sum(axis=1, skipna=True, min_count=1)

    cols_pm = [c for c in ['pm10', 'pm25'] if c in df_agg_district.columns]
    if not cols_pm:
        df_final['Particulas'] = pd.NA
    else:
        df_final['Particulas'] = df_agg_district[cols_pm].sum(axis=1, skipna=True, min_count=1)

    df_out = pd.DataFrame()
    for col in NEW_COLUMNS:
        df_out[col] = df_final.get(col, pd.NA)

    df_out = df_out.fillna(0)
    return df_out.sort_values(by=['año', 'mes', 'dia', 'numero de distrito']).reset_index(drop=True)

def main():
    start_date = DEFAULT_START
    end_date = DEFAULT_END
    print(f"[Ventana] {start_date.isoformat()} -> {end_date.isoformat()}")

    station_lookup = build_station_lookup()
    if not station_lookup:
        print("[Aviso] No se pudo cargar catálogo de estaciones ni fallback útil. Distritos 'NA'.")

    parts = []
    for url in PAGES:
        daily_all = process_one(url)
        daily_win = filter_window_daily(daily_all, start_date, end_date)
        print(f"  [Filtrado 2m] {daily_win.shape} filas")
        if not daily_win.empty:
            parts.append(daily_win)

    if not parts:
        print("\n[Resultado] Sin filas en la ventana.")
        pd.DataFrame(columns=NEW_COLUMNS).to_csv(
            OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
        )
        print(f"[OK] Datasheet vacío: {OUT_FILE.resolve()}")
        return

    daily = pd.concat(parts, ignore_index=True, sort=False)
    datasheet = build_custom_datasheet(daily, station_lookup)

    datasheet.to_csv(
        OUT_FILE, index=False, sep=";", encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
    )
    
    print(f"\n[OK] Datasheet escrito: {OUT_FILE.resolve()}")
    print(f"[Filas] {len(datasheet)}")

if __name__ == "__main__":
    main()