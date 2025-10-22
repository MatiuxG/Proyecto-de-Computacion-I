import re
import io
import sys
import math
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta

# ----------- Fuentes -----------
ACCUWEATHER_AQI = "https://www.accuweather.com/es/es/madrid/308526/air-quality-index/308526"
PAGES = [
    # Ficha de Datos Abiertos (calidad del aire - datos diarios)
    "https://datos.madrid.es/sites/v/index.jsp?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD",
]

# ----------- Salida -----------
OUTPUT_DIR = Path("./CalidadAire_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE_LOG = OUTPUT_DIR / "calidad_aire_filtrado.csv"
OUT_FILE_COMPARE = OUTPUT_DIR / "calidad_aire_comparado.csv"

# ----------- Red -----------
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ClimaComparer/1.0"
}
REQ_TIMEOUT = 45
CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

# Excluir ruido en detección de enlaces de datos
EXCLUDE_PATTERNS = [
    "wms", "wmts", "ogc", "service", "arcgis", "esri",
    ".zip", ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".pdf", ".rdf", ".xml", "sparql", "mailto:", "javascript:", "#"
]
PREFERRED_HINTS = ["download", "descarg", "csv", "json"]

# Overrides por si hubiera una URL directa conocida
OVERRIDES: Dict[str, str] = {
    # "aecb88a7e2b73410": "https://datos.madrid.es/egob/catalogo/201410-10306624-calidad-aire-diario.csv"
}

# Margen por defecto si el usuario indica fecha
MONTHS_BACK_DEFAULT = 2

# Mapas de contaminantes (Madrid suele usar códigos MAGNITUD)
MAGNITUD_MAP = {
    1:  "so2",
    6:  "co",
    7:  "no",
    8:  "no2",
    9:  "o3",
    10: "pm10",
    12: "pm25",
}
# Unidades típicas por contaminante en red de Madrid (aprox)
OPEN_DATA_UNIT = {
    "so2": "µg/m³",
    "co":  "mg/m³",   # Suele venir en mg/m³ en Madrid
    "no":  "µg/m³",
    "no2": "µg/m³",
    "o3":  "µg/m³",
    "pm10":"µg/m³",
    "pm25":"µg/m³",
}

# ----------- Utilidades de texto/fecha ----------
def normalize_text(s: str) -> str:
    # Normaliza cadenas (quita acentos, pasa a minúsculas) para comparar
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def parse_user_date(s: str) -> Optional[date_cls]:
    # Parsea la fecha introducida por el usuario
    s = (s or "").strip()
    if not s:
        return None
    try:
        return dtparser.parse(s, dayfirst=False).date()
    except Exception:
        return None

def parse_user_hour(s: str) -> Optional[Tuple[int, int]]:
    # Parsea hora tolerante (08, 08:30, 8.5, 8,25, etc.)
    st = (s or "").strip()
    if not st:
        return None
    st = st.lower().replace(" ", "").replace(",", ".").replace("h", ":")
    if ":" in st:
        hh_str, mm_str = st.split(":", 1)
        if not hh_str:
            raise ValueError("Hora inválida")
        hh = int(hh_str)
        mm = int(round(float(mm_str))) if mm_str else 0
        if mm == 60: hh, mm = hh + 1, 0
        return hh, mm
    if re.match(r"^\d+(\.\d+)?$", st):
        f = float(st)
        hh = int(f)
        mm = int(round((f - hh) * 60))
        if mm == 60: hh, mm = hh + 1, 0
        return hh, mm
    if st.isdigit():
        return int(st), 0
    raise ValueError("Formato de hora no reconocido")

def fetch_html(url: str) -> str:
    # Descarga HTML con control de codificación
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("ascii", "utf-8"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def make_soup(html: str) -> BeautifulSoup:
    # Crea objeto soup
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def absolutize(base_url: str, href: str) -> str:
    # Convierte enlaces relativos a absolutos
    from urllib.parse import urljoin
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(base_url, href)
    return href

def head_or_get_headers(url: str) -> Optional[requests.Response]:
    # Intenta HEAD y cae a GET si los headers no informan
    try:
        r = requests.head(url, headers=HEADERS, timeout=REQ_TIMEOUT, allow_redirects=True)
        if r.status_code >= 400:
            return None
        ct = (r.headers.get("Content-Type") or "").lower()
        cd = (r.headers.get("Content-Disposition") or "")
        if ct or cd:
            return r
        r2 = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT, stream=True, allow_redirects=True)
        return r2
    except Exception:
        return None

def content_says_data(resp: requests.Response, url: str) -> bool:
    # Comprueba si un URL es CSV/JSON por headers o por filename
    ct = (resp.headers.get("Content-Type") or "").lower()
    cd = (resp.headers.get("Content-Disposition") or "")
    ul = url.lower()
    if "text/csv" in ct or "application/json" in ct or "application/geo+json" in ct:
        return True
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', cd, flags=re.I)
    if m:
        fname = m.group(1).strip().lower()
        if fname.endswith(".csv") or fname.endswith(".json") or fname.endswith(".geojson"):
            return True
    if ul.endswith(".csv") or ul.endswith(".json") or ul.endswith(".geojson"):
        return True
    return False

def find_valid_data_url_from_page(page_url: str, soup: BeautifulSoup) -> Optional[str]:
    # Localiza el mejor enlace de datos en la ficha
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
        looks_data = low_href.endswith(CSV_EXT) or low_href.endswith(JSON_EXT) or any(h in (low_href + " " + low_label) for h in PREFERRED_HINTS)
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
    # Carga robusta para CSV/JSON
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()
    # CSV
    if u.endswith(".csv") or "csv" in ctype:
        try:
            return pd.read_csv(io.StringIO(r.text), sep=None, engine="python")
        except Exception:
            for sep in (";", ",", "\t", "|"):
                try:
                    return pd.read_csv(io.StringIO(r.text), sep=sep)
                except Exception:
                    pass
            raise
    # JSON / GeoJSON
    if u.endswith(".json") or u.endswith(".geojson") or "json" in ctype:
        data = r.json()
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            return pd.json_normalize(data["features"])
        return pd.json_normalize(data)
    return None

# ----------- Ventana temporal ----------
def in_date_window(d: date_cls, target_date: Optional[date_cls], months_back: int) -> bool:
    # Verdadero si d está entre [target_date - months_back, target_date]
    if target_date is None:
        return True
    start = target_date - relativedelta(months=months_back)
    return start <= d <= target_date

# ----------- Localización de columnas ----------
def find_timestamp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(k in str(c).lower() for k in
            ["fecha_hora","fechahora","datetime","timestamp","fecha","date","hora","time"])]

def find_location_columns(df: pd.DataFrame) -> List[str]:
    keys = [
        "calle","vía","via","direccion","dirección","ubicacion","ubicación",
        "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
        "street","road","address","addr","localizacion","localización",
        "distrito","barrio","municipio","localidad","seccion","sección","zona",
        "estacion","estación","station","punto_muestreo","site"
    ]
    return [c for c in df.columns if any(k in str(c).lower() for k in keys)]

# ----------- Hora auxiliar ----------
def to_hour_minute(value) -> Optional[Tuple[int, int]]:
    # Convierte distintos formatos en (hh,mm) si procede
    if pd.isna(value): return None
    if isinstance(value, pd.Timestamp): return value.hour, value.minute
    if isinstance(value, datetime): return value.hour, value.minute
    sv = str(value).strip()
    try:
        f = float(sv.replace(",", "."))
        if 0 <= f < 24 and abs(f - round(f)) > 1e-9:
            hh = int(math.floor(f)); mm = int(round((f - hh) * 60))
            return hh, mm
    except Exception:
        pass
    s2 = sv.replace(".", ":")
    m = re.match(r"^\s*(\d{1,2})\s*[:hH]?\s*(\d{1,2})?\s*$", s2)
    if m:
        return int(m.group(1)), int(m.group(2) or 0)
    if sv.isdigit():
        hh = int(sv); 
        if 0 <= hh < 24: return hh, 0
    try:
        dt = dtparser.parse(sv, dayfirst=True, fuzzy=True)
        return dt.hour, dt.minute
    except Exception:
        return None

# ----------- Filtro fila a fila ----------
def row_matches_filters(row: pd.Series,
                        target_date: Optional[date_cls],
                        hour_mm: Optional[Tuple[int,int]],
                        location: Optional[str],
                        df_cols: List[str],
                        months_back: int) -> bool:
    # ---- Fecha/Hora ----
    if target_date or hour_mm:
        match_time = False
        for c in df_cols:
            lc = str(c).lower()
            if any(k in lc for k in ["fecha_hora","fechahora","datetime","timestamp","fecha","date","hora","time"]):
                ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                if pd.notna(ts):
                    ok_date = in_date_window(ts.date(), target_date, months_back)
                    if hour_mm:
                        hh, mm = hour_mm
                        ok_hour = (ts.hour == hh and ts.minute == mm) or (ts.hour == hh)
                    else:
                        ok_hour = True
                    if ok_date and ok_hour:
                        match_time = True
                        break
                else:
                    if hour_mm:
                        hhmm = to_hour_minute(row[c])
                        if hhmm:
                            hh, mm = hour_mm
                            ok_hour = (hhmm[0] == hh and hhmm[1] == mm) or (hhmm[0] == hh)
                            if ok_hour and target_date is None:
                                match_time = True
                                break
        if not match_time:
            return False

        # Intento adicional de fecha si no hubo timestamp claro
        if target_date:
            for c in df_cols:
                if "fecha" in str(c).lower() or "date" in str(c).lower():
                    ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                    if pd.notna(ts):
                        if not in_date_window(ts.date(), target_date, months_back):
                            return False
                        break

    # ---- Ubicación ----
    if location:
        target = normalize_text(location)
        loc_cols = find_location_columns(pd.DataFrame(columns=df_cols))
        if loc_cols:
            found_loc = False
            for c in loc_cols:
                try:
                    val = normalize_text(str(row.get(c, "")))
                    if target and target in val:
                        found_loc = True
                        break
                except Exception:
                    pass
            if not found_loc:
                return False

    return True

# ----------- AQI y contaminantes (AccuWeather) -----------
def category_from_aqi(aqi: Optional[str]) -> Optional[str]:
    # Mapea categoría desde el valor AQI (EPA)
    if aqi is None:
        return None
    try:
        v = int(aqi)
    except Exception:
        return None
    if   0 <= v <= 50:   return "Buena"
    if  51 <= v <= 100:  return "Moderada"
    if 101 <= v <= 150:  return "Perjudicial para grupos sensibles"
    if 151 <= v <= 200:  return "Mala"
    if 201 <= v <= 300:  return "Muy mala"
    if 301 <= v <= 500:  return "Peligrosa"
    return None

def parse_accuweather_aqi(html: str) -> Dict[str, Optional[str]]:
    """
    Extrae AQI y contaminantes con valor y unidad (µg/m³ o ppm si aparece).
    """
    soup = make_soup(html)
    text = soup.get_text("\n", strip=True)

    # AQI numérico
    aqi = None
    m_aqi = re.search(r"\b(\d{1,3})\s*AQI\b", text, flags=re.I)
    if m_aqi:
        aqi = m_aqi.group(1)
    aqi_cat = category_from_aqi(aqi)

    # Config de contaminantes
    pol_keys = {
        "PM 2.5": "pm25", "PM 2,5": "pm25", "PM2.5": "pm25",
        "PM 10": "pm10",  "PM10": "pm10",
        "O3": "o3", "O 3": "o3",
        "NO2": "no2", "NO 2": "no2",
        "SO2": "so2", "SO 2": "so2",
        "CO": "co",
    }

    # Columnas de salida
    out = {
        "pm25_ugm3": None, "pm10_ugm3": None, "o3_ugm3": None,
        "no2_ugm3": None, "so2_ugm3": None, "co_ugm3": None,
        "pm25_raw": None, "pm25_unit": None,
        "pm10_raw": None, "pm10_unit": None,
        "o3_raw": None,  "o3_unit": None,
        "no2_raw": None, "no2_unit": None,
        "so2_raw": None, "so2_unit": None,
        "co_raw": None,  "co_unit": None,
    }

    # Busca "valor + unidad" cerca del nombre del contaminante
    for key, short in pol_keys.items():
        patt = rf"{re.escape(key)}(.{{0,300}}?)(\d+(?:[.,]\d+)?)\s*(µg/m³|ug/m3|ppm)"
        matches = list(re.finditer(patt, text, flags=re.I | re.S))
        if not matches:
            continue
        val = matches[-1].group(2).replace(",", ".")
        unit = matches[-1].group(3).lower().replace("ug/m3", "µg/m³")
        out[f"{short}_raw"] = val
        out[f"{short}_unit"] = unit
        if unit == "µg/m³":
            out[f"{short}_ugm3"] = val

    return {"aqi": aqi, "aqi_category": aqi_cat, **out}

def scrape_accuweather_row() -> pd.DataFrame:
    # Devuelve una fila con AQI+contaminantes y fecha/hora actuales
    try:
        html = fetch_html(ACCUWEATHER_AQI)
    except Exception:
        return pd.DataFrame()
    parsed = parse_accuweather_aqi(html)
    if not any(parsed.values()):
        return pd.DataFrame()
    now = datetime.now()
    row = {
        "__fuente": ACCUWEATHER_AQI,
        "fecha": now.strftime("%Y-%m-%d"),
        "hora": now.strftime("%H:%M"),
        "origen": "AccuWeather",
        **parsed
    }
    return pd.DataFrame([row])

# ----------- Procesado de Datos Abiertos -----------
def process_page_filtered_full(page_url: str,
                               target_date: Optional[date_cls],
                               hour_mm: Optional[Tuple[int,int]],
                               location: Optional[str],
                               months_back: int) -> pd.DataFrame:
    # Descarga ficha y encuentra URL de datos
    try:
        html = fetch_html(page_url)
    except Exception:
        return pd.DataFrame()
    soup = make_soup(html)

    data_url = None
    for key, forced in OVERRIDES.items():
        if key in page_url:
            data_url = forced
            break
    if not data_url:
        data_url = find_valid_data_url_from_page(page_url, soup)
    if not data_url:
        return pd.DataFrame()

    df = load_remote_table(data_url)
    if df is None or df.empty:
        return pd.DataFrame()

    df2 = df.copy()
    df2["__fuente"] = data_url
    df2["origen"] = "DatosAbiertosMadrid"

    # Detectar fecha
    fecha_col = None
    for c in df2.columns:
        if any(k in str(c).lower() for k in ["fecha_hora","fechahora","datetime","timestamp","fecha","date"]):
            fecha_col = c
            break

    if fecha_col is not None:
        df2["__fecha_dt"] = pd.to_datetime(df2[fecha_col], errors="coerce", dayfirst=True).dt.date
    else:
        cols_low = {str(c).lower(): c for c in df2.columns}
        ycol = cols_low.get("ano") or cols_low.get("año") or cols_low.get("anio") or cols_low.get("year")
        mcol = cols_low.get("mes") or cols_low.get("month")
        dcol = cols_low.get("dia") or cols_low.get("día") or cols_low.get("day")
        if ycol and mcol and dcol:
            try:
                df2["__fecha_dt"] = pd.to_datetime(
                    df2[[ycol, mcol, dcol]].rename(columns={ycol: "Y", mcol: "M", dcol: "D"}),
                    errors="coerce"
                ).dt.date
            except Exception:
                df2["__fecha_dt"] = pd.NaT
        else:
            df2["__fecha_dt"] = pd.NaT

    # Filtro por ventana temporal (si hay target date)
    if target_date:
        mask = df2["__fecha_dt"].apply(lambda x: pd.notna(x) and in_date_window(x, target_date, months_back))
        df2 = df2[mask]

    # Filtro por ubicación si procede
    if location:
        loc_cols = find_location_columns(df2)
        if loc_cols:
            target = normalize_text(location)
            m2 = False
            for c in loc_cols:
                normcol = df2[c].astype(str).map(normalize_text)
                m2 = m2 | normcol.str.contains(target, na=False)
            df2 = df2[m2]

    return df2.reset_index(drop=True)

# ----------- Agregación ciudad por día (Datos Abiertos) -----------
def normalize_pollutant_from_row(row: pd.Series) -> Optional[str]:
    """
    Intenta inferir el contaminante estándar ('pm25','pm10','o3','no2','so2','co')
    a partir de códigos o nombres en la fila.
    """
    # Por código MAGNITUD
    for key in ["MAGNITUD", "magnitud", "cod_magnitud", "codigo_magnitud"]:
        if key in row.index:
            try:
                code = int(row[key])
                return MAGNITUD_MAP.get(code)
            except Exception:
                pass

    # Por nombre en alguna columna
    txt = " ".join([str(row.get(c, "")) for c in row.index]).lower()
    mapping = {
        "pm2.5": "pm25", "pm 2.5": "pm25", "pm 2,5": "pm25", "pm25": "pm25",
        "pm10": "pm10", "pm 10": "pm10",
        "ozono": "o3", "o3": "o3",
        "no2": "no2", "dióxido de nitrógeno": "no2", "dioxido de nitrogeno": "no2",
        "so2": "so2", "dióxido de azufre": "so2", "dioxido de azufre": "so2",
        "co": "co", "monoxido de carbono": "co", "monóxido de carbono": "co",
    }
    for k, v in mapping.items():
        if k in txt:
            return v
    return None

def extract_value_from_row(row: pd.Series) -> Optional[float]:
    """
    Extrae un valor numérico de la fila. Intenta:
      - 'valor' o 'concentracion' directas
      - si hay columnas horarias H01..H24: usa media (ignorando nulos)
    """
    # Valor/Concentración directa
    for key in ["VALOR","valor","concentracion","concentración","value","median","media"]:
        if key in row.index:
            try:
                return float(str(row[key]).replace(",", "."))
            except Exception:
                pass

    # Columnas horarias H01..H24 (o variantes)
    hours = []
    for h in range(1, 25):
        for cand in (f"H{h:02d}", f"h{h:02d}", f"hora_{h:02d}", f"hora{h:02d}", f"V{h:02d}"):
            if cand in row.index:
                try:
                    hours.append(float(str(row[cand]).replace(",", ".")))
                except Exception:
                    pass
                break
    if hours:
        vals = [v for v in hours if isinstance(v, (int, float)) and not pd.isna(v)]
        if vals:
            return float(sum(vals) / len(vals))

    return None

def build_city_daily_agg(df_open: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla ciudad por día:
      columnas: fecha, pollutant, open_value, open_unit
      - 'open_value' es media entre estaciones (si hay varias).
      - 'open_unit' se toma de OPEN_DATA_UNIT según el contaminante.
    """
    if df_open is None or df_open.empty or "__fecha_dt" not in df_open.columns:
        return pd.DataFrame(columns=["fecha","pollutant","open_value","open_unit"])

    rows = []
    for _, row in df_open.iterrows():
        pol = normalize_pollutant_from_row(row)
        if not pol:
            continue
        val = extract_value_from_row(row)
        if val is None:
            continue
        fecha = row["__fecha_dt"]
        rows.append({
            "fecha": fecha,
            "pollutant": pol,
            "value": val
        })

    if not rows:
        return pd.DataFrame(columns=["fecha","pollutant","open_value","open_unit"])

    tmp = pd.DataFrame(rows)
    agg = tmp.groupby(["fecha","pollutant"], as_index=False)["value"].mean().rename(columns={"value":"open_value"})
    agg["open_unit"] = agg["pollutant"].map(OPEN_DATA_UNIT).fillna("NA")
    return agg

# ----------- Accu → formato largo -----------
def accuweather_to_long(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la fila AccuWeather a formato largo por contaminante:
      columnas: fecha, pollutant, acc_value_ugm3, acc_raw, acc_unit, aqi, aqi_category
    """
    if df_acc is None or df_acc.empty:
        return pd.DataFrame(columns=["fecha","pollutant","acc_value_ugm3","acc_raw","acc_unit","aqi","aqi_category"])
    row = df_acc.iloc[0].to_dict()
    fecha = row.get("fecha")
    out = []
    for pol in ["pm25","pm10","o3","no2","so2","co"]:
        ug = row.get(f"{pol}_ugm3")
        raw = row.get(f"{pol}_raw")
        unit = row.get(f"{pol}_unit")
        out.append({
            "fecha": pd.to_datetime(fecha, errors="coerce").date() if fecha else None,
            "pollutant": pol,
            "acc_value_ugm3": float(ug) if ug not in (None, "NA") else None,
            "acc_raw": raw if raw not in (None, "NA") else None,
            "acc_unit": unit if unit not in (None, "NA") else None,
            "aqi": row.get("aqi"),
            "aqi_category": row.get("aqi_category"),
        })
    return pd.DataFrame(out)

# ----------- Comparación entre fuentes (con blindajes) -----------
def compare_sources(df_acc_long: pd.DataFrame, df_open_agg: pd.DataFrame,
                    target_date: Optional[date_cls], months_back: int) -> pd.DataFrame:
    # --- Normaliza df_open_agg ---
    required_cols = ["fecha", "pollutant", "open_value", "open_unit"]
    if df_open_agg is None or not isinstance(df_open_agg, pd.DataFrame) or df_open_agg.empty:
        df_open_agg = pd.DataFrame(columns=required_cols)
    else:
        for c in required_cols:
            if c not in df_open_agg.columns:
                df_open_agg[c] = pd.NA
        df_open_agg["fecha"] = pd.to_datetime(df_open_agg["fecha"], errors="coerce").dt.date

    # --- Normaliza df_acc_long ---
    acc_cols = ["fecha","pollutant","acc_value_ugm3","acc_raw","acc_unit","aqi","aqi_category"]
    if df_acc_long is None or not isinstance(df_acc_long, pd.DataFrame) or df_acc_long.empty:
        df_acc_long = pd.DataFrame(columns=acc_cols)
    else:
        for c in acc_cols:
            if c not in df_acc_long.columns:
                df_acc_long[c] = pd.NA
        df_acc_long["fecha"] = pd.to_datetime(df_acc_long["fecha"], errors="coerce").dt.date

    # --- Filtro temporal (si target_date) ---
    if target_date and not df_open_agg.empty:
        mask = df_open_agg["fecha"].apply(lambda d: pd.notna(d) and in_date_window(d, target_date, months_back))
        df_open_agg = df_open_agg[mask].copy()

    # --- Unión por fecha+contaminante (histórico = open data como base) ---
    merged = df_open_agg.merge(
        df_acc_long,
        on=["fecha","pollutant"],
        how="left",
        suffixes=("_open","_acc")
    )

    # --- Delta en µg/m³ cuando ambas fuentes están en µg/m³ ---
    def delta_row(r):
        try:
            v_open = float(r["open_value"]) if r["open_value"] not in (None, "NA", "") else None
            v_acc  = float(r["acc_value_ugm3"]) if r["acc_value_ugm3"] not in (None, "NA", "") else None
            if r.get("open_unit") == "µg/m³" and v_open is not None and v_acc is not None:
                return v_acc - v_open
            return None
        except Exception:
            return None

    if not merged.empty:
        merged["delta_ugm3_acc_minus_open"] = merged.apply(delta_row, axis=1)
        merged = merged.sort_values(["fecha","pollutant"]).reset_index(drop=True)
    else:
        merged["delta_ugm3_acc_minus_open"] = pd.Series(dtype=float)

    return merged

# ----------- Main ----------
def main():
    print("=== COMPARADOR CLIMA / CALIDAD DEL AIRE (Madrid) ===")
    user_date = input("Fecha (YYYY-MM-DD) o Enter para omitir: ").strip()
    target_date = parse_user_date(user_date)
    if user_date and target_date is None:
        print("Fecha inválida. Usa YYYY-MM-DD.")
        sys.exit(1)

    months_back = MONTHS_BACK_DEFAULT
    user_months = input(f"Meses hacia atrás (Enter={MONTHS_BACK_DEFAULT}): ").strip()
    if user_months:
        try:
            months_back = max(0, int(user_months))
        except Exception:
            print("Valor de meses inválido. Usando valor por defecto:", MONTHS_BACK_DEFAULT)
            months_back = MONTHS_BACK_DEFAULT

    location = input("Estación / Ubicación (opcional) o Enter para omitir: ").strip() or None

    # Hora opcional (se usa solo si alguna fuente tiene hora por registro)
    user_hour = input("Hora (ej: 08, 08:30, 8.5, 8,25) o Enter para omitir: ").strip()
    try:
        hour_mm = parse_user_hour(user_hour) if user_hour else None
        if hour_mm:
            hh, mm = hour_mm
            assert 0 <= hh < 24 and 0 <= mm < 60
    except Exception:
        print("Hora inválida. Usa 08, 08:30, 8.5, 8,25 …")
        sys.exit(1)

    # 1) AccuWeather (foto actual)
    acc_df = scrape_accuweather_row()

    # 2) Datos Abiertos (filtrado)
    parts = []
    for url in PAGES:
        try:
            dfp = process_page_filtered_full(url, target_date, hour_mm, location, months_back)
            if not dfp.empty:
                parts.append(dfp)
        except Exception as e:
            print(f"[Aviso] Error procesando {url}: {e}")

    open_df = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

    # ---- Log consolidado (todas las filas filtradas) ----
    log_parts = []
    if not acc_df.empty:
        log_parts.append(acc_df.copy())
    if not open_df.empty:
        # Para el log, intentar rellenar 'fecha' legible
        if "__fecha_dt" in open_df.columns and "fecha" not in open_df.columns:
            tmp = open_df.copy()
            tmp["fecha"] = tmp["__fecha_dt"].astype(str)
        else:
            tmp = open_df.copy()
        log_parts.append(tmp)

    if log_parts:
        out_log = pd.concat(log_parts, ignore_index=True, sort=False)
        # Relleno NA/espacios
        out_log = out_log.fillna("NA")
        for c in out_log.columns:
            try:
                out_log[c] = out_log[c].astype(str).replace(r"^\s*$", "NA", regex=True)
            except Exception:
                pass
        out_log.to_csv(OUT_FILE_LOG, index=False, encoding="utf-8-sig", sep=";")
        print(f"[OK] Log guardado: {OUT_FILE_LOG.resolve()}")
    else:
        pd.DataFrame(columns=["__fuente"]).to_csv(OUT_FILE_LOG, index=False, encoding="utf-8-sig", sep=";")
        print(f"[OK] Log vacío (sin datos): {OUT_FILE_LOG.resolve()}")

    # ---- Comparación por fecha/contaminante ----
    open_agg = build_city_daily_agg(open_df) if not open_df.empty else pd.DataFrame(
        columns=["fecha","pollutant","open_value","open_unit"]
    )
    acc_long = accuweather_to_long(acc_df) if not acc_df.empty else pd.DataFrame(
        columns=["fecha","pollutant","acc_value_ugm3","acc_raw","acc_unit","aqi","aqi_category"]
    )
    comp_df = compare_sources(acc_long, open_agg, target_date, months_back)

    # Columnas ordenadas para la comparación
    prefer_cols = [
        "fecha","pollutant",
        "open_value","open_unit",
        "acc_value_ugm3","acc_raw","acc_unit",
        "aqi","aqi_category",
        "delta_ugm3_acc_minus_open"
    ]
    if not comp_df.empty:
        cols = [c for c in prefer_cols if c in comp_df.columns] + [c for c in comp_df.columns if c not in prefer_cols]
        comp_df = comp_df[cols]
    else:
        comp_df = pd.DataFrame(columns=prefer_cols)

    # Guardar comparación
    comp_df_out = comp_df.copy()
    comp_df_out = comp_df_out.fillna("NA")
    for c in comp_df_out.columns:
        try:
            comp_df_out[c] = comp_df_out[c].astype(str).replace(r"^\s*$", "NA", regex=True)
        except Exception:
            pass

    comp_df_out.to_csv(OUT_FILE_COMPARE, index=False, encoding="utf-8-sig", sep=";")
    print(f"[OK] Comparación guardada: {OUT_FILE_COMPARE.resolve()}")

    # Muestra breve por consola
    try:
        print("\n=== PREVIEW COMPARACIÓN ===")
        print(comp_df.head(12))
    except Exception:
        pass

if __name__ == "__main__":
    main()
