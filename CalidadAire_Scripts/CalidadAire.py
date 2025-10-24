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

USE_SELENIUM_FALLBACK = True

SELENIUM_AVAILABLE = False
try:
    if USE_SELENIUM_FALLBACK:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        import traceback
        SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False

# Adjust this path to your ChromeDriver location if you plan to use Selenium
CHROMEDRIVER_PATH = r"C:/Users/Medias/.wdm/drivers/chromedriver/win64/141.0.7390.123/chromedriver-win32/chromedriver.exe"

# ============== Sources ==============
ACCUWEATHER_AQI = "https://www.accuweather.com/es/es/madrid/308526/air-quality-index/308526"
PAGES = [
    # Madrid Open Data ficha (daily air quality)
    "https://datos.madrid.es/sites/v/index.jsp?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD",
]

# ============== Output ==============
OUTPUT_DIR = Path("./CalidadAire_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE_LOG = OUTPUT_DIR / "calidad_aire_filtrado.csv"
OUT_FILE_COMPARE = OUTPUT_DIR / "calidad_aire_comparado.csv"

# ============== Network ==============
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ClimaComparer/1.0"
}
REQ_TIMEOUT = 45
CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

EXCLUDE_PATTERNS = [
    "wms", "wmts", "ogc", "service", "arcgis", "esri",
    ".zip", ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".pdf", ".rdf", ".xml", "sparql", "mailto:", "javascript:", "#"
]
PREFERRED_HINTS = ["download", "descarg", "csv", "json"]

OVERRIDES: Dict[str, str] = {
    "aecb88a7e2b73410": "https://datos.madrid.es/egob/catalogo/201410-10306624-calidad-aire-diario.csv"
}


MONTHS_BACK_DEFAULT = 2

MAGNITUD_MAP = {
    1:  "so2",
    6:  "co",
    7:  "no",
    8:  "no2",
    9:  "o3",
    10: "pm10",
    12: "pm25",
}
OPEN_DATA_UNIT = {
    "so2": "µg/m³",
    "co":  "mg/m³",
    "no":  "µg/m³",
    "no2": "µg/m³",
    "o3":  "µg/m³",
    "pm10":"µg/m³",
    "pm25":"µg/m³",
}

# ============== Text/Date Utils ==============
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def parse_user_date(s: str) -> Optional[date_cls]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return dtparser.parse(s, dayfirst=False).date()
    except Exception:
        return None

def parse_user_hour(s: str) -> Optional[Tuple[int, int]]:
    st = (s or "").strip()
    if not st:
        return None
    st = st.lower().replace(" ", "").replace(",", ".").replace("h", ":")
    if ":" in st:
        hh_str, mm_str = st.split(":", 1)
        if not hh_str:
            raise ValueError("Invalid hour")
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
    raise ValueError("Unrecognized hour format")

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
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(base_url, href)
    return href

def head_or_get_headers(url: str) -> Optional[requests.Response]:
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

# ============== Time Window ==============
def in_date_window(d: date_cls, target_date: Optional[date_cls], months_back: int) -> bool:
    if target_date is None:
        return True
    start = target_date - relativedelta(months=months_back)
    return start <= d <= target_date

# ============== Column Helpers ==============
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

# ============== Hour Utils ==============
def to_hour_minute(value) -> Optional[Tuple[int, int]]:
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
        hh = int(sv)
        if 0 <= hh < 24: return hh, 0
    try:
        dt = dtparser.parse(sv, dayfirst=True, fuzzy=True)
        return dt.hour, dt.minute
    except Exception:
        return None

# ============== Row Filter ==============
def row_matches_filters(row: pd.Series,
                        target_date: Optional[date_cls],
                        hour_mm: Optional[Tuple[int,int]],
                        location: Optional[str],
                        df_cols: List[str],
                        months_back: int) -> bool:
    # Date/Time
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

        # Additional date check
        if target_date:
            for c in df_cols:
                if "fecha" in str(c).lower() or "date" in str(c).lower():
                    ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                    if pd.notna(ts):
                        if not in_date_window(ts.date(), target_date, months_back):
                            return False
                        break

    # Location
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

# ============== AQI Category ==============
def category_from_aqi(aqi: Optional[str]) -> Optional[str]:
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

# ============== Selenium Helpers ==============
def _map_by_key(by_key: str):
    """Map a simple string to Selenium's By.* constant."""
    if not SELENIUM_AVAILABLE:
        return None
    by_key = (by_key or "").strip().lower()
    mapping = {
        "id": By.ID,
        "name": By.NAME,
        "xpath": By.XPATH,
        "css_selector": By.CSS_SELECTOR,
        "css": By.CSS_SELECTOR,
        "class_name": By.CLASS_NAME,
        "link_text": By.LINK_TEXT,
        "partial_link_text": By.PARTIAL_LINK_TEXT,
        "tag_name": By.TAG_NAME,
    }
    return mapping.get(by_key)

def fetch_air_quality_index_selenium(url: str, selector: dict) -> Optional[str]:
    """
    Minimal Selenium fetch to get dynamic AQI when static HTML parsing fails.
    selector example: {"by": "class_name", "value": "aq-number"}
    """
    if not SELENIUM_AVAILABLE:
        return None

    chrome_options = Options()
    # Uncomment headless if you prefer no browser UI:
    # chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=chrome_options)
    try:
        driver.get(url)
        by = _map_by_key(selector.get("by", "class_name"))
        value = selector.get("value", "aq-number")
        if by is None:
            return None
        aqi_element = WebDriverWait(driver, 40).until(
            EC.visibility_of_element_located((by, value))
        )
        txt = (aqi_element.text or "").strip()
        return txt if txt else None
    except Exception as e:
        print(f"[Selenium] Error fetching AQI: {e}")
        traceback.print_exc()
        return None
    finally:
        driver.quit()

# ============== AccuWeather Parsing ==============
def parse_accuweather_aqi(html: str) -> Dict[str, Optional[str]]:
    """
    Extract AQI and pollutants (value + unit) from AccuWeather page text.
    """
    soup = make_soup(html)
    text = soup.get_text("\n", strip=True)

    # Numeric AQI
    aqi = None
    m_aqi = re.search(r"\b(\d{1,3})\s*AQI\b", text, flags=re.I)
    if m_aqi:
        aqi = m_aqi.group(1)
    aqi_cat = category_from_aqi(aqi)

    pol_keys = {
        "PM 2.5": "pm25", "PM 2,5": "pm25", "PM2.5": "pm25",
        "PM 10": "pm10",  "PM10": "pm10",
        "O3": "o3", "O 3": "o3",
        "NO2": "no2", "NO 2": "no2",
        "SO2": "so2", "SO 2": "so2",
        "CO": "co",
    }

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

    # Capture "value + unit" near pollutant name
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
    """
    Try static HTML parse first. If no data is found and Selenium fallback is enabled
    and available, try to fetch at least the AQI using Selenium.
    """
    parsed = {}
    try:
        html = fetch_html(ACCUWEATHER_AQI)
        parsed = parse_accuweather_aqi(html)
    except Exception:
        parsed = {}

    # If static parse yielded nothing, try Selenium AQI fallback
    if not any(parsed.values()) and USE_SELENIUM_FALLBACK and SELENIUM_AVAILABLE:
        aqi_val = fetch_air_quality_index_selenium(
            ACCUWEATHER_AQI,
            {"by": "class_name", "value": "aq-number"}  # Known AQI class in AccuWeather
        )
        if aqi_val:
            parsed = {"aqi": aqi_val, "aqi_category": category_from_aqi(aqi_val)}

    # If still nothing, return empty
    if not parsed or not any(parsed.values()):
        return pd.DataFrame()

    now = datetime.now()
    row = {
        "__fuente": ACCUWEATHER_AQI,
        "fecha": now.strftime("%Y-%m-%d"),
        "hora": now.strftime("%H:%M"),
        "origen": "AccuWeather",
        **{
            # Ensure all pollutant fields exist in the output schema
            "aqi": parsed.get("aqi"),
            "aqi_category": parsed.get("aqi_category"),
            "pm25_ugm3": parsed.get("pm25_ugm3"),
            "pm10_ugm3": parsed.get("pm10_ugm3"),
            "o3_ugm3": parsed.get("o3_ugm3"),
            "no2_ugm3": parsed.get("no2_ugm3"),
            "so2_ugm3": parsed.get("so2_ugm3"),
            "co_ugm3": parsed.get("co_ugm3"),
            "pm25_raw": parsed.get("pm25_raw"),
            "pm25_unit": parsed.get("pm25_unit"),
            "pm10_raw": parsed.get("pm10_raw"),
            "pm10_unit": parsed.get("pm10_unit"),
            "o3_raw": parsed.get("o3_raw"),
            "o3_unit": parsed.get("o3_unit"),
            "no2_raw": parsed.get("no2_raw"),
            "no2_unit": parsed.get("no2_unit"),
            "so2_raw": parsed.get("so2_raw"),
            "so2_unit": parsed.get("so2_unit"),
            "co_raw": parsed.get("co_raw"),
            "co_unit": parsed.get("co_unit"),
        }
    }
    return pd.DataFrame([row])

# ============== Open Data Processing ==============
def process_page_filtered_full(page_url: str,
                               target_date: Optional[date_cls],
                               hour_mm: Optional[Tuple[int,int]],
                               location: Optional[str],
                               months_back: int) -> pd.DataFrame:
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

    # Date detection
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

    if target_date:
        mask = df2["__fecha_dt"].apply(lambda x: pd.notna(x) and in_date_window(x, target_date, months_back))
        df2 = df2[mask]

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

# ============== City Daily Aggregation ==============
def normalize_pollutant_from_row(row: pd.Series) -> Optional[str]:
    for key in ["MAGNITUD", "magnitud", "cod_magnitud", "codigo_magnitud"]:
        if key in row.index:
            try:
                code = int(row[key])
                return MAGNITUD_MAP.get(code)
            except Exception:
                pass
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
    for key in ["VALOR","valor","concentracion","concentración","value","median","media"]:
        if key in row.index:
            try:
                return float(str(row[key]).replace(",", "."))
            except Exception:
                pass

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

# ============== Accu → Long Format ==============
def accuweather_to_long(df_acc: pd.DataFrame) -> pd.DataFrame:
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

# ============== Compare Sources ==============
def compare_sources(df_acc_long: pd.DataFrame, df_open_agg: pd.DataFrame,
                    target_date: Optional[date_cls], months_back: int) -> pd.DataFrame:
    required_cols = ["fecha", "pollutant", "open_value", "open_unit"]
    if df_open_agg is None or not isinstance(df_open_agg, pd.DataFrame) or df_open_agg.empty:
        df_open_agg = pd.DataFrame(columns=required_cols)
    else:
        for c in required_cols:
            if c not in df_open_agg.columns:
                df_open_agg[c] = pd.NA
        df_open_agg["fecha"] = pd.to_datetime(df_open_agg["fecha"], errors="coerce").dt.date

    acc_cols = ["fecha","pollutant","acc_value_ugm3","acc_raw","acc_unit","aqi","aqi_category"]
    if df_acc_long is None or not isinstance(df_acc_long, pd.DataFrame) or df_acc_long.empty:
        df_acc_long = pd.DataFrame(columns=acc_cols)
    else:
        for c in acc_cols:
            if c not in df_acc_long.columns:
                df_acc_long[c] = pd.NA
        df_acc_long["fecha"] = pd.to_datetime(df_acc_long["fecha"], errors="coerce").dt.date

    if target_date and not df_open_agg.empty:
        mask = df_open_agg["fecha"].apply(lambda d: pd.notna(d) and in_date_window(d, target_date, months_back))
        df_open_agg = df_open_agg[mask].copy()

    merged = df_open_agg.merge(
        df_acc_long,
        on=["fecha","pollutant"],
        how="left",
        suffixes=("_open","_acc")
    )

    def delta_row(r):
        try:
            v_open = float(r["open_value"]) if r["open_value"] not in (None, "NA", "") else None
            v_acc  = float(r["acc_value_ugm3"]) if r["acc_value_ugm3"] not in (None, "NA", "") else None
            if r.get("open_unit") == "µg/m³" and v_open is not None and v_acc is not None:
                return v_acc - v_open
            return None
        except Exception:
            return None

        # If open data is empty but we have AccuWeather, return AccuWeather rows
    if merged.empty and not df_acc_long.empty:
        acc_only = df_acc_long.copy()
        acc_only["open_value"] = pd.NA
        acc_only["open_unit"] = pd.NA
        acc_only["delta_ugm3_acc_minus_open"] = pd.NA
        # Orden final consistente
        cols = ["fecha","pollutant","open_value","open_unit",
                "acc_value_ugm3","acc_raw","acc_unit",
                "aqi","aqi_category","delta_ugm3_acc_minus_open"]
        for c in cols:
            if c not in acc_only.columns:
                acc_only[c] = pd.NA
        return acc_only[cols]

    return merged


# ============== Main ==============
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

    user_hour = input("Hora (ej: 08, 08:30, 8.5, 8,25) o Enter para omitir: ").strip()
    try:
        hour_mm = parse_user_hour(user_hour) if user_hour else None
        if hour_mm:
            hh, mm = hour_mm
            assert 0 <= hh < 24 and 0 <= mm < 60
    except Exception:
        print("Hora inválida. Usa 08, 08:30, 8.5, 8,25 …")
        sys.exit(1)

    # 1) AccuWeather (current snapshot) with Selenium fallback
    acc_df = scrape_accuweather_row()

    # 2) Open Data (filtered)
    parts = []
    for url in PAGES:
        try:
            dfp = process_page_filtered_full(url, target_date, hour_mm, location, months_back)
            if not dfp.empty:
                parts.append(dfp)
        except Exception as e:
            print(f"[Aviso] Error procesando {url}: {e}")

    open_df = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

    # ---- Log consolidated ----
    log_parts = []
    if not acc_df.empty:
        log_parts.append(acc_df.copy())
    if not open_df.empty:
        if "__fecha_dt" in open_df.columns and "fecha" not in open_df.columns:
            tmp = open_df.copy()
            tmp["fecha"] = tmp["__fecha_dt"].astype(str)
        else:
            tmp = open_df.copy()
        log_parts.append(tmp)

    if log_parts:
        out_log = pd.concat(log_parts, ignore_index=True, sort=False)
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

    # ---- Comparison (date/pollutant) ----
    open_agg = build_city_daily_agg(open_df) if not open_df.empty else pd.DataFrame(
        columns=["fecha","pollutant","open_value","open_unit"]
    )
    acc_long = accuweather_to_long(acc_df) if not acc_df.empty else pd.DataFrame(
        columns=["fecha","pollutant","acc_value_ugm3","acc_raw","acc_unit","aqi","aqi_category"]
    )
    comp_df = compare_sources(acc_long, open_agg, target_date, months_back)

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

    comp_df_out = comp_df.copy()
    comp_df_out = comp_df_out.fillna("NA")
    for c in comp_df_out.columns:
        try:
            comp_df_out[c] = comp_df_out[c].astype(str).replace(r"^\s*$", "NA", regex=True)
        except Exception:
            pass

    comp_df_out.to_csv(OUT_FILE_COMPARE, index=False, encoding="utf-8-sig", sep=";")
    print(f"[OK] Comparación guardada: {OUT_FILE_COMPARE.resolve()}")

    try:
        print("\n=== PREVIEW COMPARACIÓN ===")
        print(comp_df.head(12))
    except Exception:
        pass

if __name__ == "__main__":
    main()
