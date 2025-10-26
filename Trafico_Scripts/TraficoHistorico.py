import re
import io
import sys
import math
import zipfile
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta

# ================== Config ==================
PAGES = [
    # 1) Enabled, supports ZIP/CSV
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=33cb30c367e78410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default", "enabled": True},
    # 2) Enabled
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=3732db7ff8cc5910VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default", "enabled": True},
    # 3) Incidents — disabled by design
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=255e0ff725b93410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default", "enabled": False},
    # 4) Enabled but strict validations (will try and skip if no valid data link)
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=02f2c23866b93410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD", "enabled": True},
]

OUTPUT_DIR = Path("./Trafico_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "traffic_filtrado.csv"

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,application/zip,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TraficoETL/3.1"
}
REQ_TIMEOUT = 60
CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")
ZIP_EXT = (".zip",)

EXCLUDE_PATTERNS = [
    "wms", "wmts", "ogc", "service", "arcgis", "esri",
    ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".pdf", ".rdf", ".xml", "sparql", "mailto:", "javascript:", "#"
]
PREFERRED_HINTS = ["download", "descarg", "csv", "json", "zip"]

# Direct URL overrides if a ficha requires a hard link (optional)
OVERRIDES: Dict[str, str] = {
    # "33cb30c367e78410": "https://.../direct.csv",
}

# Filters defaults
MONTHS_BACK_DEFAULT = 2
HOUR_MARGIN_DEFAULT_MIN = 60

# Concurrency / limits
MAX_WORKERS = 6          # tune based on your network/CPU
MAX_CSV_PER_ZIP = 8      # avoid opening dozens of internal CSVs in huge zips
MAX_MEMBER_SIZE_MB = 80  # skip very large internal members

# ================== Precompiled regex ==================
RE_FILENAME = re.compile(r'filename\s*=\s*"?([^";]+)"?', re.I)
RE_RANGE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
RE_HHMM = re.compile(r"^\s*(\d{1,2})[:hH](\d{1,2})\s*$")
RE_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ================== Utilities ==================
def normalize_text(s: str) -> str:
    """ASCII lowercase normalization for robust 'contains' matching."""
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
        hh = int(hh_str); mm = int(round(float(mm_str))) if mm_str else 0
        if mm == 60: hh, mm = hh + 1, 0
        return hh, mm
    if re.match(r"^\d+(\.\d+)?$", st):
        f = float(st); hh = int(f); mm = int(round((f - hh) * 60))
        if mm == 60: hh, mm = hh + 1, 0
        return hh, mm
    if st.isdigit():
        return int(st), 0
    raise ValueError("Formato de hora no reconocido")

def session_with_retries() -> requests.Session:
    """HTTP session with retries & keep-alive."""
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s

def fetch_html(sess: requests.Session, url: str) -> str:
    r = sess.get(url, timeout=REQ_TIMEOUT, allow_redirects=True)
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

def infer_type_from_headers(resp: requests.Response, url: str) -> str:
    ct = (resp.headers.get("Content-Type") or "").lower()
    cd = (resp.headers.get("Content-Disposition") or "")
    ul = url.lower()
    if "zip" in ct: return "zip"
    if "text/csv" in ct: return "csv"
    if "application/json" in ct or "application/geo+json" in ct: return "json"
    m = RE_FILENAME.search(cd)
    if m:
        fname = m.group(1).strip().lower()
        if fname.endswith(".zip"): return "zip"
        if fname.endswith(".csv"): return "csv"
        if fname.endswith(".json") or fname.endswith(".geojson"): return "json"
    if ul.endswith(".zip"): return "zip"
    if ul.endswith(".csv"): return "csv"
    if ul.endswith(".json") or ul.endswith(".geojson"): return "json"
    return ""

def find_data_url(sess: requests.Session, page_url: str, soup: BeautifulSoup) -> Optional[Tuple[str, str]]:
    """Find a suitable data link (csv/json/zip) in the ficha page."""
    # 1) Direct override (fast path)
    for key, forced in OVERRIDES.items():
        if key in page_url:
            r = sess.get(forced, timeout=REQ_TIMEOUT, allow_redirects=True, stream=False)
            if r.status_code < 400:
                t = infer_type_from_headers(r, str(r.url))
                if t: return str(r.url), t
            return None

    # 2) Scan <a> links that look like data
    candidates = []
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).split())
        href = absolutize(page_url, a["href"])
        low_href, low_label = href.lower(), label.lower()
        if not (low_href.startswith("http://") or low_href.startswith("https://") or low_href.startswith("//")):
            continue
        if any(pat in low_href for pat in EXCLUDE_PATTERNS):
            continue
        looks_data = low_href.endswith(CSV_EXT) or low_href.endswith(JSON_EXT) or low_href.endswith(ZIP_EXT) \
                     or any(h in (low_href + " " + low_label) for h in PREFERRED_HINTS)
        if not looks_data:
            continue
        score = 1000
        if low_href.endswith(".csv"): score -= 300
        elif low_href.endswith(".json") or low_href.endswith(".geojson"): score -= 250
        elif low_href.endswith(".zip"): score -= 200
        if "download" in low_href or "descarg" in low_href or "descarga" in low_label: score -= 150
        if not (low_href.endswith(".csv") or low_href.endswith(".json") or low_href.endswith(".geojson") or low_href.endswith(".zip")):
            score += 100
        candidates.append((score, href))

    for _, u in sorted(candidates, key=lambda t: t[0]):
        r = sess.get(u, timeout=REQ_TIMEOUT, allow_redirects=True, stream=False)
        if r.status_code >= 400:
            continue
        final_url = str(r.url)
        typ = infer_type_from_headers(r, final_url)
        if typ:
            return final_url, typ

    # 3) Fallback: buttons or nodes with data-url / data-href
    data_links = soup.select('[data-url], [data-href]')
    for node in data_links:
        href = node.get('data-url') or node.get('data-href')
        if not href:
            continue
        href = absolutize(page_url, href)
        low_href = href.lower()
        if any(p in low_href for p in EXCLUDE_PATTERNS):
            continue
        try:
            r = sess.get(href, timeout=REQ_TIMEOUT, allow_redirects=True, stream=False)
            if r.status_code < 400:
                typ = infer_type_from_headers(r, str(r.url))
                if typ in ("csv", "json", "zip"):
                    return str(r.url), typ
        except Exception:
            continue

    return None

def load_csv_text(text: str) -> pd.DataFrame:
    """Load CSV text, trying dialect inference and safe fallbacks."""
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        for sep in (";", ",", "\t", "|"):
            try:
                return pd.read_csv(io.StringIO(text), sep=sep)
            except Exception:
                pass
        raise

def load_remote_table(sess: requests.Session, url: str, typ: str, target_date: Optional[date_cls] = None) -> List[pd.DataFrame]:
    """Return a list of DataFrames (multiple if a ZIP has many CSVs)."""
    r = sess.get(url, timeout=REQ_TIMEOUT)
    r.raise_for_status()

    if typ == "csv":
        return [load_csv_text(r.text)]

    if typ == "json":
        data = r.json()
        if isinstance(data, list):
            return [pd.json_normalize(data)]
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            return [pd.json_normalize(data["features"])]
        return [pd.json_normalize(data)]

    if typ == "zip":
        dfs = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]

            # Prefer members that match the month/year of target_date (if provided)
            prefer, rest = [], members
            if target_date:
                yyyy_hint = f"{target_date.year}"
                mm_hint = f"{target_date.month:02d}"
                prefer = [m for m in members if (yyyy_hint in m.filename and mm_hint in m.filename)]
                rest = [m for m in members if m not in prefer]
            else:
                # Try to prefer YYYY-MM from URL like '09-2025.zip'
                try:
                    m = re.search(r'(\d{2})[-_](\d{4})', str(r.url))
                    if m:
                        mm_hint, yyyy_hint = m.group(1), m.group(2)
                        prefer = [memb for memb in members if yyyy_hint in memb.filename and mm_hint in memb.filename]
                        rest = [memb for memb in members if memb not in prefer]
                except Exception:
                    pass

            selected = prefer + rest
            if len(selected) > MAX_CSV_PER_ZIP:
                print(f"[Info] ZIP with {len(members)} CSVs; processing only {MAX_CSV_PER_ZIP}.")
                selected = selected[:MAX_CSV_PER_ZIP]

            for i, member in enumerate(selected, 1):
                if getattr(member, "file_size", 0) > MAX_MEMBER_SIZE_MB * 1024 * 1024:
                    print(f"[Info] Skip {member.filename} (> {MAX_MEMBER_SIZE_MB}MB)")
                    continue

                print(f"[ZIP] Reading {i}/{len(selected)}: {member.filename}")
                with zf.open(member) as f:
                    raw = f.read()
                    for enc in ("utf-8", "latin-1"):
                        try:
                            dfs.append(load_csv_text(raw.decode(enc, errors="replace")))
                            break
                        except Exception:
                            continue
        return dfs

    return []

# ======== Vectorized helpers ========
LOCATION_KEYS = [
    "calle","vía","via","direccion","dirección","ubicacion","ubicación",
    "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
    "street","road","address","addr","localizacion","localización",
    "distrito","barrio","municipio","localidad","seccion","sección","zona",
    "pk","p.k","punto_km","kilometro","kilómetro","coordenada","coordenadas"
]
TIME_KEYS = ["fecha_hora","fechahora","datetime","timestamp","fecha","date","dia","hora","time","franja","tramo"]

def is_event_level(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    has_time = any(any(k in c for k in ["hora","time","franja","tramo"]) for c in cols)
    has_date = any(any(k in c for k in ["fecha_hora","fechahora","datetime","timestamp","fecha","date","dia"]) for c in cols)
    has_month = any(c in ("mes","month") for c in cols)
    has_year  = any(c in ("año","ano","anio","year") for c in cols)
    if (has_month or has_year) and not (has_time or has_date):
        return False
    return has_time or has_date

def mid_from_range_series(s: pd.Series) -> pd.Series:
    """Vectorized midpoint from time ranges; also parses HH:MM, HH or float hours."""
    s = s.astype(str)
    r = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # Ranges HH:MM-HH:MM
    m = s.str.extract(RE_RANGE)
    mask_range = m.notna().all(axis=1)
    if mask_range.any():
        h1 = m.loc[mask_range, 0].astype(int); m1 = m.loc[mask_range, 1].astype(int)
        h2 = m.loc[mask_range, 2].astype(int); m2 = m.loc[mask_range, 3].astype(int)
        t1 = h1*60 + m1; t2 = h2*60 + m2
        mid = ((t1 + t2)//2).astype(int)
        r.loc[mask_range] = pd.to_datetime((mid//60).astype(str)+":"+(mid%60).astype(str).str.zfill(2), format="%H:%M", errors="coerce")

    # HH:MM or HhMm variants
    mask_hhmm = s.str.match(RE_HHMM)
    if mask_hhmm.any():
        s2 = (s.loc[mask_hhmm]
                .str.replace("h", ":", regex=False)
                .str.replace("H", ":", regex=False))
        r.loc[mask_hhmm] = pd.to_datetime(s2, format="%H:%M", errors="coerce")

    # HH or float hours
    mask_num = (~mask_range) & (~mask_hhmm)
    if mask_num.any():
        def num_to_time(x):
            try:
                x = str(x).replace(",", ".")
                f = float(x); hh = int(math.floor(f)); mm = int(round((f - hh) * 60))
                if mm == 60: hh, mm = hh + 1, 0
                return f"{hh:02d}:{mm:02d}"
            except Exception:
                return None
        tmp = s.loc[mask_num].map(num_to_time)
        r.loc[mask_num] = pd.to_datetime(tmp, format="%H:%M", errors="coerce")

    return r

def _fast_date_parse(series: pd.Series) -> pd.Series:
    """Faster date parsing: use explicit format when possible, else fallback with dayfirst."""
    s = series.astype(str).str.strip()
    iso_mask = s.str.match(RE_ISO_DATE)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso_mask.any():
        out.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], format="%Y-%m-%d", errors="coerce")
    if (~iso_mask).any():
        out.loc[~iso_mask] = pd.to_datetime(s.loc[~iso_mask], errors="coerce", dayfirst=True)
    return out

def build_ts(df: pd.DataFrame) -> pd.Series:
    """Create a best-effort timestamp combining date + hour-like fields."""
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]

    # direct datetime-like columns
    for i, c in enumerate(cols):
        if any(k in low[i] for k in ["fecha_hora","fechahora","datetime","timestamp"]):
            ts = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if ts.notna().any():
                return ts

    # split date + hour-like
    fecha_col = next((cols[i] for i, c in enumerate(low) if any(k in c for k in ["fecha","date","dia"])), None)
    hora_col  = next((cols[i] for i, c in enumerate(low) if any(k in c for k in ["hora","time","franja","tramo"])), None)

    if fecha_col and hora_col:
        hrs = mid_from_range_series(df[hora_col])
        date_parsed = _fast_date_parse(df[fecha_col])
        time_str = hrs.dt.strftime("%H:%M").fillna("00:00")
        return pd.to_datetime(date_parsed.dt.strftime("%Y-%m-%d") + " " + time_str, errors="coerce")

    if fecha_col:
        return _fast_date_parse(df[fecha_col])

    # last resort: NA series
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

def filter_df(df: pd.DataFrame,
              target_date: Optional[date_cls],
              hour_mm: Optional[Tuple[int,int]],
              hour_margin_min: int,
              location: Optional[str],
              months_back: int) -> pd.DataFrame:
    """Vectorized filtering by date window, hour margin, and location."""
    if df is None or df.empty:
        return df

    # Early reject if event-level required but dataset is aggregated
    if (target_date or hour_mm) and not is_event_level(df):
        return pd.DataFrame()

    # Build timestamp vectorized once
    ts = build_ts(df)
    mask = pd.Series(True, index=df.index)

    # Date window (compare Timestamps, normalized to discard time)
    if target_date:
        start = pd.Timestamp(target_date - relativedelta(months=months_back))
        end   = pd.Timestamp(target_date)
        ts_norm = ts.dt.normalize()
        mask &= ((ts_norm >= start) & (ts_norm <= end)) | ts.isna()

    # Hour margin if requested
    if hour_mm:
        tgt_min = hour_mm[0]*60 + hour_mm[1]
        if ts.notna().any():
            mins = ts.dt.hour*60 + ts.dt.minute
            mask &= (mins - tgt_min).abs() <= hour_margin_min
        else:
            # Fallback: try all hour-like columns (OR across them)
            low_map = {str(c).lower(): c for c in df.columns}
            hour_cols = [orig for lowc, orig in low_map.items() if any(k in lowc for k in ["hora","time","franja","tramo"])]
            if hour_cols:
                cond = pd.Series(False, index=df.index)
                for hc in hour_cols:
                    mid = mid_from_range_series(df[hc])
                    mmins = mid.dt.hour*60 + mid.dt.minute
                    cond |= ((mmins - tgt_min).abs() <= hour_margin_min)
                mask &= cond

    # Location contains (normalized in any location-like column)
    if location:
        target_norm = normalize_text(location)
        loc_cols = [c for c in df.columns if any(k in str(c).lower() for k in LOCATION_KEYS)]
        if not loc_cols:
            return pd.DataFrame()
        joined = (
            df[loc_cols]
            .astype(str)
            .applymap(normalize_text)
            .agg(" ".join, axis=1)
        )
        mask &= joined.str.contains(re.escape(target_norm), na=False)

    out = df.loc[mask]
    return out

# =========== Page processing (parallel) ===========
def process_page(sess: requests.Session,
                 page_url: str,
                 target_date: Optional[date_cls],
                 hour_mm: Optional[Tuple[int,int]],
                 hour_margin_min: int,
                 location: Optional[str],
                 months_back: int) -> pd.DataFrame:
    try:
        html = fetch_html(sess, page_url)
    except Exception as e:
        print(f"[Error] Ficha {page_url}: {e}")
        return pd.DataFrame()

    soup = make_soup(html)
    data = find_data_url(sess, page_url, soup)
    if not data:
        print(f"[Aviso] Sin URL de datos válida en {page_url}")
        return pd.DataFrame()

    data_url, typ = data
    print(f"[Datos] {data_url} (tipo: {typ})")
    try:
        dfs = load_remote_table(sess, data_url, typ, target_date)
    except Exception as e:
        print(f"[Error] Carga fallida {data_url}: {e}")
        return pd.DataFrame()

    kept = []
    for df in dfs:
        if df is None or df.empty:
            continue
        filtered = filter_df(df, target_date, hour_mm, hour_margin_min, location, months_back)
        if not filtered.empty:
            filtered = filtered.copy()
            filtered["__fuente"] = data_url
            kept.append(filtered)

    if kept:
        res = pd.concat(kept, ignore_index=True, sort=False)
        print(f"[OK] {page_url}: {len(res)} filas tras filtro")
        return res
    else:
        print(f"[OK] {page_url}: 0 filas tras filtro")
        return pd.DataFrame()

# ================== Main ==================
def main():
    print("=== FILTRO TRÁFICO (Madrid) — rápido/vectorizado ===")

    user_date = input("Fecha (YYYY-MM-DD) o Enter para omitir: ").strip()
    target_date = parse_user_date(user_date)
    if user_date and target_date is None:
        print("Fecha inválida. Usa YYYY-MM-DD.")
        sys.exit(1)

    months_back = MONTHS_BACK_DEFAULT
    user_months = input(f"Margen hacia atrás en meses (Enter={MONTHS_BACK_DEFAULT}): ").strip()
    if user_months:
        try:
            months_back = max(0, int(user_months))
        except Exception:
            print("Valor de meses inválido. Usando valor por defecto:", MONTHS_BACK_DEFAULT)
            months_back = MONTHS_BACK_DEFAULT

    location = input("Zona/Ubicación (calle, tramo, distrito/barrio) o Enter para omitir: ").strip() or None

    user_hour = input("Hora aprox (ej: 08, 08:30, 8.5) o Enter para omitir: ").strip()
    try:
        hour_mm = parse_user_hour(user_hour) if user_hour else None
        if hour_mm:
            hh, mm = hour_mm
            assert 0 <= hh < 24 and 0 <= mm < 60
    except Exception:
        print("Hora inválida. Usa 08, 08:30, 8.5 …")
        sys.exit(1)

    user_margin = input(f"Margen horario ±min (Enter={HOUR_MARGIN_DEFAULT_MIN}): ").strip()
    try:
        hour_margin_min = int(user_margin) if user_margin else HOUR_MARGIN_DEFAULT_MIN
        if hour_margin_min < 0: hour_margin_min = HOUR_MARGIN_DEFAULT_MIN
    except Exception:
        hour_margin_min = HOUR_MARGIN_DEFAULT_MIN

    sess = session_with_retries()

    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for cfg in PAGES:
            if not cfg.get("enabled", True):
                print(f"[Omitida] {cfg['url']} (deshabilitada)")
                continue
            futures.append(
                ex.submit(
                    process_page, sess, cfg["url"], target_date, hour_mm, hour_margin_min, location, months_back
                )
            )
        for f in as_completed(futures):
            try:
                dfp = f.result()
                if dfp is not None and not dfp.empty:
                    results.append(dfp)
            except Exception as e:
                print(f"[Error inesperado] {e}")

    if not results:
        print("\n[Resultado] 0 filas cumplen los filtros.")
        pd.DataFrame(columns=["__fuente"]).to_csv(OUT_FILE, index=False, encoding="utf-8-sig", sep=";")
        print(f"[OK] Archivo vacío con cabecera: {OUT_FILE.resolve()}")
        return

    out = pd.concat(results, ignore_index=True, sort=False)

    # Fill NA and empty strings → "NA"
    out = out.fillna("NA")
    for c in out.columns:
        try:
            out[c] = out[c].astype(str).replace(r"^\s*$", "NA", regex=True)
        except Exception:
            pass

    # Order by best-effort timestamp (re-compute on merged frame)
    try:
        ts = build_ts(out)
        if ts is not None:
            out = out.iloc[ts.sort_values().index]
    except Exception:
        pass

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", sep=";")
    print(f"\n[OK] Guardado: {OUT_FILE.resolve()}")
    print(f"[Filas] {len(out)}  [Columnas] {len(out.columns)}")
    try:
        print(out.head(10))
    except Exception:
        pass

if __name__ == "__main__":
    main()
