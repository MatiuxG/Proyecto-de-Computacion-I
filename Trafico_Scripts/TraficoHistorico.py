
import re
import io
import sys
import math
import zipfile
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, date as date_cls, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta

# ----------- Fichas de tráfico (configurable) -----------
PAGES = [
    # 1) Habilitada, soporta ZIP
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=33cb30c367e78410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default", "enabled": True},
    # 2) Habilitada
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=3732db7ff8cc5910VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default", "enabled": True},
    # 3) Incidentes — deshabilitada
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=255e0ff725b93410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default", "enabled": False},
    # 4) Habilitada pero con validaciones estrictas; si no sirve para filtros pedidos, se omite
    {"url": "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=02f2c23866b93410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD", "enabled": True},
]

# ----------- Salida -----------
OUTPUT_DIR = Path("./Trafico_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "traffic_filtrado.csv"

# ----------- Red -----------
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,application/zip,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TraficoETL/2.0"
}
REQ_TIMEOUT = 60
CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")
ZIP_EXT = (".zip",)

# Excluir ruido
EXCLUDE_PATTERNS = [
    "wms", "wmts", "ogc", "service", "arcgis", "esri",
    ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".pdf", ".rdf", ".xml", "sparql", "mailto:", "javascript:", "#"
]
PREFERRED_HINTS = ["download", "descarg", "csv", "json", "zip"]

# Overrides si alguna ficha requiere URL directa (opcional)
OVERRIDES: Dict[str, str] = {
    # "33cb30c367e78410": "https://.../zip_o_csv_directo.zip",
}

# ----------- Config de filtros -----------
MONTHS_BACK_DEFAULT = 2
HOUR_MARGIN_DEFAULT_MIN = 60  # margen horario ± minutos

# ----------- Utilidades ----------
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
        if not hh_str: raise ValueError("Hora inválida")
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

def head_or_get_headers(url: str):
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

def content_says_data(resp: requests.Response, url: str) -> str:
    """
    Devuelve 'csv' | 'json' | 'zip' | '' según el tipo detectado (headers + filename).
    """
    ct = (resp.headers.get("Content-Type") or "").lower()
    cd = (resp.headers.get("Content-Disposition") or "")
    ul = url.lower()
    # Por headers
    if "zip" in ct: return "zip"
    if "text/csv" in ct: return "csv"
    if "application/json" in ct or "application/geo+json" in ct: return "json"
    # Por filename en Content-Disposition
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', cd, flags=re.I)
    if m:
        fname = m.group(1).strip().lower()
        if fname.endswith(".zip"): return "zip"
        if fname.endswith(".csv"): return "csv"
        if fname.endswith(".json") or fname.endswith(".geojson"): return "json"
    # Por URL
    if ul.endswith(".zip"): return "zip"
    if ul.endswith(".csv"): return "csv"
    if ul.endswith(".json") or ul.endswith(".geojson"): return "json"
    return ""

def find_valid_data_url_from_page(page_url: str, soup: BeautifulSoup) -> Optional[Tuple[str,str]]:
    """
    Retorna (url, tipo) donde tipo es 'csv' | 'json' | 'zip'
    """
    for key, forced in OVERRIDES.items():
        if key in page_url:
            # No sabemos tipo -> inferir
            resp = head_or_get_headers(forced)
            if resp is None:
                return None
            typ = content_says_data(resp, str(resp.url))
            return str(resp.url), typ or "csv"

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
        elif low_href.endswith(".zip"): score -= 200  # preferimos CSV, pero ZIP es válido
        if "download" in low_href or "descarg" in low_href or "descarga" in low_label: score -= 150
        if not (low_href.endswith(".csv") or low_href.endswith(".json") or low_href.endswith(".geojson") or low_href.endswith(".zip")):
            score += 100
        candidates.append((score, href))

    for _, u in sorted(candidates, key=lambda t: t[0]):
        resp = head_or_get_headers(u)
        if resp is None:
            continue
        final_url = str(resp.url)
        typ = content_says_data(resp, final_url)
        if typ:
            return final_url, typ
    return None

def load_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        for sep in (";", ",", "\t", "|"):
            try:
                return pd.read_csv(io.StringIO(text), sep=sep)
            except Exception:
                pass
        raise

def load_remote_table(url: str, typ: str) -> List[pd.DataFrame]:
    """
    Devuelve una lista de DataFrames (varios si ZIP trae múltiples CSV).
    """
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
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
        # Abrir ZIP en memoria y leer todos los CSV internos
        dfs = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for name in zf.namelist():
                low = name.lower()
                if low.endswith(".csv"):
                    with zf.open(name) as f:
                        raw = f.read().decode("utf-8", errors="replace")
                        try:
                            dfs.append(load_csv_text(raw))
                        except Exception:
                            # reintentos con latin-1 si hiciera falta
                            raw2 = f.read().decode("latin-1", errors="replace") if hasattr(f, "read") else raw
                            try:
                                dfs.append(load_csv_text(raw2))
                            except Exception:
                                pass
        return dfs

    return []

def is_event_level(df: pd.DataFrame) -> bool:
    cols = [c.lower() for c in df.columns]
    has_time = any("hora" in c or "time" in c or "franja" in c or "tramo" in c for c in cols)
    has_date = any(x in c for c in cols for x in ["fecha_hora","fechahora","datetime","timestamp","fecha","date","dia"])
    has_month = any(c == "mes" or c == "month" for c in cols)
    has_year  = any(c in ["año","ano","anio","year"] for c in cols)
    if (has_month or has_year) and not (has_time or has_date):
        return False
    return has_time or has_date

def find_timestamp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(k in str(c).lower() for k in
            ["fecha_hora","fechahora","datetime","timestamp","fecha","date","dia","hora","time","franja","tramo"])]

def find_location_columns(df: pd.DataFrame) -> List[str]:
    keys = [
        "calle","vía","via","direccion","dirección","ubicacion","ubicación",
        "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
        "street","road","address","addr","localizacion","localización",
        "distrito","barrio","municipio","localidad","seccion","sección","zona",
        "pk","p.k","punto_km","kilometro","kilómetro","coordenada","coordenadas"
    ]
    return [c for c in df.columns if any(k in str(c).lower() for k in keys)]

def parse_hour_cell(cell) -> Optional[Tuple[int,int]]:
    """
    Devuelve (hh,mm) si la celda representa una hora (o el centro de un intervalo HH:MM-HH:MM).
    """
    if pd.isna(cell): return None
    s = str(cell).strip()
    # Intervalo HH:MM-HH:MM => tomar el punto medio
    miv = re.match(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$", s)
    if miv:
        h1, m1, h2, m2 = map(int, miv.groups())
        t1 = h1*60 + m1
        t2 = h2*60 + m2
        mid = (t1 + t2)//2
        return mid//60, mid%60
    # HH:MM
    m = re.match(r"^\s*(\d{1,2})[:hH](\d{1,2})\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # HH
    if s.isdigit():
        return int(s), 0
    # HH.dec
    try:
        f = float(s.replace(",", "."))
        hh = int(math.floor(f)); mm = int(round((f - hh) * 60))
        if mm == 60: hh, mm = hh + 1, 0
        return hh, mm
    except Exception:
        pass
    # parse natural
    try:
        p = dtparser.parse(s, dayfirst=True, fuzzy=True)
        return p.hour, p.minute
    except Exception:
        return None

def hour_within_margin(cell_value, target_hhmm: Tuple[int,int], margin_min: int) -> bool:
    hhmm = parse_hour_cell(cell_value)
    if not hhmm: return False
    t = hhmm[0]*60 + hhmm[1]
    tgt = target_hhmm[0]*60 + target_hhmm[1]
    return abs(t - tgt) <= margin_min

def in_date_window(d: date_cls, target_date: Optional[date_cls], months_back: int) -> bool:
    if target_date is None:
        return True
    start = target_date - relativedelta(months=months_back)
    return start <= d <= target_date

def row_matches_filters(
    row: pd.Series,
    target_date: Optional[date_cls],
    hour_mm: Optional[Tuple[int, int]],
    hour_margin_min: int,
    location: Optional[str],
    df_cols: List[str],
    months_back: int
) -> bool:
    # -------- Fecha/Hora --------
    if target_date or hour_mm:
        match_time = False
        for c in df_cols:
            lc = str(c).lower()
            if any(k in lc for k in ["fecha_hora","fechahora","datetime","timestamp","fecha","date","dia","hora","time","franja","tramo"]):
                ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                if pd.notna(ts):
                    ok_date = in_date_window(ts.date(), target_date, months_back)
                    ok_hour = hour_within_margin(ts.strftime("%H:%M"), hour_mm, hour_margin_min) if hour_mm else True
                    if ok_date and ok_hour:
                        match_time = True
                        break
                else:
                    # Hora o franja no parseable como datetime -> intentar solo hora/tramo
                    if hour_mm and any(k in lc for k in ["hora","time","franja","tramo"]):
                        if hour_within_margin(row[c], hour_mm, hour_margin_min):
                            # fecha se validará más abajo si hay columna fecha en otra parte
                            if target_date is None:
                                match_time = True
                                break
        if not match_time:
            return False

        if target_date:
            # Validar ventana de fecha si existe alguna columna de fecha explícita
            seen_date_col = False
            for c in df_cols:
                if any(k in str(c).lower() for k in ["fecha","date","dia","fecha_hora","datetime","timestamp"]):
                    ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                    if pd.notna(ts):
                        seen_date_col = True
                        if not in_date_window(ts.date(), target_date, months_back):
                            return False
                        break
            # Si no existe fecha explícita, no descartamos.

    # -------- Zona --------
    if location:
        target = normalize_text(location)
        loc_cols = [c for c in df_cols if any(k in str(c).lower() for k in
                   ["calle","vía","via","direccion","dirección","ubicacion","ubicación",
                    "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
                    "street","road","address","addr","localizacion","localización",
                    "distrito","barrio","municipio","localidad","seccion","sección","zona",
                    "pk","p.k","punto_km","kilometro","kilómetro","coordenada","coordenadas"])]
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
        else:
            # No hay columnas de zona -> este dataset no sirve para el filtro solicitado
            return False

    return True

# -------- Proceso por ficha --------
def process_page_filtered_full(page_url: str,
                               target_date: Optional[date_cls],
                               hour_mm: Optional[Tuple[int,int]],
                               hour_margin_min: int,
                               location: Optional[str],
                               months_back: int) -> pd.DataFrame:
    print(f"\n[Procesando] {page_url}")
    try:
        html = fetch_html(page_url)
    except Exception as e:
        print(f"  [Error] al abrir ficha: {e}")
        return pd.DataFrame()
    soup = make_soup(html)

    data = find_valid_data_url_from_page(page_url, soup)
    if not data:
        print("  [Aviso] sin URL de datos válida")
        return pd.DataFrame()
    data_url, typ = data
    print(f"  [Datos] {data_url} (tipo: {typ})")

    dfs = load_remote_table(data_url, typ)
    if not dfs:
        print("  [Aviso] descarga vacía")
        return pd.DataFrame()

    # Filtrado dataset por dataset
    kept_frames = []
    for i, df in enumerate(dfs, 1):
        if df is None or df.empty:
            continue

        # Si el dataset no tiene fecha/hora y pediste alguno de esos filtros, se omite.
        if (target_date or hour_mm) and not is_event_level(df):
            print(f"  [Skip] dataset #{i} agregado/sin fecha-hora (no sirve para filtros de fecha/hora).")
            continue

        # Si pediste zona y no tiene columnas de zona, se omite.
        if location:
            if not find_location_columns(df):
                print(f"  [Skip] dataset #{i} sin columnas de zona (no sirve para filtro de ubicación).")
                continue

        cols = list(df.columns)
        kept = []
        for _, row in df.iterrows():
            try:
                if row_matches_filters(row, target_date, hour_mm, hour_margin_min, location, cols, months_back):
                    d = row.to_dict()
                    d["__fuente"] = data_url
                    kept.append(d)
            except Exception:
                continue

        if kept:
            out = pd.DataFrame(kept)
            print(f"  [OK] dataset #{i}: {len(out)} filas pasan filtro (cols={len(out.columns)})")
            kept_frames.append(out)
        else:
            print(f"  [OK] dataset #{i}: 0 filas pasan filtro")

    if not kept_frames:
        return pd.DataFrame()
    return pd.concat(kept_frames, ignore_index=True, sort=False)

# -------- Main --------
def main():
    print("=== FILTRO TRÁFICO (Madrid) — robusto ===")
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

    parts: List[pd.DataFrame] = []
    for cfg in PAGES:
        if not cfg.get("enabled", True):
            print(f"\n[Omitida] {cfg['url']} (deshabilitada por configuración)")
            continue
        try:
            dfp = process_page_filtered_full(cfg["url"], target_date, hour_mm, hour_margin_min, location, months_back)
            if not dfp.empty:
                parts.append(dfp)
        except Exception as e:
            print(f"  [Error inesperado] {e}")

    if not parts:
        print("\n[Resultado] 0 filas cumplen los filtros.")
        pd.DataFrame(columns=["__fuente"]).to_csv(OUT_FILE, index=False, encoding="utf-8-sig", sep=";")
        print(f"[OK] Archivo vacío con cabecera: {OUT_FILE.resolve()}")
        return

    out = pd.concat(parts, ignore_index=True, sort=False)

    # Rellenar blancos: NaN y strings vacíos -> "NA"
    out = out.fillna("NA")
    for c in out.columns:
        try:
            out[c] = out[c].astype(str).replace(r"^\s*$", "NA", regex=True)
        except Exception:
            pass

    # Ordenar por fecha/hora si existen columnas compatibles
    try:
        ts = None
        ts_col = None
        for c in out.columns:
            if any(k in str(c).lower() for k in ["fecha_hora","fechahora","datetime","timestamp"]):
                ts_col = c; break
        if ts_col:
            ts = pd.to_datetime(out[ts_col], errors="coerce", dayfirst=True)
        else:
            fecha_col = next((c for c in out.columns if any(k in str(c).lower() for k in ["fecha","date","dia"])), None)
            hora_col  = next((c for c in out.columns if any(k in str(c).lower() for k in ["hora","time","franja","tramo"])), None)
            if fecha_col and hora_col:
                # si la hora es un tramo, intenta extraer el medio para concatenar
                def hour_mid(s):
                    hhmm = parse_hour_cell(s)
                    return f"{hhmm[0]:02d}:{hhmm[1]:02d}" if hhmm else "00:00"
                ts = pd.to_datetime(out[fecha_col].astype(str) + " " + out[hora_col].astype(str).map(hour_mid), errors="coerce", dayfirst=True)
            elif fecha_col:
                ts = pd.to_datetime(out[fecha_col], errors="coerce", dayfirst=True)
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
