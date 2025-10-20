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

# ----------- Fichas -----------
PAGES = [
    # Bomberos (agregado mensual: se omitirá al filtrar eventos con fecha/hora)
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=fa677996afc6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    # SAMUR activaciones
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=50d7d35982d6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    # Tercera ficha (otra fuente de emergencias)
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
]

# ----------- Salida -----------
OUTPUT_DIR = Path("./Emergencias_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "emergencias_filtrado.csv"

# ----------- Red -----------
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}
REQ_TIMEOUT = 45
CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

# Excluir ruido
EXCLUDE_PATTERNS = [
    "wms", "wmts", "ogc", "service", "arcgis", "esri",
    ".zip", ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".pdf", ".rdf", ".xml", "sparql", "mailto:", "javascript:", "#"
]
PREFERRED_HINTS = ["download", "descarg", "csv", "json"]

# Overrides si alguna ficha requiere URL directa
OVERRIDES: Dict[str, str] = {
    # "50d7d35982d6f510": "https://datos.madrid.es/egobfiles/.../activaciones_samur_2025.csv",
}

# Margen por defecto cuando el usuario indica fecha
MONTHS_BACK_DEFAULT = 2

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

def is_event_level(df: pd.DataFrame) -> bool:
    """True si hay fecha/hora por evento. False si es agregado (solo AÑO/MES etc.)."""
    cols = [c.lower() for c in df.columns]
    has_time = any("hora" in c or "time" in c for c in cols)
    has_date = any(x in c for c in cols for x in ["fecha_hora","fechahora","datetime","timestamp","fecha","date"])
    has_month = any(c == "mes" or c == "month" for c in cols)
    has_year  = any(c in ["año","ano","anio","year"] for c in cols)
    if (has_month or has_year) and not (has_time or has_date):
        return False
    return has_time or has_date

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

# ---------- Filtros ----------
def in_date_window(d: date_cls, target_date: Optional[date_cls], months_back: int) -> bool:
    if target_date is None:
        return True
    start = target_date - relativedelta(months=months_back)
    return start <= d <= target_date

def find_timestamp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(k in str(c).lower() for k in
            ["fecha_hora","fechahora","datetime","timestamp","fecha","date","hora","time"])]

def find_location_columns(df: pd.DataFrame) -> List[str]:
    keys = [
        "calle","vía","via","direccion","dirección","ubicacion","ubicación",
        "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
        "street","road","address","addr","localizacion","localización",
        "distrito","barrio","municipio","localidad","seccion","sección","zona"
    ]
    return [c for c in df.columns if any(k in str(c).lower() for k in keys)]

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
        hh = int(sv); 
        if 0 <= hh < 24: return hh, 0
    try:
        dt = dtparser.parse(sv, dayfirst=True, fuzzy=True)
        return dt.hour, dt.minute
    except Exception:
        return None

def row_matches_filters(
    row: pd.Series,
    target_date: Optional[date_cls],
    hour_mm: Optional[Tuple[int, int]],
    location: Optional[str],
    df_cols: List[str],
    months_back: int
) -> bool:
    # -------- Fecha / Hora --------
    if target_date or hour_mm:
        match_time = False

        for c in df_cols:
            lc = str(c).lower()
            if any(k in lc for k in ["fecha_hora","fechahora","datetime","timestamp","fecha","date","hora","time"]):
                # 1) Intentar parsear como timestamp/fecha
                ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                if pd.notna(ts):
                    # fecha dentro de ventana
                    ok_date = in_date_window(ts.date(), target_date, months_back)
                    # hora exacta (si la pide el usuario)
                    if hour_mm:
                        hh, mm = hour_mm
                        ok_hour = (ts.hour == hh and ts.minute == mm) or (ts.hour == hh)
                    else:
                        ok_hour = True

                    if ok_date and ok_hour:
                        match_time = True
                        break
                else:
                    # 2) Si la columna parece ser solo "hora", permitir match de hora sin fecha explícita
                    if hour_mm:
                        hhmm = to_hour_minute(row[c])
                        if hhmm:
                            hh, mm = hour_mm
                            ok_hour = (hhmm[0] == hh and hhmm[1] == mm) or (hhmm[0] == hh)
                            # Sin fecha en esa celda: si hay target_date, podría estar en otra columna; seguimos buscando.
                            if ok_hour and target_date is None:
                                match_time = True
                                break

        if not match_time:
            return False

        # Si el usuario dio fecha pero la fecha NO apareció en ninguna columna timestamp explícita,
        # intentamos reconstruir fecha desde otras columnas posibles (year/month/day en texto).
        if target_date:
            any_date_seen = False
            for c in df_cols:
                if "fecha" in str(c).lower() or "date" in str(c).lower():
                    ts = pd.to_datetime(row[c], errors="coerce", dayfirst=True)
                    if pd.notna(ts):
                        any_date_seen = True
                        if not in_date_window(ts.date(), target_date, months_back):
                            return False
                        break
            # Si no vimos fecha en ninguna columna, no descartamos solo por no encontrar campo.

    # -------- Ubicación --------
    if location:
        target = normalize_text(location)
        loc_cols = [c for c in df_cols if any(k in str(c).lower() for k in
                   ["calle","vía","via","direccion","dirección","ubicacion","ubicación",
                    "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
                    "street","road","address","addr","localizacion","localización",
                    "distrito","barrio","municipio","localidad","seccion","sección","zona"])]
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

# ---------- Proceso por ficha (conservando filas completas) ----------
def process_page_filtered_full(page_url: str, target_date: Optional[date_cls], hour_mm: Optional[Tuple[int,int]], location: Optional[str], months_back: int) -> pd.DataFrame:
    print(f"\n[Procesando] {page_url}")
    # HTML de la ficha
    try:
        html = fetch_html(page_url)
    except Exception as e:
        print(f"  [Error] al abrir ficha: {e}")
        return pd.DataFrame()
    soup = make_soup(html)

    # URL de datos validada
    data_url = None
    for key, forced in OVERRIDES.items():
        if key in page_url:
            data_url = forced
            break
    if not data_url:
        data_url = find_valid_data_url_from_page(page_url, soup)
    if not data_url:
        print("  [Aviso] sin URL CSV/JSON válida")
        return pd.DataFrame()

    # Carga tabla
    df = load_remote_table(data_url)
    if df is None or df.empty:
        print("  [Aviso] descarga vacía")
        return pd.DataFrame()

    # Descarta agregados sin fecha/hora de evento
    if not is_event_level(df):
        print("  [Skip] dataset agregado (sin fecha/hora por evento)")
        return pd.DataFrame()

    # Filtro fila a fila conservando todas las columnas
    kept = []
    cols = list(df.columns)
    for _, row in df.iterrows():
        try:
            if row_matches_filters(row, target_date, hour_mm, location, cols, months_back=months_back):
                d = row.to_dict()
                d["__fuente"] = data_url
                kept.append(d)
        except Exception:
            continue

    if not kept:
        print("  [OK] 0 filas pasan filtro")
        return pd.DataFrame()

    out = pd.DataFrame(kept)
    print(f"  [OK] {len(out)} filas pasan filtro (cols={len(out.columns)})")
    return out

# ---------- Main ----------
def main():
    user_date = input("Fecha (YYYY-MM-DD) o Enter para omitir: ").strip()
    target_date = parse_user_date(user_date)
    if user_date and target_date is None:
        print("Fecha inválida. Usa YYYY-MM-DD.")
        sys.exit(1)

    # margen en meses (por defecto 2)
    months_back = MONTHS_BACK_DEFAULT
    user_months = input(f"Margen hacia atrás en meses (Enter={MONTHS_BACK_DEFAULT}): ").strip()
    if user_months:
        try:
            months_back = max(0, int(user_months))
        except Exception:
            print("Valor de meses inválido. Usando valor por defecto:", MONTHS_BACK_DEFAULT)
            months_back = MONTHS_BACK_DEFAULT

    location = input("Vía / Ubicación (calle o distrito/barrio) o Enter para omitir: ").strip() or None

    user_hour = input("Hora (ej: 08, 08:30, 8.5, 8,25) o Enter para omitir: ").strip()
    try:
        hour_mm = parse_user_hour(user_hour) if user_hour else None
        if hour_mm:
            hh, mm = hour_mm
            assert 0 <= hh < 24 and 0 <= mm < 60
    except Exception:
        print("Hora inválida. Usa 08, 08:30, 8.5, 8,25 …")
        sys.exit(1)

    parts: List[pd.DataFrame] = []
    for url in PAGES:
        try:
            dfp = process_page_filtered_full(url, target_date, hour_mm, location, months_back)
            if not dfp.empty:
                parts.append(dfp)
        except Exception as e:
            print(f"  [Error inesperado] {e}")

    if not parts:
        print("\n[Resultado] 0 filas cumplen los filtros.")
        # crea CSV vacío con BOM y una columna de referencia
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
        # busca mejor columna de fecha/hora compuesta
        ts_col = None
        for c in out.columns:
            if any(k in str(c).lower() for k in ["fecha_hora","fechahora","datetime","timestamp"]):
                ts_col = c; break
        if ts_col:
            ts = pd.to_datetime(out[ts_col], errors="coerce", dayfirst=True)
        else:
            # intenta combinar 'fecha' y 'hora' si existieran
            fecha_col = next((c for c in out.columns if "fecha" in str(c).lower() or "date" in str(c).lower()), None)
            hora_col  = next((c for c in out.columns if "hora"  in str(c).lower() or "time" in str(c).lower()), None)
            if fecha_col and hora_col:
                ts = pd.to_datetime(out[fecha_col] + " " + out[hora_col], errors="coerce", dayfirst=True)
            elif fecha_col:
                ts = pd.to_datetime(out[fecha_col], errors="coerce", dayfirst=True)
            else:
                ts = None
        if ts is not None:
            out = out.iloc[ts.sort_values().index]
    except Exception:
        pass

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", sep=";")
    print(f"\n[OK] Guardado: {OUT_FILE.resolve()}")
    print(f"[Filas] {len(out)}  [Columnas] {len(out.columns)}")
    print(out.head(10))

if __name__ == "__main__":
    main()
