# -*- coding: utf-8 -*-
# Code in English, comments in Spanish

import io
import re
import csv
import math
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, date as date_cls

import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
from dateutil import parser as dtparser

# ================================
# Configuración general (solo Obras)
# ================================

OBRAS_CSV = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"

OUTPUT_DIR = Path("./Obras_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_UNIFIED = OUTPUT_DIR / "datasheet_infraestructura.csv"

HEADERS = {
    "User-Agent": "MateoScraperBot/2.4 (+contact: your-email@example.com)",
    "Accept": "text/csv,application/json,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}
REQ_TIMEOUT = 60

# Ventana por defecto: hoy y 2 meses atrás
TODAY = datetime.today().date()
DEFAULT_START = TODAY - relativedelta(months=2)
DEFAULT_END = TODAY

# Contrato común (14 columnas)
CONTRACT_BASE = [
    "dataset","event_type","date","time","datetime",
    "district_code","district_name","lat","lon","location",
    "severity","value","units","source_id",
]

# Mapeo códigos → nombres de distritos del Ayuntamiento de Madrid
MADRID_DISTRICTS = {
    "1":"Centro","01":"Centro",
    "2":"Arganzuela","02":"Arganzuela",
    "3":"Retiro","03":"Retiro",
    "4":"Salamanca","04":"Salamanca",
    "5":"Chamartín","05":"Chamartín",
    "6":"Tetuán","06":"Tetuán",
    "7":"Chamberí","07":"Chamberí",
    "8":"Fuencarral-El Pardo","08":"Fuencarral-El Pardo",
    "9":"Moncloa-Aravaca","09":"Moncloa-Aravaca",
    "10":"Latina","11":"Carabanchel","12":"Usera","13":"Puente de Vallecas",
    "14":"Moratalaz","15":"Ciudad Lineal","16":"Hortaleza","17":"Villaverde",
    "18":"Villa de Vallecas","19":"Vicálvaro","20":"San Blas-Canillejas","21":"Barajas",
}

# ================================
# Helpers de normalización
# ================================

def nfd_lower(s: str) -> str:
    """Minúsculas + sin acentos."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columna."""
    mapping = {}
    for c in df.columns:
        base = nfd_lower(str(c)).replace("\ufeff","")
        base = re.sub(r"\s+", "_", base)
        mapping[c] = base
    return df.rename(columns=mapping)

def fetch(url: str) -> requests.Response:
    """Descarga con headers/timeout estándar."""
    r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r

def load_csv_robust(url: str) -> pd.DataFrame:
    """Carga CSV con detección robusta de separador/encoding."""
    r = fetch(url)
    data = r.content
    for enc in ("utf-8-sig","utf-8","latin-1",None):
        for sep in (";","\t",",","|"):
            try:
                if enc is None:
                    df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False)
                else:
                    df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, low_memory=False, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    raise RuntimeError("Unable to parse CSV with common encodings/separators")

def parse_mixed_date_series(series: pd.Series) -> pd.Series:
    """Parsea fechas mezcladas: YYYY-MM-DD con/ sin HH:MM y, si no, dayfirst=True."""
    s = series.astype(str)
    iso_mask = s.str.match(r"^\s*\d{4}-\d{2}-\d{2}(\s+\d{2}:\d{2}(:\d{2})?)?\s*$", na=False)
    result = pd.to_datetime(pd.Series([pd.NaT]*len(s)), errors="coerce")
    if iso_mask.any():
        with_time = s[iso_mask].str.contains(r"\d{2}:\d{2}", regex=True)
        result.loc[iso_mask & ~with_time] = pd.to_datetime(s[iso_mask & ~with_time], format="%Y-%m-%d", errors="coerce")
        result.loc[iso_mask & with_time] = pd.to_datetime(s[iso_mask & with_time], errors="coerce")
    if (~iso_mask).any():
        result.loc[~iso_mask] = pd.to_datetime(s[~iso_mask], errors="coerce", dayfirst=True)
    return result

def as_date(s) -> str:
    """Fecha ISO o '' si no válida."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    try:
        txt = str(s).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", txt):
            dt = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
        else:
            dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
        if pd.isna(dt): return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def to_hour_minute(value) -> Optional[Tuple[int,int]]:
    """Convierte representaciones comunes de hora a (HH,MM)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    sv = str(value).strip().replace(",", ".")
    if re.match(r"^\d+(\.\d+)?$", sv):
        f = float(sv)
        hh = int(math.floor(f))
        mm = int(round((f - hh) * 60))
        if mm == 60: mm = 0
        return hh, mm
    m = re.match(r"^\s*(\d{1,2})\s*[:hH]?\s*(\d{1,2})?\s*$", sv.replace(".", ":"))
    if m:
        return int(m.group(1)), int(m.group(2) or 0)
    try:
        dt = dtparser.parse(sv, dayfirst=True, fuzzy=True)
        return dt.hour, dt.minute
    except Exception:
        return None

def as_time(s) -> str:
    """Hora HH:MM o '' si no válida."""
    t = to_hour_minute(s)
    return f"{t[0]:02d}:{t[1]:02d}" if t else ""

def guess_datetime_cols(df: pd.DataFrame) -> List[str]:
    """Candidatas de fecha/hora por nombre (incluye abreviaturas INIC/FINA)."""
    keys = [
        "fecha_hora","fechahora","datetime","timestamp","fecha","date",
        "inicio","fin","f_inicio","f_fin",
        "inic","fina"  # <— clave para FECHA_INIC y FECHA_FINA
    ]
    cols = []
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in keys):
            cols.append(c)
    return cols

def guess_street_cols(df: pd.DataFrame) -> List[str]:
    """Columnas candidatos de ubicación/vía (añadimos denominaci y viario_afe)."""
    keys = [
        "calle","via","vía","direccion","dirección","ubicacion","ubicación",
        "lugar","tramo","punto","cruce","descripcion","descripción","observaciones","detalle",
        "denominaci","viario_afe"  # <— campos del dataset de Obras
    ]
    return [c for c in df.columns if any(k in c for k in keys)]

def guess_district_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detecta columnas distrito código/nombre."""
    code_keys = ["codigo_distrito","cod_distrito","coddistrito","c_distrito","district_code","codigo__distrito"]
    name_keys = ["distrito","nombre_distrito","district","district_name","distrito_s"]
    code = next((c for c in df.columns if any(k in c for k in code_keys)), None)
    name = next((c for c in df.columns if any(k in c for k in name_keys)), None)
    return code, name

def filter_by_window_any_datecols(df: pd.DataFrame, start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    """
    Filtra filas que caigan en la ventana en cualquier columna temporal detectada.
    Si NO hay columnas temporales, NO filtra (devuelve df completo).
    """
    if df.empty: return df
    date_cols = guess_datetime_cols(df)
    if not date_cols:
        return df
    mask = None
    for c in date_cols:
        try:
            ts = parse_mixed_date_series(df[c])
            m = (ts.dt.date >= start_date) & (ts.dt.date <= end_date)
            mask = m if mask is None else (mask | m)
        except Exception:
            continue
    if mask is None:
        return df
    return df[mask]

def finalize_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena NA/orden columnas exactamente al contrato y aplica fallback de distrito."""
    if df.empty:
        return pd.DataFrame(columns=CONTRACT_BASE)
    out = df.copy()
    out = out.fillna("NA").replace(r"^\s*$", "NA", regex=True)
    # district_name <- location si queda 'NA' (para RapidMiner)
    if "district_name" in out.columns and "location" in out.columns:
        out.loc[out["district_name"].isin(["", "NA"]), "district_name"] = out["location"]
        out["district_name"] = out["district_name"].replace(r"^\s*$", "NA", regex=True)
    for c in CONTRACT_BASE:
        if c not in out.columns: out[c] = "NA"
    return out[CONTRACT_BASE]

# ================================
# Procesamiento: OBRAS (CSV)
# ================================

def process_obras(url: str, start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    """Procesa el CSV de obras y lo mapea al contrato."""
    raw = load_csv_robust(url)
    raw = normalize_cols(raw)

    # 1) Filtrado por ventana (si hay fechas detectables)
    win = filter_by_window_any_datecols(raw, start_date, end_date)

    # 2) Si quedó vacío, Fallback: sin ventana (para no devolver 0 filas)
    if win.empty:
        print("[Aviso] Sin filas en ventana. Usando TODO el CSV (fallback sin ventana).")
        win = raw

    # Construcción de contrato
    out = pd.DataFrame()
    out["dataset"] = "obras"
    out["event_type"] = "obra_planificada"

    # Fecha/hora -> ahora incluye 'fecha_inic' y 'fecha_fina'
    dt_cols = guess_datetime_cols(win)
    chosen = None
    for pref in ["fecha_inicio","fecha_inic","inicio","fechainicio","fecha","fecha_prevista_inicio","f_inicio","datetime","fecha_fina","f_fin"]:
        if pref in win.columns:
            chosen = pref; break
    if chosen is None and dt_cols:
        chosen = dt_cols[0]

    if chosen:
        dtp = parse_mixed_date_series(win[chosen])
        out["datetime"] = dtp.dt.strftime("%Y-%m-%d %H:%M:%S")
        out["date"] = dtp.dt.strftime("%Y-%m-%d")
        out["time"] = dtp.dt.strftime("%H:%M")
        for col in ("date","time","datetime"):
            out.loc[out[col].isin(["NaT","nan","None"]), col] = ""
    else:
        dcol = next((c for c in win.columns if c in ("fecha","date","fecha_inic","fecha_fina")), None)
        tcol = next((c for c in win.columns if c in ("hora","time")), None)
        out["date"] = win[dcol].map(as_date) if dcol else ""
        out["time"] = win[tcol].map(as_time) if tcol else ""
        out["datetime"] = ""

    # Distrito (código/nombre)
    dcode, dname = guess_district_cols(win)
    code_series = None
    if dcode:
        code_series = win[dcode].astype(str).str.extract(r"(\d+)")[0]
    elif dname and win[dname].astype(str).str.match(r"^\s*\d+\s*$").all():
        code_series = win[dname].astype(str).str.extract(r"(\d+)")[0]

    out["district_code"] = (code_series.fillna("") if code_series is not None
                            else (win.get(dcode, "").astype(str) if dcode else ""))

    if dname and not win[dname].astype(str).str.match(r"^\s*\d+\s*$").all():
        out["district_name"] = win[dname].astype(str)
    else:
        if code_series is not None:
            out["district_name"] = code_series.map(
                lambda x: MADRID_DISTRICTS.get(x, MADRID_DISTRICTS.get(str(x).zfill(2), ""))
            )
        else:
            out["district_name"] = ""

    # Lat/Lon (no suelen estar; quedarán en NA)
    lat_col = next((c for c in win.columns if c in ("lat","latitud","latitude","y")), None)
    lon_col = next((c for c in win.columns if c in ("lon","longitud","longitude","x")), None)
    out["lat"] = win.get(lat_col, "") if lat_col else ""
    out["lon"] = win.get(lon_col, "") if lon_col else ""

    # Ubicación (mejorada): DENOMINACI + VIARIO_AFE + (DESCRIPCIO si existe)
    street_cols = guess_street_cols(win)
    pieces = []
    for key in ["denominaci","viario_afe","descripcion","descripción","tramo","ubicacion","ubicación"]:
        if key in win.columns:
            pieces.append(win[key].astype(str))
    if pieces:
        try:
            location = pieces[0].fillna("")
            for p in pieces[1:]:
                location = (location + " " + p.fillna("")).str.strip()
        except Exception:
            location = pieces[0].astype(str)
        out["location"] = location
    else:
        out["location"] = "NA"

    # Severidad/Value/Units
    sev_candidates = ["severidad","impacto","prioridad","estado","fase"]
    sev_col = next((c for c in win.columns if c in sev_candidates), None)
    out["severity"] = win.get(sev_col, "") if sev_col else ""
    out["value"] = "NA"
    out["units"] = "NA"

    # source_id -> ID expediente o URL
    id_candidates = ["id","codigo","n_expedien","cod_obra","id_obra","expediente","num_exp","referencia"]
    id_col = next((c for c in win.columns if c in id_candidates), None)
    out["source_id"] = win.get(id_col, "") if id_col else OBRAS_CSV

    return finalize_contract(out)

# ================================
# Main (solo Obras)
# ================================

def main():
    start_date = DEFAULT_START
    end_date = DEFAULT_END
    print(f"[Ventana] {start_date.isoformat()} -> {end_date.isoformat()}")

    try:
        obras = process_obras(OBRAS_CSV, start_date, end_date)
        print(f"[OK] Obras: {len(obras)} filas")
    except Exception as e:
        obras = pd.DataFrame(columns=CONTRACT_BASE)
        print(f"[Error OBRAS] {e}")

    obras.to_csv(
        OUT_UNIFIED, index=False, sep=";", encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
    )
    print(f"[OK] Datasheet unificado → {OUT_UNIFIED.resolve()} ({len(obras)} filas)")

if __name__ == "__main__":
    main()
