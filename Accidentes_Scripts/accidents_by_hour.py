
import re
import io
import sys
import math
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

# --- Config fuentes ---
DATASET_PAGES = [
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=40085fb0e70b7410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default",
    "https://datos.comunidad.madrid/catalogos/#/dataset/1908061?view=info",
]

# Overrides para la CAM (RDF que compartiste)
HARDCODED_DOWNLOADS: Dict[str, List[str]] = {
    "comunidad.madrid/catalogos/#/dataset/1908061": [
        "https://datos.comunidad.madrid/dataset/fb9c5a17-afb0-4e95-a7b1-186e7cacc901/resource/58e39362-fbd1-45f6-865b-91505f6bd199/download/accidentes-de-circulacion-con-victimas-por-ubicacion-y-resultado-del-accidente.csv",
        "https://datos.comunidad.madrid/dataset/fb9c5a17-afb0-4e95-a7b1-186e7cacc901/resource/69a6b3e0-f711-47c5-aa2d-a87b0f82fd31/download/accidentes-de-circulacion-con-victimas-por-ubicacion-y-resultado-del-accidente.json",
    ],
}
OUTPUT_DIR = Path("./Accidentes_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "MateoScraperBot/1.0 (+contact: tu-email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,application/rdf+xml,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

CSV_EXT = (".csv",)
JSON_EXT = (".json", ".geojson")

# ---------------- Red / scraping ----------------

def fetch(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return r

def fetch_text(url: str) -> str:
    r = fetch(url)
    if not r.encoding or r.encoding.lower() in ("ascii", "utf-8"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def find_download_links_html(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
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
        if u.endswith(".csv"):
            return 0
        if u.endswith(".json") or u.endswith(".geojson"):
            return 1
        return 5
    links.sort(key=score)
    return links

def parse_rdf_for_downloads(xml_text: str) -> List[Tuple[str, str, str]]:
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

def find_downloads(page_url: str, html_text: str) -> List[str]:
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
    print(f"  [Descarga] {url}")
    r = fetch(url)
    ctype = (r.headers.get("Content-Type") or "").lower()
    u = url.lower()
    # CSV
    if u.endswith(".csv") or "csv" in ctype:
        try:
            df = pd.read_csv(io.StringIO(r.text), sep=None, engine="python")
        except Exception:
            df = None
            for sep in (";", ",", "\t", "|"):
                try:
                    df = pd.read_csv(io.StringIO(r.text), sep=sep); break
                except Exception:
                    pass
            if df is None:
                raise
        print(f"    [OK CSV] {df.shape}")
        return df
    # JSON / GeoJSON
    if u.endswith(".json") or u.endswith(".geojson") or "json" in ctype:
        data = r.json()
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict) and "features" in data and isinstance(data["features"], list):
            df = pd.json_normalize(data["features"])
        else:
            df = pd.json_normalize(data)
        print(f"    [OK JSON] {df.shape}")
        return df
    print("    [Aviso] Tipo no soportado (solo CSV/JSON).")
    return None

# ---------------- Normalización texto ----------------

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # quita acentos
    return s

# ---------------- Fechas / horas / columnas ----------------

def parse_user_hour(s: str) -> Optional[Tuple[int, int]]:
    """
    Acepta:
      - "08", "8"                 -> 08:00
      - "08:30", "8:30", "08:05"  -> HH:MM
      - "8.5", "8,25", "7.75"     -> fracción de hora (min = round(fracción*60))
    Vacío -> None (sin filtro por hora)
    """
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
        if mm == 60:
            hh, mm = hh + 1, 0
        return hh, mm
    # decimal
    if re.match(r"^\d+(\.\d+)?$", st):
        f = float(st)
        hh = int(f)
        mm = int(round((f - hh) * 60))
        if mm == 60:
            hh, mm = hh + 1, 0
        return hh, mm
    # entero
    if st.isdigit():
        return int(st), 0
    raise ValueError("Formato de hora no reconocido")

def parse_user_date(s: str) -> Optional[date_cls]:
    s = s.strip()
    if not s:
        return None
    try:
        dt = dtparser.parse(s, dayfirst=False)  # espera YYYY-MM-DD
        return dt.date()
    except Exception:
        return None

def detect_date_parts(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    low = {c.lower(): c for c in df.columns}
    ano = low.get("año") or low.get("ano") or low.get("year") or low.get("anio")
    mes = low.get("mes") or low.get("month")
    dia = low.get("dia") or low.get("día") or low.get("day")
    if ano and mes and dia:
        return ano, mes, dia
    return None

def find_timestamp_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(k in str(c).lower() for k in
            ["fecha_hora","fechahora","datetime","timestamp","fecha","date"])]

def find_street_columns(df: pd.DataFrame) -> List[str]:
    """Detecta columnas de vía/dirección más comunes."""
    STREET_KEYS = [
        "calle","via","vía","direccion","dirección","ubicacion","ubicación",
        "localizacion","localización","lugar","punto","domicilio","via_publica",
        "carretera","tramo","pk","interseccion","intersección","cruce","kilometro","kilómetro",
        "street","road","address","addr"
    ]
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in STREET_KEYS):
            cols.append(c)
    return cols

# ---------------- Filtrado principal ----------------

def to_hour_minute(value) -> Optional[Tuple[int, int]]:
    if pd.isna(value): return None
    if isinstance(value, pd.Timestamp): return value.hour, value.minute
    if isinstance(value, datetime): return value.hour, value.minute
    sv = str(value).strip()
    # decimal 8.5 -> 8:30
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

def filter_df(df: pd.DataFrame,
              start_date: Optional[date_cls],
              end_date: Optional[date_cls],
              street_query: Optional[str],
              hour_mm: Optional[Tuple[int, int]]) -> pd.DataFrame:
    """
    Aplica filtros:
      - fecha en [start_date, end_date] si se proporcionan
      - calle (contains insensible a acentos)
      - hora (exacta HH:MM; si dataset no trae minutos, cae a solo HH)
    """
    if df.empty:
        return df

    # 1) Fecha
    date_mask: Optional[pd.Series] = None
    if start_date and end_date:
        # timestamp completo
        for c in find_timestamp_cols(df):
            try:
                ts = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if ts.notna().mean() > 0.5:
                    m = (ts.dt.date >= start_date) & (ts.dt.date <= end_date)
                    date_mask = m if date_mask is None else (date_mask | m)
            except Exception:
                pass
        # partes ANO/MES/DIA
        parts = detect_date_parts(df)
        if parts:
            ano, mes, dia = parts
            y = pd.to_numeric(df[ano], errors="coerce")
            m = pd.to_numeric(df[mes], errors="coerce")
            d = pd.to_numeric(df[dia], errors="coerce")
            # construimos date como strings para comparar rango
            try:
                built = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
                m2 = (built.dt.date >= start_date) & (built.dt.date <= end_date)
                date_mask = m2 if date_mask is None else (date_mask | m2)
            except Exception:
                pass
        # si no logramos ninguna máscara de fecha y se pidió fecha -> devolvemos vacío
        if date_mask is None:
            return df.iloc[0:0]

    # 2) Calle
    street_mask: Optional[pd.Series] = None
    if street_query:
        target = normalize_text(street_query)
        cols = find_street_columns(df)
        for c in cols:
            try:
                col_norm = df[c].astype(str).map(normalize_text)
                m = col_norm.str.contains(re.escape(target), na=False)
                street_mask = m if street_mask is None else (street_mask | m)
            except Exception:
                continue
        # si no hay columnas de calle y se pidió filtro -> vacío
        if street_mask is None:
            return df.iloc[0:0]

    # 3) Hora
    hour_mask: Optional[pd.Series] = None
    if hour_mm is not None:
        hh, mm = hour_mm
        # timestamp
        for c in find_timestamp_cols(df):
            try:
                ts = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if ts.notna().mean() > 0.5:
                    m = (ts.dt.hour == hh) & (ts.dt.minute == mm)
                    if m.sum() == 0:
                        m = (ts.dt.hour == hh)  # fallback solo hora
                    hour_mask = m if hour_mask is None else (hour_mask | m)
            except Exception:
                pass
        # columnas de hora explícitas
        H_KEYS = ["hora","time","tiempo","h_acc","acc_hora"]
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in H_KEYS):
                vals = df[c].apply(to_hour_minute)
                m = vals.apply(lambda t: t is not None and t[0] == hh and t[1] == mm)
                if m.sum() == 0:
                    m = vals.apply(lambda t: t is not None and t[0] == hh)
                hour_mask = m if hour_mask is None else (hour_mask | m)

    # 4) Combinar
    mask = pd.Series([True] * len(df))
    if date_mask is not None:   mask &= date_mask.fillna(False)
    if street_mask is not None: mask &= street_mask.fillna(False)
    if hour_mask is not None:   mask &= hour_mask.fillna(False)
    return df[mask]

# ---------------- Proceso por dataset ----------------

def process_one(page_url: str,
                start_date: Optional[date_cls],
                end_date: Optional[date_cls],
                street_query: Optional[str],
                hour_mm: Optional[Tuple[int, int]]) -> pd.DataFrame:
    print(f"\n[Procesando] {page_url}")
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
            print(f"  [Error al descargar datos] {e}")
            continue
        if df is None or df.empty:
            continue

        filtered = filter_df(df, start_date, end_date, street_query, hour_mm)
        print(f"  [Filtrado] {filtered.shape} filas")
        if not filtered.empty:
            filtered = filtered.copy()
            filtered["__origen__"] = page_url
            filtered["__descarga__"] = dl
            return filtered

    print("  [Aviso] Ninguna descarga produjo filas para ese filtro.")
    return pd.DataFrame()

# ---------------- Main ----------------

def main():
    # Fecha (opcional) → si se da, usaremos [fecha-3 meses, fecha]
    user_d = input("Introduce la FECHA (YYYY-MM-DD) o pulsa Enter para omitir: ").strip()
    target_date = parse_user_date(user_d)
    if user_d and target_date is None:
        print("Fecha inválida. Usa formato YYYY-MM-DD (ej: 2025-10-15).")
        sys.exit(1)

    start_date = end_date = None
    if target_date:
        end_date = target_date
        start_date = target_date - relativedelta(months=3)
        print(f"[Rango de fechas] {start_date.isoformat()} → {end_date.isoformat()}")

    # Calle (opcional)
    street_query = input("Introduce la CALLE (parcial, ej: 'gran via') o pulsa Enter para omitir: ").strip()
    if not street_query:
        street_query = None

    # Hora (opcional; admite fracciones)
    user_h = input("Introduce la HORA (opcional: ej 08, 08:30, 8.5, 8,25) o Enter para omitir: ").strip()
    try:
        hour_mm = parse_user_hour(user_h) if user_h else None
        if hour_mm:
            hh, mm = hour_mm
            assert 0 <= hh < 24 and 0 <= mm < 60
    except Exception:
        print("Hora inválida. Usa 08, 08:30, 8.5, 8,25 …")
        sys.exit(1)

    parts = []
    for url in DATASET_PAGES:
        try:
            df = process_one(url, start_date, end_date, street_query, hour_mm)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  [Error inesperado] {e}")

    if not parts:
        print("\n[Resultado] 0 filas encontradas con ese filtro.")
        return

    out = pd.concat(parts, ignore_index=True, sort=False)
    # ordenar si hay fecha
    for c in ["fecha_hora","FechaHora","FECHA_HORA","fecha","FECHA","date","DATE"]:
        if c in out.columns:
            out = out.sort_values(by=c)
            break

    # Nombre de archivo
    suffix_parts = []
    if target_date:
        suffix_parts.append(f"{start_date.isoformat()}_{end_date.isoformat()}")
    if street_query:
        slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(street_query)).strip("_")
        if slug:
            suffix_parts.append(slug)
    if hour_mm:
        suffix_parts.append(f"{hour_mm[0]:02d}{hour_mm[1]:02d}")
    suffix = "__".join(suffix_parts) if suffix_parts else "sin_filtros"

    out_file = OUTPUT_DIR / f"accidentes_{suffix}.csv"
    out.to_csv(out_file, index=False, sep=";", encoding="utf-8")

    print(f"\n[OK] Guardado: {out_file.resolve()}")
    print(f"[Filas totales] {len(out)}")
    print("\n[Preview]")
    print(out.head(10))

if __name__ == "__main__":
    main()
