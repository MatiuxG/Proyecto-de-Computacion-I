
import re
import io
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
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

OUTPUT_DIR = Path(r".\Resultados\Accidentes")
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


# ---------------- Fechas / horas ----------------

def parse_user_hour(s: str) -> Tuple[int, int]:
    s = s.strip().replace(".", ":")
    if ":" in s:
        hh, mm = s.split(":", 1)
        return int(hh), int(mm)
    return int(s), 0

def parse_user_date(s: str) -> Optional[date_cls]:
    s = s.strip()
    if not s:
        return None
    try:
        dt = dtparser.parse(s, dayfirst=False)  # espera YYYY-MM-DD por defecto
        return dt.date()
    except Exception:
        return None

def detect_date_parts(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    """Busca columnas ANO/AÑO + MES + DIA (insensible a mayúsculas/acentos)."""
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

def find_time_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["hora","time","tiempo","h_acc","acc_hora"]):
            cols.append(c)
    if not cols:
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(60).str.strip()
            hhmm = sample.str.contains(r"^\d{1,2}[:hH]\d{2}$", regex=True).mean()
            only_h = sample.str.match(r"^\d{1,2}$", na=False).mean()
            if hhmm > 0.4 or only_h > 0.6:
                cols.append(c)
    return cols

def filter_df_by_date_hour(df: pd.DataFrame, target_date: Optional[date_cls], hh: int, mm: int) -> pd.DataFrame:
    """Filtra por fecha (si viene) y hora."""
    if df.empty:
        return df

    # 1) Si hay timestamp claro
    for c in find_timestamp_cols(df):
        try:
            ts = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if ts.notna().mean() > 0.5:
                mask = pd.Series([True] * len(df))
                if target_date is not None:
                    mask &= (ts.dt.date == target_date)
                mask &= (ts.dt.hour == hh) & (ts.dt.minute == mm)
                out = df[mask]
                if not out.empty:
                    return out
        except Exception:
            pass

    # 2) Partes separadas ANO/AÑO + MES + DIA
    parts = detect_date_parts(df)
    if parts and target_date is not None:
        ano, mes, dia = parts
        y = pd.to_numeric(df[ano], errors="coerce")
        m = pd.to_numeric(df[mes], errors="coerce")
        d = pd.to_numeric(df[dia], errors="coerce")
        mask_date = (y == target_date.year) & (m == target_date.month) & (d == target_date.day)
        # hora por columnas separadas
        for c in find_time_columns(df):
            vals = df[c].apply(to_hour_minute)
            mask_hour = vals.apply(lambda t: t is not None and (t[0] == hh and t[1] == mm))
            out = df[mask_date & mask_hour]
            if not out.empty:
                return out
            mask_hour2 = vals.apply(lambda t: t is not None and (t[0] == hh))
            out2 = df[mask_date & mask_hour2]
            if not out2.empty:
                return out2

    # 3) Sin fecha: filtra solo por hora
    if target_date is None:
        for c in find_time_columns(df):
            vals = df[c].apply(to_hour_minute)
            mask = vals.apply(lambda t: t is not None and (t[0] == hh and t[1] == mm))
            out = df[mask]
            if not out.empty:
                return out
            mask2 = vals.apply(lambda t: t is not None and (t[0] == hh))
            out2 = df[mask2]
            if not out2.empty:
                return out2

    return df.iloc[0:0]


# ---------------- Proceso por dataset ----------------

def process_one(page_url: str, target_date: Optional[date_cls], hh: int, mm: int) -> pd.DataFrame:
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

        filtered = filter_df_by_date_hour(df, target_date, hh, mm)
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
    # Pide hora
    user_h = input("Introduce la HORA (ej: 08 o 08:30): ").strip()
    try:
        hh, mm = parse_user_hour(user_h)
        assert 0 <= hh < 24 and 0 <= mm < 60
    except Exception:
        print("Hora inválida. Usa 08, 8, 08:30, 8:30 …")
        sys.exit(1)

    # Pide fecha (opcional)
    user_d = input("Introduce la FECHA (YYYY-MM-DD) o pulsa Enter para omitir: ").strip()
    target_date = parse_user_date(user_d)
    if user_d and target_date is None:
        print("Fecha inválida. Usa formato YYYY-MM-DD (ej: 2025-10-15).")
        sys.exit(1)

    parts = []
    for url in DATASET_PAGES:
        try:
            df = process_one(url, target_date, hh, mm)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  [Error inesperado] {e}")

    if not parts:
        print("\n[Resultado] 0 filas encontradas con ese filtro (fecha/hora).")
        return

    out = pd.concat(parts, ignore_index=True, sort=False)
    # ordenar si hay fecha
    for c in ["fecha_hora","FechaHora","FECHA_HORA","fecha","FECHA","date","DATE"]:
        if c in out.columns:
            out = out.sort_values(by=c)
            break

    hhmm = f"{hh:02d}{mm:02d}"
    suffix = hhmm if target_date is None else f"{target_date.isoformat()}_{hhmm}"
    out_file = OUTPUT_DIR / f"accidentes_{suffix}.csv"
    out.to_csv(out_file, index=False, sep=";", encoding="utf-8")

    print(f"\n[OK] Guardado: {out_file.resolve()}")
    print(f"[Filas totales] {len(out)}")
    print("\n[Preview]")
    print(out.head(10))


if __name__ == "__main__":
    main()
