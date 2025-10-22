
import re
import io
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Dict

import requests
import pandas as pd
from bs4 import BeautifulSoup

# -------- Config: FICHA --------
PAGE_URL = ("https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/"
            "?vgnextoid=4678f7de62435510VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310"
            "VgnVCM100000171f5a0aRCRD&vgnextfmt=default")

# -------- Salida --------
OUTPUT_DIR = Path("./Camaras_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Red --------
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) UbicacionETL/1.0"
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

# Overrides si conocieras una URL de datos directa (opcional)
OVERRIDES: Dict[str, str] = {
    # "4678f7de62435510": "https://datos.madrid.es/egobfiles/.../dataset.csv"
}

# -------- Utilidades --------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

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
    # Respeta overrides
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

        looks_data = low_href.endswith(CSV_EXT) or low_href.endswith(JSON_EXT) \
                     or any(h in (low_href + " " + low_label) for h in PREFERRED_HINTS)
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

def find_location_columns(df: pd.DataFrame):
    keys = [
        "calle","vía","via","direccion","dirección","ubicacion","ubicación",
        "lugar","punto","domicilio","via_publica","carretera","tramo","cruce",
        "street","road","address","addr","localizacion","localización",
        "distrito","barrio","municipio","localidad","seccion","sección","zona",
        "estacion","estación","station","punto_muestreo","site"
    ]
    cols = [c for c in df.columns if any(k in str(c).lower() for k in keys)]
    return cols

# -------- Proceso principal --------
def main():
    print("[INFO] Abriendo ficha…")
    try:
        html = fetch_html(PAGE_URL)
    except Exception as e:
        print(f"[ERROR] No pude abrir la ficha.\n{e}")
        sys.exit(1)

    soup = make_soup(html)
    data_url = find_valid_data_url_from_page(PAGE_URL, soup)

    if not data_url:
        print("[ERROR] No encontré una URL de datos CSV/JSON válida en la ficha.")
        sys.exit(1)

    print(f"[INFO] URL de datos: {data_url}")

    df = load_remote_table(data_url)
    if df is None or df.empty:
        print("[ERROR] Descarga vacía o no parseable.")
        sys.exit(1)

    print(f"[INFO] Dataset: {len(df)} filas, {len(df.columns)} columnas.")

    # --- Filtro de ubicación ---
    user_loc = input("Ubicación (calle/distrito/barrio/…); Enter para descargar TODO: ").strip()
    out_df = df

    if user_loc:
        loc_cols = find_location_columns(df)
        if not loc_cols:
            print("[AVISO] No se detectaron columnas de ubicación. Se guardará TODO el fichero.")
        else:
            target = normalize_text(user_loc)
            mask = pd.Series(False, index=df.index)
            for c in loc_cols:
                try:
                    col_norm = df[c].astype(str).map(normalize_text)
                    mask = mask | col_norm.str.contains(re.escape(target), na=False)
                except Exception:
                    pass
            out_df = df[mask]
            print(f"[INFO] Filtrado por ubicación: {len(out_df)} filas (de {len(df)}).")

    # --- Relleno 'NA' y guardado ---
    out_df = out_df.fillna("NA")
    for c in out_df.columns:
        try:
            out_df[c] = out_df[c].astype(str).replace(r"^\s*$", "NA", regex=True)
        except Exception:
            pass

    if user_loc:
        slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(user_loc)).strip("_") or "ubicacion"
        out_file = OUTPUT_DIR / f"dataset_ubicacion_{slug}.csv"
    else:
        out_file = OUTPUT_DIR / "dataset_completo.csv"

    out_df.to_csv(out_file, index=False, sep=";", encoding="utf-8-sig")
    print(f"[OK] Guardado: {out_file.resolve()}")

    # Vista rápida
    try:
        print(out_df.head(10))
    except Exception:
        pass

if __name__ == "__main__":
    main()
