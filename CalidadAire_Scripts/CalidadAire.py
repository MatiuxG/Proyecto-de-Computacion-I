# -*- coding: utf-8 -*-
"""
Datasheet unificado de Calidad del Aire (Madrid)
Versión profesional Modo B con catálogo de estaciones.
- SIN NA en distritos (excepto estaciones desconocidas)
- Expansión diaria desde datasets mensuales
- Compatible con RapidMiner
"""

# ============================================================
# IMPORTS
# ============================================================

import csv
import io
import re
import unicodedata
from pathlib import Path
from datetime import date as date_cls

import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# CONFIG
# ============================================================

HEADERS = {
    "User-Agent": "MateoAirScraper/2.0",
    "Accept": "*/*"
}

PAGES = [
    # Ficha original
    "https://datos.madrid.es/sites/v/index.jsp?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD"
]

# Catálogos oficiales
STATION_CATALOG_CANDIDATES = [
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.csv",
]

# Fallback completo (del script original)
STATION_FALLBACK = {
    "004": ("09", "MONCLOA ARAVACA"),
    "008": ("04", "SALAMANCA"),
    "011": ("05", "CHAMARTIN"),
    "016": ("15", "CIUDAD LINEAL"),
    "017": ("17", "VILLAVERDE"),
    "018": ("11", "CARABANCHEL"),
    "024": ("09", "MONCLOA ARAVACA"),
    "027": ("21", "BARAJAS"),
    "035": ("01", "CENTRO"),
    "036": ("14", "MORATALAZ"),
    "038": ("06", "TETUAN"),
    "039": ("08", "FUENCARRAL EL PARDO"),
    "040": ("13", "PUENTE DE VALLECAS"),
    "047": ("02", "ARGANZUELA"),
    "048": ("05", "CHAMARTIN"),
    "049": ("03", "RETIRO"),
    "050": ("05", "CHAMARTIN"),
    "054": ("18", "VILLA DE VALLECAS"),
    "055": ("21", "BARAJAS"),
    "056": ("11", "CARABANCHEL"),
    "057": ("16", "HORTALEZA"),
    "058": ("08", "FUENCARRAL EL PARDO"),
    "059": ("21", "BARAJAS"),
    "060": ("08", "FUENCARRAL EL PARDO"),
}

OUTPUT_DIR = Path("CalidadAire_Scripts\Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "datasheet_calidad_aire.csv"


# ============================================================
# NORMALIZATION (MODO B)
# ============================================================

def normalize_text(s):
    if not s:
        return "NA"
    s = str(s).upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"-", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("", "NAN", "NONE", "NULL"):
        return "NA"
    return s


# ============================================================
# CATALOG LOADING
# ============================================================

def load_catalog_from_url(url):
    try:
        df = pd.read_csv(url, dtype=str, sep=";", encoding="latin-1")
    except:
        try:
            df = pd.read_csv(url, dtype=str)
        except:
            return None

    df.columns = [normalize_text(c) for c in df.columns]

    candidates_code = ["CODIGO_ESTACION", "COD_ESTACION", "CODIGO", "ESTACION"]
    candidates_name = ["NOMBRE", "NOMBRE_ESTACION"]
    candidates_district = ["DISTRITO", "NOMBRE_DISTRITO"]
    candidates_dcode = ["COD_DISTRITO", "CODIGO_DISTRITO"]

    stc = next((c for c in candidates_code if c in df.columns), None)
    name = next((c for c in candidates_name if c in df.columns), None)
    dist_name = next((c for c in candidates_district if c in df.columns), None)
    dist_code = next((c for c in candidates_dcode if c in df.columns), None)

    if not stc:
        return None

    lookup = {}

    for _, r in df.iterrows():
        code = re.sub(r"\D", "", str(r.get(stc, ""))).zfill(3)
        if not code:
            continue

        dname = normalize_text(r.get(dist_name, "")) if dist_name else "NA"
        dcode = re.sub(r"\D", "", str(r.get(dist_code, ""))).zfill(2) if dist_code else "NA"

        lookup[code] = (dcode, dname)

    return lookup


def build_station_lookup():
    # Try online catalogs
    for url in STATION_CATALOG_CANDIDATES:
        lookup = load_catalog_from_url(url)
        if lookup:
            print(f"[Lookup] Loaded: {url} ({len(lookup)} stations)")
            return lookup

    # Fallback
    print("[Lookup] Using fallback station catalog.")
    return {
        k: (v[0], normalize_text(v[1]))
        for k, v in STATION_FALLBACK.items()
    }


# ============================================================
# SCRAPING
# ============================================================

def find_csvs(url):
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        from urllib.parse import urljoin
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".csv"):
                out.append(href if href.startswith("http") else urljoin(url, href))
        return out
    except:
        return []


def load_table(url):
    try:
        r = requests.get(url, headers=HEADERS)
        data = r.content

        for enc in ["utf-8-sig", "utf-8", "latin-1", None]:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, dtype=str)
                    if df.shape[1] > 1:
                        return df
                except:
                    pass
        return pd.DataFrame()
    except:
        return pd.DataFrame()


# ============================================================
# EXPANSION DIARIA (D01...D31)
# ============================================================

def extract_station_code(s):
    if not s:
        return None
    m = re.search(r"28079(\d{3})", str(s))
    return m.group(1) if m else None


def expand_month(df, lookup):
    if df.empty:
        return pd.DataFrame()

    df.columns = [normalize_text(c) for c in df.columns]

    ycol = next((c for c in df.columns if "ANO" in c or "AÑO" in c or "YEAR" in c), None)
    mcol = "MES" if "MES" in df.columns else None
    pcol = next((c for c in df.columns if "PUNTO_MUESTREO" in c), None)

    if not ycol or not mcol or not pcol:
        return pd.DataFrame()

    dcols = [c for c in df.columns if re.match(r"^D\d{2}$", c)]
    if not dcols:
        return pd.DataFrame()

    out = []

    for _, r in df.iterrows():
        year = re.findall(r"\d{4}", str(r.get(ycol, "2022")))
        year = year[0] if year else "2022"
        month = r.get(mcol, "01").strip().zfill(2)

        station_raw = r.get(pcol, "")
        station = extract_station_code(station_raw)

        # lookup
        if station in lookup:
            no_dist, name_dist = lookup[station]
        else:
            no_dist, name_dist = ("NA", "NA")

        for dcol in dcols:
            day = int(dcol[1:])
            val = r.get(dcol, "").strip()

            if val in ("", "-", "NA", None):
                continue

            m = re.search(r"-?\d+(?:[.,]\d+)?", val)
            if not m:
                continue

            num_val = float(m.group(0).replace(",", "."))

            out.append({
                "dia": str(day).zfill(2),
                "mes": month,
                "año": year,
                "no_distrito": no_dist,
                "nombre_distrito": name_dist,
                "valor_calidad_aire": num_val
            })

    return pd.DataFrame(out)


# ============================================================
# MAIN
# ============================================================

def process_page(url, lookup):
    csvs = find_csvs(url)
    dfs = []
    for u in csvs:
        df_raw = load_table(u)
        df_exp = expand_month(df_raw, lookup)
        if not df_exp.empty:
            dfs.append(df_exp)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    df["año"] = df["año"].astype(str).str.extract(r"(\d{4})")[0]
    df = df[df["año"].astype(int) >= 2022]

    return df


def main():
    print("=== Generando datasheet Calidad del Aire ===")

    lookup = build_station_lookup()

    parts = []
    for url in PAGES:
        df = process_page(url, lookup)
        if not df.empty:
            parts.append(df)

    if not parts:
        pd.DataFrame(columns=[
            "dia","mes","año","no_distrito","nombre_distrito","valor_calidad_aire"
        ]).to_csv(OUT_FILE, sep=";", index=False, encoding="utf-8-sig")
        print("[OK] Archivo vacío generado")
        return

    final = pd.concat(parts, ignore_index=True)

    final.to_csv(
        OUT_FILE,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE
    )

    print("[OK] Archivo generado →", OUT_FILE.resolve())
    print("Filas:", len(final))
    print(final.head())


if __name__ == "__main__":
    main()
