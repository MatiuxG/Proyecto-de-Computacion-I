# -*- coding: utf-8 -*-
"""
Datasheet unificado OBRAS Madrid
Versión PRO reconstruida (Modo B distritos: SIN acentos, SIN guiones, MAYÚSCULAS)

✔ Descarga y unifica datos desde 2022
✔ Normaliza distritos con 4 niveles (exacto / alias / fuzzy / fallback)
✔ Limpieza total de fechas y campos corruptos
✔ Compatible con RapidMiner
✔ Igual arquitectura interna que emergencias_scraper.py
"""

import csv
import io
import re
import unicodedata
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
from difflib import get_close_matches

# ============================================================
# CONFIG
# ============================================================

CSV_URL = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"

HEADERS = {
    "User-Agent": "MateoScraperBot/9.0",
    "Accept": "*/*"
}

OUTPUT_DIR = Path("Obras_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "datasheet_obras.csv"

TIMEOUT = 60

# ============================================================
# NORMALIZACIÓN MODO B
# ============================================================

def normalize_text(s):
    """Modo B — mayúsculas, sin acentos, sin guiones, espacios simples."""
    if not s:
        return ""
    s = str(s).upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"-", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

MADRID_DISTRICTS = {
    normalize_text("CENTRO"): 1,
    normalize_text("ARGANZUELA"): 2,
    normalize_text("RETIRO"): 3,
    normalize_text("SALAMANCA"): 4,
    normalize_text("CHAMARTIN"): 5,
    normalize_text("TETUAN"): 6,
    normalize_text("CHAMBERI"): 7,
    normalize_text("FUENCARRAL EL PARDO"): 8,
    normalize_text("MONCLOA ARAVACA"): 9,
    normalize_text("LATINA"): 10,
    normalize_text("CARABANCHEL"): 11,
    normalize_text("USERA"): 12,
    normalize_text("PUENTE DE VALLECAS"): 13,
    normalize_text("MORATALAZ"): 14,
    normalize_text("CIUDAD LINEAL"): 15,
    normalize_text("HORTALEZA"): 16,
    normalize_text("VILLAVERDE"): 17,
    normalize_text("VILLA DE VALLECAS"): 18,
    normalize_text("VICALVARO"): 19,
    normalize_text("SAN BLAS CANILLEJAS"): 20,
    normalize_text("BARAJAS"): 21,
}

ALIAS = {
    "VALLECAS PTE": "PUENTE DE VALLECAS",
    "VALLECAS-PTE": "PUENTE DE VALLECAS",
    "FUENCARRAL EL-PARDO": "FUENCARRAL EL PARDO",
    "SAN BLAS": "SAN BLAS CANILLEJAS",
}

# ============================================================
# UTILIDADES
# ============================================================

def clean(s):
    if not s or str(s).strip().upper() in ("", "NAN", "NONE", "NULL"):
        return "NA"
    return normalize_text(s)

def clean_date(value):
    if not value or value in ("0", "0.0", "nan"):
        return ""
    return str(value).strip()

def parse_date_safe(v):
    """Parsea fecha robustamente con varios formatos."""
    v = clean_date(v)
    if not v:
        return pd.NaT

    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(v, fmt)
        except:
            pass

    return pd.NaT

# ============================================================
# DISTRITOS — Lógica completa
# ============================================================

def resolve_district(raw_name):
    """Devuelve (no_distrito, nombre_distrito) con 4 niveles de resolución."""
    if not raw_name:
        return "NA", "NA"

    name = clean(raw_name)

    # Nivel 1 — exacto
    if name in MADRID_DISTRICTS:
        return str(MADRID_DISTRICTS[name]), name

    # Nivel 2 — alias
    if name in ALIAS:
        key = normalize_text(ALIAS[name])
        return str(MADRID_DISTRICTS[key]), key

    # Nivel 3 — fuzzy
    candidates = list(MADRID_DISTRICTS.keys())
    match = get_close_matches(name, candidates, n=1, cutoff=0.75)
    if match:
        k = match[0]
        return str(MADRID_DISTRICTS[k]), k

    # Nivel 4 — fallback
    return "NA", name

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    print("\n=== Generando datasheet OBRAS ===")

    r = requests.get(CSV_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content), sep=";", dtype=str).fillna("")

    # Fechas
    df["FECHA_INIC"] = df["FECHA_INIC"].apply(parse_date_safe)
    df["FECHA_FINA"] = df["FECHA_FINA"].apply(parse_date_safe)

    # Elegir fecha válida
    df["FECHA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])
    df = df.dropna(subset=["FECHA"])

    rows = []

    for _, r in df.iterrows():
        fecha = r["FECHA"]
        if fecha.year < 2022:
            continue

        dia = f"{fecha.day:02d}"
        mes = f"{fecha.month:02d}"
        año = str(fecha.year)

        no_dist, nom_dist = resolve_district(
            r.get("DISTRITO_S", "") or r.get("DENOMINACI", "")
        )

        finished = False
        if isinstance(r["FECHA_FINA"], datetime):
            finished = r["FECHA_FINA"] < datetime.today()

        rows.append({
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": no_dist,
            "nombre_distrito": nom_dist,
            "terminada": str(finished).lower()
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUT_FILE,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE
    )

    print("\n[OK] Archivo generado →", OUT_FILE.resolve())
    print("Filas:", len(out))
    print(out.head(10))


if __name__ == "__main__":
    main()
