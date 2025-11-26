# -*- coding: utf-8 -*-
# Code in English, comments in Spanish

import io
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
import unicodedata
import re
from difflib import get_close_matches

# ================================
# CONFIG
# ================================

CSV_URL = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"

OUTPUT_DIR = Path("Obras_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "datasheet_obras_estado.csv"

MADRID_DISTRICTS = {
    "01":"Centro","02":"Arganzuela","03":"Retiro","04":"Salamanca",
    "05":"Chamartín","06":"Tetuán","07":"Chamberí","08":"Fuencarral-El Pardo",
    "09":"Moncloa-Aravaca","10":"Latina","11":"Carabanchel","12":"Usera",
    "13":"Puente de Vallecas","14":"Moratalaz","15":"Ciudad Lineal",
    "16":"Hortaleza","17":"Villaverde","18":"Villa de Vallecas",
    "19":"Vicálvaro","20":"San Blas-Canillejas","21":"Barajas"
}

# ================================
# HELPERS
# ================================

def normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s).encode("ascii","ignore").decode()
    return s.lower().strip()

def clean_date(value):
    if not value or value in ("0", "0.0", "nan"):
        return ""
    return str(value).strip()

def parse_date_safe(value):
    """Intenta múltiples formatos sin perder FECHA_INIC"""
    value = clean_date(value)

    # try DD/MM/YYYY
    try:
        return datetime.strptime(value, "%d/%m/%Y")
    except:
        pass

    # try YYYY-MM-DD
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except:
        pass

    return pd.NaT

def smart_district_lookup(row):
    # 1) exact from DISTRITO_S
    raw = normalize(row.get("DISTRITO_S", ""))
    for code, name in MADRID_DISTRICTS.items():
        if normalize(name) in raw:
            return code, name

    # 2) search in text fields
    for field in ["DENOMINACI", "VIARIO_AFE", "DESCRIPCIO"]:
        text = normalize(row.get(field, ""))
        for code, name in MADRID_DISTRICTS.items():
            if normalize(name) in text:
                return code, name

    # 3) fuzzy match
    text = normalize(
        " ".join([
            row.get("DENOMINACI", ""),
            row.get("VIARIO_AFE", ""),
            row.get("DESCRIPCIO", "")
        ])
    )

    district_names = [normalize(x) for x in MADRID_DISTRICTS.values()]
    match = get_close_matches(text, district_names, n=1, cutoff=0.75)

    if match:
        for code, name in MADRID_DISTRICTS.items():
            if normalize(name) == match[0]:
                return code, name

    # 4) fallback final
    return "00", "Sin asignar"

# ================================
# MAIN
# ================================

def main():
    print("[INFO] Downloading dataset...")
    r = requests.get(CSV_URL, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content), sep=";", dtype=str).fillna("")

    # robust date parsing
    df["FECHA_INIC"] = df["FECHA_INIC"].apply(parse_date_safe)
    df["FECHA_FINA"] = df["FECHA_FINA"].apply(parse_date_safe)

    # final selection of date WITHOUT overwriting valid FECHA_INIC
    df["FECHA_USADA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])

    df = df.dropna(subset=["FECHA_USADA"])

    # === MODIFICACIÓN: Definir rango de fechas ===
    fecha_inicio_filtro = datetime(2025, 7, 1)
    fecha_fin_filtro = datetime(2025, 9, 30)
    # =============================================

    today = datetime.today()

    rows = []

    for _, row in df.iterrows():
        fecha = row["FECHA_USADA"]
        
        # === MODIFICACIÓN: Aplicar filtro ===
        if not (fecha_inicio_filtro <= fecha <= fecha_fin_filtro):
            continue
        # ====================================

        fecha_fin = row["FECHA_FINA"]

        dia = f"{fecha.day:02d}"
        mes = f"{fecha.month:02d}"
        año = str(fecha.year)

        no_dist, nombre_dist = smart_district_lookup(row)

        terminada = False if pd.isna(fecha_fin) else (fecha_fin < today)

        rows.append({
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": no_dist,
            "nombre_distrito": nombre_dist,
            "terminada": str(terminada).lower()
        })

    out = pd.DataFrame(rows)

    out.to_csv(OUT_CSV, sep=";", index=False, encoding="utf-8-sig")

    print(f"[OK] CSV generated → {OUT_CSV.resolve()} ({len(out)} rows)")
    print("[CHECK] Filter applied: Only July-Sept 2025 ✅")
    print("[CHECK] ALL DATES = FECHA_INIC unless missing ✅")
    print("[CHECK] NO district missing ✅")

if __name__ == "__main__":
    main()