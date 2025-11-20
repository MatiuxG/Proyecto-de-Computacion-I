# -*- coding: utf-8 -*-
# Code in English, comments in Spanish

import io
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

# ================================
# CONFIG
# ================================

CSV_URL = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"

OUTPUT_DIR = Path("./Obras_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "datasheet_plazo_ejecucion.csv"

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
    return s.strip()

def extract_date_from_expediente(exp):
    """
    Convert formats like:
    711/2020/05971
    711202005971
    711/2020/05971-L2-012
    Into day/month/year
    """
    if not exp:
        return "", "", "", ""

    exp = str(exp).strip()

    # Remove suffixes like "-L2-012"
    exp = re.split(r"[- ]", exp)[0]

    # Remove all non-digits
    digits = re.sub(r"[^0-9]", "", exp)

    if len(digits) < 7:
        return "", "", "", ""

    dia = digits[0]
    mes = digits[1:3]
    año = digits[3:7]

    try:
        fecha = datetime(int(año), int(mes), int(dia)).strftime("%Y-%m-%d")
    except:
        fecha = ""

    return dia, mes, año, fecha

def get_first_district(name):
    if not name:
        return "", ""

    parts = re.split(r"[-–.,;/]+", name)
    first = normalize(parts[0]).lower()

    for code, dname in MADRID_DISTRICTS.items():
        if first.startswith(normalize(dname).lower()):
            return code, dname

    return "", name

# ================================
# MAIN
# ================================

def main():
    print("[INFO] Downloading CSV...")
    r = requests.get(CSV_URL, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content), sep=";", dtype=str).fillna("")

    if "DISTRITO_S" not in df.columns or "N_EXPEDIEN" not in df.columns:
        print("[ERROR] Missing expected columns in dataset")
        print("Columns found:", df.columns.tolist())
        return

    rows = []

    for _, row in df.iterrows():
        distrito_raw = row["DISTRITO_S"]
        expediente = row["N_EXPEDIEN"]

        no_dist, nombre_dist = get_first_district(distrito_raw)
        dia, mes, año, iso = extract_date_from_expediente(expediente)

        rows.append({
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": no_dist,
            "nombre_distrito": nombre_dist,
            "plazo_ejecucion_fecha_normalizada": iso
        })

    out = pd.DataFrame(rows)

    out = out[~(out.eq("").all(axis=1))]

    out = out[~((out["dia"] == "") & (out["mes"] == "") & (out["año"] == ""))]

    out = out.reset_index(drop=True)

    out.to_csv(OUT_CSV, sep=";", index=False, encoding="utf-8-sig")

    print(f"[OK] CSV generado → {OUT_CSV.resolve()} ({len(out)} filas)")

if __name__ == "__main__":
    main()