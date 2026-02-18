import csv
import io
import re
import unicodedata
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
from difflib import get_close_matches

CSV_URL = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"
HEADERS = {
    "User-Agent": "MateoScraperBot/9.0",
    "Accept": "*/*"
}

OUTPUT_DIR = Path("Obras_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_obras.csv"

TIMEOUT = 60

def normalize_text(s):
    """Modo B — mayúsculas, sin acentos, sin guiones, espacios simples."""
    res = "" 
    if s:
        s = str(s).upper()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = re.sub(r"-", " ", s)
        res = re.sub(r"\s+", " ", s).strip()
    return res

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

def clean(s):
    res = "NA"
    if s and str(s).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        res = normalize_text(s)
    return res

def clean_date(value):
    res = ""
    if value and value not in ("0", "0.0", "nan"):
        res = str(value).strip()
    return res

def parse_date_safe(v):
    """Parsea fecha robustamente con varios formatos."""
    v = clean_date(v)
    res = pd.NaT
    
    if v:
        # Usamos una lista de formatos y un flag para no usar breaks ni returns
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        found = False
        for fmt in formats:
            if not found:
                try:
                    res = datetime.strptime(v, fmt)
                    found = True
                except:
                    pass
    return res

def resolve_district(raw_name):
    res_no, res_nom = "NA", "NA"
    
    if raw_name:
        name = clean(raw_name)
        res_nom = name
        
        if name in MADRID_DISTRICTS:
            res_no = str(MADRID_DISTRICTS[name])
            res_nom = name
        elif name in ALIAS:
            key = normalize_text(ALIAS[name])
            res_no = str(MADRID_DISTRICTS[key])
            res_nom = key
        else:
            candidates = list(MADRID_DISTRICTS.keys())
            match = get_close_matches(name, candidates, n=1, cutoff=0.75)
            if match:
                k = match[0]
                res_no = str(MADRID_DISTRICTS[k])
                res_nom = k
                
    return res_no, res_nom

def main():
    print("\n=== Generando datasheet OBRAS ===")

    r = requests.get(CSV_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content), sep=";", dtype=str).fillna("")

    df["FECHA_INIC"] = df["FECHA_INIC"].apply(parse_date_safe)
    df["FECHA_FINA"] = df["FECHA_FINA"].apply(parse_date_safe)

    df["FECHA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])
    df = df.dropna(subset=["FECHA"])

    rows = []

    for _, r in df.iterrows():
        fecha = r["FECHA"]
        if fecha.year >= 2022:
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
        quoting=csv.QUOTE_NONE,
        escapechar="\\" # Añadido para evitar errores con QUOTE_NONE si hay ';'
    )

    print("\n[OK] Archivo generado →", OUT_FILE.resolve())
    print("Filas:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()