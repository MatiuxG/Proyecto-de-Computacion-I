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

OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_obras.csv"

TIMEOUT = 60

def normalize_text(text, salida):
    #convierte a mayusculas y quita tildes y guiones
    res = ""
    if text:
        text = str(text).upper()
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = re.sub(r"-", " ", text)
        res = re.sub(r"\s+", " ", text).strip()
    salida.append(res)

def build_madrid_districts(salida):
    #crea diccionario de distritos usando texto normalizado
    items = [
        ("CENTRO", 1),
        ("ARGANZUELA", 2),
        ("RETIRO", 3),
        ("SALAMANCA", 4),
        ("CHAMARTIN", 5),
        ("TETUAN", 6),
        ("CHAMBERI", 7),
        ("FUENCARRAL EL PARDO", 8),
        ("MONCLOA ARAVACA", 9),
        ("LATINA", 10),
        ("CARABANCHEL", 11),
        ("USERA", 12),
        ("PUENTE DE VALLECAS", 13),
        ("MORATALAZ", 14),
        ("CIUDAD LINEAL", 15),
        ("HORTALEZA", 16),
        ("VILLAVERDE", 17),
        ("VILLA DE VALLECAS", 18),
        ("VICALVARO", 19),
        ("SAN BLAS CANILLEJAS", 20),
        ("BARAJAS", 21),
    ]
    dicc = {}
    for name, code in items:
        out_norm = []
        normalize_text(name, out_norm)
        dicc[out_norm[0]] = code
    salida.append(dicc)

_dist_out = []
build_madrid_districts(_dist_out)
MADRID_DISTRICTS = _dist_out[0]

ALIAS = {
    "VALLECAS PTE": "PUENTE DE VALLECAS",
    "VALLECAS-PTE": "PUENTE DE VALLECAS",
    "FUENCARRAL EL-PARDO": "FUENCARRAL EL PARDO",
    "SAN BLAS": "SAN BLAS CANILLEJAS",
}

def clean(text, salida):
    #limpia texto y devuelve NA si esta vacio
    res = "NA"
    if text and str(text).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        out_norm = []
        normalize_text(text, out_norm)
        res = out_norm[0]
    salida.append(res)

def clean_date(value, salida):
    #limpia fecha como texto
    res = ""
    if value and value not in ("0", "0.0", "nan"):
        res = str(value).strip()
    salida.append(res)

def parse_date_safe(value, salida):
    #parsea fecha con varios formatos
    out_clean = []
    clean_date(value, out_clean)
    v = out_clean[0]
    res = pd.NaT

    if v:
        #usamos una lista de formatos y un flag para no usar breaks
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        found = False
        for fmt in formats:
            if not found:
                try:
                    res = datetime.strptime(v, fmt)
                    found = True
                except:
                    pass
    salida.append(res)

def resolve_district(raw_name, salida):
    #resuelve el distrito usando nombre directo, alias o parecido
    res_no, res_nom = "NA", "NA"

    if raw_name:
        out_clean = []
        clean(raw_name, out_clean)
        name = out_clean[0]
        res_nom = name

        if name in MADRID_DISTRICTS:
            res_no = str(MADRID_DISTRICTS[name])
            res_nom = name
        elif name in ALIAS:
            out_norm = []
            normalize_text(ALIAS[name], out_norm)
            key = out_norm[0]
            res_no = str(MADRID_DISTRICTS[key])
            res_nom = key
        else:
            candidates = list(MADRID_DISTRICTS.keys())
            match = get_close_matches(name, candidates, n=1, cutoff=0.75)
            if match:
                k = match[0]
                res_no = str(MADRID_DISTRICTS[k])
                res_nom = k

    salida.append((res_no, res_nom))

def main():
    print("\n=== Generando datasheet OBRAS ===")

    r = requests.get(CSV_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content), sep=";", dtype=str).fillna("")

    #convierte fechas con funcion segura
    fecha_inic_vals = []
    for value in df["FECHA_INIC"]:
        out_date = []
        parse_date_safe(value, out_date)
        fecha_inic_vals.append(out_date[0])
    df["FECHA_INIC"] = fecha_inic_vals

    fecha_fina_vals = []
    for value in df["FECHA_FINA"]:
        out_date = []
        parse_date_safe(value, out_date)
        fecha_fina_vals.append(out_date[0])
    df["FECHA_FINA"] = fecha_fina_vals

    df["FECHA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])
    df = df.dropna(subset=["FECHA"])

    rows = []

    for _, r in df.iterrows():
        fecha = r["FECHA"]
        if fecha.year >= 2022:
            dia = f"{fecha.day:02d}"
            mes = f"{fecha.month:02d}"
            año = str(fecha.year)

            out_dist = []
            resolve_district(
                r.get("DISTRITO_S", "") or r.get("DENOMINACI", ""),
                out_dist
            )
            no_dist, nom_dist = out_dist[0]

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