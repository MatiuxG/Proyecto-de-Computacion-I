import csv
import io
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches

import pandas as pd
import requests

#configuracion general
CSV_URL = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"
HEADERS = {
    "User-Agent": "MateoScraperBot/9.0",
    "Accept": "*/*",
}
TIMEOUT = 60
OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_obras.csv"

def normalize_text(text):
    #pasa a mayusculas y quita tildes, guiones y espacios sobrantes
    resultado = ""
    if text:
        text = str(text).upper()
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = re.sub(r"-", " ", text)
        resultado = re.sub(r"\s+", " ", text).strip()
    return resultado

def clean_text_or_na(text):
    #normaliza el texto o devuelve NA si esta vacio
    resultado = "NA"
    if text and str(text).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        resultado = normalize_text(text)
    return resultado

def clean_date_str(value):
    #limpia el texto de la fecha o lo deja vacio
    resultado = ""
    if value and value not in ("0", "0.0", "nan"):
        resultado = str(value).strip()
    return resultado

def parse_date_safe(value):
    #convierte el texto a fecha probando varios formatos
    resultado = pd.NaT
    date_str = clean_date_str(value)
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        if date_str and pd.isna(resultado):
            try:
                resultado = datetime.strptime(date_str, fmt)
            except ValueError:
                resultado = pd.NaT
    return resultado

DISTRICT_ITEMS = [
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

MADRID_DISTRICTS = {normalize_text(name): code for name, code in DISTRICT_ITEMS}
DISTRICT_ALIASES = {
    "VALLECAS PTE": "PUENTE DE VALLECAS",
    "VALLECAS-PTE": "PUENTE DE VALLECAS",
    "FUENCARRAL EL-PARDO": "FUENCARRAL EL PARDO",
    "SAN BLAS": "SAN BLAS CANILLEJAS",
}

def resolve_district(raw_name):
    #devuelve (codigo, nombre) del distrito a partir del nombre crudo
    district_code, district_name = "NA", "NA"
    if raw_name:
        cleaned_name = clean_text_or_na(raw_name)
        district_name = cleaned_name
        if cleaned_name in MADRID_DISTRICTS:
            district_code = str(MADRID_DISTRICTS[cleaned_name])
        elif cleaned_name in DISTRICT_ALIASES:
            alias_norm = normalize_text(DISTRICT_ALIASES[cleaned_name])
            district_code = str(MADRID_DISTRICTS.get(alias_norm, "NA"))
            district_name = alias_norm
        else:
            #coincidencia aproximada para erratas de escritura
            match = get_close_matches(cleaned_name, list(MADRID_DISTRICTS.keys()), n=1, cutoff=0.75)
            if match:
                district_code = str(MADRID_DISTRICTS[match[0]])
                district_name = match[0]
    return district_code, district_name

def load_raw_data():
    #descarga el csv remoto de obras
    response = requests.get(CSV_URL, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), sep=";", dtype=str).fillna("")

def add_parsed_dates(df):
    #convierte fecha inicio/fin a datetime y elige la fecha de referencia
    df = df.copy()
    df["FECHA_INIC"] = [parse_date_safe(value) for value in df["FECHA_INIC"]]
    df["FECHA_FINA"] = [parse_date_safe(value) for value in df["FECHA_FINA"]]
    #usa la de inicio y, si no hay, la de fin
    df["FECHA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])
    return df.dropna(subset=["FECHA"])

def build_output_rows(df):
    #convierte cada obra en una fila estandar (dia/mes/año/distrito)
    rows = []
    today = datetime.today()
    for _, row in df.iterrows():
        fecha = row["FECHA"]
        if fecha.year >= 2022:
            dia = f"{fecha.day:02d}"
            mes = f"{fecha.month:02d}"
            anio = str(fecha.year)
            #el dataset usa a veces DISTRITO_S y otras DENOMINACI
            raw_district = row.get("DISTRITO_S", "") or row.get("DENOMINACI", "")
            district_code, district_name = resolve_district(raw_district)
            #la obra esta terminada si su fecha fin ya paso
            finished = False
            if isinstance(row["FECHA_FINA"], datetime):
                finished = row["FECHA_FINA"] < today
            rows.append({
                "dia": dia,
                "mes": mes,
                "año": anio,
                "no_distrito": district_code,
                "nombre_distrito": district_name,
                "terminada": str(finished).lower(),
            })
    return rows

def save_output(rows):
    #guarda el resultado final en csv
    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        OUT_FILE,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    return out_df

def main():
    raw_df = load_raw_data()
    df_with_dates = add_parsed_dates(raw_df)
    rows = build_output_rows(df_with_dates)
    out_df = save_output(rows)
    print("\n[OK] Archivo generado ", OUT_FILE.resolve())
    print(out_df.head(10))

if __name__ == "__main__":
    main()
