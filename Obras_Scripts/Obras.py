import csv
import io
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta
from difflib import get_close_matches

import pandas as pd
import requests


#csv del portal con las obras en ejecucion
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
    #mayusculas, sin tildes ni guiones
    res = ""
    if text:
        texto = str(text).upper()
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(char for char in texto if unicodedata.category(char) != "Mn")
        texto = re.sub(r"-", " ", texto)
        res = re.sub(r"\s+", " ", texto).strip()
    return res


def clean_text_or_na(text):
    #texto normalizado, o "NA" si esta vacio
    res = "NA"
    if text and str(text).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        res = normalize_text(text)
    return res


def clean_date_str(value):
    #descarta marcadores "0"/"nan" del campo fecha
    res = ""
    if value and value not in ("0", "0.0", "nan"):
        res = str(value).strip()
    return res


def parse_date_safe(value):
    #parsea la fecha probando formatos conocidos
    res = pd.NaT
    date_str = clean_date_str(value)

    if date_str:
        encontrado = False
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            if not encontrado:
                try:
                    res = datetime.strptime(date_str, fmt)
                    encontrado = True
                except ValueError:
                    pass

    return res


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

#nombres alternativos del csv
DISTRICT_ALIASES = {
    "VALLECAS PTE": "PUENTE DE VALLECAS",
    "VALLECAS-PTE": "PUENTE DE VALLECAS",
    "FUENCARRAL EL-PARDO": "FUENCARRAL EL PARDO",
    "SAN BLAS": "SAN BLAS CANILLEJAS",
}


def resolve_district(raw_name):
    #devuelve (codigo, nombre) del distrito normalizando el nombre crudo
    district_code = "NA"
    district_name = "NA"

    if raw_name:
        cleaned_name = clean_text_or_na(raw_name)
        district_name = cleaned_name

        #1) coincidencia directa
        if cleaned_name in MADRID_DISTRICTS:
            district_code = str(MADRID_DISTRICTS[cleaned_name])
            district_name = cleaned_name
        #2) alias conocido
        elif cleaned_name in DISTRICT_ALIASES:
            alias_norm = normalize_text(DISTRICT_ALIASES[cleaned_name])
            district_code = str(MADRID_DISTRICTS.get(alias_norm, "NA"))
            district_name = alias_norm
        else:
            #3) coincidencia aproximada (cutoff alto para evitar fallos)
            candidates = list(MADRID_DISTRICTS.keys())
            match = get_close_matches(cleaned_name, candidates, n=1, cutoff=0.75)
            if match:
                closest = match[0]
                district_code = str(MADRID_DISTRICTS[closest])
                district_name = closest

    return district_code, district_name


def load_raw_data():
    #descarga el csv remoto y rellena vacios con ""
    response = requests.get(CSV_URL, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    res = pd.read_csv(io.BytesIO(response.content), sep=";", dtype=str).fillna("")
    return res


def add_parsed_dates(df):
    #parsea FECHA_INIC y FECHA_FINA y crea la columna FECHA
    df = df.copy()
    df["FECHA_INIC"] = [parse_date_safe(value) for value in df["FECHA_INIC"]]
    df["FECHA_FINA"] = [parse_date_safe(value) for value in df["FECHA_FINA"]]
    df["FECHA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])
    res = df.dropna(subset=["FECHA"])
    return res


FECHA_MINIMA = datetime(2022, 1, 1)


def build_output_rows(df):
    #expande cada obra en una fila por dia activo en su distrito
    rows = []
    today = datetime.today()

    for _, row in df.iterrows():
        fecha_inicio = row["FECHA_INIC"]
        fecha_fin = row["FECHA_FINA"]
        valido = isinstance(fecha_inicio, datetime)

        if valido:
            #sin fecha de fin asumimos que sigue activa hasta hoy
            if not isinstance(fecha_fin, datetime):
                fecha_fin = today
            #cap por fechas absurdas en el origen
            if fecha_fin > today + timedelta(days=365 * 2):
                fecha_fin = today + timedelta(days=365 * 2)
            if fecha_inicio > fecha_fin:
                valido = False

        if valido:
            #recortamos al rango que nos interesa
            dia_actual = max(fecha_inicio, FECHA_MINIMA)
            fecha_final = fecha_fin

            #la columna del distrito cambia entre versiones del csv
            raw_district = row.get("DISTRITO_S", "") or row.get("DENOMINACI", "")
            district_code, district_name = resolve_district(raw_district)

            #la obra esta terminada si la fecha fin ya paso
            finished = isinstance(row["FECHA_FINA"], datetime) and row["FECHA_FINA"] < today
            terminada_txt = str(finished).lower()

            while dia_actual <= fecha_final:
                rows.append({
                    "dia": f"{dia_actual.day:02d}",
                    "mes": f"{dia_actual.month:02d}",
                    "año": str(dia_actual.year),
                    "no_distrito": district_code,
                    "nombre_distrito": district_name,
                    "terminada": terminada_txt,
                })
                dia_actual = dia_actual + timedelta(days=1)

    return rows


def save_output(rows):
    #escribe el csv final
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
    #baja el csv, parsea fechas, expande por dias y guarda
    raw_df = load_raw_data()
    df_with_dates = add_parsed_dates(raw_df)
    rows = build_output_rows(df_with_dates)
    out_df = save_output(rows)
    print("\n[OK] Archivo generado ", OUT_FILE.resolve())
    print(out_df.head(10))


if __name__ == "__main__":
    main()
