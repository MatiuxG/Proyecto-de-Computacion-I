import csv
import io
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta
from difflib import get_close_matches

import pandas as pd
import requests

#conf general
CSV_URL = "https://datos.madrid.es/egob/catalogo/300538-11514071-obras-planificadas-ejecucion.csv"
HEADERS = {
    "User-Agent": "MateoScraperBot/9.0",
    "Accept": "*/*",
}
TIMEOUT = 60
OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_obras.csv"

#limpieza de texto
def normalize_text(text): 
    if not text:
        return ""
    text = str(text).upper()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"-", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_text_or_na(text):
    if text and str(text).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        return normalize_text(text)
    return "NA"

def clean_date_str(value):
    if value and value not in ("0", "0.0", "nan"):
        return str(value).strip()
    return ""

def parse_date_safe(value):
    date_str = clean_date_str(value)
    if not date_str:
        return pd.NaT

    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return pd.NaT

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
    district_code, district_name = "NA", "NA"
    if not raw_name:
        return district_code, district_name
    cleaned_name = clean_text_or_na(raw_name)
    district_name = cleaned_name

    #coincidencia directa
    if cleaned_name in MADRID_DISTRICTS:
        district_code = str(MADRID_DISTRICTS[cleaned_name])
        return district_code, cleaned_name

    #alias conocido
    if cleaned_name in DISTRICT_ALIASES:
        alias_norm = normalize_text(DISTRICT_ALIASES[cleaned_name])
        district_code = str(MADRID_DISTRICTS.get(alias_norm, "NA"))
        return district_code, alias_norm

    #coincidencia aproximada (para errores de escritura)
    candidates = list(MADRID_DISTRICTS.keys())
    match = get_close_matches(cleaned_name, candidates, n=1, cutoff=0.75)
    if match:
        closest = match[0]
        district_code = str(MADRID_DISTRICTS[closest])
        district_name = closest

    return district_code, district_name

#proceso principal
def load_raw_data():
    #descarga el csv remoto y lo devuelve como DataFrame
    response = requests.get(CSV_URL, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), sep=";", dtype=str).fillna("")


def add_parsed_dates(df):
    #convierte FECHA_INIC y FECHA_FINA a datetime y crea columna FECHA preferente
    df = df.copy()
    df["FECHA_INIC"] = [parse_date_safe(value) for value in df["FECHA_INIC"]]
    df["FECHA_FINA"] = [parse_date_safe(value) for value in df["FECHA_FINA"]]
    # fecha se toma de inicio si existe, si no de fin
    df["FECHA"] = df["FECHA_INIC"].combine_first(df["FECHA_FINA"])
    return df.dropna(subset=["FECHA"])


FECHA_MINIMA = datetime(2022, 1, 1)


def build_output_rows(df):
    #convierte cada obra en una fila por cada dia que estuvo activa en su distrito
    #(antes ponia una sola fila por obra, asi que obras_activas en main.py
    #siempre quedaba a 1 - aqui lo expandimos para que el modelo sepa
    #cuantas obras hay realmente cada dia en cada distrito)
    rows = []
    today = datetime.today()

    for _, row in df.iterrows():
        fecha_inicio = row["FECHA_INIC"]
        fecha_fin = row["FECHA_FINA"]

        #si no hay fecha de inicio no sabemos cuando empezo, la saltamos
        if not isinstance(fecha_inicio, datetime):
            continue
        #si no hay fecha de fin asumimos que sigue en marcha (hasta hoy)
        if not isinstance(fecha_fin, datetime):
            fecha_fin = today
        #cap por si hay fechas absurdas en el origen
        if fecha_fin > today + timedelta(days=365 * 2):
            fecha_fin = today + timedelta(days=365 * 2)
        if fecha_inicio > fecha_fin:
            continue

        #recortamos al rango que nos interesa (no nos sirve nada anterior a 2022)
        dia_actual = max(fecha_inicio, FECHA_MINIMA)
        fecha_final = fecha_fin

        # el dataset a veces usa DISTRITO_S y otras DENOMINACI
        raw_district = row.get("DISTRITO_S", "") or row.get("DENOMINACI", "")
        district_code, district_name = resolve_district(raw_district)
        # una obra esta "terminada" si su fecha de fin ya paso
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
    #guarda el resultado final en csv y devuelve el DataFrame
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