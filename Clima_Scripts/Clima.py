import csv
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

#variables del .env de la raiz; si no hay dotenv tiramos del entorno normal
try:
    from dotenv import load_dotenv
    ROOT_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

API_KEY = os.getenv("AEMET_API_KEY", "")

if not API_KEY:
    print("[ERROR] Falta AEMET_API_KEY. Configurala en el archivo .env.")
    sys.exit(1)

BASE_URL = "https://opendata.aemet.es/opendata/api"
DAILY_URL_TPL = f"{BASE_URL}/valores/climatologicos/diarios/datos/fechaini/{{start_str}}/fechafin/{{end_str}}/estacion/{{station_id}}"

OUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUT_FILE = OUT_DIR / "datasheet_clima.csv"

HEADERS = {"accept": "application/json", "api_key": API_KEY}
ENCODING = "utf-8"
CSV_SEP = ";"

#cada estacion AEMET cubre varios distritos; replicamos los datos para no dejar huecos
STATION_MAPPING = {
    "3195": [   #Retiro
        ("00", "Madrid"), ("01", "Centro"), ("02", "Arganzuela"),
        ("03", "Retiro"), ("04", "Salamanca"), ("14", "Moratalaz"), ("15", "Ciudad Lineal"),
    ],
    "3129": [   #Barajas
        ("16", "Hortaleza"), ("20", "San Blas-Canillejas"), ("21", "Barajas"),
    ],
    "3194U": [  #Ciudad Universitaria
        ("05", "Chamartín"), ("06", "Tetuán"), ("07", "Chamberí"), ("09", "Moncloa-Aravaca"),
    ],
    "3196": [   #Vallecas
        ("13", "Puente de Vallecas"), ("18", "Villa de Vallecas"), ("19", "Vicálvaro"),
    ],
    "3200": [   #Fuencarral / El Goloso
        ("08", "Fuencarral-El Pardo"),
    ],
    "3191": [   #Cuatro Vientos
        ("10", "Latina"), ("11", "Carabanchel"), ("12", "Usera"), ("17", "Villaverde"),
    ],
}

#columnas finales del csv en orden
FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name",
    "Temp_Media_°C", "Temp_Max_°C", "Temp_Min_°C", "Hora_Temp_Max", "Hora_Temp_Min",
    "Precipitacion_mm", "Vel_Viento_Media_m/s", "Racha_Max_m/s",
    "Presion_Max_hPa", "Presion_Min_hPa",
]


def req_aemet(url, label, result_list):
    #peticion en dos pasos a AEMET; vuelca los registros en result_list
    time.sleep(0.6)  #no saturar la api
    try:
        response = requests.get(url, headers=HEADERS, verify=True)
        #si nos limitan, esperamos y reintentamos
        if response.status_code == 429:
            print(f"[{label}] cazados, esperando 5s...")
            time.sleep(5)
            response = requests.get(url, headers=HEADERS, verify=True)

        response.raise_for_status()
        data = response.json()

        if data.get("estado") == 200:
            link_datos = data["datos"]
            #segunda peticion para el json real
            response_data = requests.get(link_datos, headers=HEADERS, verify=True)
            response_data.raise_for_status()
            print(f"[{label}] OK")
            #AEMET sirve este json en iso-8859-1
            records = json.loads(response_data.content.decode("iso-8859-1"))
            if records:
                result_list.extend(records)
        else:
            print(f"[{label}] Error API: {data.get('descripcion')} (Status: {data.get('estado')})")
    except Exception as e:
        print(f"  [Excepcion] {e}")


def limpiar_numero(val):
    #cambia coma decimal por punto; "0" si esta vacio
    res = "0"
    if val:
        res = str(val).replace(",", ".")
    return res


def parse_record(record, district_code, district_name, output_list):
    #convierte un registro de la api al formato del csv
    date_str = record.get("fecha", "")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dia, mes, anio = dt.day, dt.month, dt.year
    except ValueError:
        #fecha invalida; se filtra mas adelante
        dia, mes, anio = 0, 0, 0

    parsed_record = {
        "Dia": dia,
        "Mes": mes,
        "Año": anio,
        "district_code": district_code,
        "district_name": district_name,
        "Temp_Media_°C": limpiar_numero(record.get("tmed")),
        "Temp_Max_°C": limpiar_numero(record.get("tmax")),
        "Temp_Min_°C": limpiar_numero(record.get("tmin")),
        "Hora_Temp_Max": record.get("horatmax", "0"),
        "Hora_Temp_Min": record.get("horatmin", "0"),
        "Precipitacion_mm": limpiar_numero(record.get("prec")),
        "Vel_Viento_Media_m/s": limpiar_numero(record.get("velmedia")),
        "Racha_Max_m/s": limpiar_numero(record.get("racha")),
        "Presion_Max_hPa": limpiar_numero(record.get("presMax")),
        "Presion_Min_hPa": limpiar_numero(record.get("presMin")),
    }
    output_list.append(parsed_record)


def main():
    #descarga mes a mes y replica los datos de cada estacion en sus distritos
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global_start = datetime(2022, 1, 1)
    global_end = datetime(2025, 10, 31)

    print(f"[Info] Iniciando extraccion desde {global_start.date()} hasta {global_end.date()}")
    all_data = []

    current_date = global_start
    while current_date <= global_end:
        #ultimo dia del mes saltando al dia 1 del mes siguiente
        next_month = current_date.replace(day=28) + timedelta(days=4)
        next_month_start = next_month.replace(day=1)
        chunk_end = next_month_start - timedelta(days=1)
        if chunk_end > global_end:
            chunk_end = global_end

        #formato que pide AEMET
        start_str = current_date.strftime("%Y-%m-%dT00:00:00UTC")
        end_str = chunk_end.strftime("%Y-%m-%dT23:59:59UTC")

        for station_id, districts_list in STATION_MAPPING.items():
            url = DAILY_URL_TPL.format(start_str=start_str, end_str=end_str, station_id=station_id)
            records = []
            req_aemet(url, label=f"St:{station_id}", result_list=records)
            if records:
                #cada registro se duplica en todos los distritos de la estacion
                for record in records:
                    for district_code, district_name in districts_list:
                        parse_record(record, district_code, district_name, all_data)

        current_date = next_month_start

    if all_data:
        df = pd.DataFrame(all_data)

        #aseguramos que estan todas las columnas finales
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "0"

        df = df[FINAL_COLUMNS]
        df.sort_values(by=["Año", "Mes", "Dia", "district_code"], inplace=True)
        df.drop_duplicates(subset=["Año", "Mes", "Dia", "district_code"], inplace=True)

        df.to_csv(OUT_FILE, sep=CSV_SEP, encoding=ENCODING, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\n[EXITO] Archivo generado: {OUT_FILE.resolve()}")
        print(f"        Total registros: {len(df)}")
    else:
        print("no se obtuvieron datos.")


if __name__ == "__main__":
    main()
