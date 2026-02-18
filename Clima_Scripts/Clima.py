import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJFZGR5ZnJhdGVyMkBnbWFpbC5jb20iLCJqdGkiOiIwZTRmNjVjYy03YTNiLTRjMzUtYjZiNS02YzJkOWM3YmNiZTMiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc2Mjk3MjI2NywidXNlcklkIjoiMGU0ZjY1Y2MtN2EzYi00YzM1LWI2YjUtNmMyZDljN2JjYmUzIiwicm9sZSI6IiJ9.31mijI-aiAuAuqZjYK9JrsK_1I7Jt3NRdkt0dgplHDg"
BASE_URL = "https://opendata.aemet.es/opendata/api"
DAILY_URL_TPL = f"{BASE_URL}/valores/climatologicos/diarios/datos/fechaini/{{start_str}}/fechafin/{{end_str}}/estacion/{{station_id}}"

OUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUT_FILE = OUT_DIR / "datasheet_clima.csv"

#para las peticiones de la api
HEADERS = {"accept": "application/json", "api_key": API_KEY}
ENCODING = "utf-8"
CSV_SEP = ";"

STATION_MAPPING = {
    "3195": [   #Retiro
        ("00", "Madrid"), ("01", "Centro"), ("02", "Arganzuela"), 
        ("03", "Retiro"), ("04", "Salamanca"), ("14", "Moratalaz"), ("15", "Ciudad Lineal")
    ],
    "3129": [   #Barajas
        ("16", "Hortaleza"), ("20", "San Blas-Canillejas"), ("21", "Barajas")
    ],
    "3194U": [  #Ciudad universitaria
        ("05", "Chamartín"), ("06", "Tetuán"), ("07", "Chamberí"), ("09", "Moncloa-Aravaca")
    ],
    "3196": [   #Vallecas
        ("13", "Puente de Vallecas"), ("18", "Villa de Vallecas"), ("19", "Vicálvaro")
    ],
    "3200": [   #Fuencarral/El Goloso
        ("08", "Fuencarral-El Pardo")
    ],
    "3191": [   #Cuatro Vientos
        ("10", "Latina"), ("11", "Carabanchel"), ("12", "Usera"), ("17", "Villaverde")
    ]
}

#columnas finales del csv
FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name", 
    "Temp_Media_°C", "Temp_Max_°C", "Temp_Min_°C", "Hora_Temp_Max", "Hora_Temp_Min", 
    "Precipitacion_mm", "Vel_Viento_Media_m/s", "Racha_Max_m/s", 
    "Presion_Max_hPa", "Presion_Min_hPa"
]

def req_aemet(url, label, result_list):
    #hace peticion a la api de aemet y guarda los datos en result_list
    time.sleep(0.6) #no quitar, es para no saturar a la api
    try:
        #primera peticion para obtener el enlace a los datos
        response = requests.get(url, headers=HEADERS, verify=True)
        if response.status_code == 429:
            #si nos bloquea por muchas peticiones, esperamos
            print(f"[{label}] cazados, esperando 5s...")
            time.sleep(5)
            response = requests.get(url, headers=HEADERS, verify=True)
            
        response.raise_for_status()
        data = response.json()
        
        if data.get("estado") == 200:
            link_datos = data["datos"]
            #segunda peticion para bajar el json real con los datos
            response_data = requests.get(link_datos, headers=HEADERS, verify=True)
            response_data.raise_for_status()
            print(f"[{label}] OK          ")
            #decodifica y guarda los datos en la lista
            records = json.loads(response_data.content.decode('iso-8859-1'))
            if records:
                result_list.extend(records)
        else:
            print(f"[{label}] Error API: {data.get('descripcion')} (Status: {data.get('estado')})")
    except Exception as e:
        print(f"  [Excepción] {e}")

def parse_record(record, district_code, district_name, output_list):
    #convierte un registro de la api a formato csv y lo agrega a output_list
    date_str = record.get("fecha", "")
    try:
        #extrae dia, mes y año de la fecha
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dia, mes, anio = dt.day, dt.month, dt.year
    except ValueError:
        dia, mes, anio = 0, 0, 0

    #limpia numeros convirtiendo comas a puntos
    def clean(val):
        if not val:
            val = "0"
        else:
            val = str(val).replace(",", ".")
        return val

    #crea el registro limpio y lo pone a la lista
    parsed_record = {
        "Dia": dia,
        "Mes": mes,
        "Año": anio,
        "district_code": district_code,
        "district_name": district_name,
        "Temp_Media_°C": clean(record.get("tmed")),
        "Temp_Max_°C": clean(record.get("tmax")),
        "Temp_Min_°C": clean(record.get("tmin")),
        "Hora_Temp_Max": record.get("horatmax", "0"),
        "Hora_Temp_Min": record.get("horatmin", "0"),
        "Precipitacion_mm": clean(record.get("prec")),
        "Vel_Viento_Media_m/s": clean(record.get("velmedia")),
        "Racha_Max_m/s": clean(record.get("racha")),
        "Presion_Max_hPa": clean(record.get("presMax")),
        "Presion_Min_hPa": clean(record.get("presMin"))
    }
    output_list.append(parsed_record)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global_start = datetime(2022, 1, 1)
    global_end = datetime(2025, 10, 31)
    
    print(f"[Info] Iniciando extracción desde {global_start.date()} hasta {global_end.date()}")
    all_data = []

    #vamos de mes a mes para no saturar la api
    current_date = global_start
    while current_date <= global_end:
        #calcula el ultimo dia del mes actual
        next_month = current_date.replace(day=28) + timedelta(days=4)
        next_month_start = next_month.replace(day=1)
        chunk_end = next_month_start - timedelta(days=1)
        if chunk_end > global_end:
            chunk_end = global_end
        #formatea las fechas para la url de la api
        start_str = current_date.strftime("%Y-%m-%dT00:00:00UTC")
        end_str = chunk_end.strftime("%Y-%m-%dT23:59:59UTC")        
        #procesa cada estacion meteorologica
        for station_id, districts_list in STATION_MAPPING.items():
            url = DAILY_URL_TPL.format(start_str=start_str, end_str=end_str, station_id=station_id)
            records = []
            #descarga los datos de esta estacion
            req_aemet(url, label=f"St:{station_id}", result_list=records)
            #procesa cada registro y lo replica para todos los distritos asociados
            if records:
                for record in records:
                    for district_code, district_name in districts_list:
                        parse_record(record, district_code, district_name, all_data)
        
        #avanza al siguiente mes
        current_date = next_month_start

    if not all_data:
        print("no se obtuvieron datos.")
    else:
        #convierte la lista a dataframe
        df = pd.DataFrame(all_data)
        #asegura que todas las columnas existen
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "0"
        
        #reordena columnas segun el orden definido
        df = df[FINAL_COLUMNS]
        #ordena por fecha y distrito
        df.sort_values(by=["Año", "Mes", "Dia", "district_code"], inplace=True)
        
        #elimina duplicados si los hay
        df.drop_duplicates(subset=["Año", "Mes", "Dia", "district_code"], inplace=True)

        #guarda el resultado final
        df.to_csv(OUT_FILE, sep=CSV_SEP, encoding=ENCODING, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\n[EXITO] Archivo generado: {OUT_FILE.resolve()}")
        print(f"        Total registros: {len(df)}")

if __name__ == "__main__":
    main()
