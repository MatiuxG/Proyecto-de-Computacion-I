import sys, csv, re, json, unicodedata
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJFZGR5ZnJhdGVyMkBnbWFpbC5jb20iLCJqdGkiOiIwZTRmNjVjYy03YTNiLTRjMzUtYjZiNS02YzJkOWM3YmNiZTMiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc2Mjk3MjI2NywidXNlcklkIjoiMGU0ZjY1Y2MtN2EzYi00YzM1LWI2YjUtNmMyZDljN2JjYmUzIiwicm9sZSI6IiJ9.31mijI-aiAuAuqZjYK9JrsK_1I7Jt3NRdkt0dgplHDg"
BASE_URL = "https://opendata.aemet.es/opendata/api"
DAILY_URL_TPL = f"{BASE_URL}/valores/climatologicos/diarios/datos/fechaini/{{start_str}}/fechafin/{{end_str}}/estacion/{{station_id}}"
OUT_DIR = Path("Clima_Scripts/Resultados")
OUT_FILE = OUT_DIR / "datasheet_clima.csv"
HEADERS = {"accept": "application/json", "api_key": API_KEY}
ENCODING = "utf-8"
CSV_SEP = ";"

STATION_MAPPING = {
    "3195": [   # Madrid - Retiro
        ("00", "Madrid"), ("01", "Centro"), ("02", "Arganzuela"), 
        ("03", "Retiro"), ("04", "Salamanca"), ("14", "Moratalaz"), ("15", "Ciudad Lineal")
    ],
    "3129": [   # Madrid - Barajas
        ("16", "Hortaleza"), ("20", "San Blas-Canillejas"), ("21", "Barajas")
    ],
    "3194U": [  # Madrid - Ciudad Universitaria
        ("05", "Chamartín"), ("06", "Tetuán"), ("07", "Chamberí"), ("09", "Moncloa-Aravaca")
    ],
    "3196": [   # Madrid - Vallecas
        ("13", "Puente de Vallecas"), ("18", "Villa de Vallecas"), ("19", "Vicálvaro")
    ],
    "3200": [   # Madrid - Fuencarral/El Goloso
        ("08", "Fuencarral-El Pardo")
    ],
    "3191": [   # Madrid - Cuatro Vientos
        ("10", "Latina"), ("11", "Carabanchel"), ("12", "Usera"), ("17", "Villaverde")
    ]
}

FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name", 
    "Temp_Media_°C", "Temp_Max_°C", "Temp_Min_°C", "Hora_Temp_Max", "Hora_Temp_Min", 
    "Precipitacion_mm", "Vel_Viento_Media_m/s", "Racha_Max_m/s", 
    "Presion_Max_hPa", "Presion_Min_hPa"
]


def req_aemet(url: str, label: str = None) -> dict:
    #pausa para no saturar la API
    time.sleep(0.6) 
    print(f"[{label or 'req'}] GET ...", end="\r")
    try:
        r = requests.get(url, headers=HEADERS, verify=True)
        if r.status_code == 429:
            print(f"[{label}] Rate Limit! Esperando 5s...")
            time.sleep(5)
            r = requests.get(url, headers=HEADERS, verify=True)
            
        r.raise_for_status()
        data = r.json()
        
        if data.get("estado") == 200:
            link_datos = data["datos"]
            # Segunda petición para bajar el JSON real
            r_data = requests.get(link_datos, headers=HEADERS, verify=True)
            r_data.raise_for_status()
            print(f"[{label}] OK          ")
            return json.loads(r_data.content.decode('iso-8859-1')) #FLAG
            
        print(f"[{label}] Error API: {data.get('descripcion')} (Status: {data.get('estado')})")
        return {} #FLAG
    except Exception as e:
        print(f"  [Excepción] {e}")
        return {} #FLAG

def parse_record(record: dict, district_code: str, district_name: str) -> dict:
    date_str = record.get("fecha", "")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dia, mes, anio = dt.day, dt.month, dt.year
    except ValueError:
        dia, mes, anio = 0, 0, 0 

    #para limpiar números (coma a punto)
    def clean(val):
        if not val: return "0"
        return str(val).replace(",", ".")

    return {
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

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_start = datetime(2022, 1, 1)
    global_end = datetime(2025, 10, 31)
    
    print(f"[Info] Iniciando extracción desde {global_start.date()} hasta {global_end.date()}")
    all_data = []

    #vamos de mes a mes para no saturar
    current_date = global_start
    while current_date <= global_end:
        next_month = current_date.replace(day=28) + timedelta(days=4)
        next_month_start = next_month.replace(day=1)
        chunk_end = next_month_start - timedelta(days=1)
        
        if chunk_end > global_end:
            chunk_end = global_end
        start_str = current_date.strftime("%Y-%m-%dT00:00:00UTC")
        end_str = chunk_end.strftime("%Y-%m-%dT23:59:59UTC")
        
        print(f"\n--- Procesando bloque: {current_date.date()} al {chunk_end.date()} ---")
        
        for station_id, districts_list in STATION_MAPPING.items():
            url = DAILY_URL_TPL.format(start_str=start_str, end_str=end_str, station_id=station_id)
            records = req_aemet(url, label=f"St:{station_id}")
            
            if records:
                for record in records:
                    for d_code, d_name in districts_list:
                        all_data.append(parse_record(record, d_code, d_name))
        
        current_date = next_month_start

    if not all_data:
        print("[Aviso] No se obtuvieron datos.")
    else:
        df = pd.DataFrame(all_data)
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "0"
        
        df = df[FINAL_COLUMNS]
        df.sort_values(by=["Año", "Mes", "Dia", "district_code"], inplace=True)
        
        #buscamos y eliminamos duplicados
        df.drop_duplicates(subset=["Año", "Mes", "Dia", "district_code"], inplace=True)

        df.to_csv(OUT_FILE, sep=CSV_SEP, encoding=ENCODING, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\n[EXITO] Archivo generado: {OUT_FILE.resolve()}")
        print(f"        Total registros: {len(df)}")

if __name__ == "__main__":
    main()
