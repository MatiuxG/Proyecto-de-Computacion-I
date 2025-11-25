# -*- coding: utf-8 -*-
import sys, csv, re, json, unicodedata
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests

# ===================== Config =====================
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJFZGR5ZnJhdGVyMkBnbWFpbC5jb20iLCJqdGkiOiIwZTRmNjVjYy03YTNiLTRjMzUtYjZiNS02YzJkOWM3YmNiZTMiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc2Mjk3MjI2NywidXNlcklkIjoiMGU0ZjY1Y2MtN2EzYi00YzM1LWI2YjUtNmMyZDljN2JjYmUzIiwicm9sZSI6IiJ9.31mijI-aiAuAuqZjYK9JrsK_1I7Jt3NRdkt0dgplHDg"
BASE_URL = "https://opendata.aemet.es/opendata/api"
STATIONS_URL = f"{BASE_URL}/valores/climatologicos/inventarioestaciones/todasestaciones"
DAILY_URL_TPL = f"{BASE_URL}/valores/climatologicos/diarios/datos/fechaini/{{start_str}}/fechafin/{{end_str}}/estacion/{{station_id}}"

OUT_DIR = Path("Clima_Scripts/Clima_Scripts/Resultados") # Ajuste de ruta para compatibilidad Linux/Windows
OUT_FILE = OUT_DIR / "datasheet_clima.csv"

HEADERS = {"accept": "application/json", "api_key": API_KEY}
ENCODING = "utf-8"
CSV_SEP = ";"

# Mapeo de Estaciones AEMET -> Lista de Distritos (Código, Nombre)
# Se asigna la estación más cercana a cada grupo de distritos para cubrir del 00 al 21.
STATION_MAPPING = {
    "3195": [   # Estación: Madrid - Retiro
        ("00", "Madrid"), 
        ("01", "Centro"), 
        ("02", "Arganzuela"), 
        ("03", "Retiro"), 
        ("04", "Salamanca"), 
        ("14", "Moratalaz"), 
        ("15", "Ciudad Lineal")
    ],
    "3129": [   # Estación: Madrid - Barajas
        ("16", "Hortaleza"), 
        ("20", "San Blas-Canillejas"), 
        ("21", "Barajas")
    ],
    "3194U": [  # Estación: Madrid - Ciudad Universitaria
        ("05", "Chamartín"), 
        ("06", "Tetuán"), 
        ("07", "Chamberí"), 
        ("09", "Moncloa-Aravaca")
    ],
    "3196": [   # Estación: Madrid - Vallecas
        ("13", "Puente de Vallecas"), 
        ("18", "Villa de Vallecas"), 
        ("19", "Vicálvaro")
    ],
    "3200": [   # Estación: Madrid - Fuencarral/El Goloso
        ("08", "Fuencarral-El Pardo")
    ],
    "3191": [   # Estación: Madrid - Cuatro Vientos
        ("10", "Latina"), 
        ("11", "Carabanchel"), 
        ("12", "Usera"), 
        ("17", "Villaverde")
    ]
}

FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name", 
    "Temp_Media_°C", "Temp_Max_°C", "Temp_Min_°C", "Hora_Temp_Max", "Hora_Temp_Min", 
    "Precipitacion_mm", "Vel_Viento_Media_m/s", "Racha_Max_m/s", 
    "Presion_Max_hPa", "Presion_Min_hPa"
]

# ===================== Funciones =====================

def req_aemet(url: str, label: str = None) -> dict:
    print(f"[{label or 'req'}] GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, verify=True)
        r.raise_for_status()
        data = r.json()
        
        if data.get("estado") == 200:
            r_data = requests.get(data["datos"], headers=HEADERS, verify=True)
            r_data.raise_for_status()
            return json.loads(r_data.content.decode('iso-8859-1'))
            
        print(f"[{label or 'req'}] HTTP {r.status_code} -> {url}")
        if data.get("descripcion"):
            print(f"  [Error AEMET] {data['descripcion']}")
        return {}
    except Exception as e:
        print(f"  [Excepción] {e}")
        return {}

def parse_record(record: dict, district_code: str, district_name: str) -> dict:
    # Procesamos la fecha
    date_str = record.get("fecha", "")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dia = dt.day
        mes = dt.month
        anio = dt.year
    except ValueError:
        dia, mes, anio = 0, 0, 0 

    return {
        "Dia": dia,
        "Mes": mes,
        "Año": anio,
        "district_code": district_code,
        "district_name": district_name,
        "Temp_Media_°C": record.get("tmed", "0").replace(",","."),
        "Temp_Max_°C": record.get("tmax", "0").replace(",","."),
        "Temp_Min_°C": record.get("tmin", "0").replace(",","."),
        "Hora_Temp_Max": record.get("horatmax", "0"),
        "Hora_Temp_Min": record.get("horatmin", "0"),
        "Precipitacion_mm": record.get("prec", "0").replace(",","."),
        "Vel_Viento_Media_m/s": record.get("velmedia", "0").replace(",","."),
        "Racha_Max_m/s": record.get("racha", "0").replace(",","."),
        "Presion_Max_hPa": record.get("presMax", "0").replace(",","."),
        "Presion_Min_hPa": record.get("presMin", "0").replace(",","."),
        "date_iso_debug": date_str 
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- FECHAS: JULIO-SEPTIEMBRE 2025 ---
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 9, 30)
    
    start_str = start_date.strftime("%Y-%m-%dT00:00:00UTC")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59UTC")
    
    print(f"[Info] Extrayendo datos desde {start_str} hasta {end_str}")

    all_data = []
    
    # Iterar por cada estación definida en el mapeo
    for station_id, districts_list in STATION_MAPPING.items():
        print(f"--- Procesando Estación {station_id} para {len(districts_list)} distritos ---")
        try:
            url = DAILY_URL_TPL.format(start_str=start_str, end_str=end_str, station_id=station_id)
            records = req_aemet(url, label=f"clima-{station_id}")
            
            if records:
                # Si obtenemos datos de la estación, los replicamos para cada distrito asociado
                for record in records:
                    for d_code, d_name in districts_list:
                        all_data.append(parse_record(record, d_code, d_name))
            else:
                print(f"  [Aviso] Sin datos devueltos para estación {station_id}")

        except Exception as e:
            print(f"  [Error] Fallo al procesar estación {station_id}: {e}")

    if not all_data:
        print("[Aviso] No se obtuvieron datos de clima para ninguna zona.")
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        df = pd.DataFrame(all_data)
        # Asegurar columnas finales
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "0"
        df = df[FINAL_COLUMNS]

    # Ordenar por fecha y distrito para limpieza
    df.sort_values(by=["Año", "Mes", "Dia", "district_code"], inplace=True)

    df.to_csv(
        OUT_FILE, 
        sep=CSV_SEP, 
        encoding=ENCODING, 
        index=False,
        quoting=csv.QUOTE_MINIMAL
    )
    
    print(f"[OK] Datasheet de clima escrito -> {OUT_FILE.resolve()} ({len(df)} filas)")

if __name__ == "__main__":
    main()