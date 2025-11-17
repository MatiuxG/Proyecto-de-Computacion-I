# -*- coding: utf-8 -*-
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

OUT_DIR = Path("Clima_Scripts\Clima_Scripts\Resultados")
OUT_FILE = OUT_DIR / "datasheet_clima.csv"

HEADERS = {"accept": "application/json", "api_key": API_KEY}
ENCODING = "utf-8"
CSV_SEP = ";"

# Ventana de fechas (ejemplo: 2 meses atrás)
try:
    MONTHS_BACK = int(sys.argv[1]) if len(sys.argv) > 1 else 2
except Exception:
    MONTHS_BACK = 2

# Mapa de estaciones de Madrid a Códigos de Distrito
# (Simplificado, puedes ampliarlo si usas más estaciones)
STATION_TO_DISTRICT = {
    "3195": ("03", "Retiro"),           # Madrid, Retiro
    "3129": ("21", "Barajas"),          # Madrid, Aeropuerto
    "3194U": ("07", "Chamberí"),        # Madrid, C. Universitaria
    "3196": ("13", "Puente de Vallecas"),# Madrid, Vallecas
    "3200": ("08", "Fuencarral-El Pardo"),# Madrid, El Goloso
}

# Columnas finales solicitadas
FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "district_code", "district_name", 
    "Temp_Media_°C", "Temp_Max_°C", "Temp_Min_°C", "Hora_Temp_Max", "Hora_Temp_Min", 
    "Precipitacion_mm", "Vel_Viento_Media_m/s", "Racha_Max_m/s", 
    "Presion_Max_hPa", "Presion_Min_hPa", "Insolacion_h"
]

# ===================== Funciones =====================

def req_aemet(url: str, label: str = None) -> dict:
    print(f"[{label or 'req'}] GET {url}")
    r = requests.get(url, headers=HEADERS, verify=True)
    r.raise_for_status()
    data = r.json()
    
    if data.get("estado") == 200:
        r_data = requests.get(data["datos"], headers=HEADERS, verify=True)
        r_data.raise_for_status()
        # Decodificar manualmente para manejar acentos
        return json.loads(r_data.content.decode('iso-8859-1'))
        
    print(f"[{label or 'req'}] HTTP {r.status_code} -> {url}")
    if data.get("descripcion"):
        print(f"  [Error AEMET] {data['descripcion']}")
    return {}

def parse_record(record: dict, station_id: str) -> dict:
    district_code, district_name = STATION_TO_DISTRICT.get(station_id, ("NA", "NA"))
    
    # Extraer y formatear fecha
    date_str = record.get("fecha", "")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dia = dt.day
        mes = dt.month
        anio = dt.year
    except ValueError:
        dia, mes, anio = "NA", "NA", "NA"

    return {
        "Dia": dia,
        "Mes": mes,
        "Año": anio,
        "district_code": district_code,
        "district_name": district_name,
        "Temp_Media_°C": record.get("tmed", "NA").replace(",","."),
        "Temp_Max_°C": record.get("tmax", "NA").replace(",","."),
        "Temp_Min_°C": record.get("tmin", "NA").replace(",","."),
        "Hora_Temp_Max": record.get("horatmax", "NA"),
        "Hora_Temp_Min": record.get("horatmin", "NA"),
        "Precipitacion_mm": record.get("prec", "NA").replace(",","."),
        "Vel_Viento_Media_m/s": record.get("velmedia", "NA").replace(",","."),
        "Racha_Max_m/s": record.get("racha", "NA").replace(",","."),
        "Presion_Max_hPa": record.get("presMax", "NA").replace(",","."),
        "Presion_Min_hPa": record.get("presMin", "NA").replace(",","."),
        "Insolacion_h": record.get("insolac", "NA").replace(",","."),
        "date_iso_debug": date_str # Para depuración
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(months=MONTHS_BACK)
    start_str = start_date.strftime("%Y-%m-%dT00:00:00UTC")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59UTC")
    
    all_data = []
    
    for station_id in STATION_TO_DISTRICT.keys():
        try:
            url = DAILY_URL_TPL.format(start_str=start_str, end_str=end_str, station_id=station_id)
            records = req_aemet(url, label=f"clima-{station_id}")
            if records:
                all_data.extend([parse_record(r, station_id) for r in records])
        except Exception as e:
            print(f"  [Error] Fallo al procesar estación {station_id}: {e}")

    if not all_data:
        print("[Aviso] No se obtuvieron datos de clima.")
        # Guardar CSV vacío con cabeceras si no hay datos
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        df = pd.DataFrame(all_data)
        # Asegurar que solo tengamos las columnas finales y en el orden correcto
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "NA"
        df = df[FINAL_COLUMNS]

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