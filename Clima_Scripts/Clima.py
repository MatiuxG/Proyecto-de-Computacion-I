# -*- coding: utf-8 -*-
"""
Clima (AEMET) — Datasheet único normalizado (contrato común)
- Descarga DIARIOS (últimos 180 días) y MENSUALES (año actual) de la estación Madrid Retiro (3195)
- Emite un ÚNICO CSV: ./Clima_Scripts/Resultados/datasheet_clima.csv
- Contrato común: dataset,event_type,date,time,datetime,district_code,district_name,lat,lon,location,severity,value,units,source_id
- Formato largo por métrica (cada métrica = 1 fila)
"""

import csv
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================
# Configuración
# ==============================

# Estación AEMET (Madrid, Retiro)
STATION_RETIRO = "3195"

# Enriquecimiento común para joins (alineado con tus otros datasheets)
DISTRICT_CODE = "03"                               # Retiro
DISTRICT_NAME = "Retiro"
LOCATION_NAME = "Madrid — Estación Retiro (AEMET 3195)"
# Coordenadas opcionales (útiles si luego mapeas). Déjalas vacías si no quieres coords.
LAT = "40.413"    # opcional
LON = "-3.683"    # opcional

# Período para DIARIOS (≈ 6 meses)
DAILY_DAYS = 180

# Año MENSUAL (año actual)
MONTHLY_YEAR = datetime.today().year

# API Keys (proporcionadas)
API_KEY_DAILY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJFZGR5ZnJhdGVyMkBnbWFpbC5jb20iLCJqdGkiOiJhZTQ4Zjg0Zi1hZTMxLTQ5MzgtYTFkNy1jYzlmODhjOTI5MWQiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc2MTQxNDUxOSwidXNlcklkIjoiYWU0OGY4NGYtYWUzMS00OTM4LWExZDctY2M5Zjg4YzkyOTFkIiwicm9sZSI6IiJ9.z0VMMvrTjwl5MsQuf5YWTdaOtXP7ctRYfasHDfZSE30"
API_KEY_MONTHLY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtYXRlb2dhbDI0MDlAZ21haWwuY29tIiwianRpIjoiOTZjZjYxMjMtN2EzMy00OTkxLWJkMGMtNjNmZDFiYmFkN2E0IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE3NTkzMTg5NTgsInVzZXJJZCI6Ijk2Y2Y2MTIzLTdhMzMtNDk5MS1iZDBjLTYzZmQxYmJhZDdhNCIsInJvbGUiOiIifQ.pnpqxv1fmE9ZeMVTb4VkvZZF8NuffxQrcSFWpYqBVKg"

# Endpoints AEMET
AEMET_DAILY_META = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{ini}/fechafin/{fin}/estacion/{est}"
AEMET_MONTHLY_META = "https://opendata.aemet.es/opendata/api/valores/climatologicos/mensualesanuales/datos/anioini/{anio}/aniofin/{anio}/estacion/{est}"

# Salida ÚNICA
OUT_DIR = Path("./Clima_Scripts/Resultados")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "datasheet_clima.csv"

# CSV RapidMiner-friendly
CSV_SEP = ";"
CSV_QUOTING = csv.QUOTE_NONE
CSV_ESCAPE = "\\"
CSV_ENCODING = "utf-8-sig"

# Contrato común (orden fijo)
CONTRACT = [
    "dataset","event_type","date","time","datetime",
    "district_code","district_name","lat","lon","location",
    "severity","value","units","source_id",
]

# ==============================
# Utilidades de red / IO
# ==============================

def fetch_json(url: str, headers: Optional[Dict]=None, timeout: int=60, retries: int=3, label: str="") -> Optional[dict|list]:
    """Descarga JSON con reintentos (mensajes de diagnóstico mínimos)."""
    last_exc = None
    for i in range(1, retries+1):
        try:
            r = requests.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code >= 400:
                print(f"[{label or 'req'}] HTTP {r.status_code} -> {url}")
                time.sleep(1.2*i); continue
            try:
                return r.json()
            except json.JSONDecodeError:
                print(f"[{label or 'req'}] Respuesta no es JSON válido.")
                time.sleep(1.2*i); continue
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"[{label or 'req'}] Error de red: {e}")
            time.sleep(1.2*i)
    if last_exc:
        print(f"[{label or 'req'}] Falló tras reintentos: {last_exc}")
    return None

def export_contract_csv(rows: List[Dict], out_path: Path):
    """Escribe el contrato común, vacíos como 'NA', separador ';', sin comillas, UTF-8 BOM."""
    prepared: List[Dict] = []
    for row in rows:
        base = {c: "" for c in CONTRACT}
        base.update({k: ("" if v is None else v) for k, v in row.items()})
        prepared.append(base)

    with open(out_path, "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=CONTRACT, delimiter=CSV_SEP,
                           quoting=CSV_QUOTING, escapechar=CSV_ESCAPE)
        w.writeheader()
        for r in prepared:
            out = {k: ("NA" if str(v).strip()=="" else v) for k, v in r.items()}
            w.writerow(out)

# ==============================
# Transformaciones → contrato
# ==============================

def daily_to_contract(items: List[Dict], source_url: str) -> List[Dict]:
    """Convierte DIARIOS AEMET a formato largo por métrica (con distrito Retiro poblado)."""
    rows: List[Dict] = []
    if not items:
        return rows

    unit_map = {
        "tmed": "°C", "tmax": "°C", "tmin": "°C",
        "prec": "mm",
        "velmedia": "m/s", "racha": "m/s",
        "presmax": "hPa", "presmin": "hPa",
        "sol": "h",
    }
    key_map = {
        "tmed": "tmed",
        "tmax": "tmax",
        "tmin": "tmin",
        "prec": "prec",
        "velmedia": "velmedia",
        "racha": "racha",
        "presmax": "presMax",
        "presmin": "presMin",
        "sol": "sol",
    }

    for rec in items:
        date = rec.get("fecha","")
        for etype, k in key_map.items():
            val = rec.get(k, "")
            rows.append({
                "dataset": "clima",
                "event_type": etype,
                "date": date,
                "time": "",
                "datetime": "",
                "district_code": DISTRICT_CODE,
                "district_name": DISTRICT_NAME,
                "lat": LAT,
                "lon": LON,
                "location": LOCATION_NAME,
                "severity": "",
                "value": val,
                "units": unit_map.get(etype, ""),
                "source_id": source_url,
            })
    return rows

def monthly_to_contract(items: List[Dict], year: int, source_url: str) -> List[Dict]:
    """Convierte MENSUALES AEMET (un año) a formato largo por métrica (con distrito Retiro poblado)."""
    rows: List[Dict] = []
    if not items:
        return rows

    unit_map = {
        "tm_mes": "°C", "tm_max": "°C", "tm_min": "°C",
        "ta_max": "°C", "ta_min": "°C",
        "hr": "%", "p_mes": "mm", "n_sol": "h",
    }
    key_map = {
        "tm_mes": "tm_mes", "tm_max": "tm_max", "tm_min": "tm_min",
        "ta_max": "ta_max", "ta_min": "ta_min",
        "hr": "hr", "p_mes": "p_mes", "n_sol": "n_sol",
    }

    for rec in items:
        mtxt = str(rec.get("mes","")).strip()
        try:
            m = int(float(mtxt)) if mtxt else 0
        except Exception:
            m = 0
        # Fecha representativa del mes (o promedio anual)
        if 1 <= m <= 12:
            date = f"{year:04d}-{m:02d}-01"
        elif m == 13:
            date = f"{year:04d}-12-31"
        else:
            date = f"{year:04d}-01-01"

        for etype, k in key_map.items():
            val = rec.get(k, "")
            rows.append({
                "dataset": "clima",
                "event_type": etype,
                "date": date,
                "time": "",
                "datetime": "",
                "district_code": DISTRICT_CODE,
                "district_name": DISTRICT_NAME,
                "lat": LAT,
                "lon": LON,
                "location": LOCATION_NAME,
                "severity": "",
                "value": val,
                "units": unit_map.get(etype, ""),
                "source_id": source_url,
            })
    return rows

# ==============================
# Descargas AEMET
# ==============================

def download_daily(est: str, days: int) -> Tuple[List[Dict], str]:
    """Descarga DIARIOS últimos N días; devuelve (json, url_datos)."""
    end = datetime.today()
    start = end - timedelta(days=days)
    ini = start.strftime("%Y-%m-%dT00:00:00UTC")
    fin = end.strftime("%Y-%m-%dT23:59:59UTC")

    meta_url = AEMET_DAILY_META.format(ini=ini, fin=fin, est=est)
    meta = fetch_json(meta_url, headers={"api_key": API_KEY_DAILY},
                      timeout=60, retries=3, label="aemet_diarios_meta")
    if not meta:
        return [], meta_url
    data_url = meta.get("datos","")
    if not data_url:
        return [], meta_url

    items = fetch_json(data_url, headers=None, timeout=90, retries=3, label="aemet_diarios_datos")
    if not items or not isinstance(items, list):
        return [], data_url

    try:
        items = sorted(items, key=lambda x: x.get("fecha",""), reverse=True)
    except Exception:
        pass
    return items, data_url

def download_monthly(est: str, year: int) -> Tuple[List[Dict], str]:
    """Descarga MENSUALES del año indicado; devuelve (json, url_datos)."""
    meta_url = AEMET_MONTHLY_META.format(anio=year, est=est)
    meta = fetch_json(meta_url, headers={"accept":"application/json","api_key": API_KEY_MONTHLY},
                      timeout=60, retries=3, label="aemet_mensuales_meta")
    if not meta:
        return [], meta_url
    data_url = meta.get("datos","")
    if not data_url:
        return [], meta_url

    items = fetch_json(data_url, headers=None, timeout=90, retries=3, label="aemet_mensuales_datos")
    if not items or not isinstance(items, list):
        return [], data_url
    return items, data_url

# ==============================
# Main (batch, una sola salida)
# ==============================

def main():
    print("[Clima] Batch start…")

    # Descarga DIARIOS
    daily_json, daily_src = download_daily(STATION_RETIRO, DAILY_DAYS)
    daily_rows = daily_to_contract(daily_json, daily_src)
    print(f"[Clima] Daily rows: {len(daily_rows)}")

    # Descarga MENSUALES
    monthly_json, monthly_src = download_monthly(STATION_RETIRO, MONTHLY_YEAR)
    monthly_rows = monthly_to_contract(monthly_json, MONTHLY_YEAR, monthly_src)
    print(f"[Clima] Monthly rows: {len(monthly_rows)}")

    # Combina y exporta en un ÚNICO CSV
    combined = daily_rows + monthly_rows
    export_contract_csv(combined, OUT_FILE)
    print(f"[Clima] Combined → {OUT_FILE.resolve()} (rows={len(combined)})")
    print("[Clima] Done.")

if __name__ == "__main__":
    main()
