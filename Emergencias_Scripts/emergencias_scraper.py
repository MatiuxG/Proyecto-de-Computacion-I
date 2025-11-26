# -*- coding: utf-8 -*-
"""
Datasheet unificado Emergencias Madrid - MODIFICADO
Reglas:
- Bomberos -> hora 00:00, código_emergencia="Incendio", código_emergencia_num=10
- SAMUR -> usa Año, Mes, Hora Solicitud, Código, Distrito
- Servicios Sociales -> usa Código Distrito, Distrito, Fecha Cita, Tipo Supuesto Urgente
- SALIDA: Agregado por día y distrito con conteo total (cantidad_emergencias)
"""

import csv
import io
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "MateoScraperBot/3.0",
    "Accept": "*/*"
}
TIMEOUT = 60

OUTPUT_DIR = Path("Emergencias_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FINAL = OUTPUT_DIR / "datasheet_emergencias.csv"

URL_BOMBEROS = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=fa677996afc6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SAMUR    = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=50d7d35982d6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SOCIALES = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"

DISTRITOS_MADRID = {
    "CENTRO": 1,
    "ARGANZUELA": 2,
    "RETIRO": 3,
    "SALAMANCA": 4,
    "CHAMARTIN": 5, "CHAMARTÍN": 5,
    "TETUAN": 6, "TETUÁN": 6,
    "CHAMBERI": 7, "CHAMBERÍ": 7,
    "FUENCARRAL": 8, "FUENCARRAL-EL PARDO": 8, "FUENCARRAL - EL PARDO": 8,
    "MONCLOA": 9, "MONCLOA-ARAVACA": 9, "MONCLOA - ARAVACA": 9,
    "LATINA": 10,
    "CARABANCHEL": 11,
    "USERA": 12,
    "PUENTE DE VALLECAS": 13,
    "MORATALAZ": 14,
    "CIUDAD LINEAL": 15,
    "HORTALEZA": 16,
    "VILLAVERDE": 17,
    "VILLA DE VALLECAS": 18,
    "VICALVARO": 19, "VICÁLVARO": 19,
    "SAN BLAS": 20, "SAN BLAS-CANILLEJAS": 20, "SAN BLAS - CANILLEJAS": 20,
    "BARAJAS": 21
}

def find_csv_url(page_url):
    try:
        r = requests.get(page_url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "csv" in href.lower():
                from urllib.parse import urljoin
                return href if href.startswith("http") else urljoin(page_url, href)
    except Exception as e:
        print(f"Error buscando URL CSV: {e}")
    return None

def load_csv(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.content
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str)
                if df.shape[1] > 1:
                    return df
            except:
                pass
        try:
            txt = data.decode("utf-8", errors="ignore")
            return pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str)
        except:
            print("  [AVISO] Archivo no es CSV válido → omitido")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        return pd.DataFrame()

MES_MAP = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "setiembre": "09", "octubre": "10",
    "noviembre": "11", "diciembre": "12"
}

def mes_to_num(value):
    if not value: return "NA"
    v = str(value).strip().lower()
    if v.isdigit(): return v.zfill(2)
    return MES_MAP.get(v, "NA")

def get_distrito_id(nombre_raw, codigo_raw=None):
    if codigo_raw and str(codigo_raw).isdigit():
        num = int(codigo_raw)
        if 1 <= num <= 21:
            return str(num)
    if not nombre_raw or pd.isna(nombre_raw):
        return "NA"
    clean_name = str(nombre_raw).upper().strip()
    clean_name = re.sub(r'^\d+[\.\-\s]+', '', clean_name) 
    if clean_name in DISTRITOS_MADRID:
        return str(DISTRITOS_MADRID[clean_name])
    return "NA"

def get_bomberos():
    print("[Bomberos] Buscando CSV...")
    csv_url = find_csv_url(URL_BOMBEROS)
    if not csv_url:
        print("No CSV bomberos")
        return pd.DataFrame()
    df = load_csv(csv_url)
    if df.empty: return df
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    year_col = next((c for c in df.columns if "año" in c or "year" in c or "anio" in c), None)
    month_col = next((c for c in df.columns if "mes" in c), None)
    dist_col = next((c for c in df.columns if "distrito" in c), None)
    out = []
    for _, row in df.iterrows():
        raw_dist = row.get(dist_col, "NA")
        out.append({
            "dataset": "bomberos",
            "dia": "01",
            "mes": mes_to_num(row.get(month_col, "NA")),
            "año": str(row.get(year_col, "NA")),
            "no_distrito": get_distrito_id(raw_dist),
            "nombre_distrito": str(raw_dist)
            # Quitamos hora y código específico porque agregaremos
        })
    return pd.DataFrame(out)

def get_samur():
    print("[SAMUR] Buscando CSV...")
    csv_url = find_csv_url(URL_SAMUR)
    if not csv_url:
        print("No CSV SAMUR")
        return pd.DataFrame()
    df = load_csv(csv_url)
    if df.empty: return df
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    year_col = next((c for c in df.columns if "año" in c or "anio" in c or "year" in c), None)
    month_col = next((c for c in df.columns if "mes" in c), None)
    # hora_col = next((c for c in df.columns if "hora_solicitud" in c), None) # No necesaria para el conteo diario
    dist_cols = [c for c in df.columns if "distrito" in c]
    dist_col = dist_cols[0] if dist_cols else None
    
    out = []
    for _, row in df.iterrows():
        dia = "01" # Los CSVs agregados mensuales suelen poner 1 por defecto si no hay día explícito
        mes = mes_to_num(row.get(month_col, "NA"))
        año = str(row.get(year_col, "NA"))
        raw_dist = row.get(dist_col, "NA")
        out.append({
            "dataset": "samur",
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": get_distrito_id(raw_dist),
            "nombre_distrito": str(raw_dist)
        })
    return pd.DataFrame(out)

def get_sociales():
    print("[Servicios Sociales] Buscando CSV...")
    csv_url = find_csv_url(URL_SOCIALES)
    if not csv_url:
        print("No CSV sociales")
        return pd.DataFrame()
    df = load_csv(csv_url)
    if df.empty: return df
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    dcode_col = next((c for c in df.columns if "código_distrito" in c or "cod_distrito" in c), "código_distrito")
    dname_col = next((c for c in df.columns if "distrito" in c and "código" not in c), "distrito")
    fecha_col = next((c for c in df.columns if "fecha" in c), "fecha_cita")
    
    out = []
    for _, row in df.iterrows():
        fecha_raw = str(row.get(fecha_col, "")).strip()
        dia, mes, año = "NA", "NA", "NA"
        m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", fecha_raw)
        if m: dia, mes, año = m.groups()
        
        raw_code = row.get(dcode_col, None)
        raw_name = row.get(dname_col, None)
        out.append({
            "dataset": "servicios_sociales",
            "dia": dia,
            "mes": mes_to_num(mes),
            "año": año,
            "no_distrito": get_distrito_id(raw_name, codigo_raw=raw_code),
            "nombre_distrito": str(raw_name)
        })
    return pd.DataFrame(out)

def main():
    df1 = get_bomberos()
    df2 = get_samur()
    df3 = get_sociales()

    final = pd.concat([df1, df2, df3], ignore_index=True, sort=False)

    if "dataset" in final.columns:
        final = final.drop(columns=["dataset"])
    
    # --- FILTRO FECHAS (Julio-Septiembre 2025) ---
    print("Filtrando datos para Julio-Septiembre 2025...")
    mask = (final["año"].astype(str) == "2025") & (final["mes"].astype(str).str.zfill(2).isin(["07", "08", "09"]))
    final = final[mask]

    # --- NUEVA LÓGICA: AGREGACIÓN POR DÍA Y DISTRITO ---
    print("Agrupando emergencias por día y distrito...")
    
    # Agrupamos por las columnas de fecha y lugar
    # .size() cuenta el número de filas en cada grupo
    final_agrupado = final.groupby(
        ['dia', 'mes', 'año', 'no_distrito', 'nombre_distrito'], 
        as_index=False
    ).size()
    
    # Renombramos la columna de conteo (pandas crea una columna llamada 'size' por defecto)
    final_agrupado.rename(columns={'size': 'cantidad_emergencias'}, inplace=True)

    # Guardamos el resultado agregado
    final_agrupado.to_csv(
        OUT_FINAL,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE
    )

    print("\n[OK] Archivo final generado →", OUT_FINAL.resolve())
    print("Filas totales (agrupadas):", len(final_agrupado))
    if not final_agrupado.empty:
        print("Ejemplo de filas:")
        print(final_agrupado.head())

if __name__ == "__main__":
    main()