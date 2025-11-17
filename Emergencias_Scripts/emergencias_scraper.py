# -*- coding: utf-8 -*-
"""
Datasheet unificado Emergencias Madrid
Reglas:
- Bomberos -> hora 00:00, código_emergencia="Incendio", código_emergencia_num=10
- SAMUR -> usa Año, Mes, Hora Solicitud, Código, Distrito
- Servicios Sociales -> usa Código Distrito, Distrito, Fecha Cita, Tipo Supuesto Urgente
"""

import csv
import io
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ================================
# CONFIG GENERAL
# ================================
HEADERS = {
    "User-Agent": "MateoScraperBot/3.0",
    "Accept": "*/*"
}
TIMEOUT = 60

OUTPUT_DIR = Path("./Emergencias_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FINAL = OUTPUT_DIR / "datasheet_emergencias.csv"

# URLs oficiales reales
URL_BOMBEROS = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=fa677996afc6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SAMUR    = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=50d7d35982d6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SOCIALES = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"

# ================================
# HELPERS
# ================================
def find_csv_url(page_url):
    """Encuentra el enlace CSV dentro de la ficha."""
    r = requests.get(page_url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "csv" in href.lower():
            from urllib.parse import urljoin
            return href if href.startswith("http") else urljoin(page_url, href)

    return None


def load_csv(url):
    """Carga robusta de CSV, evita explosiones si no es un CSV válido."""
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.content

    # intento seguro con separadores básicos
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str)
            if df.shape[1] > 1:
                return df
        except:
            pass

    # autodetect
    try:
        txt = data.decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str)
    except:
        print("  [AVISO] Archivo no es CSV válido → omitido")
        return pd.DataFrame()

# ================================
# 1. BOMBEROS
# ================================
def get_bomberos():
    print("[Bomberos] Buscando CSV...")
    csv_url = find_csv_url(URL_BOMBEROS)
    if not csv_url:
        print("No CSV bomberos")
        return pd.DataFrame()

    df = load_csv(csv_url)
    if df.empty:
        return df

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    year_col = next((c for c in df.columns if "año" in c or "year" in c or "anio" in c), None)
    month_col = next((c for c in df.columns if "mes" in c), None)
    dist_col = next((c for c in df.columns if "distrito" in c), None)

    out = []
    for _, row in df.iterrows():
        out.append({
            "dataset": "bomberos",
            "dia": "01",
            "mes": str(row.get(month_col, "NA")),
            "año": str(row.get(year_col, "NA")),
            "no_distrito": str(row.get(dist_col, "NA")),
            "nombre_distrito": str(row.get(dist_col, "NA")),
            "hora_emergencia": "00:00",
            "codigo_emergencia": "Incendio",
            "codigo_emergencia_num": 10
        })

    return pd.DataFrame(out)

# ================================
# 2. SAMUR
# ================================
def get_samur():
    print("[SAMUR] Buscando CSV...")
    csv_url = find_csv_url(URL_SAMUR)
    if not csv_url:
        print("No CSV SAMUR")
        return pd.DataFrame()

    df = load_csv(csv_url)
    if df.empty:
        return df

    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Columnas reales del dataset SAMUR
    year_col = next((c for c in df.columns if "año" in c or "anio" in c or "year" in c), None)
    month_col = next((c for c in df.columns if "mes" in c), None)
    hora_col = next((c for c in df.columns if "hora_solicitud" in c), None)

    # Código con o sin tilde → universal
    code_cols = [c for c in df.columns if "codigo" in c or "código" in c]
    code_col = code_cols[0] if code_cols else None

    dist_cols = [c for c in df.columns if "distrito" in c]
    dist_col = dist_cols[0] if dist_cols else None

    if code_col is None:
        print("[ERROR] No se encontró la columna de CÓDIGO en SAMUR:")
        print("Columnas CSV:", df.columns.tolist())
        return pd.DataFrame()

    # Mapa texto → número
    codigos_unique = sorted(df[code_col].dropna().unique())
    mapa_codigos = {txt: i+1 for i, txt in enumerate(codigos_unique)}

    out = []
    for _, row in df.iterrows():

        dia = "01"
        mes = str(row.get(month_col, "NA"))
        año = str(row.get(year_col, "NA"))

        hora = str(row.get(hora_col, "")).strip()
        if hora == "" or hora.lower() == "nan":
            hora = "00:00"

        codigo_txt = str(row.get(code_col, "NA")).strip()

        out.append({
            "dataset": "samur",
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": str(row.get(dist_col, "NA")),
            "nombre_distrito": str(row.get(dist_col, "NA")),
            "hora_emergencia": hora,
            "codigo_emergencia": codigo_txt,
            "codigo_emergencia_num": mapa_codigos.get(codigo_txt, "NA")
        })

    return pd.DataFrame(out)

    print("[SAMUR] Buscando CSV...")
    csv_url = find_csv_url(URL_SAMUR)
    if not csv_url:
        print("No CSV SAMUR")
        return pd.DataFrame()

    df = load_csv(csv_url)
    if df.empty:
        return df

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Columnas reales del dataset SAMUR
    year_col = next((c for c in df.columns if "año" in c or "anio" in c or "year" in c), None)
    month_col = next((c for c in df.columns if "mes" in c), None)
    hora_col = next((c for c in df.columns if "hora_solicitud" in c), None)
    code_col = next((c for c in df.columns if c == "codigo"), None)
    dist_col = next((c for c in df.columns if c == "distrito"), None)

    # Mapear códigos a números
    codigos_unique = sorted(df[code_col].dropna().unique())
    mapa_codigos = {txt: i+1 for i, txt in enumerate(codigos_unique)}

    out = []
    for _, row in df.iterrows():

        dia = "01"
        mes = str(row.get(month_col, "NA"))
        año = str(row.get(year_col, "NA"))

        hora = str(row.get(hora_col, "")).strip()
        if hora == "" or hora.lower() == "nan":
            hora = "00:00"

        codigo_txt = str(row.get(code_col, "NA")).strip()

        out.append({
            "dataset": "samur",
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": str(row.get(dist_col, "NA")),
            "nombre_distrito": str(row.get(dist_col, "NA")),
            "hora_emergencia": hora,
            "codigo_emergencia": codigo_txt,
            "codigo_emergencia_num": mapa_codigos.get(codigo_txt, "NA")
        })

    return pd.DataFrame(out)

# ================================
# 3. SERVICIOS SOCIALES
# ================================
def get_sociales():
    print("[Servicios Sociales] Buscando CSV...")
    csv_url = find_csv_url(URL_SOCIALES)
    if not csv_url:
        print("No CSV sociales")
        return pd.DataFrame()

    df = load_csv(csv_url)
    if df.empty:
        return df

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    dcode_col = "código_distrito"
    dname_col = "distrito"
    fecha_col = "fecha_cita"
    tipo_text_col = "tipo_supuesto_urgente"

    # mapa por texto
    descripciones = sorted(df[tipo_text_col].dropna().unique())
    mapa_codigos = {txt: i+1 for i, txt in enumerate(descripciones)}

    out = []
    for _, row in df.iterrows():

        fecha_raw = str(row.get(fecha_col, "")).strip()
        dia, mes, año = "NA", "NA", "NA"

        m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", fecha_raw)
        if m:
            dia, mes, año = m.groups()

        descripcion = str(row.get(tipo_text_col, "NA"))

        out.append({
            "dataset": "servicios_sociales",
            "dia": dia,
            "mes": mes,
            "año": año,
            "no_distrito": str(row.get(dcode_col, "NA")),
            "nombre_distrito": str(row.get(dname_col, "NA")),
            "hora_emergencia": "00:00",
            "codigo_emergencia": descripcion,
            "codigo_emergencia_num": mapa_codigos.get(descripcion, "NA")
        })

    return pd.DataFrame(out)

# ================================
# MAIN
# ================================
def main():
    df1 = get_bomberos()
    df2 = get_samur()
    df3 = get_sociales()

    final = pd.concat([df1, df2, df3], ignore_index=True, sort=False)

    # ELIMINAR LA COLUMNA DATASET volver a poner en caso de error
    if "dataset" in final.columns:
        final = final.drop(columns=["dataset"])

    final.to_csv(
        OUT_FINAL,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE
    )

    print("\n[OK] Archivo final generado →", OUT_FINAL.resolve())
    print("Filas totales:", len(final))


if __name__ == "__main__":
    main()
