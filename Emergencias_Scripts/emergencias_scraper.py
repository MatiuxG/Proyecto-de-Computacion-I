import csv
import io
import re
import unicodedata
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "MateoScraperBot/7.0",
    "Accept": "*/*"
}
TIMEOUT = 60
OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FINAL = OUTPUT_DIR / "datasheet_emergencias.csv"

URL_BOMBEROS = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=fa677996afc6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SAMUR    = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=50d7d35982d6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SOCIALES = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"

def normalize_text(text):
    #pasa a mayusculas y quita tildes, guiones y espacios sobrantes
    resultado = ""
    if text:
        text = text.upper()
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = re.sub(r"-", " ", text)
        resultado = re.sub(r"\s+", " ", text).strip()
    return resultado

MADRID_DISTRICTS = {
    normalize_text("CENTRO"): 1,
    normalize_text("ARGANZUELA"): 2,
    normalize_text("RETIRO"): 3,
    normalize_text("SALAMANCA"): 4,
    normalize_text("CHAMARTIN"): 5,
    normalize_text("TETUAN"): 6,
    normalize_text("CHAMBERI"): 7,
    normalize_text("FUENCARRAL EL PARDO"): 8,
    normalize_text("MONCLOA ARAVACA"): 9,
    normalize_text("LATINA"): 10,
    normalize_text("CARABANCHEL"): 11,
    normalize_text("USERA"): 12,
    normalize_text("PUENTE DE VALLECAS"): 13,
    normalize_text("MORATALAZ"): 14,
    normalize_text("CIUDAD LINEAL"): 15,
    normalize_text("HORTALEZA"): 16,
    normalize_text("VILLAVERDE"): 17,
    normalize_text("VILLA DE VALLECAS"): 18,
    normalize_text("VICALVARO"): 19,
    normalize_text("SAN BLAS CANILLEJAS"): 20,
    normalize_text("BARAJAS"): 21,
}

ALIAS_DISTRITOS = {
    "VALLECAS PTE": "PUENTE DE VALLECAS",
    "VALLECAS PTE.": "PUENTE DE VALLECAS",
    "VALLECAS-PTE": "PUENTE DE VALLECAS",
    "PUENTE VALLECAS": "PUENTE DE VALLECAS",
    "VILLAVERDE ALTO": "VILLAVERDE",
    "VILLAVERDE BAJO": "VILLAVERDE",
}

def clean_name(raw):
    #limpia el nombre del distrito y resuelve alias
    resultado = "NA"
    if raw and str(raw).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        text = normalize_text(str(raw))
        text = re.sub(r"^\d+\s*", "", text)
        if text in ALIAS_DISTRITOS:
            text = normalize_text(ALIAS_DISTRITOS[text])
        resultado = text
    return resultado

def mes_to_num(value):
    #convierte el mes (texto o numero) a formato 01-12
    resultado = "NA"
    if value:
        value_str = str(value).strip().lower()
        if value_str.isdigit():
            resultado = value_str.zfill(2)
        else:
            meses = {
                "enero":"01","febrero":"02","marzo":"03","abril":"04",
                "mayo":"05","junio":"06","julio":"07","agosto":"08",
                "septiembre":"09","setiembre":"09","octubre":"10",
                "noviembre":"11","diciembre":"12"
            }
            resultado = meses.get(value_str, "NA")
    return resultado

def find_all_csv_urls(url):
    #busca en el html los enlaces que acaban en .csv
    csv_urls = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href_value = anchor.get("href")
            href = str(href_value).strip() if href_value else ""
            if href.lower().endswith(".csv"):
                csv_urls.append(href if href.startswith("http") else urljoin(url, href))
    except Exception as error:
        print("Error al buscar csv en", url, "->", error)
    return csv_urls

def load_csv(url):
    #descarga un csv probando varios separadores
    resultado = pd.DataFrame()
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = response.content
        for separator in [";", ",", "\t"]:
            if resultado.empty:
                try:
                    dataframe = pd.read_csv(io.BytesIO(data), sep=separator, dtype=str)
                    if dataframe.shape[1] > 1:
                        resultado = dataframe
                except:
                    pass
        if resultado.empty:
            txt = data.decode("utf-8", errors="ignore")
            resultado = pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str)
    except Exception as error:
        print("Error al leer csv en", url, "->", error)
        resultado = pd.DataFrame()
    return resultado

def get_distrito_id(raw_name, raw_code=None):
    #obtiene el numero de distrito (1-21) desde el codigo o el nombre
    resultado = "NA"
    if raw_code and str(raw_code).replace(".0", "").isdigit() and 1 <= int(float(raw_code)) <= 21:
        resultado = str(int(float(raw_code)))
    elif raw_name and str(raw_name).replace(".0", "").isdigit() and 1 <= int(float(raw_name)) <= 21:
        resultado = str(int(float(raw_name)))
    else:
        name = clean_name(raw_name)
        if name in MADRID_DISTRICTS:
            resultado = str(MADRID_DISTRICTS[name])
        elif name in ALIAS_DISTRITOS:
            district_name = normalize_text(ALIAS_DISTRITOS[name])
            resultado = str(MADRID_DISTRICTS.get(district_name, "NA"))
    return resultado

def resolve_district(raw_name, raw_code):
    #devuelve el nombre normalizado del distrito
    name = clean_name(raw_name)
    resultado = name
    encontrado = False
    if name in MADRID_DISTRICTS:
        resultado = name
        encontrado = True
    elif name in ALIAS_DISTRITOS:
        resultado = normalize_text(ALIAS_DISTRITOS[name])
        encontrado = True
    if not encontrado and name.replace(".0", "").isdigit():
        code = int(float(name))
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                resultado = district_name
                encontrado = True
    if not encontrado and raw_code and str(raw_code).replace(".0", "").isdigit():
        code = int(float(raw_code))
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                resultado = district_name
    return resultado

def get_dataset(url, year_candidates, month_candidates, dist_candidates):
    #descarga los csv de la pagina y saca dia/mes/año/distrito
    resultado = pd.DataFrame()
    urls = find_all_csv_urls(url)
    dataframes = []
    for csv_url in urls:
        df_temp = load_csv(csv_url)
        if not df_temp.empty:
            dataframes.append(df_temp)
    if dataframes:
        dataframe = pd.concat(dataframes, ignore_index=True)
        dataframe.columns = [normalize_text(column) for column in dataframe.columns]
        year_col = next((column for column in dataframe.columns if column in year_candidates), None)
        month_col = next((column for column in dataframe.columns if column in month_candidates), None)
        dist_col = next((column for column in dataframe.columns if column in dist_candidates), None)
        result = []
        for _, row in dataframe.iterrows():
            raw_district = row.get(dist_col, "NA")
            year = row.get(year_col, "2022")
            month = row.get(month_col, "1")
            result.append({
                "dia": "01",
                "mes": mes_to_num(month),
                "año": str(year),
                "no_distrito": get_distrito_id(raw_district),
                "nombre_distrito": resolve_district(raw_district, None),
            })
        resultado = pd.DataFrame(result)
    return resultado

def get_bomberos():
    #scraping de la ficha de bomberos
    print("\n[Bomberos]")
    return get_dataset(URL_BOMBEROS,
                       ["AÑO","ANO","YEAR"],
                       ["MES"],
                       ["DISTRITO"])

def get_samur():
    #scraping de la ficha de samur
    print("\n[SAMUR]")
    return get_dataset(URL_SAMUR,
                       ["AÑO","ANO","YEAR"],
                       ["MES"],
                       ["DISTRITO"])

def get_sociales():
    #scraping de servicios sociales (la fecha viene completa)
    print("\n[Servicios Sociales]")
    resultado = pd.DataFrame()
    urls = find_all_csv_urls(URL_SOCIALES)
    dataframes = []
    for csv_url in urls:
        df_temp = load_csv(csv_url)
        if not df_temp.empty:
            dataframes.append(df_temp)
    if dataframes:
        dataframe = pd.concat(dataframes, ignore_index=True)
        dataframe.columns = [normalize_text(column) for column in dataframe.columns]
        fecha_col = next((column for column in dataframe.columns if "FECHA" in column), None)
        dcode_col = next((column for column in dataframe.columns if "COD" in column), None)
        dname_col = next((column for column in dataframe.columns if "DISTRITO" in column), None)
        result = []
        for _, row in dataframe.iterrows():
            fecha = str(row.get(fecha_col, "01/01/2022"))
            match = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", fecha)
            if match:
                dia, mes, año = match.groups()
            else:
                dia, mes, año = "01", "01", "2022"
            raw_name = row.get(dname_col, "")
            raw_code = row.get(dcode_col, "")
            result.append({
                "dia": dia.zfill(2),
                "mes": mes_to_num(mes),
                "año": año,
                "no_distrito": get_distrito_id(raw_name, raw_code),
                "nombre_distrito": resolve_district(raw_name, raw_code),
            })
        resultado = pd.DataFrame(result)
    return resultado

def final_district_fix(row):
    #recupera el nombre del distrito si solo tenemos el codigo
    no_dist = row["no_distrito"]
    name = row["nombre_distrito"]
    if not name or name.strip() == "" or name.upper() in ("NAN", "NONE", "NULL"):
        name = "NA"
    if no_dist.isdigit() and 1 <= int(no_dist) <= 21 and name == "NA":
        code = int(no_dist)
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                name = district_name
    return pd.Series({
        "dia": row["dia"],
        "mes": row["mes"],
        "año": row["año"],
        "no_distrito": no_dist,
        "nombre_distrito": name,
        "cantidad_emergencias": row["cantidad_emergencias"],
    })

def main():
    #junta bomberos, samur y servicios sociales contando por dia y distrito
    df_bomberos = get_bomberos()
    df_samur = get_samur()
    df_sociales = get_sociales()
    final = pd.concat([df_bomberos, df_samur, df_sociales], ignore_index=True)

    if final.empty:
        print("\n[AVISO] No hay datos, se genera csv vacio")
        cleaned = pd.DataFrame(columns=[
            "dia",
            "mes",
            "año",
            "no_distrito",
            "nombre_distrito",
            "cantidad_emergencias",
        ])
    else:
        #limpia los campos de fecha
        final["año"] = final["año"].astype(str).str.extract(r"(\d{4})")[0].fillna("2022")
        final["mes"] = final["mes"].astype(str).str.extract(r"(\d{1,2})")[0].fillna("01").str.zfill(2)
        final["dia"] = final["dia"].astype(str).str.extract(r"(\d{1,2})")[0].fillna("01").str.zfill(2)
        final = final[final["año"].astype(int) >= 2022]
        #agrupa contando registros por dia/mes/año/distrito
        grouped = final.groupby(
            ["dia", "mes", "año", "no_distrito", "nombre_distrito"],
            as_index=False
        ).size().rename(columns={"size": "cantidad_emergencias"})
        #recupera nombres de distrito que falten
        cleaned = grouped.apply(final_district_fix, axis=1)
    cleaned.to_csv(
        OUT_FINAL,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE
    )

    print("\n[OK] Archivo generado: ", OUT_FINAL.resolve())
    print(cleaned.head(10))

if __name__ == "__main__":
    main()
