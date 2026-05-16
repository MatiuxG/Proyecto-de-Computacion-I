import calendar
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

#vista /downloads del portal CKAN (la antigua redirige aqui pero sin enlaces directos)
URL_BOMBEROS = "https://datos.madrid.es/dataset/300177-0-bomberos-actuaciones/downloads"
URL_SAMUR    = "https://datos.madrid.es/dataset/300178-0-samur-activaciones/downloads"
URL_SOCIALES = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"

#convierte texto a mayusculas sin tildes ni guiones
def normalize_text(text):
    if not text:
        return ""
    text = text.upper()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

#limpia nombre de distrito quitando numeros iniciales
def clean_name(raw):
    if not raw or str(raw).strip().upper() in ("", "NAN", "NONE", "NULL"):
        return "NA"
    text = normalize_text(str(raw))
    text = re.sub(r"^\d+\s*", "", text)
    if text in ALIAS_DISTRITOS:
        text = normalize_text(ALIAS_DISTRITOS[text])
    return text

#convierte mes en texto o numero a formato 01-12
def mes_to_num(value):
    if not value:
        return "NA"
    value_str = str(value).strip().lower()
    if value_str.isdigit():
        return value_str.zfill(2)
    MAP = {
        "enero":"01","febrero":"02","marzo":"03","abril":"04",
        "mayo":"05","junio":"06","julio":"07","agosto":"08",
        "septiembre":"09","setiembre":"09","octubre":"10",
        "noviembre":"11","diciembre":"12"
    }
    return MAP.get(value_str, "NA")

#busca todos los enlaces csv en la pagina
def find_all_csv_urls(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        csv_urls = []
        for anchor in soup.find_all("a", href=True):
            href_value = anchor.get("href")
            href = str(href_value).strip() if href_value else ""
            if href.lower().endswith(".csv"):
                csv_urls.append(href if href.startswith("http") else urljoin(url, href))
        return csv_urls
    except Exception as error:
        print("Error al buscar csv en", url, "->", error)
        return []

#descarga csv probando varios separadores
def load_csv(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = response.content
        for separator in [";", ",", "\t"]:
            try:
                dataframe = pd.read_csv(io.BytesIO(data), sep=separator, dtype=str)
                if dataframe.shape[1] > 1:
                    return dataframe
            except:
                pass
        txt = data.decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str)
    except Exception as error:
        print("Error al leer csv en", url, "->", error)
        return pd.DataFrame()

#obtiene numero de distrito 1-21 desde nombre o codigo
def get_distrito_id(raw_name, raw_code=None):
    if raw_code and str(raw_code).replace(".0", "").isdigit():
        num = int(float(raw_code))
        if 1 <= num <= 21:
            return str(num)

    if raw_name and str(raw_name).replace(".0", "").isdigit():
        num = int(float(raw_name))
        if 1 <= num <= 21:
            return str(num)

    name = clean_name(raw_name)
    if name in MADRID_DISTRICTS:
        return str(MADRID_DISTRICTS[name])

    if name in ALIAS_DISTRITOS:
        district_name = normalize_text(ALIAS_DISTRITOS[name])
        return str(MADRID_DISTRICTS.get(district_name, "NA"))

    return "NA"


#convierte codigo o nombre a nombre normalizado de distrito
def resolve_district(raw_name, raw_code):
    name = clean_name(raw_name)

    if name in MADRID_DISTRICTS:
        return name
    
    if name in ALIAS_DISTRITOS:
        return normalize_text(ALIAS_DISTRITOS[name])

    if name.replace(".0", "").isdigit():
        code = int(float(name))
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code: 
                return district_name

    if raw_code and str(raw_code).replace(".0", "").isdigit():
        code = int(float(raw_code))
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                return district_name

    return name

#descarga csvs de la url y le saca los registros con dia/mes/año/distrito.
#si se le pasa hour_candidates intenta tambien sacar la hora (0-23) de cada registro;
#util para SAMUR que tiene "Hora Solicitud", inutil para Bomberos que solo viene mensual.
def get_dataset(url, year_candidates, month_candidates, dist_candidates, hour_candidates=None):
    urls = find_all_csv_urls(url)
    dataframes = []
    for csv_url in urls:
        df_temp = load_csv(csv_url)
        if not df_temp.empty:
            dataframes.append(df_temp)
    if not dataframes:
        return pd.DataFrame()

    dataframe = pd.concat(dataframes, ignore_index=True)
    dataframe.columns = [normalize_text(column) for column in dataframe.columns]

    year_col  = next((column for column in dataframe.columns if column in year_candidates), None)
    month_col = next((column for column in dataframe.columns if column in month_candidates), None)
    dist_col  = next((column for column in dataframe.columns if column in dist_candidates), None)
    hour_col  = None
    if hour_candidates:
        hour_col = next((column for column in dataframe.columns if column in hour_candidates), None)

    result = []
    for _, row in dataframe.iterrows():
        raw_district = row.get(dist_col, "NA")
        year = row.get(year_col, "2022")
        month = row.get(month_col, "1")

        registro = {
            "dia": "01",
            "mes": mes_to_num(month),
            "año": str(year),
            "no_distrito": get_distrito_id(raw_district),
            "nombre_distrito": resolve_district(raw_district, None),
        }
        if hour_col is not None:
            hora_txt = str(row.get(hour_col, "")).strip()
            #formatos vistos: "9:30:21", "09:30", "9". cogemos los digitos previos al primer ":".
            antes_dos_puntos = hora_txt.split(":")[0]
            try:
                hora = int(antes_dos_puntos)
            except ValueError:
                hora = -1
            registro["hora"] = hora if 0 <= hora <= 23 else -1
        else:
            registro["hora"] = -1   #marca "sin hora" -> luego se reparte en 24h

        result.append(registro)

    return pd.DataFrame(result)

def get_bomberos():
    print("\n[Bomberos]")
    #bomberos solo viene mensual, no le pedimos hora
    return get_dataset(URL_BOMBEROS,
                       ["AÑO","ANO","YEAR"],
                       ["MES"],
                       ["DISTRITO"])

def get_samur():
    print("\n[SAMUR]")
    #en SAMUR la columna se llama "Hora Solicitud" -> tras normalize_text "HORA SOLICITUD"
    return get_dataset(URL_SAMUR,
                       ["AÑO","ANO","YEAR"],
                       ["MES"],
                       ["DISTRITO"],
                       hour_candidates=["HORA SOLICITUD", "HORA INTERVENCION"])

def get_sociales():
    print("\n[Servicios Sociales]")
    urls = find_all_csv_urls(URL_SOCIALES)
    dataframes = []
    for csv_url in urls:
        df_temp = load_csv(csv_url)
        if not df_temp.empty:
            dataframes.append(df_temp)
    if not dataframes:
        return pd.DataFrame()

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
            "hora": -1,   #servicios sociales no incluye hora
        })

    return pd.DataFrame(result)

#recupera nombre de distrito si solo hay codigo
def final_district_fix(row):
    no_dist = row["no_distrito"]
    name = row["nombre_distrito"]
    if not name or name.strip() == "" or name.upper() in ("NAN","NONE","NULL"):
        name = "NA"

    if no_dist.isdigit() and 1 <= int(no_dist) <= 21 and name == "NA":
        code = int(no_dist)
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                name = district_name
                break

    return pd.Series({
        "dia": row["dia"],
        "mes": row["mes"],
        "año": row["año"],
        "hora": row["hora"],
        "no_distrito": no_dist,
        "nombre_distrito": name,
        "cantidad_emergencias": row["cantidad_emergencias"],
    })

#reparte las emergencias mensuales en todos los dias del mes y todas las horas
#  - si la fila viene con hora real (samur) -> repartimos solo entre los dias del mes
#  - si la fila viene sin hora (bomberos, hora=-1) -> repartimos entre dias x 24 horas
def expandir_por_dia_y_hora(df_mensual):
    filas = []
    for _, fila in df_mensual.iterrows():
        try:
            año = int(fila["año"])
            mes = int(fila["mes"])
            hora_origen = int(fila["hora"])
        except ValueError:
            continue
        if not (1 <= mes <= 12):
            continue

        cantidad_total = float(fila["cantidad_emergencias"])
        dias_del_mes = calendar.monthrange(año, mes)[1]

        if hora_origen >= 0:
            #tenemos hora real, solo repartimos por dias
            cantidad_por_dia = cantidad_total / dias_del_mes
            horas = [hora_origen]
            valor_por_hora = cantidad_por_dia
        else:
            #sin hora -> distribuimos uniforme en las 24 horas del dia
            cantidad_por_dia_hora = cantidad_total / (dias_del_mes * 24)
            horas = list(range(24))
            valor_por_hora = cantidad_por_dia_hora

        for dia in range(1, dias_del_mes + 1):
            for hora in horas:
                filas.append({
                    "dia": str(dia).zfill(2),
                    "mes": str(mes).zfill(2),
                    "año": str(año),
                    "hora": int(hora),
                    "no_distrito": fila["no_distrito"],
                    "nombre_distrito": fila["nombre_distrito"],
                    "cantidad_emergencias": round(valor_por_hora, 3),
                })
    return pd.DataFrame(filas)


#junta bomberos samur y servicios sociales y reparte las emergencias por hora
def main():
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
            "hora",
            "no_distrito",
            "nombre_distrito",
            "cantidad_emergencias",
        ])
    else:
        #limpia campos de fecha
        final["año"] = final["año"].astype(str).str.extract(r"(\d{4})")[0].fillna("2022")
        final["mes"] = final["mes"].astype(str).str.extract(r"(\d{1,2})")[0].fillna("01").str.zfill(2)
        #en SAMUR/Bomberos no hay dia, viene a "01" puesto por get_dataset(); lo ignoramos
        final["hora"] = pd.to_numeric(final["hora"], errors="coerce").fillna(-1).astype(int)
        final = final[final["año"].astype(int) >= 2022]

        #agrupa por mes/año/hora/distrito contando registros.
        #hora=-1 significa "sin hora" (bomberos, sociales), hora>=0 son las de SAMUR.
        agrupado = final.groupby(
            ["mes", "año", "hora", "no_distrito", "nombre_distrito"],
            as_index=False
        ).size().rename(columns={"size": "cantidad_emergencias"})

        #expandimos a (dia, hora). las filas con hora real solo se expanden por dias,
        #las filas sin hora se reparten en las 24 horas tambien.
        expandido = expandir_por_dia_y_hora(agrupado)

        #al expandir bomberos y samur por separado pueden coincidir filas;
        #las sumamos para que la hora 9 contenga tanto las llamadas SAMUR
        #de las 9 como la parte horaria que toca de bomberos.
        if not expandido.empty:
            expandido = (
                expandido
                .groupby(["dia", "mes", "año", "hora", "no_distrito", "nombre_distrito"], as_index=False)
                ["cantidad_emergencias"]
                .sum()
            )
            expandido["cantidad_emergencias"] = expandido["cantidad_emergencias"].round(3)

        #corrige nombres de distrito faltantes
        cleaned = expandido.apply(final_district_fix, axis=1)
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
