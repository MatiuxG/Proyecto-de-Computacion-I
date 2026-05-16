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

#vista /downloads del CKAN, la antigua no tiene enlaces directos
URL_BOMBEROS = "https://datos.madrid.es/dataset/300177-0-bomberos-actuaciones/downloads"
URL_SAMUR    = "https://datos.madrid.es/dataset/300178-0-samur-activaciones/downloads"
URL_SOCIALES = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"


def normalize_text(text):
    #mayusculas, sin tildes ni guiones, espacios colapsados
    res = ""
    if text:
        res = str(text).upper()
        res = unicodedata.normalize("NFD", res)
        res = "".join(char for char in res if unicodedata.category(char) != "Mn")
        res = re.sub(r"-", " ", res)
        res = re.sub(r"\s+", " ", res).strip()
    return res


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

#nombres alternativos que aparecen en los csv antiguos
ALIAS_DISTRITOS = {
    "VALLECAS PTE": "PUENTE DE VALLECAS",
    "VALLECAS PTE.": "PUENTE DE VALLECAS",
    "VALLECAS-PTE": "PUENTE DE VALLECAS",
    "PUENTE VALLECAS": "PUENTE DE VALLECAS",
    "VILLAVERDE ALTO": "VILLAVERDE",
    "VILLAVERDE BAJO": "VILLAVERDE",
}


def clean_name(raw):
    #nombre de distrito normalizado o "NA" si esta vacio
    res = "NA"
    if raw and str(raw).strip().upper() not in ("", "NAN", "NONE", "NULL"):
        texto = normalize_text(str(raw))
        #quitamos prefijo numerico tipo "1 CENTRO"
        texto = re.sub(r"^\d+\s*", "", texto)
        if texto in ALIAS_DISTRITOS:
            texto = normalize_text(ALIAS_DISTRITOS[texto])
        res = texto
    return res


def mes_to_num(value):
    #devuelve el mes como "01".."12", acepta numero o nombre
    res = "NA"
    if value:
        value_str = str(value).strip().lower()
        if value_str.isdigit():
            res = value_str.zfill(2)
        else:
            mapa = {
                "enero":"01","febrero":"02","marzo":"03","abril":"04",
                "mayo":"05","junio":"06","julio":"07","agosto":"08",
                "septiembre":"09","setiembre":"09","octubre":"10",
                "noviembre":"11","diciembre":"12"
            }
            res = mapa.get(value_str, "NA")
    return res


def find_all_csv_urls(url):
    #lista los .csv de la pagina
    csv_urls = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href_value = anchor.get("href")
            href = str(href_value).strip() if href_value else ""
            if href.lower().endswith(".csv"):
                url_absoluta = href if href.startswith("http") else urljoin(url, href)
                csv_urls.append(url_absoluta)
    except Exception as error:
        print("Error al buscar csv en", url, "->", error)
    return csv_urls


def load_csv(url):
    #descarga el csv probando separadores hasta que uno encaje
    res = pd.DataFrame()
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        data = response.content

        #probamos los separadores mas tipicos
        encontrado = False
        for separator in [";", ",", "\t"]:
            if not encontrado:
                try:
                    dataframe = pd.read_csv(io.BytesIO(data), sep=separator, dtype=str)
                    if dataframe.shape[1] > 1:
                        res = dataframe
                        encontrado = True
                except Exception:
                    pass

        #fallback: pandas detecta el separador solo
        if not encontrado:
            txt = data.decode("utf-8", errors="ignore")
            res = pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str)
    except Exception as error:
        print("Error al leer csv en", url, "->", error)
    return res


def get_distrito_id(raw_name, raw_code=None):
    #devuelve el codigo de distrito 1..21 a partir del nombre o codigo
    res = "NA"

    #1: codigo numerico explicito
    if res == "NA" and raw_code and str(raw_code).replace(".0", "").isdigit():
        num = int(float(raw_code))
        if 1 <= num <= 21:
            res = str(num)

    #2: el nombre venia como numero
    if res == "NA" and raw_name and str(raw_name).replace(".0", "").isdigit():
        num = int(float(raw_name))
        if 1 <= num <= 21:
            res = str(num)

    #3: nombre normalizado contra el diccionario
    if res == "NA":
        nombre = clean_name(raw_name)
        if nombre in MADRID_DISTRICTS:
            res = str(MADRID_DISTRICTS[nombre])
        elif nombre in ALIAS_DISTRITOS:
            district_name = normalize_text(ALIAS_DISTRITOS[nombre])
            res = str(MADRID_DISTRICTS.get(district_name, "NA"))

    return res


def resolve_district(raw_name, raw_code):
    #devuelve el nombre normalizado del distrito
    nombre = clean_name(raw_name)
    res = nombre

    if nombre in MADRID_DISTRICTS:
        res = nombre
    elif nombre in ALIAS_DISTRITOS:
        res = normalize_text(ALIAS_DISTRITOS[nombre])
    elif nombre.replace(".0", "").isdigit():
        #el nombre era un codigo, lo mapeamos al nombre oficial
        code = int(float(nombre))
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                res = district_name
    elif raw_code and str(raw_code).replace(".0", "").isdigit():
        code = int(float(raw_code))
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                res = district_name

    return res


#descarga los csv y monta un dataframe con dia/mes/año/distrito/hora
def get_dataset(url, year_candidates, month_candidates, dist_candidates, hour_candidates=None):
    res = pd.DataFrame()

    #1) bajamos los csv y los apilamos
    urls = find_all_csv_urls(url)
    dataframes = []
    for csv_url in urls:
        df_temp = load_csv(csv_url)
        if not df_temp.empty:
            dataframes.append(df_temp)

    if dataframes:
        dataframe = pd.concat(dataframes, ignore_index=True)
        dataframe.columns = [normalize_text(column) for column in dataframe.columns]

        #2) localizamos las columnas (varian de nombre entre años)
        year_col  = next((column for column in dataframe.columns if column in year_candidates), None)
        month_col = next((column for column in dataframe.columns if column in month_candidates), None)
        dist_col  = next((column for column in dataframe.columns if column in dist_candidates), None)
        hour_col  = None
        if hour_candidates:
            hour_col = next((column for column in dataframe.columns if column in hour_candidates), None)

        #3) un registro por fila con fecha y distrito
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

            #la hora es el numero antes del ":"; -1 si no hay
            hora = -1
            if hour_col is not None:
                hora_txt = str(row.get(hour_col, "")).strip()
                antes_dos_puntos = hora_txt.split(":")[0]
                try:
                    candidata = int(antes_dos_puntos)
                    if 0 <= candidata <= 23:
                        hora = candidata
                except ValueError:
                    pass
            registro["hora"] = hora

            result.append(registro)

        res = pd.DataFrame(result)

    return res


def get_bomberos():
    #bomberos viene agregado por mes y sin hora; expandimos despues
    print("\n[Bomberos]")
    res = get_dataset(
        URL_BOMBEROS,
        ["AÑO", "ANO", "YEAR"],
        ["MES"],
        ["DISTRITO"],
    )
    return res


def get_samur():
    #SAMUR si trae hora real de la activacion
    print("\n[SAMUR]")
    res = get_dataset(
        URL_SAMUR,
        ["AÑO", "ANO", "YEAR"],
        ["MES"],
        ["DISTRITO"],
        hour_candidates=["HORA SOLICITUD", "HORA INTERVENCION"],
    )
    return res


def get_sociales():
    #sociales trae fecha pero no hora; marcamos hora=-1 para repartir en 24h
    print("\n[Servicios Sociales]")
    res = pd.DataFrame()

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
                "hora": -1,
            })

        res = pd.DataFrame(result)

    return res


def final_district_fix(row):
    #si hay codigo pero falta el nombre, lo rellenamos desde MADRID_DISTRICTS
    no_dist = row["no_distrito"]
    name = row["nombre_distrito"]

    if not name or name.strip() == "" or name.upper() in ("NAN", "NONE", "NULL"):
        name = "NA"

    if no_dist.isdigit() and 1 <= int(no_dist) <= 21 and name == "NA":
        code = int(no_dist)
        for district_name, district_code in MADRID_DISTRICTS.items():
            if district_code == code:
                name = district_name

    res = pd.Series({
        "dia": row["dia"],
        "mes": row["mes"],
        "año": row["año"],
        "hora": row["hora"],
        "no_distrito": no_dist,
        "nombre_distrito": name,
        "cantidad_emergencias": row["cantidad_emergencias"],
    })
    return res


#reparte las emergencias mensuales por dia y hora
def expandir_por_dia_y_hora(df_mensual):
    filas = []
    for _, fila in df_mensual.iterrows():
        valido = True
        año = 0
        mes = 0
        hora_origen = -1
        try:
            año = int(fila["año"])
            mes = int(fila["mes"])
            hora_origen = int(fila["hora"])
        except ValueError:
            valido = False

        if valido and not (1 <= mes <= 12):
            valido = False

        if valido:
            cantidad_total = float(fila["cantidad_emergencias"])
            dias_del_mes = calendar.monthrange(año, mes)[1]

            if hora_origen >= 0:
                #con hora real, solo repartimos por dias
                horas = [hora_origen]
                valor_por_hora = cantidad_total / dias_del_mes
            else:
                #sin hora repartimos en las 24 del dia
                horas = list(range(24))
                valor_por_hora = cantidad_total / (dias_del_mes * 24)

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

    res = pd.DataFrame(filas)
    return res


#junta bomberos, samur y sociales, agrupa y expande por dia/hora
def main():
    df_bomberos = get_bomberos()
    df_samur = get_samur()
    df_sociales = get_sociales()
    final = pd.concat([df_bomberos, df_samur, df_sociales], ignore_index=True)

    if final.empty:
        print("\n[AVISO] No hay datos, se genera csv vacio")
        cleaned = pd.DataFrame(columns=[
            "dia", "mes", "año", "hora",
            "no_distrito", "nombre_distrito", "cantidad_emergencias",
        ])
    else:
        #limpieza de campos de fecha
        final["año"] = final["año"].astype(str).str.extract(r"(\d{4})")[0].fillna("2022")
        final["mes"] = final["mes"].astype(str).str.extract(r"(\d{1,2})")[0].fillna("01").str.zfill(2)
        final["hora"] = pd.to_numeric(final["hora"], errors="coerce").fillna(-1).astype(int)
        final = final[final["año"].astype(int) >= 2022]

        #agrupa por mes/año/hora/distrito; hora=-1 indica que no hay hora real
        agrupado = final.groupby(
            ["mes", "año", "hora", "no_distrito", "nombre_distrito"],
            as_index=False
        ).size().rename(columns={"size": "cantidad_emergencias"})

        expandido = expandir_por_dia_y_hora(agrupado)

        #al expandir bomberos y samur pueden caer en la misma celda; sumamos
        if not expandido.empty:
            expandido = (
                expandido
                .groupby(
                    ["dia", "mes", "año", "hora", "no_distrito", "nombre_distrito"],
                    as_index=False,
                )["cantidad_emergencias"]
                .sum()
            )
            expandido["cantidad_emergencias"] = expandido["cantidad_emergencias"].round(3)

        cleaned = expandido.apply(final_district_fix, axis=1)

    cleaned.to_csv(
        OUT_FINAL,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE,
    )

    print("\n[OK] Archivo generado: ", OUT_FINAL.resolve())
    print(cleaned.head(10))


if __name__ == "__main__":
    main()
