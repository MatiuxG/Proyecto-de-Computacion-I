import csv
import io
import re
import unicodedata
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "MateoAirScraper/2.0",
    "Accept": "*/*"
}
PAGES = ["https://datos.madrid.es/sites/v/index.jsp?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD"]
STATION_CATALOG_CANDIDATES = [
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.csv",
]
#por si aca no se puede descargar el oficial
STATION_FALLBACK = {
    "004": ("09", "MONCLOA ARAVACA"),
    "008": ("04", "SALAMANCA"),
    "011": ("05", "CHAMARTIN"),
    "016": ("15", "CIUDAD LINEAL"),
    "017": ("17", "VILLAVERDE"),
    "018": ("11", "CARABANCHEL"),
    "024": ("09", "MONCLOA ARAVACA"),
    "027": ("21", "BARAJAS"),
    "035": ("01", "CENTRO"),
    "036": ("14", "MORATALAZ"),
    "038": ("06", "TETUAN"),
    "039": ("08", "FUENCARRAL EL PARDO"),
    "040": ("13", "PUENTE DE VALLECAS"),
    "047": ("02", "ARGANZUELA"),
    "048": ("05", "CHAMARTIN"),
    "049": ("03", "RETIRO"),
    "050": ("05", "CHAMARTIN"),
    "054": ("18", "VILLA DE VALLECAS"),
    "055": ("21", "BARAJAS"),
    "056": ("11", "CARABANCHEL"),
    "057": ("16", "HORTALEZA"),
    "058": ("08", "FUENCARRAL EL PARDO"),
    "059": ("21", "BARAJAS"),
    "060": ("08", "FUENCARRAL EL PARDO"),
}

OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_calidad_aire.csv"

def normalize_text(s):
    res = "NA"
    if s:
        s = str(s).upper()
        #fuera tildes
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = re.sub(r"-", " ", s)
        #fuera espacios grandes
        s = re.sub(r"\s+", " ", s).strip()
        if s not in ("", "NAN", "NONE", "NULL"):
            res = s
    return res

def load_catalog_from_url(url):
    #intenta descargar el catalogo de estaciones de una url
    lookup_res = None
    df = None
    try:
        #primer intento con separador punto y coma
        df = pd.read_csv(url, dtype=str, sep=";", encoding="latin-1")
    except: #si no sale, otro intento
        try:
            #segundo intento dejando que pandas detecte el separador
            df = pd.read_csv(url, dtype=str)
        except:
            pass

    if df is not None:
        df.columns = [normalize_text(c) for c in df.columns]
        candidates_code = ["CODIGO_ESTACION", "COD_ESTACION", "CODIGO", "ESTACION"]
        candidates_name = ["NOMBRE", "NOMBRE_ESTACION"]
        candidates_district = ["DISTRITO", "NOMBRE_DISTRITO"]
        candidates_dcode = ["COD_DISTRITO", "CODIGO_DISTRITO"]

        #busca la primera columna que coincida con cada candidato
        station_col = next((c for c in candidates_code if c in df.columns), None)
        dist_name = next((c for c in candidates_district if c in df.columns), None)
        dist_code = next((c for c in candidates_dcode if c in df.columns), None)

        if station_col:#si ha encontrad la columna de codigo de estacion...
            lookup_res = {}
            #recorre cada fila sin mirar el indice y saca lo que necesite
            for _, row in df.iterrows():
                #extrae solo digitos del codigo de estacion
                code = re.sub(r"\D", "", str(row.get(station_col, ""))).zfill(3)
                if code and code != "000":
                    #obtiene nombre y codigo de distrito
                    dname = normalize_text(row.get(dist_name, "")) if dist_name else "NA"
                    dcode = re.sub(r"\D", "", str(row.get(dist_code, ""))).zfill(2) if dist_code else "NA"
                    lookup_res[code] = (dcode, dname)
    return lookup_res


def build_station_lookup():
    #construye el diccionario estacion -> distrito
    res_lookup = {}
    found = False
    
    #intenta cada url candidata hasta que una funcione
    for url in STATION_CATALOG_CANDIDATES:
        if not found:
            lookup = load_catalog_from_url(url)
            if lookup:
                print(f"[Lookup] cargado: {url} ({len(lookup)} estaciones)")
                res_lookup = lookup
                found = True

    if not found:
        #si no funciono ninguna, usa el catalogo de respaldo
        print("[Lookup] usando la ayudita de respaldo (STATION_FALLBACK)")
        res_lookup = {}
        #recorre cada estacion del diccionario de respaldo
        for station_code, district_info in STATION_FALLBACK.items():
            #district_info es una tupla (codigo_distrito, nombre_distrito)
            district_code = district_info[0]
            district_name = normalize_text(district_info[1])
            #guarda en el diccionario: codigo_estacion -> (codigo_distrito, nombre_normalizado)
            res_lookup[station_code] = (district_code, district_name)
    return res_lookup


def find_csvs(url): #va a ser casi la misma funcion siempre 
    #busca todos los enlaces a csv en una pagina
    csv_urls = []
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        from urllib.parse import urljoin
        for anchor in soup.find_all("a", href=True):
            href_value = anchor.get("href")
            href = str(href_value).strip() if href_value is not None else ""
            if href.lower().endswith(".csv"):
                csv_urls.append(href if href.startswith("http") else urljoin(url, href))
    except:
        pass
    return csv_urls


def load_table(url):
    #descarga un csv probando distintas codificaciones y separadores
    df_res = pd.DataFrame()
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.content
        found_table = False
        #prueba distintas combinaciones hasta que funcione
        for enc in ["utf-8-sig", "utf-8", "latin-1", None]:
            for sep in [";", ",", "\t"]:
                if not found_table:
                    try:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, dtype=str)
                        #solo acepta si tiene mas de una columna
                        if df.shape[1] > 1:
                            df_res = df
                            found_table = True
                    except:
                        pass
    except:
        pass
    return df_res


def extract_station_code(s):
    #extrae el codigo de estacion del punto de muestreo
    res_code = None
    if s:
        #busca patron 28079 seguido de 3 digitos
        m = re.search(r"28079(\d{3})", str(s))
        if m:
            res_code = m.group(1)
    return res_code


def expand_month(df, lookup):
    #convierte tabla mensual a filas diarias
    df_res = pd.DataFrame()
    
    if not df.empty:
        #normaliza nombres de columnas
        df.columns = [normalize_text(c) for c in df.columns]
        #busca columnas clave
        year_col = next((c for c in df.columns if "ANO" in c or "AÑO" in c or "YEAR" in c), None)
        month_col = "MES" if "MES" in df.columns else None
        point_col = next((c for c in df.columns if "PUNTO_MUESTREO" in c), None)
        #columnas D01, D02... D31 que tienen los valores diarios
        day_cols = [c for c in df.columns if re.match(r"^D\d{2}$", c)]

        if year_col and month_col and point_col and day_cols:
            out_rows = []
            #recorre cada fila mensual
            for _, row in df.iterrows():
                #extrae el año
                year_match = re.findall(r"\d{4}", str(row.get(year_col, "2022")))
                year = year_match[0] if year_match else "2022"
                #extrae el mes
                month = row.get(month_col, "01").strip().zfill(2)
                #extrae codigo de estacion
                station = extract_station_code(row.get(point_col, ""))
                #busca distrito correspondiente a la estacion
                district_code, district_name = lookup.get(station, ("NA", "NA"))
                #recorre cada dia del mes (D01, D02... D31)
                for day_col in day_cols:
                    val = row.get(day_col, "").strip()
                    #ignora valores vacios o invalidos
                    if val not in ("", "-", "NA", None, "V"):
                        #extrae el numero aunque venga con texto
                        m = re.search(r"-?\d+(?:[.,]\d+)?", val)
                        if m:
                            num_val = float(m.group(0).replace(",", "."))
                            #agrega una fila por cada dia con valor
                            out_rows.append({
                                "dia": str(day_col[1:]).zfill(2),
                                "mes": month,
                                "año": year,
                                "no_distrito": district_code,
                                "nombre_distrito": district_name,
                                "valor_calidad_aire": num_val
                            })
            df_res = pd.DataFrame(out_rows)
            
    return df_res


def process_page(url, lookup):
    #procesa una pagina completa buscando csvs y expandiendolos
    csv_urls = find_csvs(url)
    dfs = []
    res_df = pd.DataFrame()
    
    #procesa cada csv que ha encontrado
    for csv_url in csv_urls:
        df_raw = load_table(csv_url)
        df_expanded = expand_month(df_raw, lookup)
        if not df_expanded.empty:
            dfs.append(df_expanded)

    if dfs:
        #une todos los dataframes
        res_df = pd.concat(dfs, ignore_index=True)
        #limpia el año y filtra desde el año que le digamos
        res_df["año"] = res_df["año"].astype(str).str.extract(r"(\d{4})")[0]
        res_df = res_df[res_df["año"].astype(int) >= 2022]

    return res_df


def main():
    lookup = build_station_lookup()
    parts = []
    #va pagina por pagina
    for url in PAGES:
        df_page = process_page(url, lookup)
        if not df_page.empty:
            parts.append(df_page)

    if not parts:
        #si no hay datos, genera archivo vacio con columnas
        pd.DataFrame(columns=[
            "dia","mes","año","no_distrito","nombre_distrito","valor_calidad_aire"
        ]).to_csv(OUT_FILE, sep=";", index=False, encoding="utf-8-sig")
    else:
        #une todos los datos y guarda
        final = pd.concat(parts, ignore_index=True)
        final.to_csv(
            OUT_FILE,
            sep=";",
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE,
            escapechar="\\"
        )
        print("[OK] Archivo generado →", OUT_FILE.resolve())


if __name__ == "__main__":
    main()