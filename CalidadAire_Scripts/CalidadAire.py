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
    "User-Agent": "MateoAirScraper/2.0",
    "Accept": "*/*",
}

#vista /downloads del CKAN, la antigua no tiene enlaces directos
PAGES = ["https://datos.madrid.es/dataset/201410-0-calidad-aire-diario/downloads"]

#urls del catalogo de estaciones; las probamos en orden
STATION_CATALOG_CANDIDATES = [
    "https://datos.madrid.es/egob/catalogo/201210-0-estaciones-calidad-aire.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-calidad-aire-estaciones.csv",
    "https://datos.madrid.es/egob/catalogo/201210-0-red-vigilancia-calidad-aire-estaciones.csv",
]

#fallback por si todos los catalogos remotos fallan
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
    #mayusculas, sin tildes ni guiones; "NA" si queda vacio
    res = "NA"
    if s:
        texto = str(s).upper()
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
        texto = re.sub(r"-", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        if texto not in ("", "NAN", "NONE", "NULL"):
            res = texto
    return res


def load_catalog_from_url(url):
    #carga el catalogo de estaciones en un dict {cod_estacion: (cod_distrito, nombre)}
    res = None
    df = None

    #primero ; con latin-1; si falla dejamos a pandas autodetectar
    try:
        df = pd.read_csv(url, dtype=str, sep=";", encoding="latin-1")
    except Exception:
        try:
            df = pd.read_csv(url, dtype=str)
        except Exception:
            df = None

    if df is not None:
        df.columns = [normalize_text(c) for c in df.columns]

        candidates_code = ["CODIGO_ESTACION", "COD_ESTACION", "CODIGO", "ESTACION"]
        candidates_name = ["NOMBRE", "NOMBRE_ESTACION"]
        candidates_district = ["DISTRITO", "NOMBRE_DISTRITO"]
        candidates_dcode = ["COD_DISTRITO", "CODIGO_DISTRITO"]

        stc = next((c for c in candidates_code if c in df.columns), None)
        dist_name = next((c for c in candidates_district if c in df.columns), None)
        dist_code = next((c for c in candidates_dcode if c in df.columns), None)

        if stc:
            lookup = {}
            for _, r in df.iterrows():
                #solo digitos y rellenamos a 3 cifras (28079XXX)
                code = re.sub(r"\D", "", str(r.get(stc, ""))).zfill(3)
                if code:
                    dname = normalize_text(r.get(dist_name, "")) if dist_name else "NA"
                    dcode = re.sub(r"\D", "", str(r.get(dist_code, ""))).zfill(2) if dist_code else "NA"
                    lookup[code] = (dcode, dname)
            res = lookup

    return res


def build_station_lookup():
    #monta el lookup de estaciones; si las urls fallan usa el fallback
    res = None

    for url in STATION_CATALOG_CANDIDATES:
        if res is None:
            lookup = load_catalog_from_url(url)
            if lookup:
                print(f"[Lookup] Loaded: {url} ({len(lookup)} stations)")
                res = lookup

    if res is None:
        print("[Lookup] Using fallback station catalog.")
        res = {
            k: (v[0], normalize_text(v[1]))
            for k, v in STATION_FALLBACK.items()
        }

    return res


def find_csvs(url):
    #lista los enlaces a .csv de la pagina
    out = []
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".csv"):
                url_absoluta = href if href.startswith("http") else urljoin(url, href)
                out.append(url_absoluta)
    except Exception as error:
        print("Error buscando csvs en", url, "->", error)
    return out


def load_table(url):
    #descarga el csv probando codificaciones y separadores hasta que uno encaje
    res = pd.DataFrame()
    try:
        r = requests.get(url, headers=HEADERS)
        data = r.content

        encontrado = False
        for enc in ["utf-8-sig", "utf-8", "latin-1", None]:
            for sep in [";", ",", "\t"]:
                if not encontrado:
                    try:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, dtype=str)
                        if df.shape[1] > 1:
                            res = df
                            encontrado = True
                    except Exception:
                        pass
    except Exception as error:
        print("Error leyendo csv", url, "->", error)
    return res


def extract_station_code(s):
    #saca los 3 digitos finales del codigo de estacion "28079XXX"
    res = None
    if s:
        m = re.search(r"28079(\d{3})", str(s))
        if m:
            res = m.group(1)
    return res


def expand_month(df, lookup):
    #pasa la tabla mensual (D01..D31) a formato largo, una fila por dia-estacion
    res = pd.DataFrame()

    if not df.empty:
        df.columns = [normalize_text(c) for c in df.columns]

        ycol = next((c for c in df.columns if "ANO" in c or "AÑO" in c or "YEAR" in c), None)
        mcol = "MES" if "MES" in df.columns else None
        pcol = next((c for c in df.columns if "PUNTO_MUESTREO" in c), None)
        dcols = [c for c in df.columns if re.match(r"^D\d{2}$", c)]

        #hacen falta año, mes, estacion y al menos una columna de dia
        if ycol and mcol and pcol and dcols:
            out = []
            for _, r in df.iterrows():
                year_match = re.findall(r"\d{4}", str(r.get(ycol, "2022")))
                year = year_match[0] if year_match else "2022"
                month = str(r.get(mcol, "01")).strip().zfill(2)

                station_raw = r.get(pcol, "")
                station = extract_station_code(station_raw)

                if station in lookup:
                    no_dist, name_dist = lookup[station]
                else:
                    no_dist, name_dist = ("NA", "NA")

                #para cada D01..D31 sacamos el valor numerico
                for dcol in dcols:
                    day = int(dcol[1:])
                    val = str(r.get(dcol, "")).strip()

                    #descartamos vacios y marcadores nulos
                    valor_valido = val not in ("", "-", "NA", "None")
                    if valor_valido:
                        m = re.search(r"-?\d+(?:[.,]\d+)?", val)
                        if m:
                            num_val = float(m.group(0).replace(",", "."))
                            out.append({
                                "dia": str(day).zfill(2),
                                "mes": month,
                                "año": year,
                                "no_distrito": no_dist,
                                "nombre_distrito": name_dist,
                                "valor_calidad_aire": num_val,
                            })

            res = pd.DataFrame(out)

    return res


def process_page(url, lookup):
    #procesa todos los csv de la pagina y filtra desde 2022
    res = pd.DataFrame()

    csvs = find_csvs(url)
    dfs = []
    for u in csvs:
        df_raw = load_table(u)
        df_exp = expand_month(df_raw, lookup)
        if not df_exp.empty:
            dfs.append(df_exp)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["año"] = df["año"].astype(str).str.extract(r"(\d{4})")[0]
        df = df[df["año"].astype(int) >= 2022]
        res = df

    return res


def main():
    #monta el lookup, procesa cada pagina y escribe el csv final
    print("=== Generando datasheet Calidad del Aire ===")

    lookup = build_station_lookup()

    parts = []
    for url in PAGES:
        df = process_page(url, lookup)
        if not df.empty:
            parts.append(df)

    if parts:
        final = pd.concat(parts, ignore_index=True)
        final.to_csv(
            OUT_FILE,
            sep=";",
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE,
        )
        print("[OK] Archivo generado ->", OUT_FILE.resolve())
        print("Filas:", len(final))
        print(final.head())
    else:
        #sin datos generamos solo la cabecera
        vacio = pd.DataFrame(columns=[
            "dia", "mes", "año", "no_distrito", "nombre_distrito", "valor_calidad_aire",
        ])
        vacio.to_csv(OUT_FILE, sep=";", index=False, encoding="utf-8-sig")
        print("[OK] Archivo vacio generado")


if __name__ == "__main__":
    main()
