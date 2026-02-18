import csv
import io
import re
import unicodedata
import requests
import pandas as pd
from pathlib import Path
from datetime import date as date_cls
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

OUTPUT_DIR = Path("CalidadAire_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "datasheet_calidad_aire.csv"


def normalize_text(s):
    res = "NA"
    if s:
        s = str(s).upper()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = re.sub(r"-", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s not in ("", "NAN", "NONE", "NULL"):
            res = s
    return res


def load_catalog_from_url(url):
    lookup_res = None
    df = None
    try:
        df = pd.read_csv(url, dtype=str, sep=";", encoding="latin-1")
    except:
        try:
            df = pd.read_csv(url, dtype=str)
        except:
            pass

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
            lookup_res = {}
            for _, r in df.iterrows():
                code = re.sub(r"\D", "", str(r.get(stc, ""))).zfill(3)
                if code and code != "000":
                    dname = normalize_text(r.get(dist_name, "")) if dist_name else "NA"
                    dcode = re.sub(r"\D", "", str(r.get(dist_code, ""))).zfill(2) if dist_code else "NA"
                    lookup_res[code] = (dcode, dname)
    return lookup_res


def build_station_lookup():
    res_lookup = {}
    found = False
    
    for url in STATION_CATALOG_CANDIDATES:
        if not found:
            lookup = load_catalog_from_url(url)
            if lookup:
                print(f"[Lookup] Loaded: {url} ({len(lookup)} stations)")
                res_lookup = lookup
                found = True

    if not found:
        print("[Lookup] Using fallback station catalog.")
        res_lookup = {
            k: (v[0], normalize_text(v[1]))
            for k, v in STATION_FALLBACK.items()
        }
    return res_lookup


def find_csvs(url):
    out = []
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        from urllib.parse import urljoin
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".csv"):
                out.append(href if href.startswith("http") else urljoin(url, href))
    except:
        pass
    return out


def load_table(url):
    df_res = pd.DataFrame()
    try:
        r = requests.get(url, headers=HEADERS)
        data = r.content
        found_table = False
        for enc in ["utf-8-sig", "utf-8", "latin-1", None]:
            for sep in [";", ",", "\t"]:
                if not found_table:
                    try:
                        df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, dtype=str)
                        if df.shape[1] > 1:
                            df_res = df
                            found_table = True
                    except:
                        pass
    except:
        pass
    return df_res


def extract_station_code(s):
    res_code = None
    if s:
        m = re.search(r"28079(\d{3})", str(s))
        if m:
            res_code = m.group(1)
    return res_code


def expand_month(df, lookup):
    df_res = pd.DataFrame()
    
    if not df.empty:
        df.columns = [normalize_text(c) for c in df.columns]
        ycol = next((c for c in df.columns if "ANO" in c or "AÑO" in c or "YEAR" in c), None)
        mcol = "MES" if "MES" in df.columns else None
        pcol = next((c for c in df.columns if "PUNTO_MUESTREO" in c), None)
        dcols = [c for c in df.columns if re.match(r"^D\d{2}$", c)]

        if ycol and mcol and pcol and dcols:
            out_rows = []
            for _, r in df.iterrows():
                year_match = re.findall(r"\d{4}", str(r.get(ycol, "2022")))
                year = year_match[0] if year_match else "2022"
                month = r.get(mcol, "01").strip().zfill(2)
                station = extract_station_code(r.get(pcol, ""))

                no_dist, name_dist = lookup.get(station, ("NA", "NA"))

                for dcol in dcols:
                    val = r.get(dcol, "").strip()
                    if val not in ("", "-", "NA", None, "V"):
                        m = re.search(r"-?\d+(?:[.,]\d+)?", val)
                        if m:
                            num_val = float(m.group(0).replace(",", "."))
                            out_rows.append({
                                "dia": str(dcol[1:]).zfill(2),
                                "mes": month,
                                "año": year,
                                "no_distrito": no_dist,
                                "nombre_distrito": name_dist,
                                "valor_calidad_aire": num_val
                            })
            df_res = pd.DataFrame(out_rows)
            
    return df_res


def process_page(url, lookup):
    csvs = find_csvs(url)
    dfs = []
    res_df = pd.DataFrame()
    
    for u in csvs:
        df_raw = load_table(u)
        df_exp = expand_month(df_raw, lookup)
        if not df_exp.empty:
            dfs.append(df_exp)

    if dfs:
        res_df = pd.concat(dfs, ignore_index=True)
        res_df["año"] = res_df["año"].astype(str).str.extract(r"(\d{4})")[0]
        res_df = res_df[res_df["año"].astype(int) >= 2022]

    return res_df


def main():
    print("=== Generando datasheet Calidad del Aire ===")
    lookup = build_station_lookup()
    parts = []
    
    for url in PAGES:
        df_page = process_page(url, lookup)
        if not df_page.empty:
            parts.append(df_page)

    if not parts:
        pd.DataFrame(columns=[
            "dia","mes","año","no_distrito","nombre_distrito","valor_calidad_aire"
        ]).to_csv(OUT_FILE, sep=";", index=False, encoding="utf-8-sig")
        print("[OK] Archivo vacío generado")
    else:
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
        print("Filas:", len(final))
        print(final.head())


if __name__ == "__main__":
    main()