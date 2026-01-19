import csv
import io
import re
import unicodedata
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "MateoScraperBot/7.0",
    "Accept": "*/*"
}

TIMEOUT = 60

OUTPUT_DIR = Path("Emergencias_Scripts/Resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FINAL = OUTPUT_DIR / "datasheet_emergencias.csv"

URL_BOMBEROS = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=fa677996afc6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SAMUR    = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=50d7d35982d6f510VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"
URL_SOCIALES = "https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=0b006dace9578610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default"

def normalize_text(s):
    if not s:
        return "" #FLAG
    s = s.upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"-", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
    if not raw or str(raw).strip().upper() in ("", "NAN", "NONE", "NULL"):
        return "NA" #FLAG
    s = normalize_text(str(raw))
    s = re.sub(r"^\d+\s*", "", s)
    if s in ALIAS_DISTRITOS:
        s = normalize_text(ALIAS_DISTRITOS[s])
    return s

def mes_to_num(value):
    if not value:
        return "NA" #FLAG
    v = str(value).strip().lower()
    if v.isdigit():
        return v.zfill(2) #FLAG
    MAP = {
        "enero":"01","febrero":"02","marzo":"03","abril":"04",
        "mayo":"05","junio":"06","julio":"07","agosto":"08",
        "septiembre":"09","setiembre":"09","octubre":"10",
        "noviembre":"11","diciembre":"12"
    }
    return MAP.get(v, "NA")

def find_all_csv_urls(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        from urllib.parse import urljoin
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".csv"):
                out.append(href if href.startswith("http") else urljoin(url, href))
        return out #FLAG
    except:
        return [] #FLAG

def load_csv(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        data = r.content
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str)
                if df.shape[1] > 1:
                    return df #FLAG
            except:
                pass
        txt = data.decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(txt), sep=None, engine="python", dtype=str) #FLAG
    except:
        return pd.DataFrame() #FLAG

def get_distrito_id(raw_name, raw_code=None):
    if raw_code and str(raw_code).replace(".0", "").isdigit():
        num = int(float(raw_code))
        if 1 <= num <= 21:
            return str(num) #FLAG

    if raw_name and str(raw_name).replace(".0", "").isdigit():
        num = int(float(raw_name))
        if 1 <= num <= 21:
            return str(num) #FLAG

    name = clean_name(raw_name)
    if name in MADRID_DISTRICTS:
        return str(MADRID_DISTRICTS[name]) #FLAG

    if name in ALIAS_DISTRITOS:
        k = normalize_text(ALIAS_DISTRITOS[name])
        return str(MADRID_DISTRICTS.get(k, "NA")) #FLAG

    return "NA"


def resolve_district(raw_name, raw_code):
    name = clean_name(raw_name)

    if name in MADRID_DISTRICTS:
        return name #FLAG

    if name in ALIAS_DISTRITOS:
        return normalize_text(ALIAS_DISTRITOS[name]) #FLAG

    if name.replace(".0", "").isdigit():
        code = int(float(name))
        for k, v in MADRID_DISTRICTS.items():
            if v == code: 
                return k #FLAG

    if raw_code and str(raw_code).replace(".0", "").isdigit():
        code = int(float(raw_code))
        for k, v in MADRID_DISTRICTS.items():
            if v == code:
                return k #FLAG

    return name

def get_dataset(url, year_candidates, month_candidates, dist_candidates):
    urls = find_all_csv_urls(url)
    dfs = []
    for u in urls:
        df_tmp = load_csv(u)
        if not df_tmp.empty:
            dfs.append(df_tmp)
    if not dfs:
        return pd.DataFrame() #FLAG

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [normalize_text(c) for c in df.columns]

    year_col  = next((c for c in df.columns if c in year_candidates), None)
    month_col = next((c for c in df.columns if c in month_candidates), None)
    dist_col  = next((c for c in df.columns if c in dist_candidates), None)

    out = []
    for _, r in df.iterrows():
        raw = r.get(dist_col, "NA")
        year = r.get(year_col, "2022")
        month = r.get(month_col, "1")

        out.append({
            "dia": "01",
            "mes": mes_to_num(month),
            "año": str(year),
            "no_distrito": get_distrito_id(raw),
            "nombre_distrito": resolve_district(raw, None),
        })

    return pd.DataFrame(out)

def get_bomberos():
    print("\n[Bomberos]")
    return get_dataset(URL_BOMBEROS,
                       ["AÑO","ANO","YEAR"],
                       ["MES"],
                       ["DISTRITO"])

def get_samur():
    print("\n[SAMUR]")
    return get_dataset(URL_SAMUR,
                       ["AÑO","ANO","YEAR"],
                       ["MES"],
                       ["DISTRITO"])

def get_sociales():
    print("\n[Servicios Sociales]")

    urls = find_all_csv_urls(URL_SOCIALES)
    dfs = []
    for u in urls:
        df_tmp = load_csv(u)
        if not df_tmp.empty:
            dfs.append(df_tmp)
    if not dfs:
        return pd.DataFrame() #FLAG

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [normalize_text(c) for c in df.columns]

    fecha_col = next((c for c in df.columns if "FECHA" in c), None)
    dcode_col = next((c for c in df.columns if "COD" in c), None)
    dname_col = next((c for c in df.columns if "DISTRITO" in c), None)

    out = []
    for _, r in df.iterrows():

        fecha = str(r.get(fecha_col, "01/01/2022"))
        m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", fecha)
        if m:
            dia, mes, año = m.groups()
        else:
            dia, mes, año = "01", "01", "2022"

        raw_name = r.get(dname_col, "")
        raw_code = r.get(dcode_col, "")

        out.append({
            "dia": dia.zfill(2),
            "mes": mes_to_num(mes),
            "año": año,
            "no_distrito": get_distrito_id(raw_name, raw_code),
            "nombre_distrito": resolve_district(raw_name, raw_code),
        })

    return pd.DataFrame(out)

def final_district_fix(row):
    no_dist = row["no_distrito"]
    name = row["nombre_distrito"]
    if not name or name.strip() == "" or name.upper() in ("NAN","NONE","NULL"):
        name = "NA"

    if no_dist.isdigit() and 1 <= int(no_dist) <= 21 and name == "NA":
        code = int(no_dist)
        for k, v in MADRID_DISTRICTS.items():
            if v == code:
                name = k
                break #FLAG

    return pd.Series({
        "dia": row["dia"],
        "mes": row["mes"],
        "año": row["año"],
        "no_distrito": no_dist,
        "nombre_distrito": name,
        "cantidad_emergencias": row["cantidad_emergencias"],
    })

def main():
    print("\n=== Generando datasheet emergencias ===")

    dfB = get_bomberos()
    dfS = get_samur()
    dfSS = get_sociales()

    final = pd.concat([dfB, dfS, dfSS], ignore_index=True)

    final["año"] = final["año"].astype(str).str.extract(r"(\d{4})")[0].fillna("2022")
    final["mes"] = final["mes"].astype(str).str.extract(r"(\d{1,2})")[0].fillna("01").str.zfill(2)
    final["dia"] = final["dia"].astype(str).str.extract(r"(\d{1,2})")[0].fillna("01").str.zfill(2)

    final = final[final["año"].astype(int) >= 2022]

    grouped = final.groupby(
        ["dia", "mes", "año", "no_distrito", "nombre_distrito"],
        as_index=False
    ).size().rename(columns={"size": "cantidad_emergencias"})

    cleaned = grouped.apply(final_district_fix, axis=1)

    cleaned.to_csv(
        OUT_FINAL,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE
    )

    print("\n[OK] Archivo generado →", OUT_FINAL.resolve())
    print("Filas:", len(cleaned))
    print(cleaned.head(10))


if __name__ == "__main__":
    main()
