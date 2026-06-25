import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

#pagina de observacion de aemet con los ultimos datos de la provincia de madrid
URL = "https://www.aemet.es/es/eltiempo/observacion/ultimosdatos?k=mad&w=0&datos=det&x=h24&f=temperatura"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
OUT_DIR = Path(__file__).resolve().parent / "Resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "datasheet_clima_scraper.csv"

def limpiar_numero(texto):
    #la celda repite el valor (uno visible y otro de orden), coge solo el primer numero
    resultado = ""
    encontrado = re.search(r"-?\d+[.,]\d|-?\d+", texto)
    if encontrado:
        resultado = encontrado.group(0).replace(",", ".")
    return resultado

def scrapear_clima_aemet():
    #pide la pagina de aemet y saca la temperatura de cada estacion
    filas = []
    try:
        #paso 1: pide la pagina html a aemet
        respuesta = requests.get(URL, headers=HEADERS, timeout=20)
        if respuesta.status_code == 200:
            #paso 2: lee el html con beautifulsoup
            sopa = BeautifulSoup(respuesta.text, "html.parser")
            #paso 3: busca la tabla de temperaturas
            tabla = sopa.find("table")
            if tabla is not None:
                ahora = datetime.now()
                #paso 4: recorre cada fila (cada fila es una estacion)
                for fila in tabla.find_all("tr")[1:]:
                    celdas = [td.get_text(strip=True) for td in fila.find_all("td")]
                    if len(celdas) >= 3:
                        #paso 5: coge el nombre (col 0) y la temperatura (col 2)
                        estacion = celdas[0]
                        temperatura = limpiar_numero(celdas[2])
                        #guarda la estacion solo si tiene temperatura
                        if estacion and temperatura:
                            filas.append({
                                "dia": ahora.day,
                                "mes": ahora.month,
                                "año": ahora.year,
                                "hora": ahora.strftime("%H:%M"),
                                "estacion": estacion,
                                "temperatura_C": temperatura,
                            })
    except Exception as e:
        print("Error en el scraping de clima:", e)
    return pd.DataFrame(filas)

def main():
    df = scrapear_clima_aemet()
    if df.empty:
        print("No se obtuvieron datos de clima.")
    else:
        df.to_csv(OUT_FILE, sep=";", index=False, encoding="utf-8-sig")
        print(f"[OK] Clima scrapeado: {len(df)} estaciones -> {OUT_FILE.resolve()}")
        print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
