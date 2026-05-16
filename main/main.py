"""
Unifica los CSV de cada fuente (trafico, accidentes, calidad del aire,
emergencias, clima y obras) en un unico dataset listo para entrenar.

Genera features de calendario (dia de la semana, fin de semana, festivo)
y construye targets binarios usando el percentil 90 del historico
(una incidencia "alta" es estar en el top 10% del historico, no por
encima de la mediana como tenia la version anterior).
"""

from pathlib import Path

import pandas as pd
import holidays


#raiz del proyecto
BASE = Path(__file__).resolve().parent.parent
OUT_FILE = BASE / "Resultados" / "dataset_unificado.csv"

#percentil que marca incidencia alta; 90 = top 10% del historico
PERCENTIL_INCIDENCIA = 90

#festivos de la Comunidad de Madrid
FESTIVOS_MADRID = holidays.Spain(subdiv="MD", years=range(2020, 2030))


#fuentes del dataset: (nombre, ruta, mapeo_columnas, columna_valor, operacion)
FUENTES = [
    (
        "accidentes",
        "Accidentes_Scripts/Resultados/datasheet_accidentes.csv",
        {"district_code": "codigo_de_distrito"},
        "total_de_accidentes",
        "sum",
    ),
    (
        "aire",
        "CalidadAire_Scripts/Resultados/datasheet_calidad_aire.csv",
        {"no_distrito": "codigo_de_distrito"},
        "valor_calidad_aire",
        "mean",
    ),
    (
        "emergencias",
        "Emergencias_Scripts/Resultados/datasheet_emergencias.csv",
        {"no_distrito": "codigo_de_distrito"},
        "cantidad_emergencias",
        "sum",
    ),
    (
        "trafico",
        "Trafico_Scripts/Resultados/resultado.csv",
        {"id_distrito": "codigo_de_distrito"},
        "trafico_medio",
        "mean",
    ),
    (
        "obras",
        "Obras_Scripts/Resultados/datasheet_obras.csv",
        {"no_distrito": "codigo_de_distrito"},
        "terminada",
        "count",
    ),
]

#el clima tiene varias columnas que nos interesan, no una sola. lo procesamos aparte
CLIMA_RUTA = "Clima_Scripts/Resultados/datasheet_clima.csv"
CLIMA_COLUMNAS_RENOMBRE = {
    "district_code": "codigo_de_distrito",
    "Temp_Media_°C": "temp_media",
    "Temp_Max_°C": "temp_max",
    "Temp_Min_°C": "temp_min",
    "Precipitacion_mm": "precipitacion",
    "Vel_Viento_Media_m/s": "viento_medio",
}
CLIMA_COLUMNAS_FINALES = ["temp_media", "temp_max", "temp_min", "precipitacion", "viento_medio"]


def replicar_en_24_horas(df):
    #para fuentes sin granularidad horaria (clima diario, obras), replicamos
    #cada fila 24 veces (una por hora) para que el merge con las fuentes
    #horarias case en TODAS las horas y no solo en hora=0
    horas = pd.DataFrame({"hora": list(range(24))})
    res = df.merge(horas, how="cross")
    return res


def cargar_y_agrupar(nombre, ruta_csv, mapeo, columna_valor, operacion):
    #lee un csv, normaliza columnas y agrupa por hora-distrito.
    #si el csv trae columna `hora` agrupa por (dia, mes, año, hora, distrito);
    #si no la trae se agrupa por dia y se replica el valor en 24 horas.
    #devuelve None si el archivo no existe o falla la carga
    df_resultado = None

    if not ruta_csv.exists():
        print(f"  [Aviso] No se encuentra {ruta_csv} (se omite {nombre})")
    else:
        try:
            df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8")
            df.columns = df.columns.str.lower().str.strip()

            #renombramos columnas del origen al nombre interno (claves a minusculas)
            mapeo_lower = {k.lower(): v for k, v in mapeo.items()}
            df = df.rename(columns=mapeo_lower)

            #si la fuente trae una columna "fecha" suelta, sacamos dia/mes/año de ella
            if "fecha" in df.columns and "dia" not in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                df["dia"] = df["fecha"].map(lambda x: x.day if pd.notna(x) else None)
                df["mes"] = df["fecha"].map(lambda x: x.month if pd.notna(x) else None)
                df["año"] = df["fecha"].map(lambda x: x.year if pd.notna(x) else None)

            tiene_hora = "hora" in df.columns

            #aseguramos tipos numericos en las claves de agrupacion
            claves_base = ["dia", "mes", "año", "codigo_de_distrito"]
            for col in claves_base:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if tiene_hora:
                df["hora"] = pd.to_numeric(df["hora"], errors="coerce")
                claves = ["dia", "mes", "año", "hora", "codigo_de_distrito"]
            else:
                claves = claves_base

            #para "obras" usamos count en lugar de sum porque la columna
            #"terminada" es texto (true/false); lo convertimos a 1 antes
            if operacion == "count":
                df[columna_valor] = 1
                df_resultado = (
                    df.groupby(claves)[columna_valor]
                    .count()
                    .reset_index()
                    .rename(columns={columna_valor: "obras_activas"})
                )
            else:
                df_resultado = (
                    df.groupby(claves)[columna_valor]
                    .agg(operacion)
                    .reset_index()
                )

            #si la fuente no tenia hora la replicamos en las 24 horas
            if not tiene_hora and df_resultado is not None and not df_resultado.empty:
                df_resultado = replicar_en_24_horas(df_resultado)

        except Exception as error:
            print(f"  [Error] {nombre}: {error}")

    return df_resultado


def cargar_clima(ruta_csv):
    #carga el csv de clima (AEMET solo da resumen diario, sin hora) y
    #replica el valor del dia en las 24 horas para casar con el resto
    df_resultado = None

    if not ruta_csv.exists():
        print(f"  [Aviso] No se encuentra {ruta_csv} (no habra variables de clima)")
    else:
        try:
            df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8")
            df = df.rename(columns=CLIMA_COLUMNAS_RENOMBRE)
            df.columns = df.columns.str.lower().str.strip()

            claves = ["dia", "mes", "año", "codigo_de_distrito"]
            columnas = [c.lower() for c in CLIMA_COLUMNAS_FINALES]

            for col in claves:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            for col in columnas:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            columnas_existentes = [c for c in columnas if c in df.columns]
            df_resultado = (
                df.groupby(claves)[columnas_existentes]
                .mean()
                .reset_index()
            )

            #clima diario -> replicamos cada dia-distrito en sus 24 horas
            df_resultado = replicar_en_24_horas(df_resultado)
        except Exception as error:
            print(f"  [Error] clima: {error}")

    return df_resultado


def añadir_features_calendario(df):
    #crea dia_semana (0=lunes..6=domingo), es_finde, es_festivo y es_hora_punta.
    #es_hora_punta marca 1 en franjas tipicas de trafico intenso en Madrid
    #(7-10 mañana y 17-20 tarde) para dar al modelo una pista del ritmo urbano
    fechas = pd.to_datetime(
        dict(year=df["año"], month=df["mes"], day=df["dia"]),
        errors="coerce",
    )

    df["dia_semana"] = fechas.dt.dayofweek.fillna(0).astype(int)
    df["es_finde"] = (df["dia_semana"] >= 5).astype(int)
    df["es_festivo"] = fechas.map(
        lambda f: 1 if pd.notna(f) and f.date() in FESTIVOS_MADRID else 0
    ).astype(int)

    if "hora" in df.columns:
        horas = pd.to_numeric(df["hora"], errors="coerce").fillna(0).astype(int)
        df["hora"] = horas
        df["es_hora_punta"] = (
            ((horas >= 7) & (horas <= 10)) | ((horas >= 17) & (horas <= 20))
        ).astype(int)
    else:
        df["hora"] = 0
        df["es_hora_punta"] = 0

    return df


def añadir_targets(df, percentil):
    #crea las columnas target_accidentes, target_aire y target_emergencias.
    #una jornada (dia-distrito) se marca como "incidencia alta" (=1) si el
    #valor supera el percentil indicado del historico. la version anterior
    #usaba la mediana, lo que hacia que el 50% del dataset fuera "incidencia"
    pares = [
        ("target_accidentes", "total_de_accidentes"),
        ("target_aire", "valor_calidad_aire"),
        ("target_emergencias", "cantidad_emergencias"),
    ]

    for nombre_target, columna_origen in pares:
        if columna_origen in df.columns:
            umbral = df[columna_origen].quantile(percentil / 100.0)
            df[nombre_target] = (df[columna_origen] > umbral).astype(int)
        else:
            df[nombre_target] = 0

    return df


CLAVE_MERGE = ["dia", "mes", "año", "hora", "codigo_de_distrito"]


def unir_fuentes():
    #carga todas las fuentes y devuelve el dataframe unido por
    #(dia, mes, año, hora, codigo_de_distrito)
    df_final = None

    for nombre, rel_path, mapeo, columna, operacion in FUENTES:
        ruta = BASE / rel_path
        df_fuente = cargar_y_agrupar(nombre, ruta, mapeo, columna, operacion)

        if df_fuente is not None:
            if df_final is None:
                df_final = df_fuente
            else:
                df_final = pd.merge(
                    df_final,
                    df_fuente,
                    on=CLAVE_MERGE,
                    how="outer",
                )

    #el clima se añade aparte porque trae varias columnas
    df_clima = cargar_clima(BASE / CLIMA_RUTA)
    if df_clima is not None:
        if df_final is None:
            df_final = df_clima
        else:
            df_final = pd.merge(
                df_final,
                df_clima,
                on=CLAVE_MERGE,
                how="outer",
            )

    return df_final


def crear_dataframe_vacio():
    #si no se cargo ninguna fuente, devolvemos un csv con la cabecera
    #correcta para que los modelos no fallen al leerlo
    columnas = [
        "dia", "mes", "año", "hora", "codigo_de_distrito",
        "total_de_accidentes", "valor_calidad_aire",
        "cantidad_emergencias", "trafico_medio", "obras_activas",
        "temp_media", "temp_max", "temp_min", "precipitacion", "viento_medio",
        "dia_semana", "es_finde", "es_festivo", "es_hora_punta",
        "target_accidentes", "target_aire", "target_emergencias",
    ]
    res = pd.DataFrame(columns=columnas)
    return res


def main():
    #orquesta la union: lee cada fuente, las cruza, añade features y targets,
    #y escribe el dataset final
    print(f"[Info] Uniendo fuentes desde {BASE}")
    df_final = unir_fuentes()

    if df_final is None:
        print("[Aviso] No se cargo ninguna fuente, se generara un CSV vacio.")
        df_final = crear_dataframe_vacio()
    else:
        df_final = df_final.fillna(0)
        df_final = añadir_features_calendario(df_final)
        df_final = añadir_targets(df_final, PERCENTIL_INCIDENCIA)

    #orden estable de columnas en el csv final
    orden_columnas = [
        "dia", "mes", "año", "hora", "codigo_de_distrito",
        "dia_semana", "es_finde", "es_festivo", "es_hora_punta",
        "total_de_accidentes", "valor_calidad_aire",
        "cantidad_emergencias", "trafico_medio", "obras_activas",
        "temp_media", "temp_max", "temp_min", "precipitacion", "viento_medio",
        "target_accidentes", "target_aire", "target_emergencias",
    ]
    columnas_presentes = [c for c in orden_columnas if c in df_final.columns]
    df_final = df_final[columnas_presentes]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_FILE, sep=";", index=False)

    print(f"[OK] Dataset generado: {OUT_FILE}")
    print(f"     Filas:    {len(df_final)}")
    print(f"     Columnas: {len(df_final.columns)}")
    print(f"     Percentil de incidencia: {PERCENTIL_INCIDENCIA}")


if __name__ == "__main__":
    main()
