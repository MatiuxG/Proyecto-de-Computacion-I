import pandas as pd
import os

def cargar_y_agrupar(nombre_log, ruta, mapeo, col_valor, op, resultados):
    #lee un csv, lo renombra y lo agrupa por dia/mes/año/distrito
    df_final = None
    if not os.path.exists(ruta):
        print("Advertencia: No existe", ruta)
    else:
        try:
            df = pd.read_csv(ruta, sep=";", encoding="utf-8")
            df.columns = df.columns.str.lower().str.strip()
            df = df.rename(columns={k.lower(): v for k, v in mapeo.items()})

            if "fecha" in df.columns and "dia" not in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                #1) convierte la columna fecha a tipo datetime
                #2) para cada fila, si la fecha existe, saca dia/mes/año
                #3) si la fecha esta vacia o mal, deja None
                df["dia"] = df["fecha"].map(lambda x: x.day if pd.notna(x) else None)
                df["mes"] = df["fecha"].map(lambda x: x.month if pd.notna(x) else None)
                df["año"] = df["fecha"].map(lambda x: x.year if pd.notna(x) else None)

            keys = ["dia", "mes", "año", "codigo_de_distrito"]
            for col in keys:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df_final = df.groupby(keys)[col_valor].agg(op).reset_index()
        except Exception as e:
            print("Error en", nombre_log, ":", e)

    resultados[nombre_log] = df_final

def main():
    #ruta base del proyecto
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #fuentes a unir en un dataset final
    fuentes = [
        ("accidentes", "Accidentes_Scripts/Resultados/datasheet_accidentes.csv", {'district_code': 'codigo_de_distrito'}, 'total_de_accidentes', 'sum'),
        ("aire", "CalidadAire_Scripts/Resultados/datasheet_calidad_aire.csv", {'no_distrito': 'codigo_de_distrito'}, 'valor_calidad_aire', 'mean'),
        ("emergencias", "Emergencias_Scripts/Resultados/datasheet_emergencias.csv", {'no_distrito': 'codigo_de_distrito'}, 'cantidad_emergencias', 'sum'),
        ("trafico", "Trafico_Scripts/Resultados/resultado.csv", {'id_distrito': 'codigo_de_distrito'}, 'trafico_medio', 'mean')
    ]

    resultados = {}
    #carga cada fuente con su agregacion
    for nom, rel_path, mapeo, col, op in fuentes:
        ruta = os.path.join(base, rel_path)
        cargar_y_agrupar(nom, ruta, mapeo, col, op, resultados)

    df_final = None
    for nom, _, _, _, _ in fuentes:
        df = resultados.get(nom)
        if df is not None:
            if df_final is None:
                df_final = df
            else:
                df_final = pd.merge(
                    df_final,
                    df,
                    on=["dia", "mes", "año", "codigo_de_distrito"],
                    how="outer"
                )

    if df_final is None:
        print("[AVISO] No hay datos para unir, se crea un csv vacio")
        df_final = pd.DataFrame(columns=[
            "dia",
            "mes",
            "año",
            "codigo_de_distrito",
            "total_de_accidentes",
            "valor_calidad_aire",
            "cantidad_emergencias",
            "trafico_medio",
        ])

    df_final = df_final.fillna(0)
    #crea etiquetas binarias: 1 si esta por encima de la mediana, 0 si no
    df_final["target_accidentes"] = (df_final["total_de_accidentes"] > df_final["total_de_accidentes"].median()).astype(int)
    df_final['target_aire'] = (df_final['valor_calidad_aire'] > df_final['valor_calidad_aire'].median()).astype(int)
    df_final['target_emergencias'] = (df_final['cantidad_emergencias'] > df_final['cantidad_emergencias'].median()).astype(int)

    out = os.path.join(base, "Resultados", "dataset_unificado.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_final.to_csv(out, sep=';', index=False)
    print(f"Dataset creado: {out}")

if __name__ == "__main__":
    main()
