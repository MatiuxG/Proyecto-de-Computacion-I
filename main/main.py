import pandas as pd
import os

def normalizar_texto(serie):
    if serie is None: return serie
    s = serie.astype(str).str.upper().str.strip()
    reemplazos = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
    for car, rep in reemplazos.items():
        s = s.str.replace(car, rep, regex=False)
    return s 

def cargar_y_agrupar(nombre_log, ruta, mapeo, col_valor, op='mean'):
    if not os.path.exists(ruta):
        print(f"Advertencia: No existe {ruta}")
        return None #FLAG
    try:
        df = pd.read_csv(ruta, sep=';', encoding='utf-8')
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={k.lower(): v for k, v in mapeo.items()})
        
        if 'fecha' in df.columns and 'dia' not in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df['dia'], df['mes'], df['año'] = df['fecha'].dt.day, df['fecha'].dt.month, df['fecha'].dt.year

        keys = ['dia', 'mes', 'año', 'codigo_de_distrito']
        for col in keys: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.groupby(keys)[col_valor].agg(op).reset_index() #FLAG
    except Exception as e:
        print(f"Error en {nombre_log}: {e}")
        return None

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    fuentes = [
        ("accidentes", "Accidentes_Scripts/Resultados/datasheet_accidentes.csv", {'district_code': 'codigo_de_distrito'}, 'total_de_accidentes', 'sum'),
        ("aire", "CalidadAire_Scripts/Resultados/datasheet_calidad_aire.csv", {'no_distrito': 'codigo_de_distrito'}, 'valor_calidad_aire', 'mean'),
        ("emergencias", "Emergencias_Scripts/Resultados/datasheet_emergencias.csv", {'no_distrito': 'codigo_de_distrito'}, 'cantidad_emergencias', 'sum'),
        ("trafico", "Trafico_Scripts/Resultados/resultado.csv", {'id_distrito': 'codigo_de_distrito'}, 'trafico_medio', 'mean')
    ]

    df_final = None
    for nom, rel_path, mapeo, col, op in fuentes:
        ruta = os.path.join(base, rel_path)
        df = cargar_y_agrupar(nom, ruta, mapeo, col, op)
        if df is not None:
            if df_final is None: df_final = df
            else: df_final = pd.merge(df_final, df, on=['dia', 'mes', 'año', 'codigo_de_distrito'], how='outer')

    df_final = df_final.fillna(0)
        df_final['target_accidentes'] = (df_final['total_de_accidentes'] > df_final['total_de_accidentes'].median()).astype(int)
    df_final['target_aire'] = (df_final['valor_calidad_aire'] > df_final['valor_calidad_aire'].median()).astype(int)
    df_final['target_emergencias'] = (df_final['cantidad_emergencias'] > df_final['cantidad_emergencias'].median()).astype(int)

    out = os.path.join(base, "Resultados", "dataset_unificado.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_final.to_csv(out, sep=';', index=False)
    print(f"Dataset creado: {out}")

if __name__ == "__main__":
    main()
