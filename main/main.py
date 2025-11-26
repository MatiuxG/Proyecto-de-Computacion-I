import pandas as pd
import os
import glob

def main():
    # --- Configuración de rutas ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas relativas a los CSV
    paths = {
        "clima": os.path.join(base_dir, "..", "Clima_Scripts", "Clima_Scripts", "Resultados", "datasheet_clima.csv"),
        "accidentes": os.path.join(base_dir, "..", "Accidentes_Scripts", "Accidentes_Scripts", "Resultados", "datasheet_accidentes.csv"),
        "aire": os.path.join(base_dir, "..", "CalidadAire_Scripts", "CalidadAire_Scripts", "Resultados", "datasheet_calidad_aire_agregado.csv"),
        "emergencias": os.path.join(base_dir, "..", "Emergencias_Scripts", "Emergencias_Scripts", "Resultados", "datasheet_emergencias.csv"),
        "obras": os.path.join(base_dir, "..", "Obras_Scripts", "Obras_Scripts", "Resultados", "datasheet_plazo_ejecucion.csv"),
        "trafico": os.path.join(base_dir, "..", "Trafico_Scripts", "Trafico_Scripts", "Resultados", "resultado.csv")
    }

    # Columnas clave para la unión
    merge_keys = ['Dia', 'Mes', 'Año', 'Codigo_de_distrito', 'nombre_de_distrito']

    # --- Funciones de ayuda ---
    def load_and_standardize(name, filepath, key_mapping, columns_to_keep):
        """
        Carga un CSV, renombra columnas clave, estandariza textos y filtra las columnas deseadas.
        """
        print(f"Cargando {name} desde: {filepath}")
        if not os.path.exists(filepath):
            print(f"  [AVISO] No se encontró el archivo para {name}. Se omitirá.")
            return None
        
        try:
            df = pd.read_csv(filepath, sep=';', dtype=str)
        except:
            df = pd.read_csv(filepath, sep=',', dtype=str)

        # Renombrar columnas
        df = df.rename(columns=key_mapping)

        # 1. Estandarizar Código de distrito (01, 02...)
        if 'Codigo_de_distrito' in df.columns:
            df['Codigo_de_distrito'] = df['Codigo_de_distrito'].apply(lambda x: str(x).strip().zfill(2) if pd.notna(x) else x)
        
        # 2. Estandarizar Nombre de distrito (quitar espacios y poner en formato Título)
        if 'nombre_de_distrito' in df.columns:
            df['nombre_de_distrito'] = df['nombre_de_distrito'].astype(str).str.strip().str.title()

        # Asegurar columnas numéricas de fecha
        for col in ['Dia', 'Mes', 'Año']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Filtrar columnas
        cols_existentes = [c for c in (merge_keys + columns_to_keep) if c in df.columns]
        df = df[cols_existentes]
        
        return df

    # --- 1. Carga de Datos ---

    # A. CLIMA
    cols_clima = [
        'Temp_Media_°C','Temp_Max_°C','Temp_Min_°C','Hora_Temp_Max','Hora_Temp_Min',
        'Precipitacion_mm','Vel_Viento_Media_m/s', 'Presion_Max_hPa', 'Presion_Min_hPa'
    ]
    map_clima = {'district_code': 'Codigo_de_distrito', 'district_name': 'nombre_de_distrito'}
    df_clima = load_and_standardize("Clima", paths["clima"], map_clima, cols_clima)

    # B. ACCIDENTES
    cols_acc = ['total_de_accidentes']
    map_acc = {'district_code': 'Codigo_de_distrito', 'district_name': 'nombre_de_distrito'}
    df_acc = load_and_standardize("Accidentes", paths["accidentes"], map_acc, cols_acc)

    # C. CALIDAD AIRE
    cols_aire = ['Oxidosde nitrogeno', 'Particulas']
    map_aire = {
        'dia': 'Dia', 'mes': 'Mes', 'año': 'Año', 
        'numero de distrito': 'Codigo_de_distrito', 'nombre del distrito': 'nombre_de_distrito'
    }
    df_aire = load_and_standardize("Calidad Aire", paths["aire"], map_aire, cols_aire)

    # D. EMERGENCIAS
    cols_emerg = ['cantidad_emergencias']
    map_emerg = {
        'dia': 'Dia', 'mes': 'Mes', 'año': 'Año', 
        'no_distrito': 'Codigo_de_distrito', 'nombre_distrito': 'nombre_de_distrito'
    }
    df_emerg = load_and_standardize("Emergencias", paths["emergencias"], map_emerg, cols_emerg)

    # E. OBRAS (Opcional, descomentar si se requiere)
    # df_obras = load_and_standardize("Obras", paths["obras"], map_obras, cols_obras)

    # F. TRÁFICO (Nueva sección agregada)
    cols_trafico = ['trafico_medio']
    map_trafico = {
        'dia': 'Dia', 
        'mes': 'Mes', 
        'año': 'Año', 
        'id_distrito': 'Codigo_de_distrito', 
        'nombre_distrito': 'nombre_de_distrito'
    }
    df_trafico = load_and_standardize("Tráfico", paths["trafico"], map_trafico, cols_trafico)

    # --- 2. Unión de Datasets ---
    
    # Se añade df_trafico a la lista para el merge
    dfs = [d for d in [df_clima, df_acc, df_aire, df_emerg, df_trafico] if d is not None]

    if not dfs:
        print("No se cargó ningún dataset. Finalizando.")
        return

    print("Uniendo datasets...")
    df_final = dfs[0]
    for i in range(1, len(dfs)):
        df_final = pd.merge(df_final, dfs[i], on=merge_keys, how='outer')

    # --- 3. Limpieza y Guardado ---

    df_final = df_final.fillna(0)

    if all(col in df_final.columns for col in ['Año', 'Mes', 'Dia', 'Codigo_de_distrito']):
        df_final = df_final.sort_values(by=['Año', 'Mes', 'Dia', 'Codigo_de_distrito'])

    output_dir = os.path.join(base_dir, "..", "Resultados")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "dataset_final.csv")
    
    print(f"Guardando resultado en: {output_path}")
    df_final.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
    print("¡Proceso completado!")

if __name__ == "__main__":
    main()