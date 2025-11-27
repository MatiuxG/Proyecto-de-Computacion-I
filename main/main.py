import pandas as pd
import os
import numpy as np

def normalizar_texto(serie):
    """
    Normaliza los nombres de distrito: mayúsculas, quita tildes y reemplaza guiones por espacios.
    Ayuda a cruzar datos como 'Fuencarral-El Pardo' con 'FUENCARRAL EL PARDO'.
    """
    if serie is None:
        return serie
    # Convertir a string y mayúsculas, eliminando espacios extra
    s = serie.astype(str).str.upper().str.strip()
    # Reemplazos de caracteres
    reemplazos = {
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U',
        '-': ' ',  # Reemplazar guiones por espacios
        '  ': ' '  # Quitar dobles espacios si se generan
    }
    for car, rep in reemplazos.items():
        s = s.str.replace(car, rep, regex=False)
    return s

def cargar_y_preparar(ruta, mapeo_cols, sep=';'):
    """
    Carga un CSV y renombra sus columnas según el mapeo proporcionado.
    """
    if not os.path.exists(ruta):
        print(f"Advertencia: No se encontró el archivo en {ruta}")
        return None
    
    try:
        df = pd.read_csv(ruta, sep=sep)
    except Exception as e:
        print(f"Error al leer {ruta}: {e}")
        return None
    
    # Renombrar columnas para estandarizar claves
    df = df.rename(columns=mapeo_cols)
    
    # Estandarizar claves de fusión
    if 'codigo_de_distrito' in df.columns:
        # Convertir a numérico, forzando errores a NaN
        df['codigo_de_distrito'] = pd.to_numeric(df['codigo_de_distrito'], errors='coerce')
    
    if 'nombre_de_distrito' in df.columns:
        df['nombre_de_distrito'] = normalizar_texto(df['nombre_de_distrito'])
        
    # Asegurar que las columnas de fecha sean numéricas para el ordenamiento posterior
    for col in ['dia', 'mes', 'año']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def main():
    # --- Configuración de rutas ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Definición de rutas a los archivos CSV
    rutas = {
        "clima": os.path.join(parent_dir, "Clima_Scripts", "Resultados", "datasheet_clima.csv"),
        "accidentes": os.path.join(parent_dir, "Accidentes_Scripts", "Resultados", "datasheet_accidentes.csv"),
        "aire": os.path.join(parent_dir, "CalidadAire_Scripts", "Resultados", "datasheet_calidad_aire.csv"),
        "emergencias": os.path.join(parent_dir, "Emergencias_Scripts", "Resultados", "datasheet_emergencias.csv"),
        "obras": os.path.join(parent_dir, "Obras_Scripts", "Resultados", "datasheet_obras.csv"),
        "trafico": os.path.join(parent_dir, "Trafico_Scripts", "Resultados", "resultado.csv")
    }

    # --- Mapeo de columnas para estandarizar nombres ---
    map_clima = {
        'Dia': 'dia', 'Mes': 'mes', 'Año': 'año', 
        'district_code': 'codigo_de_distrito', 'district_name': 'nombre_de_distrito'
    }
    
    map_acc = {
        'Dia': 'dia', 'Mes': 'mes', 'Año': 'año', 
        'district_code': 'codigo_de_distrito', 'district_name': 'nombre_de_distrito'
    }
    
    map_aire = {
        'no_distrito': 'codigo_de_distrito', 'nombre_distrito': 'nombre_de_distrito'
    }
    
    map_emerg = {
        'no_distrito': 'codigo_de_distrito', 'nombre_distrito': 'nombre_de_distrito'
    }
    
    map_obras = {
        'no_distrito': 'codigo_de_distrito', 'nombre_distrito': 'nombre_de_distrito',
        'terminada': 'Obra_terminada'
    }
    
    map_trafico = {
        'id_distrito': 'codigo_de_distrito', 'nombre_distrito': 'nombre_de_distrito'
    }

    # --- Carga de Datos ---
    print("Cargando datasets...")
    df_clima = cargar_y_preparar(rutas["clima"], map_clima)
    df_acc = cargar_y_preparar(rutas["accidentes"], map_acc)
    df_aire = cargar_y_preparar(rutas["aire"], map_aire)
    df_emerg = cargar_y_preparar(rutas["emergencias"], map_emerg)
    df_obras = cargar_y_preparar(rutas["obras"], map_obras)
    df_trafico = cargar_y_preparar(rutas["trafico"], map_trafico)

    # --- Unificación (Merge) ---
    print("Unificando datasets...")
    
    merge_keys = ['dia', 'mes', 'año', 'codigo_de_distrito', 'nombre_de_distrito']
    
    # Usar Clima como base inicial si existe, sino crear un DataFrame vacío con las keys
    if df_clima is not None:
        df_final = df_clima
    else:
        df_final = pd.DataFrame(columns=merge_keys) 

    if df_acc is not None:
        df_final = pd.merge(df_final, df_acc, on=merge_keys, how='outer')
    if df_aire is not None:
        df_final = pd.merge(df_final, df_aire, on=merge_keys, how='outer')
    if df_emerg is not None:
        df_final = pd.merge(df_final, df_emerg, on=merge_keys, how='outer')
    if df_obras is not None:
        df_final = pd.merge(df_final, df_obras, on=merge_keys, how='outer')
    if df_trafico is not None:
        df_final = pd.merge(df_final, df_trafico, on=merge_keys, how='outer')

    # --- Filtrado de Distritos Excluidos ---
    print("Filtrando zonas prohibidas (5, 6, 7, 9, 10, 11, 12, 17)...")
    distritos_excluidos = [5, 6, 7, 9, 10, 11, 12, 17]
    
    # Aseguramos que sea numérico y rellenamos nulos temporalmente con -1 para no borrar filas sin distrito si las hubiera
    df_final['codigo_de_distrito'] = pd.to_numeric(df_final['codigo_de_distrito'], errors='coerce').fillna(-1)
    
    # Aplicar filtro
    df_final = df_final[~df_final['codigo_de_distrito'].isin(distritos_excluidos)]

    # --- Limpieza y Selección de Columnas ---
    
    columnas_finales_orden = [
        'dia', 'mes', 'año', 'codigo_de_distrito', 'nombre_de_distrito',
        'Temp_Media_°C', 'Temp_Max_°C', 'Temp_Min_°C', 
        'Hora_Temp_Max', 'Hora_Temp_Min', 'Precipitacion_mm', 
        'Vel_Viento_Media_m/s', 'Racha_Max_m/s', 'Presion_Max_hPa', 'Presion_Min_hPa',
        'total_de_accidentes', 
        'valor_calidad_aire', 
        'cantidad_emergencias', 
        'Obra_terminada', 
        'trafico_medio'
    ]
    
    # Lista de columnas que convertiremos a INT (todas las numéricas)
    # Se excluyen 'Hora_Temp_Max' y 'Hora_Temp_Min' si son textos tipo "14:30". 
    # Si son números, se convertirán. Si no, pandas fallará al hacer astype(int), así que usamos try/except.
    cols_numericas_int = [
        'dia', 'mes', 'año', 'codigo_de_distrito',
        'Temp_Media_°C', 'Temp_Max_°C', 'Temp_Min_°C', 
        'Precipitacion_mm', 'Vel_Viento_Media_m/s', 'Racha_Max_m/s', 
        'Presion_Max_hPa', 'Presion_Min_hPa',
        'total_de_accidentes', 
        'valor_calidad_aire', 
        'cantidad_emergencias', 
        'trafico_medio'
    ]

    # Crear columnas faltantes con NaN
    for col in columnas_finales_orden:
        if col not in df_final.columns:
            df_final[col] = np.nan

    # --- Conversión a enteros (int) ---
    print("Convirtiendo valores numéricos a enteros...")
    
    for col in cols_numericas_int:
        if col in df_final.columns:
            # Rellenar NaN con 0
            df_final[col] = df_final[col].fillna(0)
            try:
                # Convertir a int
                df_final[col] = df_final[col].astype(int)
            except Exception as e:
                print(f"No se pudo convertir {col} a int: {e}")

    # Seleccionar y reordenar columnas
    df_final = df_final[columnas_finales_orden]
    
    # --- Ordenar Cronológicamente ---
    print("Ordenando datos...")
    df_final = df_final.sort_values(by=['año', 'mes', 'dia'])
    
    # --- Guardado ---
    output_folder = os.path.join(parent_dir, "Resultados")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_file = os.path.join(output_folder, "dataset_unificado.csv")
    
    # Guardar con ; como separador
    df_final.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    print(f"Proceso finalizado. Dataset guardado en: {output_file}")
    # Mostrar una muestra para verificar
    print(df_final.head())

if __name__ == "__main__":
    main()