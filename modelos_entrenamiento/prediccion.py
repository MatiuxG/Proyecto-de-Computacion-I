import joblib
import pandas as pd
import os

def realizar_prediccion(modelo_completo_nombre, datos_entrada):
    """
    modelo_completo_nombre: Nombre del archivo .pkl
    datos_entrada: Diccionario con todas las variables necesarias
    """
    ruta_modelo = os.path.join("modelos_guardados", modelo_completo_nombre)
    
    if not os.path.exists(ruta_modelo):
        return "Error: Archivo de modelo no encontrado."

    try:
        modelo = joblib.load(ruta_modelo)
        df_input = pd.DataFrame([datos_entrada])
        
        # El orden de las columnas debe ser el mismo que en el entrenamiento
        # Si añadiste tráfico y clima al entrenamiento, deben ir aquí
        columnas_entrenamiento = ['dia', 'mes', 'año', 'codigo_de_distrito', 'trafico_medio', 'valor_calidad_aire']
        
        # Filtramos solo las columnas que el modelo espera (por si sobran)
        df_input = df_input[columnas_entrenamiento]
        
        prediccion = modelo.predict(df_input)[0]
        return {"resultado": int(prediccion)}
    except Exception as e:
        return f"Error en predicción: {str(e)}"