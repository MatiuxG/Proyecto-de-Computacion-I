import joblib
import pandas as pd
import os

def realizar_prediccion(ruta_modelo, datos_entrada):
    #resultado por defecto si el modelo no existe
    resultado = {"error": "Modelo no encontrado"}
    if os.path.exists(ruta_modelo):
        try:
            modelo = joblib.load(ruta_modelo)
            df_input = pd.DataFrame([datos_entrada])
            #columnas con las que se entreno el modelo, o las basicas si no las expone
            if hasattr(modelo, "feature_names_in_"):
                cols = modelo.feature_names_in_
            else:
                cols = ["dia", "mes", "año", "codigo_de_distrito"]
            #rellena con 0 las columnas que falten en la fila
            for col in cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            prob = modelo.predict_proba(df_input[cols])[0][1]
            resultado = {"probabilidad": prob}
        except Exception as e:
            resultado = {"error": str(e)}
    return resultado
