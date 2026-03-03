import joblib
import pandas as pd
import os

def realizar_prediccion(modelo_completo_nombre, datos_entrada, salida):
    #carga el modelo y calcula la probabilidad
    resultado = {}
    ruta = os.path.join("modelos_guardados", modelo_completo_nombre)
    if not os.path.exists(ruta):
        resultado = {"error": "Modelo no encontrado"}
    else:
        try:
            modelo = joblib.load(ruta)
            df_input = pd.DataFrame([datos_entrada])

            if hasattr(modelo, "feature_names_in_"):
                cols = modelo.feature_names_in_
            else:
                cols = ["dia", "mes", "año", "codigo_de_distrito"]

            for col in cols:
                if col not in df_input.columns:
                    df_input[col] = 0

            prob = modelo.predict_proba(df_input[cols])[0][1]
            resultado = {"probabilidad": prob}
        except Exception as e:
            resultado = {"error": str(e)}

    salida.append(resultado)
