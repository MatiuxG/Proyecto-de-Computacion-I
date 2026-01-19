import joblib
import pandas as pd
import os

def realizar_prediccion(modelo_completo_nombre, datos_entrada):
    ruta = os.path.join("modelos_guardados", modelo_completo_nombre)
    if not os.path.exists(ruta):
        return {"error": "Modelo no encontrado"} #FLAG
    
    try:
        modelo = joblib.load(ruta)
        df_input = pd.DataFrame([datos_entrada])
        
        if hasattr(modelo, "feature_names_in_"):
            cols = modelo.feature_names_in_
        else:
            cols = ['dia', 'mes', 'año', 'codigo_de_distrito']
            
        for c in cols:
            if c not in df_input.columns: df_input[c] = 0
            
        prob = modelo.predict_proba(df_input[cols])[0][1]
        return {"probabilidad": prob}
    except Exception as e:
        return {"error": str(e)}
