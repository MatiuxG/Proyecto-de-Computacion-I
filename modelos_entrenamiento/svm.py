import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime

def entrenamiento_svm(target_usuario, features_seleccionadas, ruta_guardado):
    df = pd.read_csv("Resultados/dataset_unificado.csv", sep=';')
    mapa = {"Accidentes": "target_accidentes", "Calidad Aire": "target_aire", "Emergencias": "target_emergencias"}
    col_target = mapa.get(target_usuario, "target_accidentes")

    X, y = df[features_seleccionadas], df[col_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    modelo = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    modelo.fit(X_train, y_train)
        
    joblib.dump(modelo, ruta_guardado)
    preds = modelo.predict(X_test)
        
    return {
            "accuracy": modelo.score(X_test, y_test), "mae": mean_absolute_error(y_test, preds),
            "mse": mean_squared_error(y_test, preds), "n_muestras": len(df),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "archivo": ruta_guardado
        }
