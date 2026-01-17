import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenamiento_random_forest(target_usuario):
    df = pd.read_csv("Resultados/dataset_unificado.csv", sep=';')
    
    # Mapeo de lo que elige el usuario a la columna real del CSV
    mapa = {
        "Accidentes": "total_de_accidentes",
        "Calidad Aire": "valor_calidad_aire",
        "Emergencias": "cantidad_emergencias",
        "Tráfico": "trafico_medio"
    }
    
    col_objetivo = mapa[target_usuario]
    
    # Creamos un target binario (1 si es mayor a la media, 0 si no) para clasificación
    df['target'] = (df[col_objetivo] > df[col_objetivo].mean()).astype(int)
    
    X = df[['dia', 'mes', 'año', 'codigo_de_distrito']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    os.makedirs("modelos_guardados", exist_ok=True)
    joblib.dump(modelo, f"modelos_guardados/RandomForest_{target_usuario.replace(' ', '')}.pkl")
    
    return modelo.score(X_test, y_test)