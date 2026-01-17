import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Corregido: importación correcta
import joblib
import os

def entrenamiento_arbol_de_decision(target_usuario):
    # Verificación de existencia del archivo
    ruta_csv = "Resultados/dataset_unificado.csv"
    if not os.path.exists(ruta_csv):
        return "Error: No existe el dataset unificado."

    df = pd.read_csv(ruta_csv, sep=';')
    
    # Mapeo de la selección de la App al nombre real de la columna
    mapa = {
        "Accidentes": "total_de_accidentes",
        "Calidad Aire": "valor_calidad_aire",
        "Emergencias": "cantidad_emergencias",
        "Tráfico": "trafico_medio"
    }
    
    if target_usuario not in mapa:
        return f"Error: {target_usuario} no es válido."
        
    col_objetivo = mapa[target_usuario]
    
    # Crear target binario basado en la media (Clasificación)
    df['target'] = (df[col_objetivo] > df[col_objetivo].mean()).astype(int)
    
    # Variables predictoras
    X = df[['dia', 'mes', 'año', 'codigo_de_distrito']]
    y = df['target']
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuración del modelo (Sin n_estimators, eso es de Random Forest)
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Guardado del modelo
    os.makedirs("modelos_guardados", exist_ok=True)
    nombre_archivo = f"DecisionTree_{target_usuario.replace(' ', '')}.pkl"
    joblib.dump(modelo, f"modelos_guardados/{nombre_archivo}")
    
    # Retornar precisión
    return modelo.score(X_test, y_test)