import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC              # Corregido
from sklearn.preprocessing import StandardScaler # Necesario para SVM
from sklearn.pipeline import make_pipeline       # Para facilitar el escalado
import joblib                             # Corregido
import os

def entrenamiento_svm(target_usuario):
    if not os.path.exists("Resultados/dataset_unificado.csv"):
        return "Error: No existe el archivo de datos."

    df = pd.read_csv("Resultados/dataset_unificado.csv", sep=';')
    
    mapa = {
        "Accidentes": "total_de_accidentes",
        "Calidad Aire": "valor_calidad_aire",
        "Emergencias": "cantidad_emergencias",
        "Tráfico": "trafico_medio"
    }
    
    if target_usuario not in mapa:
        return f"Error: {target_usuario} no es un objetivo válido."
        
    col_objetivo = mapa[target_usuario]
    
    # Target binario
    df['target'] = (df[col_objetivo] > df[col_objetivo].mean()).astype(int)
    
    X = df[['dia', 'mes', 'año', 'codigo_de_distrito']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # USAMOS UN PIPELINE: Esto escala los datos automáticamente antes de entrenar
    modelo = make_pipeline(
        StandardScaler(), 
        SVC(probability=True, random_state=42) # Quitamos n_estimators
    )
    
    modelo.fit(X_train, y_train)
    
    os.makedirs("modelos_guardados", exist_ok=True)
    nombre_archivo = f"modelos_guardados/SVM_{target_usuario.replace(' ', '')}.pkl"
    joblib.dump(modelo, nombre_archivo)
    
    return modelo.score(X_test, y_test)