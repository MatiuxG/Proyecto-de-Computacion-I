#scikit learn - NLTK (PLN)- XGBoost

#Decision Tree Classifier: un árbol de decisión es un modelo de aprendizaje supervisado utilizado tanto para clasificación como para regresión. Funciona dividiendo los datos en subconjuntos basados en características específicas, creando una estructura similar a un árbol donde cada nodo representa una característica, cada rama representa una decisión y cada hoja representa un resultado o clase.


# PARA SABER EN GENERAL:
# x significa todo lo que el modelo usa para predecir (características), lo que usa como entrada
# y significa la etiqueta que el modelo tiene que predecir (target)

#librerias importadas
import sklearn
import pandas
import numpy
import dataclasses
import joblib
#todo el tema del modelo 
from sklearn.model_selection import train_test_split #dividir el conjunto de datos en conjuntos de entrenamiento y prueba
from sklearn.tree import DecisionTreeClassifier #importar el clasificador de árbol de decisión
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #importar la función para calcular la precisión del modelo
from dataclasses import dataclass#para crear clases de datos
from joblib import dump, load #para guardar y cargar modelos entrenados
from typing import Tuple, Dict, Any, List, Optional #tipado para escribir bonico :)
#columnas que usaremos (BASES DEL CV)
COLUMNAS_BASE=["dia",
    "mes",
    "codigo_de_distrito",
    "nombre_de_distrito",
    "Precipitacion_mm",
    "Presion_Max_hPa",
    "Presion_Min_hPa",
    "Temp_Max_ºC",
    "Temp_Media_ºC",
    "Temp_Min_ºC",
    "Racha_Max_m/s",
    "Vel_Viento_Media_m/s",
    "total_de_accidentes",
    "cantidad_emergencias",
    "valor_calidad_aire",
    "trafico_medio",
]

#el modelo tiene que saber que target ha eleigo el usuario :
"""Targets soportados:
- "accidentes"  -> etiqueta accidente_riesgo
- "atascos"     -> etiqueta atasco_riesgo
- "emergencias" -> etiqueta emergencia_riesgo
- "aire"        -> etiqueta aire_riesgo"""

TARGETS_CONFIG={ "accidentes": {
        "label": "accidente_riesgo", #etiqueta objetivo para accidentes
        "leakage_feature": "total_de_accidentes", #col que borramos para evitar data leakage
    },"atascos": {
        "label": "atasco_riesgo",
        "leakage_feature": "trafico_medio",
    },
    "emergencias": {
        "label":"emergencia_riesgo",
        "leakage_feature": "cantidad_emergencias",
    },
    "aire": {
        "label": "aire_riesgo",
        "leakage_feature": "valor_calidad_aire",
    },
}

def load_data(data_path): #la ui nos pasa la ruta del csv
    df=pandas.read_csv(data_path, sep=";", engine="python") #cargamos el csv con pandas, el sep es la forma en la que estan separados los datos y el engine es para evitar errores
    return df #devolvemos el dataframe

def load_and_prepare_data(ruta_csv, tipo_incidentes):

    if tipo_incidentes not in TARGETS_CONFIG: #miramos si el target es valido
        raise ValueError(f"Target '{tipo_incidentes}' no soportado. Targets soportados: {list(TARGETS_CONFIG.keys())}")
    
    df=load_data(ruta_csv) #cargamos los datos
    leakage_feature=TARGETS_CONFIG[tipo_incidentes]["leakage_feature"] #obtenemos la columna que causa data leakage
    label_col=TARGETS_CONFIG[tipo_incidentes]["label"] #obtenemos la columna objetivo

    if label_col not in df.columns: #miramos si la columna objetivo esta en el dataframe
        raise ValueError(f"La columna objetivo '{label_col}' no se encuentra en los datos.")
    
    columnas_entrada = [col for col in COLUMNAS_BASE if col in df.columns]

    if len(columnas_entrada) ==0: #miramos si las columnas base estan en el dataframe
        raise ValueError("Ninguna de las columnas base se encuentra en los datos.")
    if leakage_feature in columnas_entrada: #si la columna que causa data leakage esta en las columnas base, la eliminamos
        columnas_entrada.remove(leakage_feature)

    y=df[label_col] #etiqueta objetivo
    x=df[columnas_entrada].copy() #caracteristicas de entrada 

    columnas_numericas=[]
    columnas_categoricas=[]
    for col in columnas_entrada: #miramos las columnas que son numericas
        if pandas.api.types.is_numeric_dtype(x[col]):
            columnas_numericas.append(col)
        else:
            columnas_categoricas.append(col)

    for col in columnas_numericas: #rellenamos los nans de las numericas con la media
        if x[col].isna().all():
            x[col] = x[col].fillna(0) #si toda la columna es nula, rellenamos con 0
        else:
            x[col] = x[col].fillna(x[col].mean()) #si no, rellenamos con la media

    for col in columnas_categoricas: #rellenamos los nans de las categóricas con la moda o "desconocido" si toda la columna es nula
        if x[col].isna().all():
            x[col] = x[col].fillna("desconocido")
        else:
            mode_value = x[col].mode(dropna=True)
            if not mode_value.empty:
                x[col] = x[col].fillna(mode_value.iloc[0])
            else:
                x[col] = x[col].fillna("desconocido")
    X = pandas.get_dummies(x, columns=columnas_categoricas, drop_first=False) #convertir categóricas en números con get_dummies

    return X, y        

def entrenamiento_arbol_de_decision(ruta_csv, tipo_incidente, ruta_modelo_salida): 
    #ruta_csv: ruta al csv con los datos
    #tipo_incidente: objetivo a predecir
    #ruta_modelo_salida: ruta donde guardar el modelo entrenado (lo pondremos en la ui)
    #TIENE QUE DEVOLVER LAS METRICAS CON LA INFO DEL MODELO

    X, y=load_and_prepare_data(ruta_csv, tipo_incidente) #cargamos y preparamos los datos
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42) #dividimos los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba), random_state para reproducibilidad x e y son las caracteristicas y etiquetas
    #creamos el modelo
    modelo_arbol=DecisionTreeClassifier(random_state=42, criterion="gini", max_depth=10, min_samples_split=2)#creamos el modelo de árbol de decisión
    #CARCTERÍSTICAS DEL MODELO:
    #random_state=42: para reproducibilidad
    #criterion="gini": criterio para medir la calidad de una división (índice de
    #max_depth=10: profundidad máxima del árbol
    #min_samples_split=2: número mínimo de muestras necesarias para dividir un nodo interno

    modelo_arbol.fit(X_train, y_train) #entrenamos el modelo con los datos de entrenamiento
    y_pred=modelo_arbol.predict(X_test) #hacemos predicciones con los datos de prueba

    accuracy=accuracy_score(y_test, y_pred) #calculamos la precisión del modelo
    class_report=classification_report(y_test, y_pred, output_dict=True) #informe de clasificación como diccionario 
    conf_matrix=confusion_matrix(y_test, y_pred) #matriz de confusión

    # guardamos también las columnas de entrada usadas por el modelo
    columnas_modelo = list(X.columns)

    # preparamos un "paquete" con el modelo y metadatos necesarios para predecir después
    modelo_guardado = {
        "modelo": modelo_arbol,
        "tipo_incidente": tipo_incidente,
        "columnas_entrada": columnas_modelo,
    }

    dump(modelo_guardado, ruta_modelo_salida)  # guardamos el paquete completo

    metrics = {  # devolvemos las métricas del modelo 
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "target": tipo_incidente,
        "n_samples": len(X),
        "n_features": X.shape[1],
    }
    return metrics

def preparar_datos_para_prediccion(ruta_csv, tipo_incidente, columnas_entrada_modelo):
    #se prepara los datos de un csv nuevo para hacer predicciones con un modelo que ya este entrenado
    #solo usaremos las columnas base

    if tipo_incidente not in TARGETS_CONFIG: #miramos si el target es valido
        raise ValueError(f"Target '{tipo_incidente}' no soportado. Targets soportados: {list(TARGETS_CONFIG.keys())}")
    
    df=load_data(ruta_csv) #cargamos los datos
    leakage_feature=TARGETS_CONFIG[tipo_incidente]["leakage_feature"] #misma lógica que en la preparación de datos para entrenamiento
    columnas_entrada = [col for col in COLUMNAS_BASE if col in df.columns]

    if leakage_feature in columnas_entrada: #si la columna que causa data leakage esta en las columnas base, la eliminamos
        columnas_entrada.remove(leakage_feature)

    x=df[columnas_entrada].copy() #caracteristicas de entrada
    columnas_numericas=[]
    columnas_categoricas=[]

    for col in columnas_entrada: #miramos las columnas que son numericas
        if pandas.api.types.is_numeric_dtype(x[col]):
            columnas_numericas.append(col)
        else:
            columnas_categoricas.append(col)

    # rellenamos nulos en numéricas
    for col in columnas_numericas:
        if x[col].isna().all():
            x[col] = x[col].fillna(0)
        else:
            x[col] = x[col].fillna(x[col].mean())

    # rellenamos nulos en categóricas
    for col in columnas_categoricas:
        if x[col].isna().all():
            x[col] = x[col].fillna("desconocido")
        else:
            mode_value = x[col].mode(dropna=True)
            if not mode_value.empty:
                x[col] = x[col].fillna(mode_value.iloc[0])
            else:
                x[col] = x[col].fillna("desconocido")

    # get_dummies igual que en entrenamiento
    X_nuevo = pandas.get_dummies(x, columns=columnas_categoricas, drop_first=False)

    # alineamos las columnas con las que usaba el modelo entrenado
    X_nuevo = X_nuevo.reindex(columns=columnas_entrada_modelo, fill_value=0)

    # columnas identificadoras para devolver en el resultado (si existen)
    columnas_id = ["dia", "mes", "codigo_de_distrito", "nombre_de_distrito"]
    columnas_id_presentes = [c for c in columnas_id if c in df.columns]

    if len(columnas_id_presentes) > 0:
        df_identificadores = df[columnas_id_presentes].copy()
    else:
        df_identificadores = pandas.DataFrame(index=df.index)

    return X_nuevo, df_identificadores
    

def predecir_incidentes(ruta_modelo, ruta_csv_nuevos_datos):
    #ruta_modelo: ruta al modelo entrenado
    #ruta_csv_nuevos_datos: ruta al csv con los nuevos datos para pre
    #TIENE QUE DEVOLVER LAS PREDICCIONES EN FORMATO BONICO
    modelo_guardado = load(ruta_modelo)

    if not isinstance(modelo_guardado, dict) or "modelo" not in modelo_guardado:
        raise ValueError("El archivo de modelo no tiene el formato esperado.")

    modelo_arbol = modelo_guardado["modelo"]
    tipo_incidente = modelo_guardado["tipo_incidente"]
    columnas_entrada_modelo = modelo_guardado["columnas_entrada"]

    X_nuevo, df_identificadores = preparar_datos_para_prediccion(
        ruta_csv_nuevos_datos,
        tipo_incidente,
        columnas_entrada_modelo
    )

    predicciones = modelo_arbol.predict(X_nuevo)

    #intentamos sacar las probabilidades (si el modelo nos deja)
    probabilidades_alto = None
    if hasattr(modelo_arbol, "predict_proba"):
        try:
            proba = modelo_arbol.predict_proba(X_nuevo)
            # si es binaria, cogemos la probabilidad de la clase "1"
            if proba.shape[1] == 2:
                probabilidades_alto = proba[:, 1]
            else:
                # multi-clase: usamos la probabilidad máxima
                probabilidades_alto = proba.max(axis=1)
        except Exception:
            probabilidades_alto = None

    #hacemos el DataFrame de salida
    resultado = df_identificadores.copy()
    resultado["prediccion"] = predicciones

    if probabilidades_alto is not None:
        resultado["probabilidad_alto"] = probabilidades_alto

    return resultado


    

