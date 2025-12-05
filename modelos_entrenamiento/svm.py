# SVM (Support Vector Machine): es un modelo de aprendizaje supervisado que busca
# un hiperplano que separe las clases con el mayor margen posible. Suele funcionar
# muy bien con datos no lineales usando kernels (por ejemplo, RBF).

# PARA SABER EN GENERAL:
# x significa todo lo que el modelo usa para predecir (características)
# y significa la etiqueta que el modelo tiene que predecir (target)

# librerías importadas
import pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

# columnas que usaremos (BASES DEL CSV)
COLUMNAS_BASE = [
    "dia",
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

# el modelo tiene que saber qué target ha elegido el usuario:
"""
Targets soportados:
- "accidentes"  -> etiqueta accidente_riesgo
- "atascos"     -> etiqueta atasco_riesgo
- "emergencias" -> etiqueta emergencia_riesgo
- "aire"        -> etiqueta aire_riesgo
"""

TARGETS_CONFIG = {
    "accidentes": {
        "label": "accidente_riesgo",
        "leakage_feature": "total_de_accidentes",
    },
    "atascos": {
        "label": "atasco_riesgo",
        "leakage_feature": "trafico_medio",
    },
    "emergencias": {
        "label": "emergencia_riesgo",
        "leakage_feature": "cantidad_emergencias",
    },
    "aire": {
        "label": "aire_riesgo",
        "leakage_feature": "valor_calidad_aire",
    },
}

def load_data(ruta_csv):
    # la UI nos pasa la ruta del CSV. Esta función solo lo carga y devuelve el DataFrame.
    df = pandas.read_csv(ruta_csv, sep=";", engine="python")
    return df

def load_and_prepare_data(ruta_csv, tipo_incidentes):
    # carga el CSV y prepara X (características) e y (etiqueta) según el tipo de incidente

    if tipo_incidentes not in TARGETS_CONFIG:
        raise ValueError(
            f"Target '{tipo_incidentes}' no soportado. Targets soportados: {list(TARGETS_CONFIG.keys())}"
        )
    
    df = load_data(ruta_csv)

    leakage_feature = TARGETS_CONFIG[tipo_incidentes]["leakage_feature"]
    label_col = TARGETS_CONFIG[tipo_incidentes]["label"]

    if label_col not in df.columns:
        raise ValueError(f"La columna objetivo '{label_col}' no se encuentra en los datos.")
    
    columnas_entrada = [col for col in COLUMNAS_BASE if col in df.columns]

    if len(columnas_entrada) == 0:
        raise ValueError("Ninguna de las columnas base se encuentra en los datos.")
    
    if leakage_feature in columnas_entrada:
        columnas_entrada.remove(leakage_feature)

    y = df[label_col]   # etiqueta objetivo
    x = df[columnas_entrada].copy()  # características de entrada 

    columnas_numericas = []
    columnas_categoricas = []
    for col in columnas_entrada:
        if pandas.api.types.is_numeric_dtype(x[col]):
            columnas_numericas.append(col)
        else:
            columnas_categoricas.append(col)

    # rellenamos nulos en numéricas con la media
    for col in columnas_numericas:
        if x[col].isna().all():
            x[col] = x[col].fillna(0)
        else:
            x[col] = x[col].fillna(x[col].mean())

    # rellenamos nulos en categóricas con la moda o "desconocido"
    for col in columnas_categoricas:
        if x[col].isna().all():
            x[col] = x[col].fillna("desconocido")
        else:
            mode_value = x[col].mode(dropna=True)
            if not mode_value.empty:
                x[col] = x[col].fillna(mode_value.iloc[0])
            else:
                x[col] = x[col].fillna("desconocido")

    # convertir categóricas en números con get_dummies
    X = pandas.get_dummies(x, columns=columnas_categoricas, drop_first=False)

    return X, y

def entrenamiento_svm(ruta_csv, tipo_incidente, ruta_modelo_salida):
    # entrena un modelo SVM para el tipo de incidente indicado

    X, y = load_and_prepare_data(ruta_csv, tipo_incidente)

    # dividimos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # creamos el modelo SVM dentro de un pipeline con StandardScaler
    modelo_svm = make_pipeline(
        StandardScaler(),                 # escalado de las features
        SVC(
            kernel="rbf",                 # kernel radial (no lineal)
            C=1.0,                        # regularización
            gamma="scale",                # parámetro del kernel
            probability=True,             # para poder usar predict_proba
            random_state=42
        )
    )

    # entrenamos el modelo
    modelo_svm.fit(X_train, y_train)

    # hacemos predicciones sobre el conjunto de prueba
    y_pred = modelo_svm.predict(X_test)

    # calculamos métricas
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # guardamos también las columnas de entrada usadas por el modelo
    columnas_modelo = list(X.columns)

    modelo_guardado = {
        "modelo": modelo_svm,
        "tipo_incidente": tipo_incidente,
        "columnas_entrada": columnas_modelo,
    }

    dump(modelo_guardado, ruta_modelo_salida)

    metrics = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "target": tipo_incidente,
        "n_samples": len(X),
        "n_features": X.shape[1],
    }
    return metrics

def preparar_datos_para_prediccion(ruta_csv, tipo_incidente, columnas_entrada_modelo):
    # prepara los datos de un CSV nuevo para hacer predicciones con un modelo SVM entrenado

    if tipo_incidente not in TARGETS_CONFIG:
        raise ValueError(f"Target '{tipo_incidente}' no soportado. Targets soportados: {list(TARGETS_CONFIG.keys())}")
    
    df = load_data(ruta_csv)
    leakage_feature = TARGETS_CONFIG[tipo_incidente]["leakage_feature"]

    columnas_entrada = [col for col in COLUMNAS_BASE if col in df.columns]
    if len(columnas_entrada) == 0:
        raise ValueError("Ninguna de las columnas base se encuentra en los datos de predicción.")

    if leakage_feature in columnas_entrada:
        columnas_entrada.remove(leakage_feature)

    x = df[columnas_entrada].copy()
    columnas_numericas = []
    columnas_categoricas = []

    for col in columnas_entrada:
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

    # columnas identificadoras para el resultado
    columnas_id = ["dia", "mes", "codigo_de_distrito", "nombre_de_distrito"]
    columnas_id_presentes = [c for c in columnas_id if c in df.columns]

    if len(columnas_id_presentes) > 0:
        df_identificadores = df[columnas_id_presentes].copy()
    else:
        df_identificadores = pandas.DataFrame(index=df.index)

    return X_nuevo, df_identificadores

def predecir_incidentes_svm(ruta_modelo, ruta_csv_nuevos_datos):
    # carga un modelo SVM guardado y realiza predicciones sobre nuevos datos

    modelo_guardado = load(ruta_modelo)

    if not isinstance(modelo_guardado, dict) or "modelo" not in modelo_guardado:
        raise ValueError("El archivo de modelo no tiene el formato esperado.")

    modelo_svm = modelo_guardado["modelo"]
    tipo_incidente = modelo_guardado["tipo_incidente"]
    columnas_entrada_modelo = modelo_guardado["columnas_entrada"]

    X_nuevo, df_identificadores = preparar_datos_para_prediccion(
        ruta_csv_nuevos_datos,
        tipo_incidente,
        columnas_entrada_modelo
    )

    predicciones = modelo_svm.predict(X_nuevo)

    probabilidades_alto = None
    if hasattr(modelo_svm, "predict_proba"):
        try:
            proba = modelo_svm.predict_proba(X_nuevo)
            if proba.shape[1] == 2:
                probabilidades_alto = proba[:, 1]
            else:
                probabilidades_alto = proba.max(axis=1)
        except Exception:
            probabilidades_alto = None

    resultado = df_identificadores.copy()
    resultado["prediccion"] = predicciones
    if probabilidades_alto is not None:
        resultado["probabilidad_alto"] = probabilidades_alto

    return resultado
