"""
Funciones comunes para los tres modelos (Random Forest, Decision Tree, SVM).

Centraliza:
  - Carga del dataset (robusta a la ubicacion de ejecucion).
  - Split TEMPORAL (no aleatorio): el test se hace siempre con las fechas
    mas recientes para simular un escenario realista de prediccion.
  - Metricas de clasificacion bien elegidas: accuracy, precision, recall,
    F1, ROC-AUC y matriz de confusion. La version anterior usaba MAE/MSE,
    que son metricas de regresion y no tienen sentido para un clasificador
    binario.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


#carpeta raiz del proyecto (dos niveles por encima de este archivo)
ROOT = Path(__file__).resolve().parent.parent

#localizaciones donde puede estar el dataset, en orden de preferencia.
#dejamos dos por si el script se ejecuta desde la raiz o desde otra carpeta
RUTAS_DATASET = [
    ROOT / "Resultados" / "dataset_unificado.csv",
    Path.cwd() / "Resultados" / "dataset_unificado.csv",
]

#"etiqueta visible" -> "columna target" en el dataset
MAPA_TARGETS = {
    "Accidentes": "target_accidentes",
    "Calidad Aire": "target_aire",
    "Emergencias": "target_emergencias",
}


def cargar_dataset():
    #busca el dataset_unificado.csv en sitios habituales y lo carga.
    #lanza FileNotFoundError si no lo encuentra (mejor un error claro
    #que un crash con stack trace ininteligible)
    ruta_elegida = None

    for ruta in RUTAS_DATASET:
        if ruta_elegida is None and ruta.exists():
            ruta_elegida = ruta

    if ruta_elegida is None:
        raise FileNotFoundError(
            "No se encuentra dataset_unificado.csv. "
            "Ejecuta primero `python main/main.py` para generarlo."
        )

    res = pd.read_csv(ruta_elegida, sep=";")
    return res


def split_temporal(df, columna_target, features, proporcion_test=0.2):
    #divide el dataset por orden temporal: el test son las fechas mas recientes.
    #es el split correcto para un problema de prediccion de incidencias
    #futuras. la version anterior usaba train_test_split aleatorio, lo que
    #mezcla pasado y futuro y da metricas engañosamente optimistas
    #ordenamos por (año, mes, dia, hora) si esas columnas existen
    columnas_orden = [c for c in ["año", "mes", "dia", "hora"] if c in df.columns]

    if columnas_orden:
        df = df.sort_values(columnas_orden).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * (1 - proporcion_test))

    X_train = df[features].iloc[:n_train]
    y_train = df[columna_target].iloc[:n_train]
    X_test = df[features].iloc[n_train:]
    y_test = df[columna_target].iloc[n_train:]

    return X_train, X_test, y_train, y_test


def evaluar_modelo(modelo, X_test, y_test):
    #calcula el conjunto de metricas correctas para un clasificador binario.
    #devuelve un dict con accuracy, precision, recall, F1, ROC-AUC y la
    #matriz de confusion (util para la GUI o el informe)
    predicciones = modelo.predict(X_test)

    #roc_auc_score necesita probabilidades, no etiquetas. algunos modelos
    #(SVM sin probability=True) no tienen predict_proba, asi que dejamos
    #None si no se puede calcular
    roc_auc = None
    if hasattr(modelo, "predict_proba"):
        try:
            probabilidades = modelo.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probabilidades)
        except Exception:
            roc_auc = None

    #zero_division=0 evita warnings si una clase no aparece en el test
    #(situacion normal con datasets pequeños)
    res = {
        "accuracy": accuracy_score(y_test, predicciones),
        "precision": precision_score(y_test, predicciones, zero_division=0),
        "recall": recall_score(y_test, predicciones, zero_division=0),
        "f1": f1_score(y_test, predicciones, zero_division=0),
        "roc_auc": roc_auc,
        "matriz_confusion": confusion_matrix(y_test, predicciones).tolist(),
    }
    return res


def features_por_defecto(df, target_seleccionado):
    #lista de features que usaremos si la GUI no nos pasa otra cosa.
    #excluye la columna que coincide con el target para evitar data leakage
    #(predecir "muchos accidentes" usando el numero de accidentes como
    #entrada no tiene sentido)
    columnas_excluidas = {
        "target_accidentes", "target_aire", "target_emergencias",
        "total_de_accidentes" if target_seleccionado == "Accidentes" else "",
        "valor_calidad_aire" if target_seleccionado == "Calidad Aire" else "",
        "cantidad_emergencias" if target_seleccionado == "Emergencias" else "",
    }
    columnas_excluidas.discard("")

    res = [c for c in df.columns if c not in columnas_excluidas]
    return res


def resumen_entrenamiento(metricas, n_muestras, ruta_guardado):
    #empaqueta el resultado del entrenamiento en un dict listo para la GUI
    res = {
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "f1": metricas["f1"],
        "roc_auc": metricas["roc_auc"],
        "matriz_confusion": metricas["matriz_confusion"],
        "n_muestras": n_muestras,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "archivo": str(ruta_guardado),
    }
    return res
