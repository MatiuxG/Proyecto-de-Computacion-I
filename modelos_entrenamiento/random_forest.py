"""Entrenamiento de Random Forest para la prediccion de incidencias.

Refactor 2026-05-16:
  - Metricas correctas (accuracy/precision/recall/F1/ROC-AUC) en vez
    de MAE/MSE (que eran de regresion).
  - Split temporal (test = fechas mas recientes) en vez de aleatorio.
  - Busqueda pequeña de hiperparametros con GridSearchCV usando
    TimeSeriesSplit para respetar el orden temporal tambien en la
    validacion cruzada.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .utils import (
    MAPA_TARGETS,
    cargar_dataset,
    evaluar_modelo,
    resumen_entrenamiento,
    split_temporal,
)


#busqueda pequeña para que entrene rapido.
#limitamos max_depth (sin None) porque con dataset horario de 800k filas
#los arboles sin profundidad maxima salen de varios GB, no caben en git
#y consumen demasiada RAM para cargar en LORCA
PARAMETROS_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5],
}


def entrenamiento_random_forest(target_usuario, features_seleccionadas, ruta_guardado):
    #entrena un Random Forest, lo guarda comprimido y devuelve el resumen
    #con todas las metricas
    df = cargar_dataset()
    columna_target = MAPA_TARGETS.get(target_usuario, "target_accidentes")

    X_train, X_test, y_train, y_test = split_temporal(
        df, columna_target, features_seleccionadas
    )

    #TimeSeriesSplit divide el train en bloques temporales para CV.
    #asi el GridSearch tampoco mezcla pasado y futuro
    cv = TimeSeriesSplit(n_splits=3)

    busqueda = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=PARAMETROS_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)

    modelo = busqueda.best_estimator_
    #compress=3 reduce el .pkl 3-5x sin penalizar la velocidad de carga;
    #importante para que entre en git (limite 100MB) y para no inflar
    #la RAM al desplegar en LORCA
    joblib.dump(modelo, ruta_guardado, compress=3)

    metricas = evaluar_modelo(modelo, X_test, y_test)
    resumen = resumen_entrenamiento(metricas, len(df), ruta_guardado)
    resumen["mejores_parametros"] = busqueda.best_params_

    return resumen
