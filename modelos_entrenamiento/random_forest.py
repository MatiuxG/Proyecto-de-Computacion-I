"""Entrenamiento de Random Forest para la predicción de incidencias.

Refactor 2026-05-16:
  - Métricas correctas (accuracy/precision/recall/F1/ROC-AUC) en vez
    de MAE/MSE (que eran de regresión).
  - Split temporal (test = fechas más recientes) en vez de aleatorio.
  - Pequeña búsqueda de hiperparámetros con GridSearchCV usando
    TimeSeriesSplit para respetar el orden temporal también en la
    validación cruzada.
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


# Mantenemos la búsqueda pequeña para que entrene rápido.
# Limitamos max_depth (sin None) porque con dataset horario de 800k filas
# los árboles sin profundidad maxima salen de varios GB, no caben en git
# y consumen demasiada RAM para cargar en LORCA.
PARAMETROS_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5],
}


def entrenamiento_random_forest(target_usuario, features_seleccionadas, ruta_guardado):
    """Entrena un Random Forest y devuelve el resumen con todas las métricas."""

    df = cargar_dataset()
    columna_target = MAPA_TARGETS.get(target_usuario, "target_accidentes")

    X_train, X_test, y_train, y_test = split_temporal(
        df, columna_target, features_seleccionadas
    )

    # TimeSeriesSplit divide el train en bloques temporales para CV.
    # Así el GridSearch tampoco mezcla pasado y futuro.
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
    #importante para que entre en git (limite 100MB por archivo) y para
    #no inflar la RAM al desplegar en LORCA.
    joblib.dump(modelo, ruta_guardado, compress=3)

    metricas = evaluar_modelo(modelo, X_test, y_test)
    resumen = resumen_entrenamiento(metricas, len(df), ruta_guardado)
    resumen["mejores_parametros"] = busqueda.best_params_

    return resumen
