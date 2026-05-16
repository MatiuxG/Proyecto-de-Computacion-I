"""Entrenamiento de un Árbol de Decisión.

Mismo refactor que random_forest.py: métricas correctas y split temporal.
"""

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .utils import (
    MAPA_TARGETS,
    cargar_dataset,
    evaluar_modelo,
    resumen_entrenamiento,
    split_temporal,
)


PARAMETROS_GRID = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}


def entrenamiento_arbol_de_decision(target_usuario, features_seleccionadas, ruta_guardado):
    """Entrena un árbol de decisión y devuelve el resumen con métricas."""

    df = cargar_dataset()
    columna_target = MAPA_TARGETS.get(target_usuario, "target_accidentes")

    X_train, X_test, y_train, y_test = split_temporal(
        df, columna_target, features_seleccionadas
    )

    cv = TimeSeriesSplit(n_splits=3)

    busqueda = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=PARAMETROS_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)

    modelo = busqueda.best_estimator_
    joblib.dump(modelo, ruta_guardado, compress=3)

    metricas = evaluar_modelo(modelo, X_test, y_test)
    resumen = resumen_entrenamiento(metricas, len(df), ruta_guardado)
    resumen["mejores_parametros"] = busqueda.best_params_

    return resumen
