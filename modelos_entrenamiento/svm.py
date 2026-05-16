"""Entrenamiento de SVM lineal (LinearSVC) calibrado.

Con granularidad horaria el dataset crece a ~800k filas y SVC con kernel rbf
se vuelve inviable (escalado O(n^2) y muchos GB de RAM). LinearSVC escala
casi lineal con n, asi que sigue siendo "una SVM" pero entrena en segundos.

Como LinearSVC no expone predict_proba, lo envolvemos en
CalibratedClassifierCV para tener probabilidades comparables al resto
de modelos (y poder reusar evaluar_modelo, que mide ROC-AUC).
"""

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .utils import (
    MAPA_TARGETS,
    cargar_dataset,
    evaluar_modelo,
    resumen_entrenamiento,
    split_temporal,
)


#C controla la regularizacion; grid pequeño para que sea rapido
PARAMETROS_GRID = {
    "svm__estimator__C": [0.5, 1.0, 2.0],
}


def entrenamiento_svm(target_usuario, features_seleccionadas, ruta_guardado):
    #entrena un LinearSVC calibrado dentro de un pipeline con StandardScaler
    #(SVM siempre quiere features escaladas) y devuelve el resumen
    df = cargar_dataset()
    columna_target = MAPA_TARGETS.get(target_usuario, "target_accidentes")

    X_train, X_test, y_train, y_test = split_temporal(
        df, columna_target, features_seleccionadas
    )

    #LinearSVC base + Calibrated para obtener predict_proba; envolvemos
    #en pipeline con StandardScaler porque SVM siempre quiere features escaladas
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(random_state=42, max_iter=5000, dual="auto"),
            cv=3,
        )),
    ])

    cv = TimeSeriesSplit(n_splits=3)

    busqueda = GridSearchCV(
        estimator=pipeline,
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
