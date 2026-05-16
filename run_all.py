"""
Pipeline completo del Sistema de Deteccion de Incidencias.

Ejecuta en orden:
  1. Los scrapers de cada fuente (genera los CSV de Resultados/).
  2. main/main.py para unir todo en dataset_unificado.csv.
  3. Entrenamiento de los tres modelos (Random Forest, Decision Tree, SVM)
     contra el target por defecto (Accidentes).

Uso tipico:
    python run_all.py                    #todo el pipeline
    python run_all.py --skip-scrapers    #solo unir y entrenar
    python run_all.py --solo-etl         #scrapers + main, sin entrenar
    python run_all.py --solo-modelos     #solo entrenamiento
    python run_all.py --target "Calidad Aire"
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos_guardados"


#lista de scrapers a ejecutar. cada uno es independiente.
#trafico historico es el que alimenta el dataset unificado (mensual);
#el realtime se mantiene por si en algun momento queremos un snapshot del dia
SCRAPERS = [
    ("Accidentes",        ROOT / "Accidentes_Scripts"  / "Accidentes.py"),
    ("Calidad del aire",  ROOT / "CalidadAire_Scripts" / "CalidadAire.py"),
    ("Clima",             ROOT / "Clima_Scripts"       / "Clima.py"),
    ("Emergencias",       ROOT / "Emergencias_Scripts" / "emergencias_scraper.py"),
    ("Obras",             ROOT / "Obras_Scripts"       / "Obras.py"),
    ("Trafico historico", ROOT / "Trafico_Scripts"     / "TraficoHistorico.py"),
    ("Trafico realtime",  ROOT / "Trafico_Scripts"     / "TraficoScrapper.py"),
]


def ejecutar_script(ruta_script, etiqueta):
    #lanza un script python y devuelve True si termino con codigo 0
    print(f"\n=== {etiqueta} ===")
    ok = False

    if not ruta_script.exists():
        print(f"  [Saltado] No existe {ruta_script}")
    else:
        resultado = subprocess.run(
            [sys.executable, str(ruta_script)],
            cwd=ruta_script.parent,
        )

        if resultado.returncode == 0:
            ok = True
        else:
            print(f"  [Error] {ruta_script} termino con codigo {resultado.returncode}")

    return ok


def lanzar_scrapers():
    #ejecuta cada scraper en orden. si alguno falla, sigue con el resto
    print("\n[FASE 1] Scrapers")
    for etiqueta, ruta in SCRAPERS:
        ejecutar_script(ruta, etiqueta)


def lanzar_etl():
    #ejecuta main/main.py para unir las fuentes
    print("\n[FASE 2] Unificacion de datos (ETL)")
    ejecutar_script(ROOT / "main" / "main.py", "main.py")


def lanzar_entrenamientos(target):
    #entrena los tres modelos contra el target indicado y muestra metricas
    print(f"\n[FASE 3] Entrenamiento de modelos (target: {target})")

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    #import diferido para no exigir sklearn en la fase de scraping
    sys.path.insert(0, str(ROOT))
    from modelos_entrenamiento.random_forest import entrenamiento_random_forest
    from modelos_entrenamiento.decisiontree import entrenamiento_arbol_de_decision
    from modelos_entrenamiento.svm import entrenamiento_svm
    from modelos_entrenamiento.utils import cargar_dataset, features_por_defecto

    df = cargar_dataset()
    features = features_por_defecto(df, target)
    #excluimos columnas target como features para no hacer trampa
    features = [c for c in features if not c.startswith("target_")]

    print(f"  Features usadas: {features}")
    print(f"  Filas totales:   {len(df)}\n")

    entrenadores = [
        ("Random Forest", entrenamiento_random_forest,     MODELOS_DIR / "random_forest.pkl"),
        ("Decision Tree", entrenamiento_arbol_de_decision, MODELOS_DIR / "decision_tree.pkl"),
        ("SVM",           entrenamiento_svm,               MODELOS_DIR / "svm.pkl"),
    ]

    for etiqueta, funcion, ruta in entrenadores:
        print(f"--- {etiqueta} ---")
        try:
            resumen = funcion(target, features, str(ruta))
            imprimir_resumen(resumen)
        except Exception as error:
            print(f"  [Error] {etiqueta}: {error}")


def imprimir_resumen(resumen):
    #muestra las metricas del modelo entrenado en formato leible
    print(f"  Accuracy:  {resumen['accuracy']:.3f}")
    print(f"  Precision: {resumen['precision']:.3f}")
    print(f"  Recall:    {resumen['recall']:.3f}")
    print(f"  F1:        {resumen['f1']:.3f}")
    if resumen["roc_auc"] is not None:
        print(f"  ROC-AUC:   {resumen['roc_auc']:.3f}")
    print(f"  Guardado:  {resumen['archivo']}\n")


def main():
    #parsea argumentos y decide que fases ejecutar
    parser = argparse.ArgumentParser(description="Pipeline completo de PC1.")
    parser.add_argument("--skip-scrapers", action="store_true",
                        help="No ejecutar scrapers (reutiliza los CSV existentes).")
    parser.add_argument("--solo-etl", action="store_true",
                        help="Solo scrapers y union de datos, sin entrenar.")
    parser.add_argument("--solo-modelos", action="store_true",
                        help="Solo entrenamiento (asume que el dataset ya existe).")
    parser.add_argument("--target", default="Accidentes",
                        choices=["Accidentes", "Calidad Aire", "Emergencias"],
                        help="Target a predecir (por defecto: Accidentes).")

    args = parser.parse_args()

    hacer_scrapers = not args.skip_scrapers and not args.solo_modelos
    hacer_etl = not args.solo_modelos
    hacer_modelos = not args.solo_etl

    if hacer_scrapers:
        lanzar_scrapers()

    if hacer_etl:
        lanzar_etl()

    if hacer_modelos:
        lanzar_entrenamientos(args.target)

    print("\n[OK] Pipeline terminado.")


if __name__ == "__main__":
    main()
