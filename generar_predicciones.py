"""
Genera predicciones para los proximos N dias y las inserta en la tabla
`predictions` de la base de datos de PC2 (EcoTraffic).

Por cada combinacion (distrito, dia_futuro) calcula la probabilidad de
incidencia alta usando el modelo entrenado y la guarda en la BD para que
el frontend pueda dibujarlo en el mapa.

Uso:
    python generar_predicciones.py                      #7 dias, Random Forest, Accidentes
    python generar_predicciones.py --dias 14
    python generar_predicciones.py --modelo svm.pkl
    python generar_predicciones.py --target Emergencias
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import holidays
import joblib
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

try:
    import psycopg2
except ImportError:
    print("[ERROR] Falta psycopg2-binary. Instalalo con: pip install psycopg2-binary")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos_guardados"

#los 21 distritos de Madrid; generamos prediccion para todos
DISTRITOS = list(range(1, 22))

FESTIVOS_MADRID = holidays.Spain(subdiv="MD", years=range(2020, 2030))


def conexion_postgres():
    #devuelve una conexion a la BD de PC2 usando variables del .env
    res = psycopg2.connect(
        host=os.getenv("PC2_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("PC2_DB_PORT", "5432")),
        dbname=os.getenv("PC2_DB_NAME", "pc2_grupo1"),
        user=os.getenv("PC2_DB_USER", "root"),
        password=os.getenv("PC2_DB_PASSWORD", "root"),
    )
    return res


def construir_filas_futuras(dias_futuro):
    #genera el dataset de entrada (sin valores reales) con features de
    #calendario, hora a hora y distrito a distrito.
    #las variables desconocidas del futuro (trafico, calidad aire...) las
    #dejamos a 0; el modelo penalizara su ausencia, que es lo esperado.
    #por defecto: 7 dias x 24 horas x 21 distritos = 3528 filas
    filas = []
    hoy = date.today()

    for offset in range(dias_futuro):
        fecha = hoy + timedelta(days=offset)
        es_festivo = 1 if fecha in FESTIVOS_MADRID else 0
        dia_semana = fecha.weekday()
        es_finde = 1 if dia_semana >= 5 else 0

        for hora in range(24):
            es_hora_punta = 1 if (7 <= hora <= 10 or 17 <= hora <= 20) else 0
            for distrito in DISTRITOS:
                filas.append({
                    "dia": fecha.day,
                    "mes": fecha.month,
                    "año": fecha.year,
                    "hora": hora,
                    "codigo_de_distrito": distrito,
                    "dia_semana": dia_semana,
                    "es_finde": es_finde,
                    "es_festivo": es_festivo,
                    "es_hora_punta": es_hora_punta,
                    #variables desconocidas para el futuro -> 0
                    "total_de_accidentes": 0,
                    "valor_calidad_aire": 0,
                    "cantidad_emergencias": 0,
                    "trafico_medio": 0,
                    "obras_activas": 0,
                    "temp_media": 0,
                    "temp_max": 0,
                    "temp_min": 0,
                    "precipitacion": 0,
                    "viento_medio": 0,
                    "fecha": fecha,
                })

    res = pd.DataFrame(filas)
    return res


def calcular_probabilidades(modelo, df_input):
    #aplica el modelo a cada fila y devuelve la probabilidad de incidencia
    if hasattr(modelo, "feature_names_in_"):
        columnas = list(modelo.feature_names_in_)
    else:
        columnas = ["dia", "mes", "año", "codigo_de_distrito"]

    #rellenamos con 0 las columnas que el modelo espera y no tenemos
    for col in columnas:
        if col not in df_input.columns:
            df_input[col] = 0

    res = modelo.predict_proba(df_input[columnas])[:, 1]
    return res


def clasificar_nivel(probabilidad):
    #convierte una probabilidad en BAJO / MEDIO / ALTO (lo que espera PC2)
    nivel = "BAJO"
    if probabilidad > 0.6:
        nivel = "ALTO"
    elif probabilidad > 0.3:
        nivel = "MEDIO"
    return nivel


def insertar_en_postgres(conn, df_resultado, model_type, target_type):
    #inserta cada prediccion en la tabla `predictions`. limpia primero
    #las predicciones previas del mismo (model_type, target_type) para
    #no acumular duplicados
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM predictions WHERE model_type = %s AND target_type = %s",
            (model_type, target_type),
        )

        sql = (
            "INSERT INTO predictions "
            "(district, for_date, probability, level, predicted_at, model_type, target_type) "
            "VALUES (%s, %s, %s, %s, NOW(), %s, %s)"
        )

        for _, fila in df_resultado.iterrows():
            cur.execute(sql, (
                int(fila["codigo_de_distrito"]),
                fila["fecha"],
                round(float(fila["probabilidad"]), 4),
                fila["nivel"],
                model_type,
                target_type,
            ))

    conn.commit()


def main():
    #parsea argumentos, carga el modelo, genera las filas futuras,
    #calcula probabilidades, las agrega por dia (peor caso) y las inserta.
    #si algo critico falla deja codigo de error para que la shell lo vea
    parser = argparse.ArgumentParser(description="Genera predicciones y las vuelca a la BD de PC2.")
    parser.add_argument("--dias", type=int, default=7,
                        help="Cuantos dias hacia el futuro predecir (default 7).")
    parser.add_argument("--modelo", default="random_forest.pkl",
                        help="Archivo .pkl dentro de modelos_guardados/ (default random_forest.pkl).")
    parser.add_argument("--target", default="Accidentes",
                        choices=["Accidentes", "Calidad Aire", "Emergencias"],
                        help="Etiqueta del target (se guarda en la tabla).")

    args = parser.parse_args()

    ruta_modelo = MODELOS_DIR / args.modelo
    codigo_salida = 0

    if not ruta_modelo.exists():
        print(f"[ERROR] No se encuentra el modelo: {ruta_modelo}")
        print("        Entrena primero con: python run_all.py --solo-modelos")
        codigo_salida = 1

    if codigo_salida == 0:
        print(f"[Info] Modelo: {ruta_modelo}")
        print(f"[Info] Dias a predecir: {args.dias}")
        print(f"[Info] Target: {args.target}")

        #conjunto de entrada (24h x N dias x 21 distritos)
        df_input = construir_filas_futuras(args.dias)
        modelo = joblib.load(ruta_modelo)

        probabilidades = calcular_probabilidades(modelo, df_input.drop(columns=["fecha"]).copy())
        df_input["probabilidad"] = probabilidades

        #PC2 espera una probabilidad por (distrito, dia) en la tabla `predictions`.
        #tomamos el MAX de las 24 horas como "peor caso del dia": asi el mapa
        #pinta el distrito con su pico de riesgo de la jornada
        df_resultado = (
            df_input
            .groupby(["codigo_de_distrito", "fecha"], as_index=False)["probabilidad"]
            .max()
        )
        df_resultado["nivel"] = [clasificar_nivel(p) for p in df_resultado["probabilidad"]]

        #reusamos df_input como contenedor del resultado final
        df_input = df_resultado

        print("[Info] Conectando a PostgreSQL...")
        conn = None
        try:
            conn = conexion_postgres()
        except psycopg2.OperationalError as error:
            print(f"[ERROR] No se pudo conectar a la BD: {error}")
            codigo_salida = 2

        if conn is not None:
            model_type = ruta_modelo.stem
            insertar_en_postgres(conn, df_input, model_type, args.target)
            conn.close()

            total = len(df_input)
            altos = (df_input["nivel"] == "ALTO").sum()
            medios = (df_input["nivel"] == "MEDIO").sum()
            bajos = (df_input["nivel"] == "BAJO").sum()

            print(f"\n[OK] {total} predicciones insertadas")
            print(f"     ALTO:  {altos}")
            print(f"     MEDIO: {medios}")
            print(f"     BAJO:  {bajos}")

    if codigo_salida != 0:
        sys.exit(codigo_salida)


if __name__ == "__main__":
    main()
