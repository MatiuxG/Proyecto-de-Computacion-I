"""Función de predicción puntual usada por la GUI (VisualApp).

Recibe el nombre del modelo guardado (.pkl) y un diccionario con los
datos de entrada, y devuelve la probabilidad de que sea incidencia alta.
"""

from pathlib import Path

import joblib
import pandas as pd


def realizar_prediccion(modelo_completo_nombre, datos_entrada, salida=None):
    """Carga el modelo y calcula la probabilidad para una fila de datos.

    Args:
        modelo_completo_nombre: ruta absoluta al .pkl o nombre dentro de
            modelos_guardados/.
        datos_entrada: dict con los valores de las features.
        salida: lista opcional donde se acumula el resultado (compat
            con la versión antigua de la GUI). Si es None, sólo devuelve
            el dict.

    Returns:
        dict con la clave "probabilidad" o "error".
    """
    resultado = {}

    # La versión antigua llamaba pasando solo el nombre del archivo y
    # buscaba en una carpeta concreta. Mantenemos compatibilidad probando
    # primero la ruta tal cual (absoluta) y, si no existe, en la carpeta
    # modelos_guardados/.
    ruta_directa = Path(modelo_completo_nombre)
    ruta_relativa = Path("modelos_guardados") / modelo_completo_nombre

    if ruta_directa.exists():
        ruta = ruta_directa
    elif ruta_relativa.exists():
        ruta = ruta_relativa
    else:
        ruta = None

    if ruta is None:
        resultado = {"error": "Modelo no encontrado"}
    else:
        try:
            modelo = joblib.load(ruta)
            df_input = pd.DataFrame([datos_entrada])

            # Si el modelo conoce los nombres de sus features, los usamos.
            # Si no, caemos a un mínimo razonable.
            if hasattr(modelo, "feature_names_in_"):
                columnas = list(modelo.feature_names_in_)
            else:
                columnas = ["dia", "mes", "año", "codigo_de_distrito"]

            # Rellenamos con ceros las columnas que el modelo espera
            # y no vienen en los datos de entrada.
            for col in columnas:
                if col not in df_input.columns:
                    df_input[col] = 0

            probabilidad = modelo.predict_proba(df_input[columnas])[0][1]
            resultado = {"probabilidad": float(probabilidad)}
        except Exception as error:
            resultado = {"error": str(error)}

    if salida is not None:
        salida.append(resultado)

    return resultado
