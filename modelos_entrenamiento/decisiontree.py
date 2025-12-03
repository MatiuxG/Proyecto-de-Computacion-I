#scikit learn - NLTK (PLN)- XGBoost

#Decision Tree Classifier: un árbol de decisión es un modelo de aprendizaje supervisado utilizado tanto para clasificación como para regresión. Funciona dividiendo los datos en subconjuntos basados en características específicas, creando una estructura similar a un árbol donde cada nodo representa una característica, cada rama representa una decisión y cada hoja representa un resultado o clase.

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
#cosas de la UI desde donde traeremos los datos, no usaremos input sino que lo haremos desde la interfaz
DATA_PATH = '' #esto habrá que cambiarlo por la ruta del archivo en la UI

#cargar datos
data = pandas.read_csv(DATA_PATH) #cargar el conjunto de datos desde un archivo CSV

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
