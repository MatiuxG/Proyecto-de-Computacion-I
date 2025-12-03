#scikit learn - NLTK (PLN)- XGBoost

#Decision Tree Classifier: un árbol de decisión es un modelo de aprendizaje supervisado utilizado tanto para clasificación como para regresión. Funciona dividiendo los datos en subconjuntos basados en características específicas, creando una estructura similar a un árbol donde cada nodo representa una característica, cada rama representa una decisión y cada hoja representa un resultado o clase.

#librerias importadas
import sklearn
import pandas
import numpy

#todo el tema del modelo 
from sklearn.model_selection import train_test_split #dividir el conjunto de datos en conjuntos de entrenamiento y prueba
from sklearn.tree import DecisionTreeClassifier #importar el clasificador de árbol de decisión
from sklearn.metrics import accuracy_score #importar la función para calcular la precisión del modelo

#cosas de la UI desde donde traeremos los datos
DATA_PATH = ''

#cargar datos
data = pandas.read_csv(DATA_PATH) #cargar el conjunto de datos desde un archivo CSV

