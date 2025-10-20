import pandas as pd

ruta_archivo = r'D:\Repositorios\Proyecto Computacion 1\Proyecto-de-Computacion-I\Trafico_Scripts\DatosHistoricosCSV\01-2024\01-2024.csv'
df = pd.read_csv(ruta_archivo, sep=';')

print("Primeras filas del DataFrame:")
print(df.head(10))