import pandas as pd
import matplotlib.pyplot as plt

ruta_archivo = 'D:\Repositorios\Proyecto Computacion 1\Proyecto-de-Computacion-I\CalidadAire_Scripts\DatosHistoricosCSV\datos2024.csv'
df = pd.read_csv(ruta_archivo, sep=';')

print("Primeras filas del DataFrame:")
print(df.head(10))

print("\nMagnitudes disponibles (códigos):")
print(df['MAGNITUD'].unique())

# Diccionario de códigos de magnitud
magnitudes = {
    1: 'SO2',
    6: 'NO',
    7: 'NO2',
    8: 'NOx',
    9: 'CO',
    10: 'O3',
    12: 'PM10',
    14: 'PM2.5',
    20: 'Tolueno',
    30: 'Benceno',
    35: 'Etilbenceno'
}

# Códigos de magnitud
MAGNITUD_NO2 = 7 #NO2
MAGNITUD_PM10 = 12 #PM10

# Columnas de días
columnas_diarias = [f'D{str(i).zfill(2)}' for i in range(1, 32)]

# Filtrar NO2 y PM10
df_no2 = df[df['MAGNITUD'] == MAGNITUD_NO2].copy()
df_pm10 = df[df['MAGNITUD'] == MAGNITUD_PM10].copy()

# Calcular medias mensuales
df_no2['media_mensual'] = df_no2[columnas_diarias].mean(axis=1)
df_pm10['media_mensual'] = df_pm10[columnas_diarias].mean(axis=1)

# Mostrar tablas resumen
print("\nMedia mensual de NO2 por estación:")
print(df_no2[['ESTACION', 'ANO', 'MES', 'media_mensual']])

print("\nMedia mensual de PM10 por estación:")
print(df_pm10[['ESTACION', 'ANO', 'MES', 'media_mensual']])


