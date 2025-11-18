import pandas as pd

# Leer archivos CSV
ubicaciones = pd.read_csv("UbicacionEstacionesPermanentesSentidos.csv", sep=";")
datos = pd.read_csv("DATOS_ESTACIONES_MARZO_2025.csv", sep=";")

# Limpiar espacios y poner en minúsculas los nombres de columna
ubicaciones.columns = ubicaciones.columns.str.strip()
datos.columns = datos.columns.str.strip()

# Identificar la columna de estación automáticamente (con o sin tilde, mayúsculas o espacios)
def buscar_columna_estacion(cols):
    for col in cols:
        nombre = col.strip().lower().replace("ó", "o")
        if nombre == "estacion":
            return col
    return None

col_estacion = buscar_columna_estacion(ubicaciones.columns)
if col_estacion is None:
    print("Columnas encontradas en CSV de ubicaciones:", ubicaciones.columns.tolist())
    raise ValueError("No se encontró la columna de estación. Revisa la lista mostrada arriba y usa el nombre correcto.")

# Usar la columna identificada
ubicaciones['district_code'] = ubicaciones[col_estacion]
ubicaciones = ubicaciones[['district_code', 'Nombre']].drop_duplicates().rename(columns={'Nombre': 'district_name'})



# Quitar filas con FEST vacío o nulo
datos = datos[datos['FEST'].notnull() & (datos['FEST'] != '')]


# Crear district_code numérico desde FEST ('ES01' -> 1)
datos['district_code'] = datos['FEST'].str.replace('ES', '', regex=False).astype(int)
datos = datos.merge(ubicaciones, on='district_code', how='left')

# Extraer día, mes, año
datos[['Dia', 'Mes', 'Año']] = datos['FDIA'].astype(str).str.split('/', expand=True)

# Función para identificar sentido general
def sentido_general(fsen):
    fs = str(fsen).strip()
    if fs.startswith('1'):
        return 1
    elif fs.startswith('2'):
        return 2
    else:
        return None

datos['sentido'] = datos['FSEN'].apply(sentido_general)

# Asegúrate de considerar solo columnas de horas que existen en el CSV
horas = [f'HOR{i}' for i in range(1, 13) if f'HOR{i}' in datos.columns]
datos['total_12h'] = datos[horas].sum(axis=1)

# Agrupar y calcular suma total por sentido/estacion/día
por_sentido = datos.groupby(
    ['FDIA', 'district_code', 'district_name', 'sentido', 'Dia', 'Mes', 'Año']
)['total_12h'].sum().reset_index(name='total_24h')

# Función: calcular solo sobre sentidos existentes (>0)
def media_solo_existentes(df):
    sentidos_validos = df['total_24h'][df['total_24h'] > 0]
    if len(sentidos_validos) == 0:
        return 0  # O np.nan si lo prefieres
    return sentidos_validos.mean() / 24

# Calcular intensidad_media_diaria
resultado = por_sentido.groupby(
    ['FDIA', 'district_code', 'district_name', 'Dia', 'Mes', 'Año']
).apply(media_solo_existentes, include_groups=False).reset_index(name='intensidad_media_diaria')

resultado_final = resultado[['Dia', 'Mes', 'Año', 'district_code', 'district_name', 'intensidad_media_diaria']]
resultado_final.to_csv("resultado.csv", index=False)