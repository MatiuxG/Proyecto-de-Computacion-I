import requests
import pandas as pd
from datetime import datetime, timezone
import sys
from math import radians, sin, cos, sqrt, atan2
from io import StringIO
import os

# --- CONFIGURACIÓN ---

# URL del CSV oficial de parkings públicos municipales de Madrid
# Esta es más confiable que las APIs REST que tienen problemas
API_URL = "https://datos.madrid.es/egob/catalogo/202625-0-aparcamientos-publicos.csv"

# Directorio de salida
DIRECTORIO_SALIDA = os.path.join('VehiculosPorDistrito_Scripts', 'Resultados')

# Nombre del archivo de salida
NOMBRE_ARCHIVO = "parking_centro_madrid_limpio.csv"

# Ruta completa del archivo
ARCHIVO_SALIDA_CSV = os.path.join(DIRECTORIO_SALIDA, NOMBRE_ARCHIVO)

# Coordenadas de la Puerta del Sol (centro de Madrid - Km 0 de España)
CENTRO_LAT = 40.4168
CENTRO_LON = -3.7038

# Radio de búsqueda en kilómetros
RADIO_KM = 1.5  # 1.5 km alrededor de Puerta del Sol


def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos geográficos usando la fórmula de Haversine.
    """
    R = 6371.0  # Radio de la Tierra en km
    
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


def ejecutar_etl_parking():
    """
    ETL completo para parkings del centro de Madrid usando CSV estático.
    """
    print("=" * 70)
    print("ETL - PARKINGS DEL CENTRO DE MADRID")
    print("=" * 70)
    print(f"📍 Centro de búsqueda: Puerta del Sol ({CENTRO_LAT}, {CENTRO_LON})")
    print(f"📏 Radio de búsqueda: {RADIO_KM} km")
    print("=" * 70)
    
    # --------------------------------------------------------------------
    # 0. LIMPIAR ARCHIVOS ANTERIORES
    # --------------------------------------------------------------------
    print(f"\n[0/3] 🧹 LIMPIANDO archivos anteriores...")
    
    # Crear directorio si no existe
    if not os.path.exists(DIRECTORIO_SALIDA):
        os.makedirs(DIRECTORIO_SALIDA)
        print(f"📁 Directorio creado: {DIRECTORIO_SALIDA}")
    
    # Eliminar CSV anterior si existe
    if os.path.exists(ARCHIVO_SALIDA_CSV):
        os.remove(ARCHIVO_SALIDA_CSV)
        print(f"🗑️  Archivo anterior eliminado: {NOMBRE_ARCHIVO}")
    
    # Eliminar también el archivo completo si existe
    archivo_completo = os.path.join(DIRECTORIO_SALIDA, "parking_madrid_todos.csv")
    if os.path.exists(archivo_completo):
        os.remove(archivo_completo)
        print(f"🗑️  Archivo anterior eliminado: parking_madrid_todos.csv")
    
    if not os.path.exists(ARCHIVO_SALIDA_CSV) and not os.path.exists(archivo_completo):
        print("✅ No hay archivos anteriores que eliminar")
    
    # --------------------------------------------------------------------
    # 1. EXTRAER (EXTRACT)
    # --------------------------------------------------------------------
    print(f"\n[1/3] 📥 EXTRAYENDO datos desde CSV...")
    print(f"URL: {API_URL}")
    
    try:
        # Añadir headers para simular un navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv,text/plain,*/*',
        }
        
        response = requests.get(API_URL, timeout=30, headers=headers)
        response.raise_for_status()
        
        # El CSV puede tener encoding especial
        response.encoding = 'utf-8-sig'
        
        fecha_extraccion = datetime.now(timezone.utc)
        print("✅ Extracción completada exitosamente")
        
    except requests.exceptions.Timeout:
        print("❌ Error: Tiempo de espera agotado", file=sys.stderr)
        return
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al extraer datos: {e}", file=sys.stderr)
        return

    # --------------------------------------------------------------------
    # 2. TRANSFORMAR (TRANSFORM)
    # --------------------------------------------------------------------
    print(f"\n[2/3] 🔄 TRANSFORMANDO datos...")
    
    try:
        # Leer el CSV con pandas
        # Usamos sep=';' porque los CSVs de Madrid suelen usar punto y coma
        try:
            df = pd.read_csv(StringIO(response.text), sep=';', encoding='utf-8')
        except:
            # Si falla, intentar con coma
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
        
        print(f"📊 Total de parkings en el CSV: {len(df)}")
        
        if len(df) == 0:
            print("❌ Error: El CSV está vacío", file=sys.stderr)
            return
        
        print(f"\n🔍 Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        # Normalizar nombres de columnas (minúsculas y sin espacios)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '_')
        
        # Identificar columnas de coordenadas (pueden tener diferentes nombres)
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            if 'latitud' in col or 'latitude' in col or 'coordenada-y' in col:
                lat_col = col
            if 'longitud' in col or 'longitude' in col or 'coordenada-x' in col:
                lon_col = col
        
        if not lat_col or not lon_col:
            print("❌ Error: No se encontraron columnas de coordenadas", file=sys.stderr)
            print("Columnas disponibles:", df.columns.tolist())
            return
        
        print(f"\n✅ Coordenadas identificadas:")
        print(f"   Latitud: {lat_col}")
        print(f"   Longitud: {lon_col}")
        
        # Renombrar para consistencia
        df.rename(columns={lat_col: 'latitud', lon_col: 'longitud'}, inplace=True)
        
        # Añadir marca de tiempo
        df['fecha_extraccion'] = fecha_extraccion
        
        # Convertir coordenadas a numérico
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
        
        # Limpiar registros sin coordenadas válidas
        registros_antes = len(df)
        df = df.dropna(subset=['latitud', 'longitud'])
        # Eliminar coordenadas en 0
        df = df[(df['latitud'] != 0) & (df['longitud'] != 0)]
        registros_despues = len(df)
        
        print(f"🧹 Limpieza: {registros_antes} → {registros_despues} registros válidos")
        
        if len(df) == 0:
            print("❌ Error: No hay registros con coordenadas válidas", file=sys.stderr)
            return

        # Calcular distancias
        print(f"📐 Calculando distancias al centro...")
        df['distancia_centro_km'] = df.apply(
            lambda row: calcular_distancia_haversine(
                CENTRO_LAT, CENTRO_LON,
                row['latitud'], row['longitud']
            ),
            axis=1
        )

        # Estadísticas de distancia
        print(f"\n📊 DISTRIBUCIÓN DE PARKINGS:")
        print(f"   • Dentro de 0.5 km: {len(df[df['distancia_centro_km'] <= 0.5])}")
        print(f"   • Dentro de 1.0 km: {len(df[df['distancia_centro_km'] <= 1.0])}")
        print(f"   • Dentro de 1.5 km: {len(df[df['distancia_centro_km'] <= 1.5])}")
        print(f"   • Dentro de 2.0 km: {len(df[df['distancia_centro_km'] <= 2.0])}")
        print(f"   • Distancia mínima: {df['distancia_centro_km'].min():.2f} km")
        print(f"   • Distancia máxima: {df['distancia_centro_km'].max():.2f} km")

        # Filtrar por radio
        df_filtrado = df[df['distancia_centro_km'] <= RADIO_KM].copy()
        df_filtrado = df_filtrado.sort_values('distancia_centro_km')
        df_filtrado = df_filtrado.reset_index(drop=True)

        print(f"\n✅ Parkings encontrados en radio de {RADIO_KM} km: {len(df_filtrado)}")
        
        if len(df_filtrado) == 0:
            print(f"\n⚠️  No hay parkings dentro de {RADIO_KM} km")
            print(f"💡 Sugerencia: Aumenta RADIO_KM a 2.0 o 3.0 km")
            
            # Mostrar los 5 más cercanos aunque estén fuera del radio
            print(f"\n📍 Los 5 parkings MÁS CERCANOS (fuera del radio):")
            df_cercanos = df.nsmallest(5, 'distancia_centro_km')
            
            # Identificar columna de nombre
            nombre_col = None
            for col in df_cercanos.columns:
                if 'nombre' in col or 'denominacion' in col or 'title' in col:
                    nombre_col = col
                    break
            
            if nombre_col:
                for idx, row in df_cercanos.iterrows():
                    print(f"   • {row[nombre_col][:50]:50s} - {row['distancia_centro_km']:.2f} km")
            
            # Guardar dataset completo para análisis
            archivo_completo = os.path.join(DIRECTORIO_SALIDA, "parking_madrid_todos.csv")
            
            # El directorio ya existe del paso 0
            df.to_csv(archivo_completo, index=False, encoding='utf-8-sig')
            print(f"\n📁 Dataset completo guardado en: {archivo_completo}")
            return

    except Exception as e:
        print(f"❌ Error durante la transformación: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------------------
    # 3. CARGAR (LOAD)
    # --------------------------------------------------------------------
    print(f"\n[3/3] 💾 GUARDANDO resultados...")
    
    try:
        # El directorio ya fue creado y limpiado en el paso 0
        # Guardar el archivo CSV
        df_filtrado.to_csv(ARCHIVO_SALIDA_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ Archivo guardado en: {ARCHIVO_SALIDA_CSV}")
        
        print("\n" + "=" * 70)
        print("🎯 PARKINGS MÁS CERCANOS A PUERTA DEL SOL")
        print("=" * 70)
        
        # Identificar columnas relevantes para mostrar
        nombre_col = None
        direccion_col = None
        
        for col in df_filtrado.columns:
            if 'nombre' in col or 'denominacion' in col:
                nombre_col = col
            if 'direccion' in col or 'via' in col or 'calle' in col:
                direccion_col = col
        
        columnas_mostrar = ['distancia_centro_km']
        if nombre_col:
            columnas_mostrar.insert(0, nombre_col)
        if direccion_col:
            columnas_mostrar.append(direccion_col)
        
        # Añadir coordenadas
        columnas_mostrar.extend(['latitud', 'longitud'])
        
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.width', None)
        
        # Mostrar top 15
        print(df_filtrado[columnas_mostrar].head(15).to_string(index=True))
        
        print("\n" + "=" * 70)
        print("📈 ESTADÍSTICAS")
        print("=" * 70)
        print(f"Total parkings en el radio: {len(df_filtrado)}")
        print(f"Parking más cercano: {df_filtrado['distancia_centro_km'].iloc[0]:.3f} km")
        if nombre_col:
            print(f"   └─ {df_filtrado[nombre_col].iloc[0]}")
        print(f"Distancia promedio: {df_filtrado['distancia_centro_km'].mean():.3f} km")
        print("=" * 70)
        
    except IOError as e:
        print(f"❌ Error al guardar archivo: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error inesperado: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Dependencias: pip install requests pandas
    ejecutar_etl_parking()
    print("\n Proceso completado\n")