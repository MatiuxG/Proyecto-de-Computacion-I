# -*- coding: utf-8 -*-
"""
Script Unificado para Consulta de Datos Climatológicos AEMET
- Datos diarios por período
- Datos de promedios mensuales por año
"""

import requests
import json
import csv
import os
from datetime import datetime, timedelta

# ============================================
# CONFIGURACIÓN
# ============================================

# API Keys de AEMET
API_KEY_DIARIOS = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJFZGR5ZnJhdGVyMkBnbWFpbC5jb20iLCJqdGkiOiJhZTQ4Zjg0Zi1hZTMxLTQ5MzgtYTFkNy1jYzlmODhjOTI5MWQiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc2MTQxNDUxOSwidXNlcklkIjoiYWU0OGY4NGYtYWUzMS00OTM4LWExZDctY2M5Zjg4YzkyOTFkIiwicm9sZSI6IiJ9.z0VMMvrTjwl5MsQuf5YWTdaOtXP7ctRYfasHDfZSE30"
API_KEY_MENSUALES = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtYXRlb2dhbDI0MDlAZ21haWwuY29tIiwianRpIjoiOTZjZjYxMjMtN2EzMy00OTkxLWJkMGMtNjNmZDFiYmFkN2E0IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE3NTkzMTg5NTgsInVzZXJJZCI6Ijk2Y2Y2MTIzLTdhMzMtNDk5MS1iZDBjLTYzZmQxYmJhZDdhNCIsInJvbGUiOiIifQ.pnpqxv1fmE9ZeMVTb4VkvZZF8NuffxQrcSFWpYqBVKg"

# Código de estación del Retiro en Madrid
ESTACION_RETIRO = "3195"

# Directorio de salida
DIRECTORIO_RESULTADOS = "Clima_Scripts/Resultados"

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def crear_directorio_resultados():
    """Crea el directorio de resultados si no existe"""
    if not os.path.exists(DIRECTORIO_RESULTADOS):
        os.makedirs(DIRECTORIO_RESULTADOS)
        print(f"📁 Directorio '{DIRECTORIO_RESULTADOS}' creado.")

def eliminar_archivos_anteriores(fecha_fin, periodo_dias, anio):
    """Elimina archivos CSV anteriores si existen"""
    archivo_diarios = os.path.join(DIRECTORIO_RESULTADOS, 
                                   f"datos_diarios_{fecha_fin.strftime('%Y%m%d')}_p{periodo_dias}.csv")
    archivo_mensuales = os.path.join(DIRECTORIO_RESULTADOS, 
                                     f"datos_mensuales_{anio}.csv")
    
    archivos_eliminados = []
    
    if os.path.exists(archivo_diarios):
        os.remove(archivo_diarios)
        archivos_eliminados.append(archivo_diarios)
    
    if os.path.exists(archivo_mensuales):
        os.remove(archivo_mensuales)
        archivos_eliminados.append(archivo_mensuales)
    
    if archivos_eliminados:
        print(f"\n🗑️  Archivos anteriores eliminados:")
        for archivo in archivos_eliminados:
            print(f"   • {os.path.basename(archivo)}")

def validar_fecha(fecha_str):
    """
    Valida que la fecha tenga el formato correcto DD/MM/YYYY
    
    Args:
        fecha_str (str): Fecha en formato DD/MM/YYYY
    
    Returns:
        datetime or None: Objeto datetime si es válida, None si no lo es
    """
    try:
        fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
        return fecha
    except ValueError:
        return None

# ============================================
# FUNCIONES PARA DATOS DIARIOS
# ============================================

def obtener_datos_diarios(fecha_fin, periodo_dias):
    """
    Obtiene datos climatológicos diarios del Retiro (Madrid)
    
    Args:
        fecha_fin (datetime): Fecha final del período
        periodo_dias (int): Número de días hacia atrás desde fecha_fin
    
    Returns:
        list: Lista con los datos climatológicos ordenados
    """
    
    fecha_inicio = fecha_fin - timedelta(days=periodo_dias)
    
    fecha_inicio_fmt = fecha_inicio.strftime("%Y-%m-%dT00:00:00UTC")
    fecha_fin_fmt = fecha_fin.strftime("%Y-%m-%dT23:59:59UTC")
    
    url = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_inicio_fmt}/fechafin/{fecha_fin_fmt}/estacion/{ESTACION_RETIRO}"
    
    headers = {"api_key": API_KEY_DIARIOS}
    
    try:
        print(f"\n{'='*70}")
        print(f"📊 SOLICITANDO DATOS DIARIOS - MADRID RETIRO")
        print(f"{'='*70}")
        print(f"📅 Período: {fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')}")
        print(f"📆 Total de días: {periodo_dias + 1}")
        print(f"{'='*70}\n")
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        resultado = response.json()
        
        if resultado.get("estado") != 200:
            print(f"❌ Error en la respuesta de la API: {resultado.get('descripcion', 'Sin descripción')}")
            return None
        
        url_datos = resultado.get("datos")
        
        if not url_datos:
            print("❌ No se encontró URL de datos en la respuesta")
            return None
        
        print(f"⏳ Descargando datos diarios...\n")
        
        response_datos = requests.get(url_datos)
        response_datos.raise_for_status()
        
        datos = response_datos.json()
        datos_ordenados = sorted(datos, key=lambda x: x.get('fecha', ''), reverse=True)
        
        print(f"✅ Datos diarios obtenidos: {len(datos_ordenados)} registros\n")
        return datos_ordenados
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la petición de datos diarios: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error al decodificar JSON: {e}")
        return None

def mostrar_datos_diarios(datos):
    """Muestra los datos diarios en consola"""
    if not datos:
        print("❌ No hay datos diarios para mostrar")
        return
    
    print(f"\n{'='*70}")
    print(f"🌤️  DATOS CLIMATOLÓGICOS DIARIOS - MADRID RETIRO")
    print(f"{'='*70}")
    print(f"📍 Estación: {datos[0].get('nombre', 'N/A')}")
    print(f"🆔 Indicativo: {datos[0].get('indicativo', 'N/A')}")
    print(f"📊 Total de registros: {len(datos)}")
    print(f"{'='*70}\n")
    
    for i, registro in enumerate(datos, 1):
        fecha_registro = registro.get('fecha', 'N/A')
        
        print(f"{'─'*70}")
        print(f"📅 DÍA {i} - {fecha_registro}")
        print(f"{'─'*70}")
        
        print(f"\n🌡️  TEMPERATURA:")
        print(f"   • Media: {registro.get('tmed', 'N/A')} °C")
        print(f"   • Máxima: {registro.get('tmax', 'N/A')} °C")
        print(f"   • Mínima: {registro.get('tmin', 'N/A')} °C")
        
        print(f"\n💧 PRECIPITACIÓN: {registro.get('prec', 'N/A')} mm")
        
        print(f"\n💨 VIENTO:")
        print(f"   • Velocidad media: {registro.get('velmedia', 'N/A')} m/s")
        print(f"   • Racha máxima: {registro.get('racha', 'N/A')} m/s")
        
        print(f"\n🔽 PRESIÓN ATMOSFÉRICA:")
        print(f"   • Máxima: {registro.get('presMax', 'N/A')} hPa")
        print(f"   • Mínima: {registro.get('presMin', 'N/A')} hPa")
        
        sol = registro.get('sol', 'N/A')
        if sol != 'N/A':
            print(f"\n☀️  INSOLACIÓN: {sol} horas")
        
        print()

def guardar_datos_diarios_csv(datos, fecha_fin, periodo_dias):
    """Guarda los datos diarios en un archivo CSV"""
    if not datos:
        return None
    
    nombre_archivo = os.path.join(DIRECTORIO_RESULTADOS, 
                                 f"datos_diarios_{fecha_fin.strftime('%Y%m%d')}_p{periodo_dias}.csv")
    
    with open(nombre_archivo, mode='w', newline='', encoding='utf-8-sig') as archivo_csv:
        columnas = [
            'Fecha', 'Estacion', 'Indicativo',
            'Temp_Media_°C', 'Temp_Max_°C', 'Temp_Min_°C',
            'Hora_Temp_Max', 'Hora_Temp_Min',
            'Precipitacion_mm', 'Vel_Viento_Media_m/s', 
            'Racha_Max_m/s', 'Direccion_Racha', 'Hora_Racha',
            'Presion_Max_hPa', 'Presion_Min_hPa', 'Insolacion_h'
        ]
        
        escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
        escritor.writeheader()
        
        for registro in datos:
            fila = {
                'Fecha': registro.get('fecha', 'N/D'),
                'Estacion': registro.get('nombre', 'N/D'),
                'Indicativo': registro.get('indicativo', 'N/D'),
                'Temp_Media_°C': registro.get('tmed', 'N/D'),
                'Temp_Max_°C': registro.get('tmax', 'N/D'),
                'Temp_Min_°C': registro.get('tmin', 'N/D'),
                'Hora_Temp_Max': registro.get('horatmax', 'N/D'),
                'Hora_Temp_Min': registro.get('horatmin', 'N/D'),
                'Precipitacion_mm': registro.get('prec', 'N/D'),
                'Vel_Viento_Media_m/s': registro.get('velmedia', 'N/D'),
                'Racha_Max_m/s': registro.get('racha', 'N/D'),
                'Direccion_Racha': registro.get('dir', 'N/D'),
                'Hora_Racha': registro.get('horaracha', 'N/D'),
                'Presion_Max_hPa': registro.get('presMax', 'N/D'),
                'Presion_Min_hPa': registro.get('presMin', 'N/D'),
                'Insolacion_h': registro.get('sol', 'N/D')
            }
            escritor.writerow(fila)
    
    return nombre_archivo

# ============================================
# FUNCIONES PARA DATOS MENSUALES
# ============================================

def obtener_datos_mensuales(anio):
    """
    Obtiene los valores climatológicos mensuales de un año específico
    
    Args:
        anio (int): Año a consultar
    
    Returns:
        list: Lista con los datos mensuales
    """
    
    url_endpoint = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/mensualesanuales/datos/anioini/{anio}/aniofin/{anio}/estacion/{ESTACION_RETIRO}"
    
    headers = {
        'accept': 'application/json',
        'api_key': API_KEY_MENSUALES
    }
    
    try:
        print(f"\n{'='*70}")
        print(f"📊 SOLICITANDO DATOS MENSUALES - AÑO {anio}")
        print(f"{'='*70}\n")
        
        respuesta_endpoint = requests.get(url_endpoint, headers=headers)
        respuesta_endpoint.raise_for_status()
        
        datos_endpoint = respuesta_endpoint.json()
        url_datos_finales = datos_endpoint.get('datos')
        
        if not url_datos_finales:
            print("❌ Error: La respuesta no contenía una URL de 'datos'.")
            return None
        
        print(f"⏳ Descargando datos mensuales...\n")
        
        respuesta_datos = requests.get(url_datos_finales)
        respuesta_datos.raise_for_status()
        
        respuesta_datos.encoding = 'latin-1'
        datos_climatologicos = respuesta_datos.json()
        
        if not datos_climatologicos:
            print(f"⚠️ No se encontraron datos para el año {anio}")
            return None
        
        print(f"✅ Datos mensuales obtenidos: {len(datos_climatologicos)} registros\n")
        return datos_climatologicos
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la petición de datos mensuales: {e}")
        return None
    except json.JSONDecodeError:
        print("❌ Error: La respuesta no es un JSON válido.")
        return None

def mostrar_datos_mensuales(datos, anio):
    """Muestra los datos mensuales en consola"""
    if not datos:
        print("❌ No hay datos mensuales para mostrar")
        return
    
    print(f"\n{'='*70}")
    print(f"📅 DATOS CLIMATOLÓGICOS MENSUALES - AÑO {anio}")
    print(f"📍 Estación: Madrid, Retiro (ID: {ESTACION_RETIRO})")
    print(f"{'='*70}\n")
    
    meses_nombre = {
        '1': 'Enero', '2': 'Febrero', '3': 'Marzo', '4': 'Abril',
        '5': 'Mayo', '6': 'Junio', '7': 'Julio', '8': 'Agosto',
        '9': 'Septiembre', '10': 'Octubre', '11': 'Noviembre', '12': 'Diciembre',
        '13': 'PROMEDIO ANUAL'
    }
    
    for registro in datos:
        mes = registro.get('mes', 'N/D')
        nombre_mes = meses_nombre.get(str(mes), f'Mes {mes}')
        
        if mes == '13':
            print(f"\n{'='*70}")
            print(f"  🎯 {nombre_mes}")
            print(f"{'='*70}")
        else:
            print(f"\n--- {nombre_mes} ---")
        
        print(f"  🌡️  Temp. Media:           {registro.get('tm_mes', 'N/D')} °C")
        print(f"  📈  Temp. Máx. Media:      {registro.get('tm_max', 'N/D')} °C")
        print(f"  📉  Temp. Mín. Media:      {registro.get('tm_min', 'N/D')} °C")
        print(f"  🔥  Temp. Máx. Absoluta:   {registro.get('ta_max', 'N/D')} °C")
        print(f"  ❄️   Temp. Mín. Absoluta:   {registro.get('ta_min', 'N/D')} °C")
        print(f"  💧  Humedad Media:         {registro.get('hr', 'N/D')} %")
        print(f"  🌧️  Precipitación Total:   {registro.get('p_mes', 'N/D')} mm")
        print(f"  ☀️  Horas de Sol:          {registro.get('n_sol', 'N/D')} h")
    
    print(f"\n{'='*70}\n")

def guardar_datos_mensuales_csv(datos, anio):
    """Guarda los datos mensuales en un archivo CSV"""
    if not datos:
        return None
    
    nombre_archivo = os.path.join(DIRECTORIO_RESULTADOS, 
                                 f"datos_mensuales_{anio}.csv")
    
    meses_nombre = {
        '1': 'Enero', '2': 'Febrero', '3': 'Marzo', '4': 'Abril',
        '5': 'Mayo', '6': 'Junio', '7': 'Julio', '8': 'Agosto',
        '9': 'Septiembre', '10': 'Octubre', '11': 'Noviembre', '12': 'Diciembre',
        '13': 'Promedio Anual'
    }
    
    with open(nombre_archivo, mode='w', newline='', encoding='utf-8-sig') as archivo_csv:
        columnas = [
            'Año', 'Mes', 'Nombre_Mes',
            'Temp_Media_°C', 'Temp_Max_Media_°C', 'Temp_Min_Media_°C',
            'Temp_Max_Absoluta_°C', 'Temp_Min_Absoluta_°C',
            'Humedad_Media_%', 'Precipitacion_Total_mm', 'Horas_Sol'
        ]
        
        escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
        escritor.writeheader()
        
        for registro in datos:
            mes = registro.get('mes', 'N/D')
            nombre_mes = meses_nombre.get(str(mes), f'Mes {mes}')
            
            fila = {
                'Año': anio,
                'Mes': mes,
                'Nombre_Mes': nombre_mes,
                'Temp_Media_°C': registro.get('tm_mes', 'N/D'),
                'Temp_Max_Media_°C': registro.get('tm_max', 'N/D'),
                'Temp_Min_Media_°C': registro.get('tm_min', 'N/D'),
                'Temp_Max_Absoluta_°C': registro.get('ta_max', 'N/D'),
                'Temp_Min_Absoluta_°C': registro.get('ta_min', 'N/D'),
                'Humedad_Media_%': registro.get('hr', 'N/D'),
                'Precipitacion_Total_mm': registro.get('p_mes', 'N/D'),
                'Horas_Sol': registro.get('n_sol', 'N/D')
            }
            escritor.writerow(fila)
    
    return nombre_archivo

# ============================================
# FUNCIONES DE ENTRADA DE USUARIO
# ============================================

def solicitar_fecha():
    """Solicita al usuario una fecha válida"""
    while True:
        fecha_str = input("📅 Ingrese la fecha (formato DD/MM/YYYY): ").strip()
        fecha = validar_fecha(fecha_str)
        
        if fecha:
            return fecha
        else:
            print("❌ Fecha inválida. Use el formato DD/MM/YYYY (ejemplo: 24/10/2025)\n")

def solicitar_periodo():
    """Solicita al usuario un período válido en días"""
    while True:
        try:
            periodo_str = input("📆 Ingrese el período en días (ejemplo: 20): ").strip()
            periodo = int(periodo_str)
            
            if periodo > 0 and periodo <= 365:
                return periodo
            else:
                print("❌ El período debe ser un número entre 1 y 365 días\n")
        except ValueError:
            print("❌ Por favor ingrese un número válido\n")

# ============================================
# PROGRAMA PRINCIPAL
# ============================================

def main():
    print("\n" + "="*70)
    print("🌤️  CONSULTA DE DATOS CLIMATOLÓGICOS AEMET - MADRID RETIRO")
    print("="*70 + "\n")
    
    # Crear directorio de resultados
    crear_directorio_resultados()
    
    # Solicitar datos al usuario
    fecha_fin = solicitar_fecha()
    periodo_dias = solicitar_periodo()
    
    # Extraer el año de la fecha
    anio = fecha_fin.year
    
    # Eliminar archivos anteriores
    eliminar_archivos_anteriores(fecha_fin, periodo_dias, anio)
    
    # ============================================
    # OBTENER Y PROCESAR DATOS DIARIOS
    # ============================================
    
    datos_diarios = obtener_datos_diarios(fecha_fin, periodo_dias)
    
    if datos_diarios:
        mostrar_datos_diarios(datos_diarios)
        archivo_diarios = guardar_datos_diarios_csv(datos_diarios, fecha_fin, periodo_dias)
        if archivo_diarios:
            print(f"✅ Datos diarios guardados en: '{archivo_diarios}'")
    else:
        print("\n❌ No se pudieron obtener los datos diarios")
    
    # ============================================
    # OBTENER Y PROCESAR DATOS MENSUALES
    # ============================================
    
    datos_mensuales = obtener_datos_mensuales(anio)
    
    if datos_mensuales:
        mostrar_datos_mensuales(datos_mensuales, anio)
        archivo_mensuales = guardar_datos_mensuales_csv(datos_mensuales, anio)
        if archivo_mensuales:
            print(f"✅ Datos mensuales guardados en: '{archivo_mensuales}'")
    else:
        print("\n❌ No se pudieron obtener los datos mensuales")
    
    # ============================================
    # RESUMEN FINAL
    # ============================================
    
    print(f"\n{'='*70}")
    print(f"✅ CONSULTA COMPLETADA")
    print(f"{'='*70}")
    print(f"📁 Archivos guardados en: {DIRECTORIO_RESULTADOS}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()