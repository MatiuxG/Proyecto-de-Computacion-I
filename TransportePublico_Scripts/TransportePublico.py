import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransportePublicoETL:
    """
    Clase para unificar los datos de Metro, EMT y Autobuses Interurbanos de Madrid
    """
    
    def __init__(self):
        # Definir rutas base
        self.base_path = Path("TransportePublico_Scripts")
        self.metro_path = self.base_path / "Datos_Red_De_Metro"
        self.emt_path = self.base_path / "Datos_Red_EMT"
        self.interurbanos_path = self.base_path / "Datos_Autobuses_Interurbanos"
        self.output_path = self.base_path / "Resultados"
        
        # Obtener fecha actual
        self.today_date_obj = datetime.now()
        self.today_date_str = self.today_date_obj.strftime('%Y%m%d')
        self.today_day_name = self.today_date_obj.strftime('%A').lower()
        self.today_date_int = int(self.today_date_str)
        
        # DataFrame unificado
        self.unified_data = []
        
    def limpiar_csv_anterior(self):
        """Elimina el CSV anterior si existe"""
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            output_file = self.output_path / f"transporte_publico_madrid_{self.today_date_str}.csv"
            if output_file.exists():
                output_file.unlink()
                logging.info(f"CSV anterior eliminado: {output_file}")
        except Exception as e:
            logging.error(f"Error al eliminar CSV anterior: {e}")
    
    def cargar_archivo(self, file_path, file_name):
        """Carga un archivo CSV y retorna un DataFrame"""
        try:
            full_path = file_path / file_name
            df = pd.read_csv(full_path, low_memory=False)
            logging.info(f"✓ Archivo cargado: {file_name}")
            return df
        except FileNotFoundError:
            logging.warning(f"✗ Archivo no encontrado: {file_name}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"✗ Error al cargar {file_name}: {e}")
            return pd.DataFrame()
    
    def obtener_servicios_activos(self, calendar_df, calendar_dates_df):
        """Determina los servicios activos para el día actual"""
        if calendar_df.empty:
            return set()
        
        # Convertir fechas a entero
        calendar_df['start_date'] = calendar_df['start_date'].astype(int)
        calendar_df['end_date'] = calendar_df['end_date'].astype(int)
        
        if not calendar_dates_df.empty:
            calendar_dates_df['date'] = calendar_dates_df['date'].astype(int)
            
            # Obtener excepciones
            exceptions_today = calendar_dates_df[calendar_dates_df['date'] == self.today_date_int]
            added_services = set(exceptions_today[exceptions_today['exception_type'] == 1]['service_id'])
            removed_services = set(exceptions_today[exceptions_today['exception_type'] == 2]['service_id'])
        else:
            added_services = set()
            removed_services = set()
        
        # Obtener servicios regulares
        regular_services_today = calendar_df[
            (calendar_df['start_date'] <= self.today_date_int) &
            (calendar_df['end_date'] >= self.today_date_int) &
            (calendar_df[self.today_day_name] == 1)
        ]
        active_service_ids_base = set(regular_services_today['service_id'])
        
        # Combinar servicios
        active_service_ids = (active_service_ids_base - removed_services) | added_services
        
        return active_service_ids
    
    def procesar_metro(self):
        """Procesa los datos del Metro de Madrid"""
        logging.info("\n=== PROCESANDO DATOS DEL METRO ===")
        
        try:
            # Cargar archivos
            trips_df = self.cargar_archivo(self.metro_path, "trips.txt")
            calendar_df = self.cargar_archivo(self.metro_path, "calendar.txt")
            calendar_dates_df = self.cargar_archivo(self.metro_path, "calendar_dates.txt")
            stop_times_df = self.cargar_archivo(self.metro_path, "stop_times.txt")
            stops_df = self.cargar_archivo(self.metro_path, "stops.txt")
            
            if trips_df.empty or stops_df.empty:
                logging.warning("No se pudieron cargar archivos críticos del Metro")
                return
            
            # Obtener servicios activos
            active_service_ids = self.obtener_servicios_activos(calendar_df, calendar_dates_df)
            logging.info(f"Servicios activos Metro: {len(active_service_ids)}")
            
            # Filtrar viajes activos
            active_trips_df = trips_df[trips_df['service_id'].isin(active_service_ids)]
            logging.info(f"Viajes activos Metro: {len(active_trips_df)}")
            
            # Merge de datos
            if not stop_times_df.empty:
                trips_subset = active_trips_df[['trip_id', 'route_id', 'service_id', 'trip_headsign', 'direction_id']]
                stop_times_subset = stop_times_df[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']]
                stops_subset = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
                
                trip_stop_times_df = pd.merge(trips_subset, stop_times_subset, on='trip_id', how='inner')
                full_schedule_df = pd.merge(trip_stop_times_df, stops_subset, on='stop_id', how='inner')
                
                # Añadir columna de tipo de transporte
                full_schedule_df['tipo_transporte'] = 'METRO'
                
                # Ordenar y seleccionar columnas
                full_schedule_df = full_schedule_df.sort_values(by=['trip_id', 'stop_sequence'])
                
                self.unified_data.append(full_schedule_df)
                logging.info(f"✓ Metro procesado: {len(full_schedule_df)} registros")
            
        except Exception as e:
            logging.error(f"Error procesando Metro: {e}")
    
    def procesar_emt(self):
        """Procesa los datos de la Red EMT"""
        logging.info("\n=== PROCESANDO DATOS DE EMT ===")
        
        try:
            # Cargar archivos
            trips_df = self.cargar_archivo(self.emt_path, "trips.txt")
            calendar_df = self.cargar_archivo(self.emt_path, "calendar.txt")
            calendar_dates_df = self.cargar_archivo(self.emt_path, "calendar_dates.txt")
            stop_times_df = self.cargar_archivo(self.emt_path, "stop_times.txt")
            stops_df = self.cargar_archivo(self.emt_path, "stops.txt")
            
            if trips_df.empty or stops_df.empty:
                logging.warning("No se pudieron cargar archivos críticos de EMT")
                return
            
            # Limpiar stops
            stops_df['stop_lat'] = pd.to_numeric(stops_df['stop_lat'], errors='coerce')
            stops_df['stop_lon'] = pd.to_numeric(stops_df['stop_lon'], errors='coerce')
            stops_df = stops_df.dropna(subset=['stop_lat', 'stop_lon'])
            
            # Obtener servicios activos
            active_service_ids = self.obtener_servicios_activos(calendar_df, calendar_dates_df)
            logging.info(f"Servicios activos EMT: {len(active_service_ids)}")
            
            # Filtrar viajes activos
            active_trips_df = trips_df[trips_df['service_id'].isin(active_service_ids)]
            logging.info(f"Viajes activos EMT: {len(active_trips_df)}")
            
            # Merge de datos
            if not stop_times_df.empty:
                trips_subset = active_trips_df[['trip_id', 'route_id', 'service_id', 'trip_headsign', 'direction_id']]
                stop_times_subset = stop_times_df[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']]
                stops_subset = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
                
                trip_stop_times_df = pd.merge(trips_subset, stop_times_subset, on='trip_id', how='inner')
                full_schedule_df = pd.merge(trip_stop_times_df, stops_subset, on='stop_id', how='inner')
                
                # Añadir columna de tipo de transporte
                full_schedule_df['tipo_transporte'] = 'EMT'
                
                # Ordenar y seleccionar columnas
                full_schedule_df = full_schedule_df.sort_values(by=['trip_id', 'stop_sequence'])
                
                self.unified_data.append(full_schedule_df)
                logging.info(f"✓ EMT procesado: {len(full_schedule_df)} registros")
            
        except Exception as e:
            logging.error(f"Error procesando EMT: {e}")
    
    def procesar_interurbanos(self):
        """Procesa los datos de Autobuses Interurbanos"""
        logging.info("\n=== PROCESANDO DATOS DE AUTOBUSES INTERURBANOS ===")
        
        try:
            # Cargar archivos
            trips_df = self.cargar_archivo(self.interurbanos_path, "trips.txt")
            calendar_df = self.cargar_archivo(self.interurbanos_path, "calendar.txt")
            calendar_dates_df = self.cargar_archivo(self.interurbanos_path, "calendar_dates.txt")
            stop_times_df = self.cargar_archivo(self.interurbanos_path, "stop_times.txt")
            stops_df = self.cargar_archivo(self.interurbanos_path, "stops.txt")
            
            if trips_df.empty or stops_df.empty:
                logging.warning("No se pudieron cargar archivos críticos de Interurbanos")
                return
            
            # Limpiar stops
            stops_df['stop_lat'] = pd.to_numeric(stops_df['stop_lat'], errors='coerce')
            stops_df['stop_lon'] = pd.to_numeric(stops_df['stop_lon'], errors='coerce')
            stops_df = stops_df.dropna(subset=['stop_lat', 'stop_lon'])
            
            # Obtener servicios activos
            active_service_ids = self.obtener_servicios_activos(calendar_df, calendar_dates_df)
            logging.info(f"Servicios activos Interurbanos: {len(active_service_ids)}")
            
            # Filtrar viajes activos
            active_trips_df = trips_df[trips_df['service_id'].isin(active_service_ids)]
            logging.info(f"Viajes activos Interurbanos: {len(active_trips_df)}")
            
            # Merge de datos
            if not stop_times_df.empty:
                trips_subset = active_trips_df[['trip_id', 'route_id', 'service_id', 'trip_headsign', 'direction_id']]
                stop_times_subset = stop_times_df[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']]
                stops_subset = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
                
                trip_stop_times_df = pd.merge(trips_subset, stop_times_subset, on='trip_id', how='inner')
                full_schedule_df = pd.merge(trip_stop_times_df, stops_subset, on='stop_id', how='inner')
                
                # Añadir columna de tipo de transporte
                full_schedule_df['tipo_transporte'] = 'INTERURBANO'
                
                # Ordenar y seleccionar columnas
                full_schedule_df = full_schedule_df.sort_values(by=['trip_id', 'stop_sequence'])
                
                self.unified_data.append(full_schedule_df)
                logging.info(f"✓ Interurbanos procesado: {len(full_schedule_df)} registros")
            
        except Exception as e:
            logging.error(f"Error procesando Interurbanos: {e}")
    
    def unificar_y_guardar(self):
        """Unifica todos los datos y guarda el CSV final"""
        logging.info("\n=== UNIFICANDO DATOS Y GUARDANDO CSV ===")
        
        if not self.unified_data:
            logging.error("No hay datos para unificar")
            return
        
        try:
            # Concatenar todos los DataFrames
            final_df = pd.concat(self.unified_data, ignore_index=True)
            
            # Reordenar columnas
            columnas_finales = [
                'tipo_transporte',
                'trip_id',
                'route_id',
                'service_id',
                'direction_id',
                'trip_headsign',
                'stop_sequence',
                'stop_id',
                'stop_name',
                'arrival_time',
                'departure_time',
                'stop_lat',
                'stop_lon'
            ]
            
            final_df = final_df[columnas_finales]
            
            # Ordenar por tipo de transporte y trip_id
            final_df = final_df.sort_values(by=['tipo_transporte', 'trip_id', 'stop_sequence'])
            
            # Guardar CSV
            output_file = self.output_path / f"transporte_publico_madrid_{self.today_date_str}.csv"
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logging.info(f"\n{'='*60}")
            logging.info(f"✓ PROCESO ETL COMPLETADO EXITOSAMENTE")
            logging.info(f"{'='*60}")
            logging.info(f"Fecha: {self.today_date_str} ({self.today_day_name.capitalize()})")
            logging.info(f"Total de registros: {len(final_df):,}")
            logging.info(f"Archivo guardado: {output_file}")
            logging.info(f"\nDistribución por tipo de transporte:")
            logging.info(final_df['tipo_transporte'].value_counts().to_string())
            logging.info(f"{'='*60}\n")
            
            # Mostrar muestra de datos
            print("\nPrimeras 10 filas del dataset unificado:")
            print(final_df.head(10).to_markdown(index=False))
            
        except Exception as e:
            logging.error(f"Error al unificar y guardar datos: {e}")
    
    def ejecutar(self):
        """Ejecuta el proceso ETL completo"""
        logging.info("="*60)
        logging.info("INICIANDO PROCESO ETL - TRANSPORTE PÚBLICO MADRID")
        logging.info("="*60)
        
        # Limpiar CSV anterior
        self.limpiar_csv_anterior()
        
        # Procesar cada sistema de transporte
        self.procesar_metro()
        self.procesar_emt()
        self.procesar_interurbanos()
        
        # Unificar y guardar
        self.unificar_y_guardar()


if __name__ == "__main__":
    etl = TransportePublicoETL()
    etl.ejecutar()