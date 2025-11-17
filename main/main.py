import sys, csv, re, glob, subprocess, time, threading, os, platform, unicodedata
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# ===================== Runtime env =====================
PYTHON = sys.executable
IS_WIN = platform.system().lower().startswith("win")

# ===================== Quick config =====================
MAX_WORKERS = 6
HEARTBEAT_EVERY = 15

CSV_SEP = ";"
CSV_QUOTING = csv.QUOTE_NONE
CSV_ESCAPECHAR = "\\"
CSV_ENCODING = "utf-8-sig"
CSV_FLOAT_FORMAT = "%.6f"

# Scripts a ejecutar
SCRIPT_ENTRY = {
    "obras":                 "Obras.py",
    "trafico_historico":     "TraficoHistorico.py",
    "emergencias":           "emergencias_scraper.py",
    "clima":                 "Clima.py",
    "calidad_aire":          "CalidadAire.py",
    "accidentes":            "Accidentes.py",
}

# CSV esperados
EXPECTED_OUTPUTS = {
    "obras":             "datasheet_infraestructura.csv",
    "trafico_historico": "datasheet_trafico.csv",
    "emergencias":       "datasheet_emergencias.csv",
    "clima":             "datasheet_clima.csv",
    "calidad_aire":      "datasheet_calidad_aire.csv",
    "accidentes":        "datasheet_accidentes.csv",
}

HERE = Path(__file__).parent.resolve()

# ===================== Utilidades =====================
def detect_project_root(start: Path, max_up: int = 5) -> Path:
    cur = start
    for _ in range(max_up + 1):
        try:
            if any(p.is_dir() and p.name.lower().endswith("_scripts") for p in cur.iterdir()):
                return cur
        except Exception:
            pass
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

def ascii_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def safe_read_csv(p: Path) -> pd.DataFrame:
    # 1. Intentar primero con el separador oficial del proyecto (CSV_SEP = ";")
    try:
        return pd.read_csv(p, sep=CSV_SEP, dtype=str, low_memory=False)
    except Exception:
        # 2. Si falla, intentar con otros separadores comunes
        for sep in (",", "\t", "|"): # Ya no es necesario probar ";" aquí
            try:
                return pd.read_csv(p, sep=sep, dtype=str, low_memory=False)
            except Exception:
                continue
    # 3. Si todo falla, lanzar el error
    print(f"ERROR: No se pudo leer el CSV: {p}")
    raise

def _reader_thread(stream, buf: list, status: dict, name: str):
    for line in iter(stream.readline, b""):
        try:
            txt = line.decode(errors="replace")
        except Exception:
            txt = str(line)
        buf.append(txt)
        status[name]["last_touch"] = time.time()
    try:
        stream.close()
    except Exception:
        pass

def run_script(label: str, script_path: Path, status: dict) -> tuple[str, int, str]:
    cmd = [PYTHON, str(script_path)]
    proc = subprocess.Popen(cmd, cwd=str(script_path.parent),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    status[label] = {
        "pid": proc.pid, "start": time.time(), "last_touch": time.time(),
        "state": "running", "path": str(script_path)
    }
    buf_out, buf_err = [], []
    t1 = threading.Thread(target=_reader_thread, args=(proc.stdout, buf_out, status, label), daemon=True)
    t2 = threading.Thread(target=_reader_thread, args=(proc.stderr, buf_err, status, label), daemon=True)
    t1.start(); t2.start()
    rc = proc.wait()
    try:
        t1.join(timeout=1); t2.join(timeout=1)
    except Exception:
        pass
    status[label]["state"] = "done"
    status[label]["rc"] = rc
    status[label]["end"] = time.time()
    output = "".join(buf_out + buf_err)
    return label, rc, output

def heartbeat(status: dict, stop_event: threading.Event):
    while not stop_event.is_set():
        running = [(k, v) for k, v in status.items() if v.get("state") == "running"]
        if running:
            now = time.time()
            parts = [f"{k} (pid {v.get('pid')}, {int(now - v.get('start', now))}s)" for k, v in running]
            print("[HB] Running: " + ", ".join(parts), flush=True)
        stop_event.wait(HEARTBEAT_EVERY)

# ===================== Descubrimiento =====================
PROJECT_ROOT = detect_project_root(HERE)
RESULTS_GLOBS = [str(PROJECT_ROOT / "**" / "Resultados" / "*.csv")]
OUT_DIR = PROJECT_ROOT / "Resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT_DIR / "run_log.txt"
DATASHEET_UNIFICADO = OUT_DIR / "datasheet_unificado.csv"

def discover_script_paths() -> dict:
    found = {}
    for label, filename in SCRIPT_ENTRY.items():
        hits = list(PROJECT_ROOT.rglob(filename))
        if hits:
            def score(p: Path):
                parent = p.parent.name.lower()
                s = 0 if parent.endswith("_scripts") else 10
                return (s, len(str(p)))
            hits.sort(key=score)
            found[label] = hits[0]
        else:
            print(f"[WARN] Not found: {filename} under {PROJECT_ROOT}")
    return found

def discover_expected_csvs() -> dict[str, Path]:
    files = []
    for pat in RESULTS_GLOBS:
        files += [Path(x) for x in glob.glob(pat, recursive=True)]
    latest_by_dataset = {}
    for dataset, expected_name in EXPECTED_OUTPUTS.items():
        candidates = [p for p in files if p.name.lower() == expected_name.lower()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_by_dataset[dataset] = candidates[0]
    return latest_by_dataset

# ===================== Unificación Agregada =====================

# Columnas finales deseadas
FINAL_COLUMNS = [
    "Dia", "Mes", "Año", "Codigo de distrito", "Distrito/Estacion",
    "Temp_Media_°C", "Temp_Max_°C", "Temp_Min_°C", "Hora_Temp_Max", "Hora_Temp_Min",
    "Precipitacion_mm", "Vel_Viento_Media_m/s", "Racha_Max_m/s",
    "Presion_Max_hPa", "Presion_Min_hPa", "Insolacion_h",
    "FUEGOS", "DAÑOS EN CONSTRUCCION", "SALVAMENTOS Y RESCATES", "DAÑOS POR AGUA",
    "INCIDENTES DIVERSOS", "SALIDAS SIN INTERVENCION", "SERVICIOS VARIOS",
    "aqi", "aqi_category", "pollutant"
]

# Columnas de emergencias que queremos pivotar (deben coincidir con `event_type`)
EMERGENCIA_TYPES = [
    "FUEGOS", "DAÑOS EN CONSTRUCCION", "SALVAMENTOS Y RESCATES", "DAÑOS POR AGUA",
    "INCIDENTES DIVERSOS", "SALIDAS SIN INTERVENCION", "SERVICIOS VARIOS"
]

def normalize_key(s: str) -> str:
    """Normaliza strings para usarlos como claves de merge (distritos, estaciones)."""
    # Convertir a string primero para manejar NaN (floats)
    s = str(s or "").strip().lower()
    if s == "na" or s == "nan":
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def get_aqi_simple(pollutant, value) -> tuple:
    """Calculadora AQI simplificada (ejemplo)."""
    try:
        val = float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return 0, "N/A"

    pol = str(pollutant).lower()
    
    # Lógica de ejemplo basada en tu request (o3: 28 -> Buena)
    # Esto es una simplificación, la escala real es más compleja.
    if pol == 'o3':
        if val <= 60: return int(val * (28.0/60.0)), "Buena"
        if val <= 120: return int(val * 0.8), "Moderada"
        if val <= 180: return int(val * 0.9), "Mala"
        return int(val), "Muy Mala"
    elif pol == 'pm10':
        if val <= 25: return int(val * 1.5), "Buena"
        if val <= 50: return int(val * 1.5), "Moderada"
        if val <= 90: return int(val), "Mala"
        return int(val), "Muy Mala"
    elif pol == 'pm25':
        if val <= 15: return int(val * 2), "Buena"
        if val <= 30: return int(val * 2), "Moderada"
        if val <= 55: return int(val), "Mala"
        return int(val), "Muy Mala"
    elif pol == 'no2':
        if val <= 50: return int(val * 0.8), "Buena"
        if val <= 100: return int(val * 0.8), "Moderada"
        if val <= 200: return int(val), "Mala"
        return int(val), "Muy Mala"
    
    return int(val), "N/A" # Default para SO2, CO, etc.


def build_aggregated_unified(dataset_frames: dict[str, pd.DataFrame]) -> Path:
    """
    Construye el nuevo datasheet agregado, fusionando por fecha y distrito/estación.
    """
    
    # --- 1. Preparar cada DataFrame ---
    
    all_keys = set()
    dfs_processed = {}
    
    # --- CLIMA ---
    if "clima" in dataset_frames:
        df = dataset_frames["clima"].copy()
        
        # Crear 'date_iso' desde Dia, Mes, Año para la clave de merge
        # Asegurarse de que son numéricos antes de crear la fecha
        df['Dia'] = pd.to_numeric(df['Dia'], errors='coerce')
        df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')
        df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
        df = df.dropna(subset=['Dia', 'Mes', 'Año']) # Descartar filas sin fecha
        
        df["date_iso"] = pd.to_datetime(
            df['Año'].astype(int).astype(str) + '-' + \
            df['Mes'].astype(int).astype(str) + '-' + \
            df['Dia'].astype(int).astype(str),
            errors='coerce'
        ).dt.strftime("%Y-%m-%d")

        # El CSV ya tiene 'district_name' y 'district_code'
        # Usamos 'district_name' para la clave, que es más legible (ej. "retiro")
        df["key_distrito_estacion"] = df["district_name"].apply(normalize_key)
        df["key_distrito_codigo"] = df["district_code"].apply(normalize_key).str.zfill(2)
        
        # Quedarnos con la primera lectura si hay duplicados por día/estación
        keys = ["date_iso", "key_distrito_estacion", "key_distrito_codigo"]
        
        # Las columnas ya tienen los nombres finales (ej. 'Temp_Media_°C')
        cols_to_keep = keys + [c for c in FINAL_COLUMNS if c in df.columns]
        df_clima = df[cols_to_keep].drop_duplicates(subset=keys, keep="first")
        
        dfs_processed["clima"] = df_clima
        all_keys.update(df_clima[keys].apply(tuple, axis=1))
        print(f"[i] Clima procesado: {df_clima.shape[0]} filas")

    # --- EMERGENCIAS ---
    if "emergencias" in dataset_frames:
        df = dataset_frames["emergencias"].copy()
        df["date_iso"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["key_distrito_estacion"] = df["district_name"].apply(normalize_key)
        df["key_distrito_codigo"] = df["district_code"].apply(normalize_key).str.zfill(2)
        
        # Filtrar solo los tipos de evento que queremos contar
        df["event_type_norm"] = df["event_type"].str.upper().str.strip()
        df_emerg = df[df["event_type_norm"].isin(EMERGENCIA_TYPES)]
        
        # Pivotar: contar eventos por día, distrito y tipo
        keys = ["date_iso", "key_distrito_estacion", "key_distrito_codigo"]
        if not df_emerg.empty:
            df_pivot = df_emerg.groupby(keys + ["event_type_norm"]).size().unstack(fill_value=0)
            df_pivot = df_pivot.rename_axis(columns=None).reset_index()
        else:
            df_pivot = pd.DataFrame(columns=keys + EMERGENCIA_TYPES)
        
        dfs_processed["emergencias"] = df_pivot
        all_keys.update(df_pivot[keys].apply(tuple, axis=1))
        print(f"[i] Emergencias procesadas: {df_pivot.shape[0]} filas")

    # --- CALIDAD AIRE ---
    if "calidad_aire" in dataset_frames:
        df = dataset_frames["calidad_aire"].copy()
        df["date_iso"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        # Usamos 'district_name' (ej. "Centro") en lugar de 'location' (ej. "Plaza del Carmen")
        df["key_distrito_estacion"] = df["district_name"].apply(normalize_key)
        df["key_distrito_codigo"] = df["district_code"].apply(normalize_key).str.zfill(2)
        
        # Calcular AQI para cada contaminante
        aqi_data = df.apply(
            lambda row: get_aqi_simple(row["event_type"], row["value"]),
            axis=1,
            result_type="expand"
        )
        df["aqi"] = aqi_data[0]
        df["aqi_category"] = aqi_data[1]
        df["pollutant"] = df["event_type"]
        
        # Quedarnos con el PEOR (max AQI) por día y estación
        keys = ["date_iso", "key_distrito_estacion", "key_distrito_codigo"]
        # Ordenamos por AQI (peor primero) y nos quedamos con el primero de cada grupo
        df_aqi = df.sort_values("aqi", ascending=False).drop_duplicates(subset=keys, keep="first")
        
        cols_to_keep = keys + ["aqi", "aqi_category", "pollutant"]
        df_aqi = df_aqi[cols_to_keep]
        
        dfs_processed["calidad_aire"] = df_aqi
        all_keys.update(df_aqi[keys].apply(tuple, axis=1))
        print(f"[i] Calidad Aire procesada: {df_aqi.shape[0]} filas")

    # --- DATOS NO SOLICITADOS ---
    for k in ["accidentes", "obras", "trafico_historico"]:
        if k in dataset_frames:
            print(f"[i] Omitiendo '{k}'. No se solicitaron columnas en la nueva estructura.")

    
    # --- 2. Crear Base y Fusionar (Merge) ---
    if not all_keys:
        print("❌ No se encontraron datos para agregar.")
        # Devolver un DataFrame vacío con las columnas correctas
        df_empty = pd.DataFrame(columns=FINAL_COLUMNS)
        # Asegurar tipos correctos para columnas de emergencia
        for col in EMERGENCIA_TYPES:
            if col in df_empty.columns:
                df_empty[col] = df_empty[col].astype(int)
        return df_empty

    # Crear la base de todas las combinaciones fecha/distrito encontradas
    keys = ["date_iso", "key_distrito_estacion", "key_distrito_codigo"]
    base_df = pd.DataFrame(list(all_keys), columns=keys)
    base_df = base_df.sort_values(by=["date_iso", "key_distrito_codigo", "key_distrito_estacion"]).reset_index(drop=True)
    
    print(f"[i] Base unificada creada: {base_df.shape[0]} filas")

    # Fusionar (merge) todos los dataframes procesados contra la base
    merged = base_df
    if "clima" in dfs_processed:
        merged = merged.merge(dfs_processed["clima"], on=keys, how="left")
        print(f"[i] Merge con Clima... (filas: {merged.shape[0]})")
    if "emergencias" in dfs_processed:
        merged = merged.merge(dfs_processed["emergencias"], on=keys, how="left")
        print(f"[i] Merge con Emergencias... (filas: {merged.shape[0]})")
    if "calidad_aire" in dfs_processed:
        merged = merged.merge(dfs_processed["calidad_aire"], on=keys, how="left")
        print(f"[i] Merge con Calidad Aire... (filas: {merged.shape[0]})")
        
    # --- 3. Limpieza Final ---
    
    # Rellenar NaNs
    # Rellenar contadores de emergencia con 0
    for col in EMERGENCIA_TYPES:
        if col in merged.columns:
            # Corrección: Usar pd.to_numeric para manejar strings y NaNs antes de rellenar
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0).astype(int)
    
    # Rellenar el resto con "N/A"
    merged = merged.fillna("N/A")

    # Crear columnas de fecha (si no vienen de clima)
    if not all(['Dia', 'Mes', 'Año'] in merged.columns):
        dt = pd.to_datetime(merged["date_iso"], errors="coerce")
        merged["Dia"] = dt.dt.day.fillna(0).astype(int)
        merged["Mes"] = dt.dt.month.fillna(0).astype(int)
        merged["Año"] = dt.dt.year.fillna(0).astype(int)
    
    # Renombrar columnas clave
    merged = merged.rename(columns={
        "key_distrito_codigo": "Codigo de distrito",
        "key_distrito_estacion": "Distrito/Estacion"
    })
    
    # Asegurar que todas las columnas finales existan
    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = "N/A" if col not in EMERGENCIA_TYPES else 0
            
    # Ordenar y seleccionar columnas finales
    df_final = merged[FINAL_COLUMNS]
    
    print(f"[OK] Datasheet agregado finalizado: {df_final.shape[0]} filas")
    return df_final


# ===================== MAIN =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[i] Project root: {PROJECT_ROOT}")

    # 1. Ejecutar scripts
    scripts = discover_script_paths()
    status = {}
    stop_event = threading.Event()
    hb_thread = threading.Thread(target=heartbeat, args=(status, stop_event), daemon=True)
    hb_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(run_script, label, path, status): label for label, path in scripts.items()}
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    n, rc, out = fut.result()
                    print(f"[{n}] returncode={rc}")
                    existing = RUN_LOG.read_text(encoding="utf-8") if RUN_LOG.exists() else ""
                    RUN_LOG.write_text(existing + f"\n---- [{n}] ----\n{out}\n", encoding="utf-8")
                except Exception as e:
                    print(f"[{label}] ERROR when running: {e}")
    finally:
        stop_event.set()
        hb_thread.join(timeout=1)

    # 2. Recolectar CSVs
    latest = discover_expected_csvs()
    if not latest:
        print("❌ No CSVs detected under */Resultados/*.csv")
        return
    for k, v in latest.items():
        print(f"[{k}] CSV: {v}")

    # 3. Cargar y unificar (LÓGICA CAMBIADA)
    frames = {}
    for name, path in latest.items():
        try:
            frames[name] = safe_read_csv(path)
        except Exception as e:
            print(f"[{name}] ❌ Error reading {path}: {e}")

    if not frames:
        print("❌ No readable datasheets.")
        return

    # Llamar a la nueva función de agregación
    df_final = build_aggregated_unified(frames)

    # Guardar el resultado agregado
    df_final.to_csv(
        DATASHEET_UNIFICADO,
        index=False, sep=CSV_SEP, encoding=CSV_ENCODING,
        quoting=CSV_QUOTING, escapechar=CSV_ESCAPECHAR, 
        lineterminator="\n", # Corregido (sin '_')
        float_format=CSV_FLOAT_FORMAT
    )
    print(f"[OK] Datasheet agregado → {DATASHEET_UNIFICADO} (rows={len(df_final)})")

    print("\n=== DONE ===")
    print(f"Run log: {RUN_LOG}")

if __name__ == "__main__":
    main()