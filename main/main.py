import sys, csv, re, glob, subprocess, time, threading, os, platform, unicodedata
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

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

# Contrato común (14 columnas)
CONTRACT_14 = [
    "dataset","event_type","date","time","datetime",
    "district_code","district_name","lat","lon","location",
    "severity","value","units","source_id",
]

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

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: ascii_slug(str(c)) for c in df.columns}
    df = df.rename(columns=rename)
    synonyms = {
        "fecha_hora":"fecha","fechahora":"fecha","datetime":"fecha","timestamp":"fecha","date":"fecha","dia":"fecha",
        "hora_evento":"hora","hora_solicitud":"hora","hr":"hora","time":"hora",
        "distrito_nombre":"distrito","codigo_distrito":"distrito_codigo",
        "lat":"latitud","latitude":"latitud","y":"latitud",
        "lon":"longitud","long":"longitud","x":"longitud",
        "route_id":"linea","stop_lat":"latitud","stop_lon":"longitud","stop_name":"parada",
    }
    for c in list(df.columns):
        base = synonyms.get(c, c)
        if base != c and base not in df.columns:
            df = df.rename(columns={c: base})
    return df

def safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, dtype=str, low_memory=False)
    except Exception:
        for sep in (";", ",", "\t", "|"):
            try:
                return pd.read_csv(p, sep=sep, dtype=str, low_memory=False)
            except Exception:
                continue
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

# ===================== Unificación =====================
def coerce_to_contract14(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CONTRACT_14)
    dfn = normalize_columns(df.copy())
    if "dataset" not in dfn.columns:
        dfn["dataset"] = dataset_name
    for c in CONTRACT_14:
        if c not in dfn.columns:
            dfn[c] = ""
    dfn = dfn[CONTRACT_14]
    dfn = dfn.fillna("NA").replace("", "NA")
    mask_na = dfn["district_name"].str.upper().eq("NA") | dfn["district_name"].eq("")
    if "location" in dfn.columns:
        dfn.loc[mask_na, "district_name"] = dfn.loc[mask_na, "location"]
    return dfn

def build_single_unified(dataset_frames: dict[str, pd.DataFrame]) -> Path:
    commons = []
    for name, df in dataset_frames.items():
        common = coerce_to_contract14(df, dataset_name=name)
        commons.append(common)
    df_common = pd.concat(commons, ignore_index=True, sort=False) if commons else pd.DataFrame(columns=CONTRACT_14)
    df_common.to_csv(
        DATASHEET_UNIFICADO,
        index=False, sep=CSV_SEP, encoding=CSV_ENCODING,
        quoting=CSV_QUOTING, escapechar=CSV_ESCAPECHAR, lineterminator="\n"
    )
    print(f"[OK] Datasheet único → {DATASHEET_UNIFICADO} (rows={len(df_common)})")
    return DATASHEET_UNIFICADO

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

    # 3. Cargar y unificar
    frames = {}
    for name, path in latest.items():
        try:
            frames[name] = safe_read_csv(path)
        except Exception as e:
            print(f"[{name}] ❌ Error reading {path}: {e}")

    if not frames:
        print("❌ No readable datasheets.")
        return

    build_single_unified(frames)

    print("\n=== DONE ===")
    print(f"Run log: {RUN_LOG}")

if __name__ == "__main__":
    main()
