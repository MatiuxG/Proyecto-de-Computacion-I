# run_all_and_build_datasheet.py (v7) — threads + idle-timeout + kill tree (Windows)
import sys, re, glob, subprocess, traceback, time, threading, os, argparse, queue, platform
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

PYTHON = sys.executable
IS_WIN = platform.system().lower().startswith("win")

# ---------- args ----------
ap = argparse.ArgumentParser(description="Run all dataset scripts (threads) and build datasheet")
ap.add_argument("--fast", action="store_true", help="use aggressive timeouts (quick runs)")
ap.add_argument("--max-workers", type=int, default=4, help="concurrent scripts")
ap.add_argument("--idle-timeout", type=int, default=None, help="seconds with no output before kill")
ap.add_argument("--hard-timeout", type=int, default=None, help="max seconds per script before kill")
args = ap.parse_args()

# ---------- root autodetect ----------
def detect_project_root(start: Path, max_up: int = 4) -> Path:
    cur = start
    for _ in range(max_up+1):
        if any(p.is_dir() and p.name.lower().endswith('_scripts') for p in cur.iterdir()):
            return cur
        if cur.parent == cur: break
        cur = cur.parent
    return start

HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = detect_project_root(HERE)

# ---------- discovery ----------
SCRIPT_CANDIDATES = {
    "trafico_historico":      ["TraficoHistorico.py","trafico_historico.py","trafico.py","main_trafico.py"],
    "vehiculos_por_distrito": ["VehiculosPorDistrito.py","vehiculos_distrito.py","vehiculos.py","main_vehiculos.py"],
    "transporte_publico":     ["TransportePublico.py","transporte.py","main_transporte.py"],
    "emergencias":            ["emergencias_scraper.py","Emergencias.py","emergencias.py","main_emergencias.py"],
    "clima":                  ["Clima.py","clima_aemet.py","main_clima.py"],
    "radares":                ["Radares.py","Camaras.py","radares_camaras.py","main_radares.py"],
    "calidad_aire":           ["CalidadAire.py","calidad_aire.py","main_calidad_aire.py"],
    "accidentes":             ["Accidentes.py","accidentes_trafico.py","main_accidentes.py"],
}

# timeouts (segundos)
SCRIPT_TIMEOUTS = {
    "trafico_historico": 1200,
    "vehiculos_por_distrito": 600,
    "transporte_publico": 600,
    "emergencias": 900,
    "clima": 600,
    "radares": 1200,
    "calidad_aire": 900,
    "accidentes": 1200,
}
DEFAULT_HARD = 900
DEFAULT_IDLE = 300  # mata si no hay output en 5 min

if args.fast:
    # modos rápidos (ajusta si quieres aún más agresivo)
    SCRIPT_TIMEOUTS = {k: min(600, v) for k, v in SCRIPT_TIMEOUTS.items()}
    DEFAULT_HARD = 600
    DEFAULT_IDLE = 120

if args.hard_timeout is not None:
    DEFAULT_HARD = args.hard_timeout
if args.idle_timeout is not None:
    DEFAULT_IDLE = args.idle_timeout
MAX_WORKERS = args.max_workers

# ---------- outputs ----------
OUT_DIR = PROJECT_ROOT / "Resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUT_DIR / f"datasheet_madrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
OUT_CSV  = OUT_DIR / f"datasheet_madrid_unificado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
RUN_LOG  = OUT_DIR / "run_log.txt"

CSV_PATH_RE = re.compile(r'(?P<path>(?:[A-Za-z]:\\|/)?[^\s"<>|]+?\.csv)', re.IGNORECASE)
RECURSIVE_SCAN_PATTERNS = [str(PROJECT_ROOT / "**" / "Resultados" / "*.csv"),
                           str(PROJECT_ROOT / "**" / "results" / "*.csv")]
DATASET_HINTS = {"trafico":"trafico_historico","traffic":"trafico_historico","vehiculos":"vehiculos_por_distrito",
                 "parking":"vehiculos_por_distrito","transporte":"transporte_publico","bus":"transporte_publico",
                 "metro":"transporte_publico","emergenc":"emergencias","clima":"clima","aemet":"clima",
                 "radar":"radares","camara":"radares","calidad_aire":"calidad_aire","accidente":"accidentes"}

def discover_scripts():
    found = []
    for name, cands in SCRIPT_CANDIDATES.items():
        hits = []
        for cand in cands:
            hits += list(PROJECT_ROOT.rglob(cand))
        if not hits:
            for folder in PROJECT_ROOT.glob("*_Scripts"):
                if not folder.is_dir(): continue
                key = name.split("_")[0]
                if key.lower() in folder.name.lower():
                    hits += list(folder.glob("*.py"))
        if hits:
            def score(p: Path):
                s = 0 if p.parent.name.lower().endswith("_scripts") else 10
                s += 0 if any(k in p.name.lower() for k in [name.replace("_",""), name.split("_")[0]]) else 5
                return (s, len(p.name))
            hits.sort(key=score)
            found.append((name, hits[0]))
        else:
            print(f"[WARN] No se encontró script para '{name}' (en {PROJECT_ROOT})")
    return found

# ---------- stdin filtros (solo tráfico) ----------
def ask_filters_once():
    print("Introduce filtros para Tráfico (Enter = por defecto/omitir)")
    fecha   = input("Fecha (YYYY-MM-DD): ").strip()
    meses   = input("Margen hacia atrás en meses (Enter=2): ").strip() or "2"
    zona    = input("Zona/Ubicación: ").strip()
    hora    = input("Hora aprox (08, 08:30, 8.5): ").strip()
    margen  = input("Margen horario ±min (Enter=60): ").strip() or "60"
    return fecha, meses, zona, hora, margen

# ---------- runner con idle-timeout ----------
def _reader_thread(stream, buf: list, q: queue.Queue, tag: str, last_touch: dict):
    for line in iter(stream.readline, b""):
        try:
            txt = line.decode(errors="replace")
        except Exception:
            txt = str(line)
        buf.append(txt)
        q.put((tag, txt))
        last_touch["t"] = time.time()
    stream.close()

def kill_tree(proc: subprocess.Popen):
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.terminate()
            time.sleep(1)
            if proc.poll() is None:
                proc.kill()
    except Exception:
        pass

def run_script(name, path: Path, stdin_text: str|None, hard_timeout: int, idle_timeout: int, extra_env: dict):
    info = {"name": name, "path": str(path), "ok": False, "returncode": None, "stdout": "", "stderr": ""}
    if not path.exists():
        info["stderr"] = f"Script no encontrado: {path}"
        return info

    print(f"[RUN] {name} -> {path}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # flush inmediato
    env.update(extra_env or {})

    start = time.time()
    last_output = {"t": start}

    proc = subprocess.Popen(
        [PYTHON, str(path)],
        cwd=str(path.parent),
        stdin=subprocess.PIPE if stdin_text else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, bufsize=1
    )

    if stdin_text:
        try:
            proc.stdin.write(stdin_text.encode("utf-8"))
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass

    out_buf, err_buf = [], []
    qlines = queue.Queue()
    t_out = threading.Thread(target=_reader_thread, args=(proc.stdout, out_buf, qlines, "OUT", last_output), daemon=True)
    t_err = threading.Thread(target=_reader_thread, args=(proc.stderr, err_buf, qlines, "ERR", last_output), daemon=True)
    t_out.start(); t_err.start()

    # monitor loop
    killed = False
    while True:
        if proc.poll() is not None:
            break
        now = time.time()
        if now - start > hard_timeout:
            print(f"[KILL][HARD] {name} > {hard_timeout}s")
            kill_tree(proc); killed = True; break
        if now - last_output["t"] > idle_timeout:
            print(f"[KILL][IDLE] {name} sin output > {idle_timeout}s")
            kill_tree(proc); killed = True; break
        time.sleep(1)

    t_out.join(timeout=1); t_err.join(timeout=1)

    info["stdout"] = "".join(out_buf)
    info["stderr"] = "".join(err_buf)
    info["returncode"] = proc.returncode if not killed else -1
    info["ok"] = (info["returncode"] == 0)
    status = 'OK' if info['ok'] else f"ERROR ({info['returncode']})"
    print(f"[DONE] {name}: {status}")
    return info

# ---------- discover outputs ----------
def parse_csv_paths_from_logs(text: str):
    hits = []
    for m in CSV_PATH_RE.finditer(text or ""):
        p = m.group("path").strip().strip('"').strip("'").replace("file:///", "").rstrip(".,);")
        hits.append(Path(p))
    uniq, seen = [], set()
    for p in hits:
        s = str(p)
        if s not in seen:
            seen.add(s); uniq.append(p)
    return uniq

def discover_outputs_from_results(results):
    out = []
    for r in results:
        script_dir = Path(r["path"]).parent
        for p in parse_csv_paths_from_logs(r.get("stdout","")) + parse_csv_paths_from_logs(r.get("stderr","")):
            out.append(p if p.is_absolute() else (script_dir / p).resolve())
    out = [p for p in out if p.exists()]
    uniq, seen = [], set()
    for p in out:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s); uniq.append(p.resolve())
    return uniq

def scan_local_resultados(around: Path):
    return [h.resolve() for h in (around / "Resultados").glob("*.csv") if h.exists()]

def recursive_scan_outputs(patterns):
    hits = []
    for pat in patterns:
        hits += [Path(p) for p in glob.glob(pat, recursive=True)]
    hits = [p.resolve() for p in hits if p.exists()]
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    uniq, seen = [], set()
    for p in hits:
        s = str(p)
        if s not in seen:
            seen.add(s); uniq.append(p)
    return uniq

def guess_dataset_name(path: Path):
    low = str(path).lower()
    for key, ds in DATASET_HINTS.items():
        if key in low: return ds
    parent = path.parent.name.lower()
    return parent[:31] if parent else path.stem[:31]

def read_csv_safely(path: Path):
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except Exception:
        for sep in (";", ",", "\t", "|"):
            try: return pd.read_csv(path, sep=sep, low_memory=False)
            except Exception: continue
    return None

def build_datasheet(paths):
    ok, fail, combined = [], [], []
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xlw:
        for p in paths:
            df = read_csv_safely(p)
            label = guess_dataset_name(p)
            if df is None or df.empty:
                fail.append(f"{label}::{p.name}"); continue
            sheet_df = df.head(100_000)
            sheet_name = (label[:25] + "_" + p.stem[-5:])[:31]
            try:
                sheet_df.to_excel(xlw, index=False, sheet_name=sheet_name)
                ok.append(f"{label}::{p.name}")
            except Exception:
                fail.append(f"{label}::{p.name}")
            dfi = df.copy()
            dfi.insert(0, "dataset", label)
            dfi.insert(1, "source_file", str(p.relative_to(PROJECT_ROOT)))
            combined.append(dfi)
    if combined:
        pd.concat(combined, ignore_index=True, sort=False).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return ok, fail

def heartbeat(flag, names_fn):
    while flag.is_set():
        print("[HB] Aún ejecutando:", ", ".join(sorted(names_fn())) or "ninguno")
        time.sleep(30)

# ---------- main ----------
def main():
    print(f"=== RUN ALL v7 (threads) ===\nDetected PROJECT_ROOT: {PROJECT_ROOT}")

    # pedir filtros (solo tráfico)
    fecha, meses, zona, hora, margen = "", "2", "", "", "60"
    try:
        fecha, meses, zona, hora, margen = ask_filters_once()
    except Exception:
        pass
    stdin_trafico = "\n".join([fecha, meses, zona, hora, margen]) + "\n"
    env_common = {"FILTER_DATE": fecha, "FILTER_MONTHS_BACK": meses, "FILTER_LOCATION": zona,
                  "FILTER_HOUR": hora, "FILTER_HOUR_MARGIN": margen, "PYTHONUNBUFFERED":"1"}

    scripts = discover_scripts()
    if not scripts:
        print("No se encontraron scripts."); return

    # heartbeat
    running = {}
    flag = threading.Event(); flag.set()
    hb = threading.Thread(target=heartbeat, args=(flag, lambda: list(running.keys())), daemon=True); hb.start()

    # ejecutar
    results = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(scripts))) as ex:
        fut_map = {}
        for (name, path) in scripts:
            hard = SCRIPT_TIMEOUTS.get(name, DEFAULT_HARD)
            idle = DEFAULT_IDLE
            stdin_text = stdin_trafico if name == "trafico_historico" else ("\n"*6)  # Enters por si algún script pregunta
            fut = ex.submit(run_script, name, path, stdin_text, hard, idle, env_common)
            fut_map[fut] = name
            running[name] = True
        for f in as_completed(fut_map):
            name = fut_map[f]
            try:
                res = f.result()
                results.append(res)
            finally:
                running.pop(name, None)

    flag.clear(); hb.join(timeout=1)

    # log
    with open(RUN_LOG, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"[{r['name']}] rc={r['returncode']} ok={r['ok']} path={r['path']}\n")
            if r["stdout"]:
                fh.write("---- stdout (tail) ----\n" + r["stdout"][-4000:] + "\n")
            if r["stderr"]:
                fh.write("---- stderr (tail) ----\n" + r["stderr"][-4000:] + "\n")
            fh.write("\n")
    print(f"[LOG] {RUN_LOG}")

    # recolectar salidas
    from_logs = discover_outputs_from_results(results)
    near = []
    for r in results: near += scan_local_resultados(Path(r["path"]).parent)
    scanned = recursive_scan_outputs(RECURSIVE_SCAN_PATTERNS)

    all_paths, seen = [], set()
    for p in from_logs + near + scanned:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key); all_paths.append(p.resolve())

    if not all_paths:
        print("No se encontraron salidas. Revisa el log."); return

    ok, fail = build_datasheet(all_paths)
    print("\n=== RESUMEN ===")
    print(f"Excel: {OUT_XLSX}")
    print(f"CSV unificado: {OUT_CSV}")
    print(f"Sheets OK: {len(ok)}")
    if fail:
        print(f"Sheets fallidas: {len(fail)} (mira {RUN_LOG})")

if __name__ == "__main__":
    main()