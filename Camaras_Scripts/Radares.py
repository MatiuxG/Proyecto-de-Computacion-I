import pandas as pd
import unicodedata
import re
import sys

# Path to your CSV file
CSV_PATH = "D:\Repositorios\Proyecto Computacion 1\Proyecto-de-Computacion-I\Camaras_Scripts\RADARES FIJOS_vDTT.csv"

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def main():
    print(f"[Carga] Leyendo {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    print(f"[Info] {df.shape[0]} filas, columnas: {list(df.columns)}")

    # --- Input filters ---
    street_query = input("Filtrar por Ubicacion (ej: 'M-30' o 'CASTELLANA'), Enter para omitir: ").strip()
    road_query = input("Filtrar por Carretara o vial (ej: 'M-30' o 'A5'), Enter para omitir: ").strip()
    vel_query = input("Filtrar por Velocidad límite (ej: 50, 70, 90), Enter para omitir: ").strip()

    mask = pd.Series([True] * len(df))

    if street_query:
        target = normalize_text(street_query)
        col_norm = df['Ubicacion'].astype(str).map(normalize_text)
        mask &= col_norm.str.contains(re.escape(target), na=False)
    if road_query:
        target = normalize_text(road_query)
        col_norm = df['Carretara o vial'].astype(str).map(normalize_text)
        mask &= col_norm.str.contains(re.escape(target), na=False)
    if vel_query:
        try:
            vel = int(vel_query)
            mask &= (df['Velocidad límite'] == vel)
        except Exception:
            print("Velocidad límite debe ser un número")
            sys.exit(1)

    filtered = df[mask]
    print(f"[Filtrado] {filtered.shape[0]} filas encontradas")

    # --- Save output ---
    slug_parts = []
    if street_query:
        slug_parts.append(re.sub(r"[^a-z0-9]+", "_", normalize_text(street_query)).strip("_"))
    if road_query:
        slug_parts.append(re.sub(r"[^a-z0-9]+", "_", normalize_text(road_query)).strip("_"))
    if vel_query:
        slug_parts.append(vel_query)
    slug = "__".join(slug_parts) if slug_parts else "sin_filtros"
    out_file = f"radares_filtrado_{slug}.csv"
    filtered.to_csv(out_file, index=False, sep=";", encoding="utf-8")
    print(f"[Guardado] {out_file}")
    print(filtered.head(10))

if __name__ == "__main__":
    main()