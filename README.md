# Sistema de Detección de Incidencias (PC1)

**Proyecto de Computación I — Curso 2025-2026**
**Universidad Europea de Madrid**
**Doble Grado en Diseño de Videojuegos e Ingeniería Informática**

---

## Qué hace

Sistema en Python que:

1. **Extrae** datos abiertos de movilidad urbana de Madrid (tráfico, accidentes,
   calidad del aire, emergencias, obras y clima).
2. **Unifica** todas las fuentes en un único dataset por día-distrito.
3. **Entrena** modelos de Machine Learning (Random Forest, Decision Tree, SVM)
   para predecir si un día-distrito tendrá una incidencia "alta"
   (por encima del percentil 90 del histórico).
4. **Predice** la probabilidad de incidencia para distritos y fechas concretos.

El sistema alimenta a PC2 (la aplicación web EcoTraffic), que muestra las
predicciones en el mapa de Madrid.

---

## Estructura

```
Proyecto-de-Computacion-I/
├── Accidentes_Scripts/    Scraper de accidentes (Ayto. Madrid)
├── CalidadAire_Scripts/   Scraper de calidad del aire (Ayto. Madrid)
├── Clima_Scripts/         Scraper meteorológico (AEMET)
├── Emergencias_Scripts/   Scraper de emergencias (Bomberos/SAMUR)
├── Obras_Scripts/         Scraper de obras planificadas
├── Trafico_Scripts/       TraficoHistorico.py (histórico, alimenta el dataset)
│                          TraficoScrapper.py  (tiempo real, snapshot del día)
├── main/
│   └── main.py            Une todos los CSV en dataset_unificado.csv
├── modelos_entrenamiento/
│   ├── utils.py           Carga, split temporal, métricas comunes
│   ├── random_forest.py   Entrenamiento Random Forest + GridSearchCV
│   ├── decisiontree.py    Entrenamiento Árbol de Decisión + GridSearchCV
│   ├── svm.py             Entrenamiento SVM + GridSearchCV
│   └── prediccion.py      Predicción puntual desde un modelo guardado
├── modelos_guardados/     (generado) Modelos .pkl entrenados
├── Resultados/            (generado) dataset_unificado.csv
├── VisualApp.py           GUI de escritorio (CustomTkinter)
├── generar_predicciones.py Vuelca predicciones a la BD de PC2
├── run_all.py             Pipeline completo en un solo comando
├── requirements.txt
├── .env.example           Plantilla de variables de entorno
└── README.md
```

Todos los scrapers descargan los datos directamente del portal abierto del
Ayuntamiento (datos.madrid.es) y de AEMET — no necesitas CSV manuales
descargados a mano.

Cada carpeta `*_Scripts/` genera su propio `Resultados/datasheet_*.csv`
que `main.py` consume para construir el dataset unificado.

---

## Cómo se usa

### 1. Preparación del entorno

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate            # Linux/Mac
# .venv\Scripts\activate              (Windows)

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables sensibles (API key de AEMET, etc.)
cp .env.example .env
# Edita .env y pon tu clave AEMET_API_KEY
```

### 2. Ejecución del pipeline completo

```bash
# Lo hace todo: scrapers + unificación + entrenamiento de 3 modelos
python run_all.py

# Solo unificación y entrenamiento (reutiliza scrapers ya descargados)
python run_all.py --skip-scrapers

# Solo scrapers + ETL, sin entrenar
python run_all.py --solo-etl

# Solo entrenar (asume que el dataset ya existe)
python run_all.py --solo-modelos

# Entrenar para otro target
python run_all.py --target "Calidad Aire"
```

### 3. Ejecución manual paso a paso

```bash
# Cada scraper genera su CSV en su carpeta Resultados/
python Accidentes_Scripts/Accidentes.py
python CalidadAire_Scripts/CalidadAire.py
python Clima_Scripts/Clima.py
python Emergencias_Scripts/emergencias_scraper.py
python Obras_Scripts/Obras.py
python Trafico_Scripts/TraficoHistorico.py   # alimenta el dataset
python Trafico_Scripts/TraficoScrapper.py    # snapshot del día (opcional)

# Une todas las fuentes en Resultados/dataset_unificado.csv
python main/main.py

# Lanza la GUI para entrenar y predecir interactivamente
python VisualApp.py
```

> **Tráfico histórico:** el dataset original son ~150 ZIPs mensuales de
> 80-90 MB cada uno. Por defecto `TraficoHistorico.py` baja los 3 meses más
> recientes (variable `MESES_RECIENTES` al principio del archivo). Súbela
> si quieres más cobertura histórica — cada mes añadido es ~5 minutos de
> descarga + procesamiento.

### 4. Integración con PC2

```bash
# Genera predicciones para los próximos 7 días y las inserta
# en la tabla `predictions` de la base de datos de PC2.
python generar_predicciones.py --dias 7
```

---

## Datos

### Fuentes y columnas

| Fuente            | Origen                  | Granularidad      | Columna clave           |
|-------------------|-------------------------|-------------------|-------------------------|
| Accidentes        | Datos Abiertos Madrid   | día / distrito    | `total_de_accidentes`   |
| Calidad del aire  | Datos Abiertos Madrid   | día / distrito    | `valor_calidad_aire`    |
| Clima             | AEMET Open Data         | día / distrito    | `temp_media`, `precipitacion`... |
| Emergencias       | SAMUR + Bomberos        | día / distrito    | `cantidad_emergencias`  |
| Obras             | Datos Abiertos Madrid   | día / distrito    | `obras_activas`         |
| Tráfico           | Informo Madrid          | día / distrito    | `trafico_medio`         |

### Features añadidas por `main.py`

- `dia_semana` (0=lunes, 6=domingo)
- `es_finde` (0/1)
- `es_festivo` (0/1, festivos oficiales de la Comunidad de Madrid)

### Targets

Para cada uno de los 3 objetivos (`Accidentes`, `Calidad Aire`, `Emergencias`)
se crea una columna binaria `target_*`:

> `target = 1` si el valor del día-distrito está por encima del **percentil 90**
> del histórico (top 10% más alto). En otro caso, `target = 0`.

Antes se usaba la mediana, lo que hacía que el 50% del dataset fuera
"incidencia=1" y el modelo no aprendiera nada útil.

---

## Modelos y métricas

Los tres modelos comparten estructura:

1. **Split temporal** (`TimeSeriesSplit`): el conjunto de test son siempre
   las fechas más recientes. Esto evita el data leakage de mezclar pasado
   y futuro en una división aleatoria.
2. **Búsqueda de hiperparámetros** con `GridSearchCV` y validación cruzada
   temporal (3 splits).
3. **Métricas reportadas**: accuracy, precision, recall, F1, ROC-AUC y
   matriz de confusión. (En la versión anterior se reportaban MAE/MSE,
   que son métricas de regresión y no tienen sentido para clasificación.)

| Modelo         | Hiperparámetros que se buscan       |
|----------------|--------------------------------------|
| Random Forest  | `n_estimators`, `max_depth`, `min_samples_split` |
| Decision Tree  | `max_depth`, `min_samples_split`     |
| SVM (rbf)      | `C` (regularización)                 |

---

## Equipo

| Nombre                                  | Rol                              |
|------------------------------------------|----------------------------------|
| **Paula Romero Gallart**                 | Project Manager y coordinación   |
| **Eddy Misael Abisai Catú de León**      | Responsable LORCA y validación   |
| **Mahsa Simaei**                         | Analista de datos                |
| **Mateo Galvis Guayana**                 | Programador Python / ETL         |

**Profesor:** Borja Monsalve Piqueras

---

## Referencias

- Datos Abiertos del Ayuntamiento de Madrid — https://datos.madrid.es
- AEMET Open Data — https://opendata.aemet.es
- Informo Madrid — https://informo.madrid.es
- scikit-learn — https://scikit-learn.org
- pandas — https://pandas.pydata.org
