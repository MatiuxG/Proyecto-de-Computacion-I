# 🚦 Sistema de Detección de Incidencias  
**Proyecto de Computación I — Curso 2025-2026**  
**Universidad Europea de Madrid**  
**Doble Grado en Diseño de Videojuegos e Ingeniería Informática**

---

## 📖 Descripción del proyecto  
El **Sistema de Detección de Incidencias** es un proyecto de análisis de datos urbanos cuyo propósito es **anticipar incidencias de movilidad en la Comunidad de Madrid** mediante el uso de **datos abiertos** y **técnicas de aprendizaje automático**.  

El sistema integra información de diversas fuentes (Ayuntamiento de Madrid, AEMET, Informo Madrid, entre otras) a través de un proceso **ETL (Extract, Transform, Load)** que permite crear un **dataset limpio, validado y reproducible**.  
Sobre esa base, se implementará un modelo predictivo en **Python** para estimar la probabilidad de incidencias según variables como el tráfico, la meteorología o los eventos locales.  

---

## 🎯 Objetivos del proyecto  

### 🧠 Objetivo general  
Desarrollar un **prototipo funcional** que aplique procesos de **ETL y aprendizaje automático** para generar un listado de predicciones de incidencias urbanas en la Comunidad de Madrid.

### ⚙️ Objetivos específicos  
- Identificar y documentar **fuentes de datos abiertas** relevantes para la movilidad.  
- Implementar procedimientos automáticos de **extracción, limpieza y unificación** de datos.  
- Estandarizar los datos obtenidos para construir un **dataset reproducible y trazable**.  
- Aplicar algoritmos de **Machine Learning** (como Random Forest o KNN) para el análisis y predicción de incidencias.  
- Desarrollar un **prototipo en Python** que muestre los resultados y su validación.  
- Validar los resultados en el **clúster LORCA** de la universidad.  

---

## 🧩 Estructura del proyecto  

```
📁 Sistema_Deteccion_Incidencias/
├── 📂 data/
│   ├── raw/               # Datos originales descargados desde las fuentes
│   ├── processed/         # Datos limpios y unificados (dataset final)
│   └── metadata/          # Descripción de fuentes, formatos y metadatos
├── 📂 src/
│   ├── extract/           # Scripts de extracción (API, scraping, descargas)
│   ├── transform/         # Limpieza, normalización y validación de datos
│   ├── load/              # Carga final del dataset
│   └── main.py            # Script principal que ejecuta todo el flujo ETL
├── 📂 notebooks/           # Análisis y pruebas intermedias
├── 📂 reports/             # Informes y documentación (A1, A2, etc.)
└── README.md
```

---

## 🧰 Tecnologías y herramientas  

| Categoría | Herramienta / Librería |
|------------|------------------------|
| Lenguaje principal | **Python 3** |
| ETL y análisis | `pandas`, `NumPy` |
| APIs y scraping | `requests`, `BeautifulSoup`, `Selenium`, `Scrapy` |
| Visualización | `matplotlib` |
| Modelado futuro | `scikit-learn`, `RapidMiner` |
| Geoespacial | `GeoPandas`, `Shapely` |
| Infraestructura | **Clúster LORCA (UE Madrid)** |
| Datasets | Ayuntamiento de Madrid, AEMET, Informo Madrid |

---

## 🧪 Validación y pruebas  
El flujo ETL se valida mediante:
- **Integridad de extracción:** revisión de errores y logs en APIs.  
- **Detección de duplicados y valores nulos.**  
- **Consistencia temporal y geográfica** entre datasets.  
- **Reproducibilidad total:** ejecución completa del pipeline desde el script principal.  


---

## 👥 Equipo de trabajo  
| Nombre | Rol |
|---------|-----|
| **Paula Romero Gallart** |  Project Manager y coordinación general |
| **Eddy Misael Abisai Catú de León** | Responsable de LORCA y validación técnica |
| **Mahsa Simaei** | Analista de datos |
| **Mateo Galvis Guayana** | Programador Python  / ETL Developer |

**Profesor:** Borja Monsalve Piqueras  

---

## 📚 Referencias principales  
- Datos Abiertos del Ayuntamiento de Madrid — https://datos.madrid.es  
- AEMET Open Data — https://opendata.aemet.es  
- Informo Madrid — https://informo.madrid.es/#/realtime?panel=live  
- Microsoft Learn – ETL Process — https://learn.microsoft.com/es-es/azure/architecture/data-guide/relational-data/etl  
- pandas Documentation — https://pandas.pydata.org  
- GeoPandas Documentation — https://geopandas.org  

---

## 🏛️ Licencia  
Proyecto académico desarrollado en el marco de la asignatura **Proyecto de Computación I**.  
© 2025 Universidad Europea de Madrid — Todos los derechos reservados.
