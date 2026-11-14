# devalign-ml: Pipeline de Clustering de Habilidades Técnicas

Este repositorio contiene el pipeline de Machine Learning dedicado al agrupamiento de ofertas de trabajo según sus habilidades técnicas requeridas, utilizando **UMAP** para reducción de dimensionalidad y **HDBSCAN** para clustering no supervisado.

Este proyecto separa la experimentación de datos y el entrenamiento de modelos de la lógica principal del backend (`devalign-api`), siguiendo las mejores prácticas de Clean Architecture y ML Engineering.

---

## Estructura del Repositorio

```
devalign-ml/
├── data/                       # Almacenamiento local de caché de datos (gitignored)
├── models/                     # Modelos y gráficos generados (gitignored)
├── notebooks/                  # Cuadernos interactivos Jupyter
│   └── clustering_exploration.ipynb  # Cuaderno para Google Colab y exploración local
├── src/                        # Código base modular (Clean Architecture)
│   ├── domain/                 # Interfaces y contratos del dominio (SOLID DIP/OCP)
│   │   └── interfaces.py       # ClusterNamingStrategy protocol
│   ├── services/               # Lógica de aplicación y estrategias
│   │   ├── naming_service.py   # TaxonomyNamingStrategy y LLMNamingStrategy
│   │   └── salary_service.py   # SalaryNormalizer (divisas y market tiers)
│   ├── config.py               # Hiperparámetros de UMAP/HDBSCAN y llaves API
│   ├── data_loader.py          # Carga de datos de Supabase y vectorización IDF
│   ├── clustering.py           # Pipeline algorítmico y cálculo del Silhouette Score
│   ├── naming.py               # Árbol de decisiones y taxonomía semántica
│   └── utils.py                # Visualizaciones con matplotlib/seaborn
├── tests/                      # Suite de pruebas automatizadas (pytest)
│   ├── test_clustering_pipeline.py
│   ├── test_naming_service.py
│   └── test_salary_service.py
├── validation/                 # Arnés experimental para investigación académica (RQ1, RQ2, RQ3)
├── pyproject.toml              # Configuración de pytest y linter ruff
├── requirements.txt            # Dependencias del sistema
└── run_pipeline.py             # Script CLI de ejecución principal
```

---

## Cómo Ejecutar Localmente

### Requisitos previos:
* Python 3.10 o superior (o [uv](https://github.com/astral-sh/uv) para instalación ultrarrápida).
* Visual C++ Build Tools (en Windows, necesario para compilar la librería C de HDBSCAN).

### Pasos:
1. **Crear e iniciar el entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # En macOS/Linux:
   source .venv/bin/activate
   ```
2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecutar Pruebas y Linter:**
   ```bash
   # Correr suite de pruebas unitarias
   pytest tests/ -v

   # Verificar formato y reglas de código
   ruff check .
   ```
4. **Configurar variables de entorno:**
   Copia el archivo de plantilla `.env.example` como `.env` e introduce tus llaves reales:
   ```bash
   cp .env.example .env
   ```
5. **Ejecutar el Pipeline:**
   ```bash
   # Modo Simulación (Dry-Run): Entrena el modelo, calcula Silhouette Score y guarda el gráfico
   python run_pipeline.py

   # Modo Despliegue Seguro (Safe Deploy): Sincroniza clústeres in-place sin borrar datos de usuarios
   python run_pipeline.py --deploy
   ```

---

## Metodología del Algoritmo

1. **Representación Vectorial:** Se obtienen los embeddings de habilidades de Supabase (vectores Voyage AI de 1024 dimensiones) y se realiza una agregación promedio ponderada por IDF suavizado (con penalización del 20% a habilidades comodín presentes en >35% de ofertas) para representar a cada vacante en el espacio semántico.
2. **Reducción UMAP:** Se proyecta el espacio de 1024d a 15d para mitigar la maldición de la dimensionalidad, usando la métrica de similitud del coseno.
3. **Clustering HDBSCAN:** Agrupa las vacantes basándose en la densidad espacial en la dimensión reducida.
4. **Evaluación de Precisión:** Calcula el **Silhouette Score** sobre los puntos agrupados (excluyendo el ruido) para validar la separación de los clústeres.
5. **Tratamiento del Ruido:** Las ofertas marcadas como ruido (`-1`) se conservan deliberadamente como no asignadas para evitar la contaminación semántica de los centroides de especialidad.
6. **Nombrado de Especialidades:** Se extraen las habilidades más distintivas de cada clúster y se resuelven mediante `TaxonomyNamingStrategy` (árbol determinista sin costo de API ni deriva semántica), con soporte desacoplado a `LLMNamingStrategy`.
7. **Despliegue Seguro (Zero Data Loss):** Los clústeres descubiertos se emparejan con los existentes en la base de datos mediante similitud de coseno sobre centroides ($\ge 0.88$). Las tablas transaccionales de usuarios (`diagnostics`, `roadmaps`) permanecen 100% intactas.
