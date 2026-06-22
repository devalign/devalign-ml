# devalign-ml: Pipeline de Clustering de Habilidades Técnicas

Este repositorio contiene el pipeline de Machine Learning dedicado al agrupamiento de ofertas de trabajo según sus habilidades técnicas requeridas, utilizando **UMAP** para reducción de dimensionalidad y **HDBSCAN** para clustering.

Este proyecto separa la experimentación de datos y el entrenamiento de modelos de la lógica principal del backend (`devalign-api`), siguiendo las mejores prácticas de ML Engineering.

---

## Estructura del Repositorio

```
devalign-ml/
├── data/                       # Carpeta local para almacenar la caché de datos (gitignored)
├── models/                     # Carpeta local para guardar los modelos entrenados y gráficos (gitignored)
├── notebooks/                  # Cuadernos interactivos Jupyter
│   └── clustering_exploration.ipynb  # Cuaderno optimizado para Google Colab y local
├── src/                        # Código base modular (paquete python)
│   ├── __init__.py
│   ├── config.py               # Hiperparámetros de UMAP/HDBSCAN y llaves API
│   ├── data_loader.py          # Carga de datos de Supabase y caché offline
│   ├── clustering.py           # Pipeline algorítmico y cálculo del Silhouette Score
│   └── utils.py                # Generación de gráficos y nombrado de clústeres mediante Groq LLM
├── .env.example                # Plantilla de variables de entorno
├── .gitignore                  # Exclusiones de Git (datos, claves, binarios)
├── requirements.txt            # Dependencias del sistema
└── run_pipeline.py             # Script ejecutable por terminal
```

---

## Cómo Ejecutar en Google Colab (Recomendado)

Google Colab ofrece un entorno en la nube preconfigurado para Linux que evita los problemas de compilación de C++ comunes al instalar `hdbscan` o `umap-learn` en Windows.

### Paso a paso:
1. **Sube este repositorio a GitHub:**
   Crea un nuevo repositorio en tu cuenta de GitHub (ej. `devalign-ml`) y sube todos los archivos de esta carpeta.
2. **Abre el Cuaderno en Colab:**
   - Entra a [Google Colab](https://colab.research.google.com/).
   - En el diálogo inicial, ve a la pestaña **GitHub**.
   - Ingresa tu usuario o la URL de tu repositorio, y selecciona el cuaderno `notebooks/clustering_exploration.ipynb`.
3. **Configura tus credenciales (Secrets):**
   - En la barra lateral izquierda de Colab, haz clic en el ícono de la llave (**Secrets / Secretos**).
   - Añade las siguientes variables con sus respectivos valores:
     - `SUPABASE_URL`
     - `SUPABASE_SERVICE_ROLE_KEY`
     - `GROQ_API_KEY`
   - Activa el interruptor de "Acceso al cuaderno" para cada clave.
4. **Configura la URL de tu repositorio:**
   En la primera celda del cuaderno, actualiza la variable `REPO_URL` con el enlace de tu repositorio de GitHub.
5. **Ejecuta el Cuaderno:**
   Ve a **Entorno de ejecución > Ejecutar todas**. Colab clonará tu repositorio, instalará las dependencias en la nube, procesará los vectores de Supabase, calculará el Silhouette Score (> 0.50) y mostrará el gráfico que necesitas para tu PPT.

---

## Cómo Ejecutar Localmente (VS Code / Terminal)

Si deseas ejecutar o desarrollar localmente, se recomienda usar un entorno virtual de Python.

### Requisitos previos:
*   Python 3.10 o superior.
*   (Opcional pero recomendado para Windows) Visual C++ Build Tools (necesario para compilar la librería C de HDBSCAN).

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
3. **Configurar variables de entorno:**
   Copia el archivo de plantilla `.env.example` como `.env` e introduce tus llaves reales:
   ```bash
   cp .env.example .env
   ```
4. **Ejecutar el Pipeline:**
   Puedes correr el flujo completo desde la terminal:
   ```bash
   # Modo Simulación (Dry-Run): Entrena el modelo, calcula la métrica y guarda el gráfico en models/
   python run_pipeline.py
   
   # Modo Despliegue (Deploy): Además de lo anterior, borra los clústeres antiguos en Supabase y sube los nuevos resultados
   python run_pipeline.py --deploy
   ```

---

## Metodología del Algoritmo

1. **Representación Vectorial:** Se obtienen los embeddings de habilidades de la base de datos (vectores de 1024 dimensiones) y se realiza una agregación promedio (mean-pooling) para representar a cada oferta laboral en el espacio semántico.
2. **Reducción UMAP:** Se proyecta el espacio de 1024d a 15d para mitigar la "maldición de la dimensionalidad", usando la métrica de similitud del coseno.
3. **Clustering HDBSCAN:** Agrupa las vacantes basándose en la densidad espacial en la dimensión reducida. Las muestras de baja densidad se marcan inicialmente como ruido (`-1`).
4. **Evaluación de Precisión:** Calcula el **Silhouette Score** sobre los puntos agrupados (excluyendo el ruido) para validar la separación de los clústeres.
5. **Reasignación de Ruido:** Los puntos marcados como ruido se reasignan al clúster cuyo centroide esté más cercano en el espacio reducido.
6. **Nombrado de Especialidades:** Se extraen las habilidades más frecuentes de cada clúster y se le consulta a la API de Groq (`llama-3.3-70b-versatile`) para asignar un nombre único y representativo en español (ej. *Desarrollador React Frontend*).
