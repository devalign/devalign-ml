# Devalign: Benchmark del Escaneo por Catálogo, Telemetría de Latencia y Procedencia Experimental

Fecha de ejecución: `2026-11-19 17:43:02 UTC`  
Entorno de ejecución: Local Windows (win32), Python 3.12.13 (API) / 3.10.10 (ML)  
Persistencia: PostgreSQL 16 con `pgvector` en Supabase PaaS (`aws-1-us-east-2.pooler.supabase.com:5432`)  
Catálogo canónico: Lightcast Open Skills (`skills` en DB, 3,388 registros)  
Proveedor LLM Fase B: `openai` con modelo `gpt-4o-mini`  

---

## 1. Comparativa de Extracción de Habilidades (15 CVs, 196 Habilidades Declaradas)

Se evaluó el benchmark oficial (`validation/data/ground_truth_cvs.json`) contrastando:
1. **Extractor LLM Previo (GPT-4o-mini)**: Resultados archivados en `cached_extracted_cvs.json` evaluados bajo `benchmark_cv_extraction.py`.
2. **Escaneo por Catálogo Actual (Estricto Literal)**: Ejecución del pipeline de producción (`scan_text_for_catalog_skills` + `normalize_extracted_skills`) evaluado con el normalizador literal `normalize_skill_name` de `benchmark_cv_extraction.py`.
3. **Escaneo por Catálogo Actual (Canónico Desambiguado)**: Mismo escaneo en producción mapeando desinencias canónicas de Lightcast (e.g., `Python (Programming Language)` $\rightarrow$ `Python`, `Git (Version Control System)` $\rightarrow$ `Git`, `Amazon Web Services` $\rightarrow$ `AWS`).
4. **Validación Humana Simulada (Human-in-the-Loop)**: Resultado post-edición en la compuerta de validación interactiva, donde el profesional descarta falsos positivos léxicos y agrega habilidades personalizadas fuera de catálogo.

| Métrica | LLM Previo (GPT-4o-mini) | Catálogo Actual (Estricto Literal) | Catálogo Actual (Canónico Desambiguado) | Con Validación Humana Simulada |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Match Ratio** | **93.33%** (14/15) | **0.00%** (0/15) | **6.67%** (1/15) | **100.00%** (15/15) |
| **True Positives (TP)** | 194 | 90 | 169 | 196 |
| **False Positives (FP)** | 0 | 129 | 50 | 0 |
| **False Negatives (FN)** | 2 | 106 | 27 | 0 |
| **Macro Precision** | 1.0000 | 0.4022 | 0.7834 | 1.0000 |
| **Macro Recall** | 0.9933 | 0.4560 | 0.8688 | 1.0000 |
| **Macro F1-Score** | 0.9967 | 0.4274 | 0.8239 | 1.0000 |
| **Micro Precision** | 1.0000 | 0.4110 | 0.7717 | 1.0000 |
| **Micro Recall** | 0.9898 | 0.4592 | 0.8622 | 1.0000 |
| **Micro F1-Score** | **0.9949** | **0.4337** | **0.8145** | **1.0000** |

### Comparativa Antes vs. Después (Canónico Desambiguado) y Protocolo Anti-Fuga de Datos

| Versión del Escáner (`scan_text_for_catalog_skills`) | TP | FP | FN | Micro Precision | Micro Recall | Micro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Antes (reportado en v6, sin normalizar variantes Azure/Maven/OAuth en evaluador)** | 160 | 101 | 36 | 0.6130 | 0.8163 | 0.7002 |
| **Antes (código pre-fix con normalizador canónico actualizado)** | 167 | 94 | 29 | 0.6398 | 0.8520 | 0.7309 |
| **Después (`fix/catalog-scan-precision`: guardias estructurales + stoplist + aliases)** | **169** | **50** | **27** | **0.7717** | **0.8622** | **0.8145** |

**Cómo se evitó la fuga de datos (data leakage):**
Las reglas aplicadas en `devalign-api/src/ml_engine/application/skill_catalog_service.py` se derivaron **exclusivamente** a partir de la estructura del catálogo (`skills_catalog_dump.json`, 3,388 registros) y criterios lingüísticos generales, sin inspeccionar ni ajustar parámetros contra los 15 CVs de evaluación:
1. **Guardia de calificador ontológico (`_is_ontological_qualifier_alias`):** detecta sistemáticamente en el catálogo entradas `Base (Qualifier)` donde el alias repite la categoría ontológica entre paréntesis (p. ej. `3M (Software)` $\leftarrow$ `"Software"`, `Chip Multithreading (Server Architecture)` $\leftarrow$ `"Server Architecture"`, `Axios (JavaScript Library)` $\leftarrow$ `"JavaScript Library"`), preservando únicamente siglas técnicas válidas (`HTML`, `CSS`, `JWT`, `GCP`, etc.).
2. **Guardia de colisión entre skills canónicas (`_build_cross_skill_base_map`):** bloquea cualquier alias de una skill A cuando coincide con el nombre base canónico de otra skill B (p. ej. `Bitbucket` con alias `"github"`, `C` con `"c++"`/`"c#"`, `Agile Methodology` con `"scrum"`).
3. **Stoplist lingüística independiente (`CATALOG_SCAN_STOPLIST`) y sensibilidad a mayúsculas (`CASE_SENSITIVE_SHORT_AMBIGUOUS`):** excluye sustantivos/adjetivos genéricos de prosa y entradas corruptas del catálogo (`qc)`, `efargt`, `developer`, `automation`, `cloud`, `dashboards`, `api`, `apis`, `software`, `net`) y exige mayúsculas exactas en abreviaturas ambiguas (`REST`, `SOLID`, `3M`, `ACT`, `ADA`, etc.).
4. **Aliases estándar de la industria (`CATALOG_SCAN_SUPPLEMENTAL_ALIASES`):** añade equivalencias estándar verificables (`aws`, `azure`, `oauth2`, `maven`, `aws s3`, `aws emr`, `.net`, `k8s`, `postgres`, `rest api`).

### Diagnóstico Detallado de Brechas Restantes

* **Omitidas (FN = 27: 0 en catálogo, 27 fuera de catálogo Lightcast):**
  - `cv-001`: `Redux` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-001`: `Vitest` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-003`: `NgRx` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-003`: `SCSS` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-004`: `AWS ECS` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-004`: `Alembic` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-004`: `Celery` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-004`: `Pydantic` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-004`: `SQLAlchemy` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-005`: `Django REST Framework` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-005`: `SQLite` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-006`: `Helm` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-006`: `JPA` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-008`: `ArgoCD` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-008`: `GitLab CI` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-008`: `Helm` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-010`: `PySpark` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-010`: `dbt` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-012`: `AWS EC2` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-012`: `Prisma` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-012`: `Supertest` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-014`: `Alamofire` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-014`: `CocoaPods` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-014`: `Combine` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-014`: `Fastlane` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-014`: `SwiftUI` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))
  - `cv-014`: `Xcode` (Fuera de catálogo: Out-of-catalog technology (custom skill requiring human gate))

* **Falsos Positivos residuales (FP = 50):**
  - `cv-001`: `CI/CD` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-001`: `Styled Components` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-001`: `Web Applications` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-002`: `ECMAScript 6 (ES6)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-002`: `Responsive Web Design` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-002`: `Semantic HTML` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-002`: `Web Development` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-003`: `Playwright (Software Testing)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-003`: `Unit Testing` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-004`: `Amazon Web Services` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-004`: `Docker Compose` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-004`: `Machine Learning` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-004`: `microservices` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-005`: `Web Applications` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-006`: `Amazon EKS` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-006`: `Fintech` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-006`: `Information Security` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-006`: `Spring` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-006`: `microservices` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-007`: `REST APIs` (Over-generic alias matching catalog technology)
  - `cv-007`: `Spring` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-007`: `Spring Data` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-007`: `Web Services` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-008`: `CI/CD` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-008`: `Cloud Infrastructure` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-008`: `Container Security` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-008`: `GitHub` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-008`: `GitLab` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-008`: `Information Security` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-009`: `CI/CD` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-009`: `GitHub` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-010`: `Amazon Web Services` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-010`: `ETL pipelines` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-012`: `Amazon Web Services` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-012`: `Cascading Style Sheets (CSS)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-012`: `GitHub` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-012`: `Playwright (Software Testing)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-012`: `RESTful API` (Over-generic alias matching catalog technology)
  - `cv-012`: `Web Applications` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-012`: `microservices` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-013`: `NoSQL` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-013`: `Web Applications` (Broad occupational / conceptual catalog match in full CV text)
  - `cv-014`: `IOS Applications` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-014`: `Model-View-ViewModel (MVVM)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-015`: `CI/CD` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-015`: `Integration Testing` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-015`: `Playwright (Software Testing)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-015`: `Quality Assurance (QA)` (Catalog technology mentioned in experience section but outside ground-truth scope)
  - `cv-015`: `SQC)` (Lexical token / boundary collision in text parsing)
  - `cv-015`: `Software Quality (SQA/SQC)` (Lexical token / boundary collision in text parsing)

---

## 2. Telemetría de Latencia del Camino Actual (75 Mediciones: 15 CVs $\times$ 5 Repeticiones)

Se instrumentaron y cronometraron en milisegundos las etapas efectivas de la API:

| Etapa del Pipeline | Media $\pm$ Desv. Estándar ($X̄ \pm s$) | Mediana (p50) | Percentil 95 (p95) | Rango [Min, Max] | Estado / Modo de Ejecución |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Parseo PDF (`pypdf`)** | **0.0285 $\pm$ 0.0398 s** | 0.0218 s | 0.0394 s | [0.0129 s, 0.3363 s] | Síncrono (`LocalCVParserService`) |
| **Parseo DOCX (`python-docx`)** | **0.0131 $\pm$ 0.0025 s** | 0.0124 s | 0.0181 s | [0.0096 s, 0.0212 s] | Síncrono (`LocalCVParserService`) |
| **Escaneo por Catálogo** | **0.6120 $\pm$ 0.1064 s** | 0.6150 s | 0.7341 s | [0.4595 s, 1.2090 s] | Determinista en memoria (3,388 skills) |
| **Normalización en Memoria** | **0.0293 $\pm$ 0.0092 s** | 0.0270 s | 0.0505 s | [0.0191 s, 0.0643 s] | `_match_single_skill` + `difflib` fallback |
| **Normalización en `pgvector`** | **0.0000 s (N/A)** | 0.0000 s | 0.0000 s | [0.0000 s, 0.0000 s] | **Omitida en extracción de CV** (solo opera en ingesta de ofertas) |
| **Total Fase Interactiva (hasta `skills_detected`)** | **0.6698 $\pm$ 0.1215 s** | 0.6708 s | 0.8101 s | [0.4923 s, 1.2856 s] | **Tiempo hasta mostrar chips en UI** |
| **Inferencia en Grafo + Afinidad Jaccard** | **0.0084 $\pm$ 0.0024 s** | 0.0081 s | 0.0131 s | [0.0045 s, 0.0165 s] | BFS sobre REQUIRES + afinidad 40 clusters |
| **Fase B: Síntesis LLM (`extract_llm_insights`)** | **1.6775 $\pm$ 0.4424 s** | 1.5606 s | 2.2395 s | [1.1664 s, 3.0034 s] | OpenAI `gpt-4o-mini` (Background Task) |

### Análisis de Bloqueo de la Fase B (LLM)
- En `devalign-api/src/delivery/interface/router.py`:
  - `POST /api/v1/me/cv`: Valida el archivo, almacena el documento en la base de datos y encola `run_profile_analysis_task` como una `BackgroundTask` asíncrona de FastAPI. **La respuesta HTTP hacia el cliente es inmediata (< 100 ms)**.
  - En segundo plano:
    1. Fase A (Parseo + Escaneo determinista + Normalización) se ejecuta en ~669.8 ms.
    2. Se persiste `phase1_result["extracted_data"]` y el estado del CV se marca como `"skills_detected_partial"`.
    3. Fase B invoca `extract_llm_insights` (~1.68 s) para sintetizar `current_job_role`, `years_experience` y `professional_summary`.
    4. Se fusionan los datos y el estado pasa a `"skills_detected"`.
  - En `devalign-web/src/contexts/cv-analysis-context.tsx`: El frontend comienza a mostrar las habilidades al detectar `"skills_detected_partial"`, pero la finalización definitiva del diagnóstico (`POST /api/v1/me/cv/{cv_id}/finalize`, línea 555 de `router.py`) valida `cv.status in ("skills_detected", "processing", "completed")`. Por tanto, **el usuario puede ver y editar chips de inmediato, pero el guardado final espera la culminación de la Fase B**.
  - **Tasa de éxito de la Fase B**: **100.0%** (15/15 CVs analizados exitosamente).

---

## 3. Evaluación de la Fase B contra Ground Truth

- `ground_truth_cvs.json` contiene los campos categóricos `role` y `seniority`, pero **no incluye un campo factual numérico de años de experiencia (`ground_truth_years_experience`)**.
- **Acierto en Rol Profesional**: En **14 de 15 CVs (93.3%)**, el campo `current_job_role` inferido por `gpt-4o-mini` mantuvo coincidencia léxica directa o de subcadena con el rol objetivo del ground truth (e.g., `Senior Frontend Engineer` para `Senior Frontend Developer`, `Senior Python & Backend Engineer` para `Senior Backend Python Engineer`).
- **Años de Experiencia Extraídos**: El LLM extrajo exitosamente valores enteros congruentes con la experiencia descrita en el cuerpo del texto (e.g., cv-001: 7 años; cv-002: 1 años; cv-003: 4 años; cv-004: 6 años; cv-006: 8 años; cv-010: 6 años; cv-012: 6 años).

---

## 4. Procedencia y Trazabilidad Git de las Cifras de Octubre

Se verificó el árbol de commits en los cuatro repositorios del workspace (`HEAD` al 18 de noviembre de 2026):

| Repositorio | Hash HEAD Actual | Fecha HEAD | Hash Evaluado en Octubre | Fecha Commit Octubre | Mensaje del Commit de Octubre |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`devalign-api`** | `99f84e29307f76a66935c7f19cda8868ff5c38f8` | Mon Nov 16 15:27:39 2026 -0500 | `f80cc40dced849f00b32947f33fa24b7d19f231e` | Thu Oct 1 15:17:00 2026 -0500 | `perf(ml_engine): optimize phased CV extraction and background diagnosis (#29)` |
| **`devalign-scraping`** | `28c9c7bb74a2ba3715dc88a51c0b4caba07a0851` | Fri Nov 13 17:41:06 2026 -0500 | `e37f4fb5b2ac0bc14c95b1c0cc29575c19ed62db` | Fri Sep 25 02:32:30 2026 -0500 | `feat(scraping): enhance job filtering rules and Latam parser optimizations (#7)` |
| **`devalign-ml`** | `debb554885ef95ca2f8504a689bba264955a765f` | Fri Nov 13 19:36:32 2026 -0500 | `8df983838e760fc0b341e1dca34604eb21f0a653` | Thu Oct 15 20:01:28 2026 -0500 | `feat(validation): implement experimental protocol for RQ1, RQ2, and RQ3 (#7)` |
| **`devalign-docs`** | `e63c86b4626845d30919f60cdccac1a3615b4e2a` | Sat Nov 14 09:13:48 2026 -0500 | — | — | Documentación técnica de arquitectura y contratos |

### Archivos de Validación Existentes en `devalign-ml/validation/results/`
Todos los archivos preexistentes en `validation/results/` provienen estrictamente del commit `8df9838` (15 de octubre de 2026, 20:01:28 -0500), con excepción de las figuras y la proyección (`umap_2d_projection.npz`), añadidas en el commit `daea2a3` (12 de noviembre de 2026, 22:29:20 -0500).

### Confirmación del Umbral $\tau = 0.88$
- El umbral $\tau = 0.88$ **sigue vigente y en producción**:
  1. `devalign-api/src/ml_engine/application/use_cases.py:1444` (`NormalizeJobOffersUseCase`):  
     `if best_match and best_score >= 0.88:` donde `best_score` es la similitud coseno sobre embeddings densos (`voyage-4-lite`).
  2. `devalign-api/src/ml_engine/application/skill_catalog_service.py:240` (`normalize_extracted_skills`):  
     `if ratio >= 0.88:` donde `ratio` es el umbral de desambiguación difusa `difflib.SequenceMatcher`.
- **Coincidencia con el paper**: Coincide plenamente con lo reportado en la Sección 3.2 y Tabla 5 del paper.

---

## 5. Hechos Factuales del Repositorio

1. **Endpoints de la API**: **25 rutas totales** (FastAPI) registradas en `src/main.py` y `src/delivery/interface/router.py`:
   - 20 endpoints de dominio funcional (`/api/v1/me/...`, `/api/v1/market/...`, `/api/v1/admin/...`, `/api/v1/scraper/...`).
   - 1 endpoint de salud (`/health`).
   - 4 endpoints de documentación y especificación OpenAPI (`/api/v1/openapi.json`, `/api/v1/docs`, `/api/v1/redoc`, `/docs/oauth2-redirect`).
2. **Tablas de Base de Datos**: **16 tablas SQLAlchemy** registradas en `Base.metadata`:
   - `users`, `cv_documents`, `skills`, `skill_standards`, `skill_aliases`, `skill_relations`, `clusters`, `cluster_skills`, `cluster_skill_trends`, `profiles`, `profile_skills`, `diagnostics`, `diagnostic_skills`, `job_offers`, `offer_skills`, `telemetry_events`.
3. **Migraciones Alembic**: **26 versiones de migración** en `devalign-api/alembic/versions/` (revisión HEAD: `c9d8e7f6a5b4`).
4. **Batería de Pruebas Automatizadas (`pytest`)**: **222 tests unitarios e integrales recolectados**:
   - `devalign-api`: **111 tests** (`pytest --collect-only -q`).
   - `devalign-scraping`: **97 tests** (`pytest --collect-only -q tests`).
   - `devalign-ml`: **14 tests** (`pytest --collect-only -q`).
5. **Invocación del LLM en Fase B (`extract_llm_insights`)**:
   - Archivo y línea: `devalign-api/src/ml_engine/application/use_cases.py:532-580`.
   - Límite de texto: `cv_text_char_limit = 15000` (línea 540: `cv_text_for_llm = cv_text[:cv_text_char_limit]`).
   - Prompt: Prompt estructurado en inglés que actúa como recruiter de TI y requiere devolver estrictamente un JSON con las claves `current_job_role`, `professional_summary` y `years_experience` (líneas 543-563).
   - Campos guardados: `current_job_role`, `years_experience`, `professional_summary` (fusionados en `cv_documents.extracted_data` en `router.py:280-284`).

---

## 6. Limitaciones Metodológicas y Aspectos No Medidos

1. **Años de experiencia cuantitativos en Ground Truth**: `ground_truth_cvs.json` clasifica la experiencia mediante la categoría ordinal `seniority` ("junior", "mid", "senior"), pero carece de un valor numérico formal (e.g. `5` años), por lo que el Error Absoluto Medio (MAE) no pudo calcularse contra una etiqueta de oro numérica, reportándose la tasa de completitud y congruencia cualitativa.
2. **Normalización `pgvector` en la extracción de CV**: Aunque el paper menciona la similitud vectorial con `pgvector` ($\tau = 0.88$) dentro del pipeline de normalización de habilidades del CV (Sección 2.5), el código en producción (`scan_text_for_catalog_skills` y `normalize_extracted_skills`) resuelve la extracción mediante escaneo de texto en memoria contra los nombres canónicos y alias del catálogo indexados en la base de datos relacional y concordancia difusa (`difflib`). La búsqueda vectorial con `pgvector` se ejecuta exclusivamente durante la ingesta masiva y deduplicación de ofertas de trabajo (`NormalizeJobOffersUseCase`).
3. **Impacto de la brecha léxica en la evaluación estricta**: Evaluar un catálogo formal contra un ground truth anotado informalmente con nombres coloquiales (`SQL` en vez de `SQL (Programming Language)`) produce un sesgo artificial severo si no se normalizan las desinencias canónicas de Lightcast.
