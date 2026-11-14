"""
Punto de entrada de la validación del diagnóstico.

Uso:
    python -m validation.run                          # Usar datos locales cache
    python -m validation.run --no-cache                # Forzar fetch de Supabase
    python -m validation.run --deploy-summary          # Guardar resultados como CSV
    python -m validation.run --output-dir ./resultados # Directorio de salida personalizado
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validation")


def _load_clusters_from_db(args: argparse.Namespace) -> list:
    """Carga clústeres reales desde Supabase o cache local."""
    from src import config as ml_config
    from src.data_loader import load_raw_data, process_job_offer_vectors
    from src.clustering import ClusteringPipeline
    from src.services.naming_service import get_naming_strategy
    import numpy as np
    import pandas as pd

    # 1. Cargar datos
    logger.info("Loading data from Supabase / cache...")
    skills_data, offers_data = load_raw_data(use_cache=not args.no_cache)

    # 2. Procesar vectores
    logger.info("Processing job offer vectors...")
    valid_offers, alias_to_canonical = process_job_offer_vectors(skills_data, offers_data)

    if not valid_offers:
        logger.error("No valid offers found. Cannot run validation.")
        sys.exit(1)

    vectors = np.array([o["vector"] for o in valid_offers])
    logger.info(f"Loaded {len(vectors)} offer vectors (1024d each)")

    # 3. Clustering
    logger.info("Running UMAP + HDBSCAN...")
    pipeline = ClusteringPipeline()
    results = pipeline.run(vectors)

    if not results.get("non_noise_labels"):
        logger.warning("No clusters found by HDBSCAN. Cannot evaluate.")
        return []

    # 4. Construir clústeres como objetos del módulo de validación
    from validation.alignment import Cluster, Skill

    final_labels = results["final_labels"]
    non_noise_labels = results["non_noise_labels"]

    # Calcular frecuencias
    skill_counts: dict[str, int] = {}
    for offer in valid_offers:
        for s in offer["skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1
    all_skills = sorted(skill_counts.keys())

    # Matriz offer-skills
    matrix_rows = []
    for idx, offer in enumerate(valid_offers):
        row = {"offer_id": offer["id"], "cluster_label": final_labels[idx]}
        offer_set = set(offer["skills"])
        for sk in all_skills:
            row[sk] = 1 if sk in offer_set else 0
        matrix_rows.append(row)
    df = pd.DataFrame(matrix_rows)

    # Construir objetos Cluster y nombrar mediante estrategia unificada
    naming_strategy = get_naming_strategy()
    global_skill_freq = df[all_skills].mean()
    clusters = []
    for idx, k in enumerate(non_noise_labels):
        cluster_df = df[df["cluster_label"] == k]
        freqs = cluster_df[all_skills].mean()
        top_skills = freqs[freqs >= 0.15]

        centroid_skills = []
        for sk_name, freq in top_skills.items():
            weight = 3.0 if freq >= 0.6 else (2.0 if freq >= 0.3 else 1.0)
            centroid_skills.append(Skill(
                name=sk_name,
                domain_tags=[_infer_domain(sk_name)],
                weight=float(weight),
                frequency=float(freq),
                ict_score=7.0,
            ))

        distinctiveness = {}
        for s in top_skills.index[:5]:
            cf = float(freqs.get(s, 0.0))
            gf = float(global_skill_freq.get(s, 0.0))
            distinctiveness[s] = (cf / (gf + 1e-4)) * (cf ** 1.5)

        sorted_distinctive = dict(sorted(distinctiveness.items(), key=lambda x: x[1], reverse=True))
        name = naming_strategy.name_cluster(sorted_distinctive)

        clusters.append(Cluster(
            name=name,
            centroid_skills=centroid_skills,
            job_offer_count=len(cluster_df),
        ))

    return clusters


def _infer_domain(skill_name: str) -> str:
    """Heurística simple de dominio para skills sin domain_tags."""
    skill_lower = skill_name.lower()
    backend_keywords = ["python", "java", "go", "rust", "c#", "c++", "node", "express",
                        "spring", "django", "fastapi", "flask", "api", "microservice",
                        "rest", "graphql", "sql", "postgres", "mysql", "mongodb",
                        "redis", "docker", "kubernetes", "aws", "azure", "gcp"]
    frontend_keywords = ["react", "angular", "vue", "html", "css", "javascript",
                         "typescript", "next", "svelte", "ui", "web"]
    data_keywords = ["pandas", "numpy", "tensorflow", "pytorch", "scikit", "spark",
                     "hadoop", "etl", "data", "analytics", "machine learning", "ai"]

    for kw in frontend_keywords:
        if kw in skill_lower:
            return "frontend"
    for kw in data_keywords:
        if kw in skill_lower:
            return "data"
    for kw in backend_keywords:
        if kw in skill_lower:
            return "backend"
    return "general"


def _has_calibrated_clusters() -> bool:
    from src import config as ml_config
    dump_path = os.path.join(ml_config.DATA_DIR, "clusters_calibrated_dump.json")
    return os.path.exists(dump_path)


def _load_calibrated_clusters() -> list:
    import json
    from src import config as ml_config
    from validation.alignment import Cluster, Skill

    dump_path = os.path.join(ml_config.DATA_DIR, "clusters_calibrated_dump.json")
    with open(dump_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = []
    for item in data:
        centroid_skills = []
        for sk_name, freq in item.get("skills", {}).items():
            weight = 3.0 if freq >= 0.6 else (2.0 if freq >= 0.3 else 1.0)
            centroid_skills.append(
                Skill(
                    name=sk_name,
                    domain_tags=[_infer_domain(sk_name)],
                    core_domains=[_infer_domain(sk_name)],
                    weight=float(weight),
                    frequency=float(freq),
                    ict_score=7.0,
                )
            )
        clusters.append(
            Cluster(
                name=item.get("name", f"Cluster {item.get('id')}"),
                centroid_skills=centroid_skills,
                job_offer_count=item.get("offers", 0),
            )
        )
    return clusters


def _load_demo_clusters() -> list:
    """Crea clústeres de demostración si no hay datos reales."""
    from validation.alignment import Cluster, Skill

    return [
        Cluster(
            name="Python Backend & Cloud",
            centroid_skills=[
                Skill("Python", ["backend"], ["Backend"], 3.0, 0.85, 8.0),
                Skill("FastAPI", ["backend", "api"], ["Backend"], 2.5, 0.70, 7.0),
                Skill("PostgreSQL", ["database"], ["Backend"], 2.0, 0.65, 6.5),
                Skill("Docker", ["devops"], ["DevOps"], 2.5, 0.60, 7.5),
                Skill("AWS", ["cloud"], ["Cloud"], 2.0, 0.55, 6.0),
                Skill("Redis", ["database", "cache"], ["Backend"], 1.5, 0.40, 5.0),
                Skill("SQL", ["database"], ["Backend"], 1.0, 0.80, 9.0),
            ],
            job_offer_count=482,
        ),
        Cluster(
            name="Frontend React & JavaScript",
            centroid_skills=[
                Skill("JavaScript", ["frontend", "web"], ["Frontend"], 3.0, 0.90, 8.5),
                Skill("React", ["frontend", "spa"], ["Frontend"], 3.0, 0.85, 8.0),
                Skill("TypeScript", ["frontend", "language"], ["Frontend"], 2.5, 0.75, 7.5),
                Skill("HTML", ["frontend", "web"], ["Frontend"], 1.5, 0.95, 9.5),
                Skill("CSS", ["frontend", "web"], ["Frontend"], 1.5, 0.90, 9.0),
                Skill("Next.js", ["frontend", "ssr"], ["Frontend"], 2.0, 0.55, 6.5),
                Skill("Node.js", ["backend", "runtime"], ["Backend"], 2.0, 0.60, 7.0),
            ],
            job_offer_count=356,
        ),
        Cluster(
            name="DevOps & Cloud Infrastructure",
            centroid_skills=[
                Skill("Docker", ["devops"], ["DevOps"], 3.0, 0.85, 8.0),
                Skill("Kubernetes", ["devops", "orchestration"], ["DevOps"], 3.0, 0.80, 7.5),
                Skill("AWS", ["cloud"], ["Cloud"], 2.5, 0.80, 8.0),
                Skill("Terraform", ["devops", "iac"], ["DevOps"], 2.5, 0.65, 7.0),
                Skill("CI/CD", ["devops", "pipelines"], ["DevOps"], 2.0, 0.75, 7.5),
                Skill("Linux", ["devops", "os"], ["DevOps"], 1.5, 0.90, 9.0),
                Skill("Python", ["backend", "scripting"], ["Backend"], 1.5, 0.55, 6.0),
                Skill("Git", ["devops", "vcs"], ["DevOps"], 1.0, 0.95, 9.5),
            ],
            job_offer_count=245,
        ),
        Cluster(
            name="Data Science & Analytics",
            centroid_skills=[
                Skill("Python", ["data", "backend"], ["Data"], 3.0, 0.90, 8.5),
                Skill("SQL", ["database"], ["Data"], 2.0, 0.85, 8.0),
                Skill("Pandas", ["data", "analysis"], ["Data"], 2.5, 0.70, 7.5),
                Skill("Machine Learning", ["data", "ml"], ["Data"], 3.0, 0.65, 7.0),
                Skill("Power BI", ["data", "bi"], ["Data"], 1.5, 0.50, 6.0),
                Skill("Statistics", ["data", "math"], ["Data"], 2.0, 0.55, 6.5),
                Skill("NumPy", ["data", "numerical"], ["Data"], 1.5, 0.65, 7.0),
            ],
            job_offer_count=198,
        ),
        Cluster(
            name="Mobile Development",
            centroid_skills=[
                Skill("Kotlin", ["mobile", "android"], ["Mobile"], 3.0, 0.75, 7.5),
                Skill("Swift", ["mobile", "ios"], ["Mobile"], 3.0, 0.70, 7.0),
                Skill("React Native", ["mobile", "cross-platform"], ["Mobile"], 2.0, 0.60, 6.5),
                Skill("Flutter", ["mobile", "cross-platform"], ["Mobile"], 2.5, 0.65, 7.0),
                Skill("Android", ["mobile", "android"], ["Mobile"], 2.0, 0.80, 8.0),
                Skill("iOS", ["mobile", "ios"], ["Mobile"], 2.0, 0.75, 7.5),
                Skill("Firebase", ["mobile", "backend"], ["Mobile"], 1.5, 0.50, 6.0),
            ],
            job_offer_count=142,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Devalign Diagnostic Validation Suite"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Usar clústeres de demostración (sin DB)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Forzar fetch de Supabase, ignorar cache local")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida para resultados")
    args = parser.parse_args()

    # Cargar clústeres
    if args.demo:
        logger.info("Using demo clusters")
        clusters = _load_demo_clusters()
    elif _has_calibrated_clusters():
        logger.info("Loading calibrated clusters from local dump")
        clusters = _load_calibrated_clusters()
    elif not _has_supabase_creds():
        logger.info("Using demo clusters")
        clusters = _load_demo_clusters()
    else:
        logger.info("Loading clusters from database")
        clusters = _load_clusters_from_db(args)

    if not clusters:
        logger.error("No clusters available. Cannot run validation.")
        sys.exit(1)

    logger.info(f"Loaded {len(clusters)} clusters")
    for c in clusters:
        logger.info(f"  {c.name}: {len(c.centroid_skills)} skills, {c.job_offer_count} offers")

    # Determinar directorio de salida
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "results")
    logger.info(f"Output directory: {output_dir}")

    # Ejecutar validación
    from validation.ablation_study import run_all
    results = run_all(clusters, output_dir=output_dir)

    # Imprimir resumen
    agg = results["aggregate"]
    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Profiles:          {agg['total_profiles']}")
    print(f"  Accuracy:          {agg['accuracy']:.1%}")
    print(f"  Avg Precision:     {agg['avg_precision']:.3f}")
    print(f"  Avg Recall:        {agg['avg_recall']:.3f}")
    print(f"  Avg F1:            {agg['avg_f1']:.3f}")
    print(f"  Correct assignment: {agg['correct_assignment']}/{agg['total_profiles']}")
    print("=" * 60)
    print(f"  Results saved to:  {output_dir}")
    print("=" * 60)


def _has_supabase_creds() -> bool:
    from src import config as ml_config
    return bool(ml_config.SUPABASE_URL and ml_config.SUPABASE_KEY)


if __name__ == "__main__":
    main()
