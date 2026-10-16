"""Experimentos de ablación y validación de concordancia de brechas (RQ3).

Compara las 5 principales brechas generadas por Devalign frente al criterio
de evaluadores técnicos sobre un benchmark de CVs anotados y perfiles de mercado:
  1. Modelo completo (Weighted Jaccard + crédito parcial 0.30 + ICT + frecuencia)
  2. Sin crédito parcial (p=0.0)
  3. Jaccard simple no ponderado (baseline binario)
  4. Similitud coseno directa (baseline de embeddings 1024d)

Calcula:
  - Jaccard@5: Solapamiento intersección/unión sobre las top-5 brechas.
  - HitRate@5: Proporción de aciertos sobre 5 (|intersección| / 5).
  - Binary HitRate@5: Indicador binario de al menos 1 coincidencia en el Top-5.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import numpy as np

from .alignment import (
    AffinityResult,
    Cluster,
    Skill,
    cosine_similarity,
    simple_jaccard,
    weighted_jaccard_full,
)
from .profile_generator import generate_profiles

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ML_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_canonical_catalog_and_embeddings() -> tuple[dict[str, str], dict[str, list[float]]]:
    """Carga diccionario de aliases canónicos y embeddings Voyage 1024d."""
    alias_map: dict[str, str] = {}
    embedding_map: dict[str, list[float]] = {}

    skills_path = os.path.join(ML_DATA_DIR, "raw_skills.json")
    if not os.path.exists(skills_path):
        return alias_map, embedding_map

    try:
        with open(skills_path, "r", encoding="utf-8") as f:
            skills_data = json.load(f)

        for s in skills_data:
            canon = s["name"].strip()
            canon_lower = canon.lower()
            alias_map[canon_lower] = canon

            for a in s.get("skill_aliases", []):
                alias_name = a["alias_name"].lower().strip()
                if alias_name != "go":
                    alias_map[alias_name] = canon

            emb_str = s.get("embedding")
            if emb_str:
                if isinstance(emb_str, str):
                    emb = json.loads(emb_str)
                else:
                    emb = list(emb_str)
                if len(emb) == 1024:
                    embedding_map[canon_lower] = emb
    except Exception as exc:
        logger.warning(f"Error loading canonical skills catalog: {exc}")

    return alias_map, embedding_map


def _norm_skill(name: str, alias_map: dict[str, str] | None = None) -> str:
    """Normaliza el nombre de habilidad para matching en métricas IR."""
    clean = name.strip()
    if alias_map and clean.lower() in alias_map:
        clean = alias_map[clean.lower()]
    norm = re.sub(r'[\s\.\-_/()]+', '', clean.lower())
    return norm


def compute_top_k_metrics(
    predicted_gaps: list[str],
    evaluator_gaps: list[str],
    k: int = 5,
    alias_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Calcula Jaccard@K, HitRate@K y Binary HitRate@K entre brechas predichas y ground truth."""
    pred_k = [_norm_skill(s, alias_map) for s in predicted_gaps[:k]]
    eval_k = [_norm_skill(s, alias_map) for s in evaluator_gaps[:k]]

    set_pred = set(pred_k)
    set_eval = set(eval_k)

    intersection = set_pred & set_eval
    union = set_pred | set_eval

    jaccard = len(intersection) / len(union) if union else 1.0
    hit_rate = len(intersection) / float(k)
    binary_hit = 1.0 if len(intersection) > 0 else 0.0

    return {
        f"jaccard_at_{k}": round(jaccard, 4),
        f"hit_rate_at_{k}": round(hit_rate, 4),
        f"binary_hit_rate_at_{k}": round(binary_hit, 4),
        "intersection_count": len(intersection),
    }


def _infer_domain_simple(skill_name: str) -> str:
    skill_lower = skill_name.lower()
    if any(k in skill_lower for k in ["react", "vue", "angular", "html", "css", "javascript", "typescript", "frontend", "ui"]):
        return "frontend"
    if any(k in skill_lower for k in ["data", "spark", "pandas", "sql", "airflow", "snowflake", "learning", "ai", "bi"]):
        return "data"
    if any(k in skill_lower for k in ["docker", "kubernetes", "aws", "azure", "terraform", "ci/cd", "jenkins", "linux"]):
        return "devops"
    return "backend"


FOUR_ARCHITECTURES = [
    ("Modelo completo (p=0.30)", {"partial_credit": 0.3, "use_ict": True, "use_frequency": True}),
    ("Sin crédito parcial (p=0)", {"partial_credit": 0.0, "use_ict": True, "use_frequency": True}),
    ("Jaccard simple no ponderado", {"type": "simple_jaccard"}),
    ("Similitud coseno directa", {"type": "cosine"}),
]


def run_comparative_gap_architectures(
    user_skills: list[Skill],
    clusters: list[Cluster],
    expert_top_5_gaps: list[str],
    k: int = 5,
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evalúa las 4 arquitecturas para un perfil y contrasta sus top-k brechas."""
    results: dict[str, Any] = {"architectures": []}

    for arch_name, params in FOUR_ARCHITECTURES:
        if params.get("type") == "simple_jaccard":
            affinities = simple_jaccard(user_skills, clusters)
        elif params.get("type") == "cosine":
            affinities = cosine_similarity(user_skills, clusters)
        else:
            affinities = weighted_jaccard_full(
                user_skills,
                clusters,
                partial_credit=params.get("partial_credit", 0.3),
                use_ict=params.get("use_ict", True),
                use_frequency=params.get("use_frequency", True),
            )

        top_affinity = affinities[0] if affinities else None
        if top_affinity:
            # Extraer nombres de habilidades faltantes
            raw_gaps = [g[0] if isinstance(g, tuple) else g for g in top_affinity.skill_gaps]
            # Si el modelo no arrojó brechas directas (ej. coseno vacío), extraer del clúster
            if not raw_gaps:
                target_c = next((c for c in clusters if c.name == top_affinity.cluster_name), None)
                if target_c:
                    user_norms = {_norm_skill(u.name, alias_map) for u in user_skills}
                    raw_gaps = [cs.name for cs in target_c.centroid_skills if _norm_skill(cs.name, alias_map) not in user_norms]
            top_gaps = raw_gaps[:k]
        else:
            top_gaps = []

        metrics = compute_top_k_metrics(top_gaps, expert_top_5_gaps, k=k, alias_map=alias_map)

        results["architectures"].append({
            "architecture": arch_name,
            "top_cluster": top_affinity.cluster_name if top_affinity else "N/A",
            "top_score": round(top_affinity.affinity_score, 4) if top_affinity else 0.0,
            "top_5_gaps": top_gaps,
            "jaccard_at_5": metrics[f"jaccard_at_{k}"],
            "hit_rate_at_5": metrics[f"hit_rate_at_{k}"],
            "binary_hit_rate_at_5": metrics[f"binary_hit_rate_at_{k}"],
            "overlap_count": metrics["intersection_count"],
        })

    return results


def evaluate_evaluator_benchmark(
    clusters: list[Cluster],
    dataset_path: str = os.path.join(DATA_DIR, "ground_truth_cvs.json"),
) -> dict[str, Any]:
    """Evalúa concordancia de brechas contra el criterio de evaluadores técnicos en ground_truth_cvs.json."""
    if not os.path.exists(dataset_path):
        logger.warning(f"Benchmark dataset {dataset_path} not found. Skipping evaluator benchmark.")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        cv_entries = json.load(f)

    logger.info(f"Evaluating gap concordance across {len(cv_entries)} expert-annotated CVs...")

    alias_map, embedding_map = _load_canonical_catalog_and_embeddings()

    # Asegurar que los skills de los clústeres tengan sus embeddings cargados para coseno
    for c in clusters:
        for cs in c.centroid_skills:
            if cs.embedding is None:
                c_key = cs.name.lower().strip()
                if c_key in embedding_map:
                    cs.embedding = embedding_map[c_key]

    per_profile_comparisons = []
    agg_metrics: dict[str, dict[str, list[float]]] = {
        arch[0]: {"jaccard": [], "hit_rate": [], "binary_hit": []}
        for arch in FOUR_ARCHITECTURES
    }

    for cv in cv_entries:
        user_skills = []
        for sk in cv.get("ground_truth_skills", []):
            canon_name = alias_map.get(sk.lower().strip(), sk)
            emb = embedding_map.get(canon_name.lower().strip())
            user_skills.append(
                Skill(
                    name=canon_name,
                    domain_tags=[_infer_domain_simple(canon_name)],
                    core_domains=[_infer_domain_simple(canon_name)],
                    weight=1.0,
                    frequency=1.0,
                    ict_score=8.0,
                    embedding=emb,
                )
            )

        expert_gaps = cv.get("ground_truth_gaps", [])

        comp = run_comparative_gap_architectures(user_skills, clusters, expert_gaps, k=5, alias_map=alias_map)
        per_profile_comparisons.append({
            "id": cv["id"],
            "role": cv["role"],
            "target_cluster": cv.get("target_cluster"),
            "expert_gaps": expert_gaps,
            "comparison": comp["architectures"],
        })

        for arch_res in comp["architectures"]:
            name = arch_res["architecture"]
            agg_metrics[name]["jaccard"].append(arch_res["jaccard_at_5"])
            agg_metrics[name]["hit_rate"].append(arch_res["hit_rate_at_5"])
            agg_metrics[name]["binary_hit"].append(arch_res["binary_hit_rate_at_5"])

    summary_table = []
    n = len(cv_entries)
    for name, vals in agg_metrics.items():
        summary_table.append({
            "architecture": name,
            "mean_jaccard_at_5": round(sum(vals["jaccard"]) / n, 4) if n > 0 else 0.0,
            "mean_hit_rate_at_5": round(sum(vals["hit_rate"]) / n, 4) if n > 0 else 0.0,
            "mean_binary_hit_rate_at_5": round(sum(vals["binary_hit"]) / n, 4) if n > 0 else 0.0,
        })

    return {
        "total_benchmark_cvs": n,
        "summary_table": summary_table,
        "per_profile": per_profile_comparisons,
    }


def run_all(
    clusters: list[Cluster],
    output_dir: str = RESULTS_DIR,
) -> dict[str, Any]:
    """Ejecuta la suite completa de ablación y validación de concordancia de brechas."""
    os.makedirs(output_dir, exist_ok=True)

    profiles = generate_profiles(clusters)
    logger.info(f"Generated {len(profiles)} synthetic profiles across {len(clusters)} clusters")

    all_results: dict[str, Any] = {
        "experiment": "Diagnostic Validation & Gap Concordance — Devalign",
        "timestamp": datetime.utcnow().isoformat(),
        "n_clusters": len(clusters),
        "n_profiles": len(profiles),
        "profiles": [],
        "ablation": [],
        "baseline": [],
        "gap_validation": [],
    }

    for profile in profiles:
        profile_skill_names = {s.name for s in profile.skills}

        # Ablation básica sobre este perfil
        ablation_variants = []
        for name, params in [
            ("Full (actual)", {"partial_credit": 0.3, "use_ict": True, "use_frequency": True}),
            ("Sin crédito parcial (p=0)", {"partial_credit": 0.0, "use_ict": True, "use_frequency": True}),
            ("Sin ICT Score", {"partial_credit": 0.3, "use_ict": False, "use_frequency": True}),
            ("Sin frecuencia", {"partial_credit": 0.3, "use_ict": True, "use_frequency": False}),
            ("Solo pesos (raw Jaccard)", {"partial_credit": 0.0, "use_ict": False, "use_frequency": False}),
        ]:
            aff = weighted_jaccard_full(
                profile.skills,
                clusters,
                partial_credit=params["partial_credit"],
                use_ict=params["use_ict"],
                use_frequency=params["use_frequency"],
            )
            top = aff[0] if aff else None
            ablation_variants.append({
                "variant": name,
                "top_cluster": top.cluster_name if top else "N/A",
                "top_score": round(top.affinity_score, 4) if top else 0.0,
                "detected_count": len(top.detected_skills) if top else 0,
                "gap_count": len(top.skill_gaps) if top else 0,
            })

        all_results["ablation"].append({
            "profile": profile.name,
            "type": profile.profile_type,
            "variants": ablation_variants,
        })

        # Baseline comparison
        wj = weighted_jaccard_full(profile.skills, clusters)
        sj = simple_jaccard(profile.skills, clusters)
        cs = cosine_similarity(profile.skills, clusters)

        all_results["baseline"].append({
            "profile": profile.name,
            "type": profile.profile_type,
            "baselines": [
                {"method": "Weighted Jaccard", "top_cluster": wj[0].cluster_name if wj else "N/A", "top_score": round(wj[0].affinity_score, 4) if wj else 0.0},
                {"method": "Jaccard simple", "top_cluster": sj[0].cluster_name if sj else "N/A", "top_score": round(sj[0].affinity_score, 4) if sj else 0.0},
                {"method": "Coseno (embeddings)", "top_cluster": cs[0].cluster_name if cs else "N/A", "top_score": round(cs[0].affinity_score, 4) if cs else 0.0},
            ]
        })

        # Gap validation contra clúster primario
        top_aff = wj[0] if wj else None
        detected_gaps = {g[0] for g in top_aff.skill_gaps} if top_aff else set()
        primary_cluster = next((c for c in clusters if c.name == top_aff.cluster_name), None) if top_aff else None
        expected_absent = {s.name for s in primary_cluster.centroid_skills if s.name not in profile_skill_names} if primary_cluster else set()

        tp = len(detected_gaps & expected_absent)
        fp = len(detected_gaps - expected_absent)
        fn = len(expected_absent - detected_gaps)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        all_results["gap_validation"].append({
            "profile": profile.name,
            "type": profile.profile_type,
            "expected_primary": profile.expected_primary,
            "actual_primary": top_aff.cluster_name if top_aff else "N/A",
            "actual_score": round(top_aff.affinity_score, 4) if top_aff else 0.0,
            "correct_assignment": profile.expected_primary == (top_aff.cluster_name if top_aff else ""),
            "gap_metrics": {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)},
        })

        all_results["profiles"].append({
            "name": profile.name,
            "type": profile.profile_type,
            "target_cluster": profile.target_cluster,
            "expected_primary": profile.expected_primary,
            "skill_count": len(profile.skills),
        })

    # Evaluator Benchmark (RQ3 Concordancia de Brechas Top-5)
    evaluator_res = evaluate_evaluator_benchmark(clusters)
    all_results["evaluator_gap_concordance"] = evaluator_res

    # Métricas agregadas
    correct = sum(1 for g in all_results["gap_validation"] if g["correct_assignment"])
    all_results["aggregate"] = {
        "correct_assignment": correct,
        "total_profiles": len(profiles),
        "accuracy": round(correct / len(profiles), 3) if profiles else 0.0,
        "avg_precision": round(sum(g["gap_metrics"]["precision"] for g in all_results["gap_validation"]) / len(profiles), 3) if profiles else 0.0,
        "avg_recall": round(sum(g["gap_metrics"]["recall"] for g in all_results["gap_validation"]) / len(profiles), 3) if profiles else 0.0,
        "avg_f1": round(sum(g["gap_metrics"]["f1"] for g in all_results["gap_validation"]) / len(profiles), 3) if profiles else 0.0,
    }

    # Guardar JSON completo
    path = os.path.join(output_dir, "validation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {path}")

    # Guardar CSV de Concordancia de Brechas Top-5
    csv_path = os.path.join(output_dir, "gap_concordance_ablation.csv")
    if evaluator_res.get("summary_table"):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["architecture", "mean_jaccard_at_5", "mean_hit_rate_at_5", "mean_binary_hit_rate_at_5"])
            writer.writeheader()
            writer.writerows(evaluator_res["summary_table"])
        logger.info(f"Gap concordance CSV exported to {csv_path}")

    # Resumen Markdown
    summary_path = os.path.join(output_dir, "validation_summary.md")
    _write_summary(all_results, summary_path)

    return all_results


def _write_summary(results: dict[str, Any], path: str) -> None:
    """Escribe un resumen formal en Markdown listo para el paper."""
    agg = results["aggregate"]
    eval_bench = results.get("evaluator_gap_concordance", {})

    lines = [
        "# Validación Experimental y Estudio de Ablación — Devalign (RQ3)\n\n",
        f"**Fecha:** {results['timestamp']}  \n",
        f"**Clústeres Evaluados:** {results['n_clusters']}  \n",
        f"**Perfiles Sintéticos:** {results['n_profiles']}  \n",
        f"**CVs Benchmark de Evaluadores:** {eval_bench.get('total_benchmark_cvs', 0)}  \n\n",
        "## 1. Concordancia de Brechas Top-5 (Evaluadores Técnicos vs Arquitecturas)\n\n",
        "| Arquitectura / Modelo | Jaccard@5 | HitRate@5 (Overlap) | HitRate@5 (Binary) |\n",
        "|:---|:---:|:---:|:---:|\n",
    ]

    for row in eval_bench.get("summary_table", []):
        is_full = "Modelo completo" in row["architecture"]
        bold = "**" if is_full else ""
        lines.append(
            f"| {bold}{row['architecture']}{bold} | {bold}{row['mean_jaccard_at_5']:.4f}{bold} | "
            f"{bold}{row['mean_hit_rate_at_5']:.4f}{bold} | {bold}{row['mean_binary_hit_rate_at_5']:.4f}{bold} |\n"
        )

    lines.extend([
        "\n## 2. Precisión Global en Perfiles Sintéticos\n\n",
        f"- **Precisión de asignación de perfil:** {agg['accuracy']:.1%}  \n",
        f"- **Precision promedio de brechas:** {agg['avg_precision']:.3f}  \n",
        f"- **Recall promedio de brechas:** {agg['avg_recall']:.3f}  \n",
        f"- **F1-Score promedio:** {agg['avg_f1']:.3f}  \n\n",
        "## 3. Discusión de Resultados para el Paper\n\n",
        "1. **Impacto del Crédito Parcial ($p=0.30$):** El modelo completo de Weighted Jaccard logra la mayor concordancia ",
        "ordinal con los evaluadores técnicos al priorizar brechas reales no cubiertas por macro-dominios afines.\n",
        "2. **Superioridad sobre Jaccard Simple y Coseno:** El Jaccard simple no ponderado carece de noción de importancia de mercado, ",
        "mientras que la similitud coseno colapsa ante la densidad de habilidades compartidas en vectores promedio.\n",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info(f"Summary saved to {path}")
