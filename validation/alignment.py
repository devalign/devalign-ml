"""
Implementaciones del algoritmo de alineación para validación.

Tres variantes del Weighted Jaccard para ablation study,
más implementaciones baseline (Jaccard simple, coseno).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Datos mínimos (independientes del dominio de la API) ──────────────

@dataclass
class Skill:
    name: str
    domain_tags: list[str] = field(default_factory=list)
    core_domains: list[str] = field(default_factory=list)
    weight: float = 1.0
    frequency: float = 1.0
    ict_score: float = 0.0
    embedding: list[float] | None = None


@dataclass
class Cluster:
    name: str
    centroid_skills: list[Skill]
    job_offer_count: int = 0


@dataclass
class AffinityResult:
    cluster_name: str
    affinity_score: float
    detected_skills: list[str]
    skill_gaps: list[tuple[str, str]]  # (skill_name, importance)


# ── Variante 1: Full model (actual) ───────────────────────────────────

def weighted_jaccard_full(
    user_skills: list[Skill],
    clusters: list[Cluster],
    partial_credit: float = 0.3,
    use_ict: bool = True,
    use_frequency: bool = True,
) -> list[AffinityResult]:
    """
    Algoritmo completo: Weighted Jaccard con ICT Score, crédito parcial
    por dominio y ponderación por frecuencia.

    Parámetros de ablación:
      partial_credit : 0.3 (actual) | 0.0 (sin crédito parcial)
      use_ict        : True (actual) | False (user_score = 1.0)
      use_frequency  : True (actual) | False (f_s = 1.0)
    """
    results = []
    user_norms = {n(s): s for s in user_skills}
    user_tech = [s for s in user_skills]

    for cluster in clusters:
        cluster_norms = {n(s): s for s in cluster.centroid_skills}
        union = set(cluster_norms.keys()) | set(user_norms.keys())

        numerator = 0.0
        denominator = 0.0
        matched = []
        gaps = []

        for norm_name in union:
            in_user = norm_name in user_norms
            in_cluster = norm_name in cluster_norms

            if in_cluster:
                cs = cluster_norms[norm_name]
                w = cs.weight
                f_s = cs.frequency if use_frequency else 1.0
            else:
                us = user_norms[norm_name]
                w = us.weight
                f_s = 1.0

            if in_user and in_cluster:
                us = user_norms[norm_name]
                user_score = (us.ict_score / 10.0) if use_ict else 1.0
                numerator += w * f_s * user_score
                denominator += w * f_s
                matched.append(cluster_norms[norm_name].name)

            elif in_cluster:
                cs = cluster_norms[norm_name]
                cluster_domains = set(cs.domain_tags)

                partial = 0.0
                if cluster_domains and partial_credit > 0:
                    alternatives = [
                        u for u in user_tech
                        if set(u.domain_tags) & cluster_domains
                    ]
                    if alternatives:
                        best_alt = max(alternatives, key=lambda u: u.ict_score)
                        alt_score = (best_alt.ict_score / 10.0) if use_ict else 1.0
                        partial = partial_credit * alt_score

                if partial == 0.0:
                    gaps.append((cs.name, _classify_gap(w, f_s)))

                numerator += (w * f_s) * partial
                denominator += w * f_s

            else:
                us = user_norms[norm_name]
                user_score = (us.ict_score / 10.0) if use_ict else 1.0
                denominator += w * user_score

        score = (numerator / denominator) if denominator > 0.0 else 0.0
        results.append(AffinityResult(
            cluster_name=cluster.name,
            affinity_score=score,
            detected_skills=matched,
            skill_gaps=gaps,
        ))

    results.sort(key=lambda r: r.affinity_score, reverse=True)
    return results


# ── Variante 2: Jaccard simple (sin pesos, sin ICT, sin dominio) ─────

def simple_jaccard(
    user_skills: list[Skill],
    clusters: list[Cluster],
) -> list[AffinityResult]:
    """
    Baseline clásico: |intersection| / |union|.
    Sin pesos, sin ICT, sin crédito parcial.
    """
    results = []
    user_names = {n(s) for s in user_skills}

    for cluster in clusters:
        cluster_names = {n(s) for s in cluster.centroid_skills}
        intersection = user_names & cluster_names
        union = user_names | cluster_names

        score = len(intersection) / len(union) if union else 0.0
        matched = [cs.name for cs in cluster.centroid_skills if n(cs) in user_names]
        gaps = [cs.name for cs in cluster.centroid_skills if n(cs) not in user_names]

        results.append(AffinityResult(
            cluster_name=cluster.name,
            affinity_score=score,
            detected_skills=matched,
            skill_gaps=[(g, _classify_gap(1.0, 1.0)) for g in gaps],
        ))

    results.sort(key=lambda r: r.affinity_score, reverse=True)
    return results


# ── Variante 3: Coseno sobre embeddings ───────────────────────────────

def cosine_similarity(
    user_skills: list[Skill],
    clusters: list[Cluster],
) -> list[AffinityResult]:
    """
    Baseline basado en similitud coseno de los embeddings promedio.
    """
    results = []

    # Embedding promedio del usuario (media de skills con embedding)
    user_vec = _mean_embedding(user_skills)

    if user_vec is None:
        return [AffinityResult(c.name, 0.0, [], []) for c in clusters]

    for cluster in clusters:
        cluster_vec = _mean_embedding(cluster.centroid_skills)
        if cluster_vec is None:
            score = 0.0
        else:
            score = _cosine(user_vec, cluster_vec)

        user_names = {n(s) for s in user_skills}
        matched = [cs.name for cs in cluster.centroid_skills if n(cs) in user_names]
        gaps = [cs.name for cs in cluster.centroid_skills if n(cs) not in user_names]

        results.append(AffinityResult(
            cluster_name=cluster.name,
            affinity_score=score,
            detected_skills=matched,
            skill_gaps=[(g, _classify_gap(1.0, 1.0)) for g in gaps],
        ))

    results.sort(key=lambda r: r.affinity_score, reverse=True)
    return results


# ── Funciones auxiliares ──────────────────────────────────────────────

def n(s: Skill) -> str:
    """Normalized name."""
    return s.name.lower().replace(" ", "").replace(".", "")


def _classify_gap(weight: float, frequency: float) -> str:
    priority = weight * frequency
    if priority >= 2.0:
        return "critical"
    elif priority >= 1.0:
        return "high"
    return "medium"


def _mean_embedding(skills: list[Skill]) -> list[float] | None:
    vectors = [s.embedding for s in skills if s.embedding]
    if not vectors:
        return None
    dim = len(vectors[0])
    avg = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            avg[i] += v[i]
    return [x / len(vectors) for x in avg]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
