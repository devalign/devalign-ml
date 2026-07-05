"""
Generador de perfiles sintéticos para validación del diagnóstico.

Crea desarrolladores sintéticos con skill sets controlados:
  - Full match:   todas las skills del clúster objetivo
  - Partial:      ~60% de las skills del clúster
  - Low:          ~30% de las skills del clúster
  - Mismatch:     skills de otro clúster (debería tener baja afinidad)
  - Mixed:        skills mezcladas de varios clústeres
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .alignment import Cluster, Skill


@dataclass
class SyntheticProfile:
    name: str
    profile_type: str       # "full_match" | "partial" | "low" | "mismatch" | "mixed"
    target_cluster: str     # Nombre del clúster al que debería alinearse
    skills: list[Skill]
    expected_primary: str   # Clúster esperado como primary


def generate_profiles(
    clusters: list[Cluster],
    seed: int = 42,
) -> list[SyntheticProfile]:
    """
    Genera un conjunto de perfiles sintéticos para validación.

    Para cada clúster, crea 4 perfiles (full, partial, low, mismatch).
    Además, crea perfiles mixed globales.
    """
    random.seed(seed)
    profiles: list[SyntheticProfile] = []

    for cluster in clusters:
        c_skills: list[Skill] = cluster.centroid_skills
        if len(c_skills) < 3:
            continue

        # ── Full match: usuario tiene todas las skills del clúster ──
        full_skills = [
            Skill(
                name=s.name,
                domain_tags=s.domain_tags,
                core_domains=s.core_domains,
                weight=s.weight,
                frequency=s.frequency,
                ict_score=_random_ict(seed + hash(cluster.name) % 1000),
                embedding=s.embedding,
            )
            for s in c_skills
        ]
        profiles.append(SyntheticProfile(
            name=f"{cluster.name} (Full Match)",
            profile_type="full_match",
            target_cluster=cluster.name,
            skills=full_skills,
            expected_primary=cluster.name,
        ))

        # ── Partial: ~60% de las skills ──
        n_partial = max(2, int(len(c_skills) * 0.6))
        partial_skills = random.sample(c_skills, n_partial)
        partial_set = [
            Skill(
                name=s.name,
                domain_tags=s.domain_tags,
                core_domains=s.core_domains,
                weight=s.weight,
                frequency=s.frequency,
                ict_score=_random_ict(seed + hash(cluster.name) % 1000 + 1),
                embedding=s.embedding,
            )
            for s in partial_skills
        ]
        profiles.append(SyntheticProfile(
            name=f"{cluster.name} (Partial ~60%)",
            profile_type="partial",
            target_cluster=cluster.name,
            skills=partial_set,
            expected_primary=cluster.name,
        ))

        # ── Low: ~30% de las skills ──
        n_low = max(1, int(len(c_skills) * 0.3))
        low_skills = random.sample(c_skills, n_low)
        low_set = [
            Skill(
                name=s.name,
                domain_tags=s.domain_tags,
                core_domains=s.core_domains,
                weight=s.weight,
                frequency=s.frequency,
                ict_score=_random_ict(seed + hash(cluster.name) % 1000 + 2),
                embedding=s.embedding,
            )
            for s in low_skills
        ]
        profiles.append(SyntheticProfile(
            name=f"{cluster.name} (Low ~30%)",
            profile_type="low",
            target_cluster=cluster.name,
            skills=low_set,
            expected_primary=cluster.name,
        ))

        # ── Mismatch: skills del siguiente clúster (debería fallar) ──
        other_clusters = [c for c in clusters if c.name != cluster.name]
        if other_clusters:
            other = random.choice(other_clusters)
            mismatch_skills = [
                Skill(
                    name=s.name,
                    domain_tags=s.domain_tags,
                    core_domains=s.core_domains,
                    weight=s.weight,
                    frequency=s.frequency,
                    ict_score=_random_ict(seed + hash(cluster.name) % 1000 + 3),
                    embedding=s.embedding,
                )
                for s in other.centroid_skills[:max(3, len(other.centroid_skills) // 2)]
            ]
            profiles.append(SyntheticProfile(
                name=f"{cluster.name} (Mismatch ← {other.name})",
                profile_type="mismatch",
                target_cluster=cluster.name,
                skills=mismatch_skills,
                expected_primary=other.name,
            ))

    # ── Perfiles mixed globales ──
    if len(clusters) >= 3:
        for pct in [0.3, 0.5, 0.7]:
            mixed_skills: list[Skill] = []
            for c in random.sample(clusters, min(3, len(clusters))):
                n_skills = max(1, int(len(c.centroid_skills) * pct))
                sampled = random.sample(c.centroid_skills, n_skills)
                for s in sampled:
                    mixed_skills.append(Skill(
                        name=s.name,
                        domain_tags=s.domain_tags,
                        core_domains=s.core_domains,
                        weight=s.weight,
                        frequency=s.frequency,
                        ict_score=_random_ict(seed + hash(c.name) % 1000 + 4),
                        embedding=s.embedding,
                    ))
            profiles.append(SyntheticProfile(
                name=f"Mixed Profile ({int(pct*100)}% de 3 clústeres)",
                profile_type="mixed",
                target_cluster="",
                skills=mixed_skills,
                expected_primary="",  # Determinado por el algoritmo
            ))

    return profiles


def _random_ict(base_seed: int) -> float:
    rng = random.Random(base_seed)
    return round(rng.uniform(4.0, 10.0), 1)
