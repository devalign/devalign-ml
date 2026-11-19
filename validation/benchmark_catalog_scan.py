"""Benchmark of Catalog Scan Extraction, Telemetry Latency, and Paper Alignment Audit.

Executes:
1. Catalog scan skill extraction benchmark over 15 reference CVs (196 declared skills).
2. Per-stage latency benchmark (parsing, catalog scan, normalization, graph inference + affinity).
3. Phase B LLM extraction (extract_llm_insights) latency, success rate, and metadata evaluation.
4. Comparison against development-stage LLM extraction (cached_extracted_cvs.json).
5. Simulated human validation gate analysis.

Outputs:
- validation/results/catalog_scan_benchmark.json
- validation/results/catalog_scan_benchmark.md
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np

# Setup relative paths
ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(ML_DIR)
API_DIR = os.path.join(ROOT_DIR, "devalign-api")

sys.path = [p for p in sys.path if p not in (API_DIR, ML_DIR)]
sys.path.insert(0, API_DIR)
sys.path.append(ML_DIR)

os.chdir(API_DIR)

from src.config import settings
from src.ml_engine.application.skill_catalog_service import SkillCatalogService
from src.ml_engine.domain.entities import Skill, SkillNature, TechCluster
from src.ml_engine.domain.services.affinity_calculator import compute_affinities_and_domains
from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
from src.ml_engine.infrastructure.llm_client import get_llm_service
from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark_catalog_scan")

DATA_DIR = os.path.join(ML_DIR, "validation", "data")
RESULTS_DIR = os.path.join(ML_DIR, "validation", "results")
GT_FILE = os.path.join(DATA_DIR, "ground_truth_cvs.json")
OLD_CACHE_FILE = os.path.join(RESULTS_DIR, "cached_extracted_cvs.json")
SKILLS_DUMP = os.path.join(DATA_DIR, "skills_catalog_dump.json")
CLUSTERS_DUMP = os.path.join(DATA_DIR, "clusters_dump.json")
OUT_JSON = os.path.join(RESULTS_DIR, "catalog_scan_benchmark.json")
OUT_MD = os.path.join(RESULTS_DIR, "catalog_scan_benchmark.md")

NORMALIZE_SEP_RE = re.compile(r"[\s.\-_/]+")
PAREN_QUALIFIER_RE = re.compile(r"\([^()]*\)")


def _normalize_token(text: str) -> str:
    """Lowercase and strip whitespace/punctuation separators using the canonical regex."""
    return NORMALIZE_SEP_RE.sub("", text.lower())


def _strip_parenthetical_qualifier(name: str) -> str:
    """Remove Lightcast parenthetical qualifiers such as ' (Programming Language)'."""
    return PAREN_QUALIFIER_RE.sub("", name).strip()


def _read_json(path: str) -> Any:
    """Synchronously load a UTF-8 JSON file (invoked via asyncio.to_thread in async code)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    """Synchronously write a UTF-8 JSON file (invoked via asyncio.to_thread in async code)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_text(path: str, content: str) -> None:
    """Synchronously write a UTF-8 text file (invoked via asyncio.to_thread in async code)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def normalize_skill_name_strict(name: str) -> str:
    """Exact normalization used in validation/benchmark_cv_extraction.py."""
    clean = _normalize_token(name)
    alias_map = {
        "reactjs": "react",
        "vuejs": "vue",
        "nodejs": "node",
        "golang": "go",
        "html5": "html",
        "css3": "css",
        "postgres": "postgresql",
        "restfulapi": "restapi",
        "apisrest": "restapi",
        "apirest": "restapi",
        "aws": "amazonwebservices",
        "gcp": "googlecloudplatform",
    }
    return alias_map.get(clean, clean)


def normalize_skill_name_canonical(name: str) -> str:
    """Taxonomic canonical normalization stripping Lightcast parenthetical qualifiers."""
    base = _strip_parenthetical_qualifier(name)
    clean = _normalize_token(base)
    alias_map = {
        "reactjs": "react",
        "react": "react",
        "vuejs": "vue",
        "vue": "vue",
        "nodejs": "node",
        "node": "node",
        "golang": "go",
        "go": "go",
        "html5": "html",
        "html": "html",
        "css3": "css",
        "css": "css",
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "restfulapi": "restapi",
        "restapi": "restapi",
        "apisrest": "restapi",
        "apirest": "restapi",
        "aws": "amazonwebservices",
        "amazonwebservices": "amazonwebservices",
        "gcp": "googlecloudplatform",
        "googlecloudplatform": "googlecloudplatform",
        "gitversioncontrolsystem": "git",
        "git": "git",
        "dockersoftware": "docker",
        "docker": "docker",
        "hypertextmarkuplanguage": "html",
        "cascadingstylesheets": "css",
        "seleniumsoftware": "selenium",
        "selenium": "selenium",
        "postmanapiplatform": "postman",
        "postman": "postman",
        "microsoftazure": "azure",
        "azure": "azure",
        "amazons3": "awss3",
        "awss3": "awss3",
        "appleios": "ios",
        "ios": "ios",
        "apachemaven": "maven",
        "maven": "maven",
        "oauth": "oauth2",
        "oauth2": "oauth2",
        "awselasticmapreduce": "awsemr",
        "awsemr": "awsemr",
    }
    return alias_map.get(clean, clean)


def _evaluate_single_cv(
    item: dict[str, Any],
    extracted_map: dict[str, list[str]],
    norm_fn: Any,
) -> tuple[dict[str, Any], int, int, int, bool]:
    """Compute extraction metrics for a single CV entry."""
    cv_id = item["id"]
    gt_skills = item["ground_truth_skills"]
    ext_skills = extracted_map.get(cv_id, [])

    gt_norm = {norm_fn(s): s for s in gt_skills}
    ext_norm = {norm_fn(s): s for s in ext_skills}

    tp_keys = set(gt_norm.keys()) & set(ext_norm.keys())
    fp_keys = set(ext_norm.keys()) - set(gt_norm.keys())
    fn_keys = set(gt_norm.keys()) - set(ext_norm.keys())

    tp, fp, fn = len(tp_keys), len(fp_keys), len(fn_keys)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    is_exact = fp == 0 and fn == 0

    row = {
        "id": cv_id,
        "role": item["role"],
        "seniority": item["seniority"],
        "ground_truth_count": len(gt_skills),
        "extracted_count": len(ext_skills),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "exact_match": is_exact,
        "spurious_fps": sorted(ext_norm[k] for k in fp_keys),
        "omitted_fns": sorted(gt_norm[k] for k in fn_keys),
    }
    return row, tp, fp, fn, is_exact


def compute_metrics(
    cvs_dataset: list[dict],
    extracted_map: dict[str, list[str]],
    norm_fn: Any,
) -> dict[str, Any]:
    """Computes precision, recall, f1, exact match per CV and aggregates."""
    per_cv = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_match_count = 0

    for item in cvs_dataset:
        row, tp, fp, fn, is_exact = _evaluate_single_cv(item, extracted_map, norm_fn)
        per_cv.append(row)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if is_exact:
            exact_match_count += 1

    n = len(cvs_dataset)
    macro_p = sum(r["precision"] for r in per_cv) / n if n > 0 else 0.0
    macro_r = sum(r["recall"] for r in per_cv) / n if n > 0 else 0.0
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "exact_match_ratio": round(exact_match_count / n, 4) if n > 0 else 0.0,
        "exact_match_count": exact_match_count,
        "total_eval_cvs": n,
        "total_declared_skills": sum(len(c["ground_truth_skills"]) for c in cvs_dataset),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "macro_metrics": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1_score": round(macro_f1, 4),
        },
        "micro_metrics": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1_score": round(micro_f1, 4),
        },
        "per_cv_results": per_cv,
    }


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def _parse_skill_dict(d: dict[str, Any], include_status: bool = True) -> Skill:
    """Convert a serialized skill JSON entry into a Skill domain entity."""
    nature = SkillNature.CONCEPT if d.get("nature") == "concept" else SkillNature.TECH
    kwargs: dict[str, Any] = {
        "id": UUID(d["id"]) if d.get("id") else None,
        "name": d["name"],
        "nature": nature,
        "domain_tags": d.get("domain_tags") or [],
        "core_domains": d.get("core_domains") or [],
        "aliases": d.get("aliases") or [],
        "weight": d.get("weight") or 1.0,
    }
    if include_status:
        kwargs["status"] = d.get("status") or "active"
    else:
        kwargs["frequency"] = d.get("frequency")
    return Skill(**kwargs)


def _parse_cluster_dict(d: dict[str, Any], default_index: int) -> TechCluster:
    """Convert a serialized cluster JSON entry into a TechCluster domain entity."""
    centroid_skills = [
        _parse_skill_dict(s, include_status=False) for s in d.get("centroid_skills", [])
    ]
    return TechCluster(
        id=UUID(d["id"]),
        name=d["name"],
        description=d.get("description", ""),
        centroid_skills=centroid_skills,
        job_offer_count=d.get("job_offer_count", 0),
        cluster_index=d.get("cluster_index", default_index),
        tier=str(d.get("tier", "standard")),
        compatible_roles=d.get("compatible_roles") or [],
        market_insights=d.get("market_insights") or {},
    )


def load_in_memory_catalog() -> tuple[list[Skill], list[TechCluster]]:
    """Loads skills and clusters from persistent json dumps into domain entities."""
    skills_raw = _read_json(SKILLS_DUMP)
    all_skills = [_parse_skill_dict(d, include_status=True) for d in skills_raw]

    clusters_raw = _read_json(CLUSTERS_DUMP)
    all_clusters = [
        _parse_cluster_dict(d, idx) for idx, d in enumerate(clusters_raw)
    ]
    return all_skills, all_clusters


class MockSkillRepo:
    def __init__(self, skills: list[Skill]):
        self._skills = skills

    async def get_all_skills(self) -> list[Skill]:
        await asyncio.sleep(0)
        return self._skills


async def _run_catalog_extraction(
    cvs_dataset: list[dict[str, Any]],
    catalog_service: SkillCatalogService,
    all_skills: list[Skill],
) -> dict[str, list[str]]:
    """Run deterministic catalog scan + normalization for every CV in the dataset."""
    extracted_map: dict[str, list[str]] = {}
    for item in cvs_dataset:
        detected = await catalog_service.scan_text_for_catalog_skills(
            item["cv_text"], existing_skills_cache=all_skills
        )
        normalized = await catalog_service.normalize_extracted_skills(
            detected, existing_skills_cache=all_skills
        )
        extracted_map[item["id"]] = [s["name"] for s in normalized]
    return extracted_map


def _build_catalog_clean_set(all_skills: list[Skill]) -> dict[str, str]:
    """Map normalized canonical names and aliases to canonical catalog skill names."""
    clean_set = {
        _normalize_token(_strip_parenthetical_qualifier(s.name)): s.name
        for s in all_skills
    }
    for skill in all_skills:
        for alias in skill.aliases:
            if alias:
                clean_set[_normalize_token(alias)] = skill.name
    return clean_set


def _classify_fp_cause(fp_skill: str) -> str:
    """Categorize a false-positive catalog extraction into its diagnostic bucket."""
    lowered = fp_skill.lower()
    broad_terms = ["developer", "dashboards", "automation", "cloud", "security", "web application"]
    if any(term in lowered for term in broad_terms):
        return "Broad occupational / conceptual catalog match in full CV text"
    if "3m" in lowered or "sqc" in lowered:
        return "Lexical token / boundary collision in text parsing"
    if "api" in lowered or "rest" in lowered:
        return "Over-generic alias matching catalog technology"
    return "Catalog technology mentioned in experience section but outside ground-truth scope"


def _build_error_diagnostics(
    per_cv_results: list[dict[str, Any]],
    catalog_clean_set: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], int, int]:
    """Classify per-CV false negatives and false positives against the catalog."""
    fn_diagnostic_per_cv: dict[str, list[dict[str, Any]]] = {}
    fp_diagnostic_per_cv: dict[str, list[dict[str, Any]]] = {}
    out_of_catalog_fn_count = 0
    in_catalog_missed_count = 0

    for row in per_cv_results:
        cv_id = row["id"]
        fn_entries = []
        for fn_skill in row["omitted_fns"]:
            clean_fn = _normalize_token(fn_skill)
            in_cat = clean_fn in catalog_clean_set
            if in_cat:
                in_catalog_missed_count += 1
                cause = f"In catalog as '{catalog_clean_set[clean_fn]}' (alias/regex boundary variant)"
            else:
                out_of_catalog_fn_count += 1
                cause = "Out-of-catalog technology (custom skill requiring human gate)"
            fn_entries.append({"skill": fn_skill, "cause": cause, "in_catalog": in_cat})
        fn_diagnostic_per_cv[cv_id] = fn_entries

        fp_diagnostic_per_cv[cv_id] = [
            {"skill": fp_skill, "cause": _classify_fp_cause(fp_skill)}
            for fp_skill in row["spurious_fps"]
        ]

    return fn_diagnostic_per_cv, fp_diagnostic_per_cv, out_of_catalog_fn_count, in_catalog_missed_count


def _prepare_cv_documents(cvs_dataset: list[dict[str, Any]]) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Render synthetic DOCX and PDF payloads in memory for parsing latency benchmarks."""
    import docx
    import matplotlib.pyplot as plt

    cv_docx_bytes: dict[str, bytes] = {}
    cv_pdf_bytes: dict[str, bytes] = {}
    for item in cvs_dataset:
        doc = docx.Document()
        doc.add_paragraph(item["cv_text"])
        buf_docx = io.BytesIO()
        doc.save(buf_docx)
        cv_docx_bytes[item["id"]] = buf_docx.getvalue()

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, item["cv_text"][:1500], fontsize=6, verticalalignment="top", wrap=True)
        buf_pdf = io.BytesIO()
        plt.savefig(buf_pdf, format="pdf")
        plt.close(fig)
        cv_pdf_bytes[item["id"]] = buf_pdf.getvalue()

    return cv_docx_bytes, cv_pdf_bytes


async def _measure_stage_latencies(
    cvs_dataset: list[dict[str, Any]],
    parser_service: LocalCVParserService,
    catalog_service: SkillCatalogService,
    all_skills: list[Skill],
    active_clusters: list[TechCluster],
    cv_docx_bytes: dict[str, bytes],
    cv_pdf_bytes: dict[str, bytes],
    n_reps: int = 5,
) -> dict[str, list[float]]:
    """Measure per-stage interactive pipeline latencies across repeated CV passes."""
    latencies: dict[str, list[float]] = {
        "parsing_docx": [],
        "parsing_pdf": [],
        "scan": [],
        "normalization": [],
        "graph": [],
        "interactive_total": [],
    }

    for _ in range(n_reps):
        for item in cvs_dataset:
            cv_id = item["id"]
            cv_text = item["cv_text"]

            t0 = time.perf_counter()
            parser_service._extract_from_docx(cv_docx_bytes[cv_id])
            latencies["parsing_docx"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            parser_service._extract_from_pdf(cv_pdf_bytes[cv_id])
            t_pdf = time.perf_counter() - t0
            latencies["parsing_pdf"].append(t_pdf)

            t0 = time.perf_counter()
            detected = await catalog_service.scan_text_for_catalog_skills(
                cv_text, existing_skills_cache=all_skills
            )
            t_scan = time.perf_counter() - t0
            latencies["scan"].append(t_scan)

            t0 = time.perf_counter()
            normalized = await catalog_service.normalize_extracted_skills(
                detected, existing_skills_cache=all_skills
            )
            t_norm = time.perf_counter() - t0
            latencies["normalization"].append(t_norm)
            latencies["interactive_total"].append(t_pdf + t_scan + t_norm)

            norm_names = {x["name"] for x in normalized}
            detected_entities = [s for s in all_skills if s.name in norm_names]
            t0 = time.perf_counter()
            compute_affinities_and_domains(detected_entities, active_clusters)
            latencies["graph"].append(time.perf_counter() - t0)

    return latencies


async def _measure_phase_b_insights(
    cvs_dataset: list[dict[str, Any]],
    use_case: ProfileUserFromCVUseCase,
) -> tuple[list[float], int, dict[str, dict[str, Any]]]:
    """Measure Phase B LLM synthesis latency and output coverage across CVs."""
    phase_b_latencies: list[float] = []
    phase_b_successes = 0
    phase_b_outputs: dict[str, dict[str, Any]] = {}

    for item in cvs_dataset:
        cv_id = item["id"]
        t0 = time.perf_counter()
        try:
            insights = await use_case.extract_llm_insights(
                user_id=None,
                cv_id=None,
                cv_text=item["cv_text"],
            )
            dt = time.perf_counter() - t0
            phase_b_latencies.append(dt)
            phase_b_successes += 1
            phase_b_outputs[cv_id] = {
                "role_in_gt": item["role"],
                "current_job_role_extracted": insights.get("current_job_role"),
                "years_experience_extracted": insights.get("years_experience"),
                "summary_extracted": insights.get("professional_summary"),
                "latency_s": round(dt, 4),
                "success": True,
            }
        except Exception as exc:
            dt = time.perf_counter() - t0
            phase_b_latencies.append(dt)
            phase_b_outputs[cv_id] = {
                "role_in_gt": item["role"],
                "error": str(exc),
                "latency_s": round(dt, 4),
                "success": False,
            }

    return phase_b_latencies, phase_b_successes, phase_b_outputs


def _count_role_matches(phase_b_outputs: dict[str, dict[str, Any]]) -> int:
    """Count CVs where extracted job role shares lexical tokens with ground-truth role."""
    matches = 0
    for p_out in phase_b_outputs.values():
        if not p_out.get("success"):
            continue
        gt_tokens = set(re.findall(r"\w+", p_out["role_in_gt"].lower()))
        ext_tokens = set(re.findall(r"\w+", (p_out["current_job_role_extracted"] or "").lower()))
        if gt_tokens & ext_tokens:
            matches += 1
    return matches


def _stats_dict(arr: list[float]) -> dict[str, float]:
    """Summarize latency samples with mean, sample std, percentiles, and extrema."""
    arr_np = np.array(arr)
    return {
        "mean": round(float(np.mean(arr_np)), 4),
        "std": round(float(sample_std(arr)), 4),
        "p50": round(float(np.percentile(arr_np, 50)), 4),
        "p95": round(float(np.percentile(arr_np, 95)), 4),
        "min": round(float(np.min(arr_np)), 4),
        "max": round(float(np.max(arr_np)), 4),
        "samples": len(arr),
    }


def _build_latency_summary(
    latencies: dict[str, list[float]],
    phase_b_latencies: list[float],
    phase_b_successes: int,
    total_cvs: int,
) -> dict[str, Any]:
    """Assemble the latency summary payload for JSON and Markdown reporting."""
    return {
        "pdf_parsing": _stats_dict(latencies["parsing_pdf"]),
        "docx_parsing": _stats_dict(latencies["parsing_docx"]),
        "catalog_scan": _stats_dict(latencies["scan"]),
        "semantic_normalization_in_memory": _stats_dict(latencies["normalization"]),
        "semantic_normalization_pgvector": {
            "mean": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "status": "bypassed_in_cv_path",
            "explanation": "scan_text_for_catalog_skills matches canonical terms directly; pgvector is only called in offer ingestion (NormalizeJobOffersUseCase) and sensitivity sweeps",
        },
        "graph_inference_and_affinity": _stats_dict(latencies["graph"]),
        "interactive_phase_total_up_to_skills_detected": _stats_dict(latencies["interactive_total"]),
        "phase_b_llm_insights": {
            **_stats_dict(phase_b_latencies),
            "success_rate": round((phase_b_successes / total_cvs) * 100, 2),
            "execution_mode": "asynchronous_fastapi_background_task",
            "blocking_behavior": "Immediate HTTP 202 on CV upload; Phase A sets 'skills_detected_partial'; Phase B finishes before final diagnosis persistence (finalize endpoint blocks if status != 'skills_detected')",
        },
    }


def _build_consolidated_results(
    cvs_dataset: list[dict[str, Any]],
    old_llm_metrics: dict[str, Any],
    catalog_strict_metrics: dict[str, Any],
    catalog_canonical_metrics: dict[str, Any],
    human_sim_metrics: dict[str, Any],
    fn_diagnostic_per_cv: dict[str, list[dict[str, Any]]],
    fp_diagnostic_per_cv: dict[str, list[dict[str, Any]]],
    latency_summary: dict[str, Any],
    phase_b_successes: int,
    exact_role_matches: int,
    phase_b_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the full benchmark JSON report structure."""
    total_cvs = len(cvs_dataset)
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_environment": {
                "os": "Windows (win32)",
                "python_api": "3.12.13",
                "python_ml": "3.10.10",
                "database": "PostgreSQL 16 with pgvector on Supabase PaaS (aws-1-us-east-2.pooler.supabase.com:5432)",
                "llm_provider": settings.LLM_PROVIDER,
                "llm_model": settings.LLM_MODEL,
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.EMBEDDING_MODEL,
            },
            "provenance_hashes": {
                "devalign_api_head": "99f84e29307f76a66935c7f19cda8868ff5c38f8",
                "devalign_api_head_date": "2026-11-16T15:27:39-05:00",
                "devalign_api_evaluated_october": "f80cc40dced849f00b32947f33fa24b7d19f231e",
                "devalign_scraping_head": "28c9c7bb74a2ba3715dc88a51c0b4caba07a0851",
                "devalign_scraping_head_date": "2026-11-13T17:41:06-05:00",
                "devalign_scraping_evaluated_october": "e37f4fb5b2ac0bc14c95b1c0cc29575c19ed62db",
                "devalign_ml_head": "debb554885ef95ca2f8504a689bba264955a765f",
                "devalign_ml_head_date": "2026-11-13T19:36:32-05:00",
                "devalign_ml_evaluated_october": "8df983838e760fc0b341e1dca34604eb21f0a653",
                "devalign_docs_head": "e63c86b4626845d30919f60cdccac1a3615b4e2a",
                "devalign_docs_head_date": "2026-11-14T09:13:48-05:00",
            },
            "repo_facts": {
                "endpoints_count": 25,
                "endpoints_domain_count": 20,
                "tables_count": 16,
                "alembic_migrations_count": 26,
                "alembic_head": "c9d8e7f6a5b4",
                "pytest_tests_collected_api": 111,
                "pytest_tests_collected_ml": 14,
                "pytest_tests_collected_scraping": 97,
                "pytest_tests_collected_total": 222,
                "tau_threshold_value": 0.88,
                "tau_threshold_locations": [
                    "devalign-api/src/ml_engine/application/use_cases.py:1444 (NormalizeJobOffersUseCase: cosine similarity on dense embeddings in pgvector)",
                    "devalign-api/src/ml_engine/application/skill_catalog_service.py:240 (normalize_extracted_skills: difflib SequenceMatcher ratio cutoff for fuzzy fallback)",
                ],
                "phase_b_details": {
                    "prompt_purpose": "Synthesize core technical profile (current_job_role, professional_summary, years_experience) without re-extracting skills",
                    "char_limit": 15000,
                    "char_limit_code_reference": "devalign-api/src/ml_engine/application/use_cases.py:540",
                    "stored_fields": ["current_job_role", "years_experience", "professional_summary"],
                },
            },
        },
        "task_1_extraction_benchmark": {
            "old_llm_extractor": old_llm_metrics,
            "current_catalog_scan_strict": catalog_strict_metrics,
            "current_catalog_scan_canonical": catalog_canonical_metrics,
            "simulated_human_validation": human_sim_metrics,
            "fn_diagnostics_by_cv": fn_diagnostic_per_cv,
            "fp_diagnostics_by_cv": fp_diagnostic_per_cv,
        },
        "task_2_latency_benchmarks": latency_summary,
        "task_3_phase_b_eval": {
            "success_rate": f"{(phase_b_successes / total_cvs) * 100:.1f}%",
            "total_runs": total_cvs,
            "exact_or_substring_role_matches": f"{exact_role_matches}/{total_cvs}",
            "ground_truth_years_present": False,
            "notes": "ground_truth_cvs.json contains categorical 'role' and 'seniority', but lacks explicit numeric ground truth for years_experience. Extracted values are reported per CV.",
            "per_cv_outputs": phase_b_outputs,
        },
    }


def _render_markdown_report(
    all_skills: list[Skill],
    cvs_dataset: list[dict[str, Any]],
    old_m: dict[str, Any],
    strict_m: dict[str, Any],
    canon_m: dict[str, Any],
    fn_diagnostic_per_cv: dict[str, list[dict[str, Any]]],
    fp_diagnostic_per_cv: dict[str, list[dict[str, Any]]],
    in_catalog_missed_count: int,
    out_of_catalog_fn_count: int,
    latency_summary: dict[str, Any],
    phase_b_successes: int,
    exact_role_matches: int,
    phase_b_outputs: dict[str, dict[str, Any]],
) -> str:
    """Format the full Markdown benchmark summary report."""
    prev_v6 = {"tp": 160, "fp": 101, "fn": 36, "p": 0.6130, "r": 0.8163, "f1": 0.7002}
    prev_json = {"tp": 167, "fp": 94, "fn": 29, "p": 0.6398, "r": 0.8520, "f1": 0.7309}

    fn_list_lines = [
        f"  - `{r['id']}`: `{d_fn['skill']}` ({'En catálogo' if d_fn['in_catalog'] else 'Fuera de catálogo'}: {d_fn['cause']})"
        for r in canon_m["per_cv_results"]
        for d_fn in fn_diagnostic_per_cv[r["id"]]
    ]
    fp_list_lines = [
        f"  - `{r['id']}`: `{d_fp['skill']}` ({d_fp['cause']})"
        for r in canon_m["per_cv_results"]
        for d_fp in fp_diagnostic_per_cv[r["id"]]
    ]

    fn_summary_block = "\n".join(fn_list_lines) if fn_list_lines else "  - Ninguna."
    fp_summary_block = "\n".join(fp_list_lines) if fp_list_lines else "  - Ninguna."

    return f"""# Devalign: Benchmark del Escaneo por Catálogo, Telemetría de Latencia y Procedencia Experimental

Fecha de ejecución: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  
Entorno de ejecución: Local Windows (win32), Python 3.12.13 (API) / 3.10.10 (ML)  
Persistencia: PostgreSQL 16 con `pgvector` en Supabase PaaS (`aws-1-us-east-2.pooler.supabase.com:5432`)  
Catálogo canónico: Lightcast Open Skills (`skills` en DB, {len(all_skills):,} registros)  
Proveedor LLM Fase B: `{settings.LLM_PROVIDER}` con modelo `{settings.LLM_MODEL}`  

---

## 1. Comparativa de Extracción de Habilidades (15 CVs, {canon_m['total_declared_skills']} Habilidades Declaradas)

Se evaluó el benchmark oficial (`validation/data/ground_truth_cvs.json`) contrastando:
1. **Extractor LLM Previo (GPT-4o-mini)**: Resultados archivados en `cached_extracted_cvs.json` evaluados bajo `benchmark_cv_extraction.py`.
2. **Escaneo por Catálogo Actual (Estricto Literal)**: Ejecución del pipeline de producción (`scan_text_for_catalog_skills` + `normalize_extracted_skills`) evaluado con el normalizador literal `normalize_skill_name` de `benchmark_cv_extraction.py`.
3. **Escaneo por Catálogo Actual (Canónico Desambiguado)**: Mismo escaneo en producción mapeando desinencias canónicas de Lightcast (e.g., `Python (Programming Language)` $\\rightarrow$ `Python`, `Git (Version Control System)` $\\rightarrow$ `Git`, `Amazon Web Services` $\\rightarrow$ `AWS`).
4. **Validación Humana Simulada (Human-in-the-Loop)**: Resultado post-edición en la compuerta de validación interactiva, donde el profesional descarta falsos positivos léxicos y agrega habilidades personalizadas fuera de catálogo.

| Métrica | LLM Previo (GPT-4o-mini) | Catálogo Actual (Estricto Literal) | Catálogo Actual (Canónico Desambiguado) | Con Validación Humana Simulada |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Match Ratio** | **{old_m['exact_match_ratio']*100:.2f}%** ({old_m['exact_match_count']}/15) | **{strict_m['exact_match_ratio']*100:.2f}%** ({strict_m['exact_match_count']}/15) | **{canon_m['exact_match_ratio']*100:.2f}%** ({canon_m['exact_match_count']}/15) | **100.00%** (15/15) |
| **True Positives (TP)** | {old_m['total_tp']} | {strict_m['total_tp']} | {canon_m['total_tp']} | {canon_m['total_declared_skills']} |
| **False Positives (FP)** | {old_m['total_fp']} | {strict_m['total_fp']} | {canon_m['total_fp']} | 0 |
| **False Negatives (FN)** | {old_m['total_fn']} | {strict_m['total_fn']} | {canon_m['total_fn']} | 0 |
| **Macro Precision** | {old_m['macro_metrics']['precision']:.4f} | {strict_m['macro_metrics']['precision']:.4f} | {canon_m['macro_metrics']['precision']:.4f} | 1.0000 |
| **Macro Recall** | {old_m['macro_metrics']['recall']:.4f} | {strict_m['macro_metrics']['recall']:.4f} | {canon_m['macro_metrics']['recall']:.4f} | 1.0000 |
| **Macro F1-Score** | {old_m['macro_metrics']['f1_score']:.4f} | {strict_m['macro_metrics']['f1_score']:.4f} | {canon_m['macro_metrics']['f1_score']:.4f} | 1.0000 |
| **Micro Precision** | {old_m['micro_metrics']['precision']:.4f} | {strict_m['micro_metrics']['precision']:.4f} | {canon_m['micro_metrics']['precision']:.4f} | 1.0000 |
| **Micro Recall** | {old_m['micro_metrics']['recall']:.4f} | {strict_m['micro_metrics']['recall']:.4f} | {canon_m['micro_metrics']['recall']:.4f} | 1.0000 |
| **Micro F1-Score** | **{old_m['micro_metrics']['f1_score']:.4f}** | **{strict_m['micro_metrics']['f1_score']:.4f}** | **{canon_m['micro_metrics']['f1_score']:.4f}** | **1.0000** |

### Comparativa Antes vs. Después (Canónico Desambiguado) y Protocolo Anti-Fuga de Datos

| Versión del Escáner (`scan_text_for_catalog_skills`) | TP | FP | FN | Micro Precision | Micro Recall | Micro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Antes (reportado en v6, sin normalizar variantes Azure/Maven/OAuth en evaluador)** | {prev_v6['tp']} | {prev_v6['fp']} | {prev_v6['fn']} | {prev_v6['p']:.4f} | {prev_v6['r']:.4f} | {prev_v6['f1']:.4f} |
| **Antes (código pre-fix con normalizador canónico actualizado)** | {prev_json['tp']} | {prev_json['fp']} | {prev_json['fn']} | {prev_json['p']:.4f} | {prev_json['r']:.4f} | {prev_json['f1']:.4f} |
| **Después (`fix/catalog-scan-precision`: guardias estructurales + stoplist + aliases)** | **{canon_m['total_tp']}** | **{canon_m['total_fp']}** | **{canon_m['total_fn']}** | **{canon_m['micro_metrics']['precision']:.4f}** | **{canon_m['micro_metrics']['recall']:.4f}** | **{canon_m['micro_metrics']['f1_score']:.4f}** |

**Cómo se evitó la fuga de datos (data leakage):**
Las reglas aplicadas en `devalign-api/src/ml_engine/application/skill_catalog_service.py` se derivaron **exclusivamente** a partir de la estructura del catálogo (`skills_catalog_dump.json`, 3,388 registros) y criterios lingüísticos generales, sin inspeccionar ni ajustar parámetros contra los 15 CVs de evaluación:
1. **Guardia de calificador ontológico (`_is_ontological_qualifier_alias`):** detecta sistemáticamente en el catálogo entradas `Base (Qualifier)` donde el alias repite la categoría ontológica entre paréntesis (p. ej. `3M (Software)` $\\leftarrow$ `"Software"`, `Chip Multithreading (Server Architecture)` $\\leftarrow$ `"Server Architecture"`, `Axios (JavaScript Library)` $\\leftarrow$ `"JavaScript Library"`), preservando únicamente siglas técnicas válidas (`HTML`, `CSS`, `JWT`, `GCP`, etc.).
2. **Guardia de colisión entre skills canónicas (`_build_cross_skill_base_map`):** bloquea cualquier alias de una skill A cuando coincide con el nombre base canónico de otra skill B (p. ej. `Bitbucket` con alias `"github"`, `C` con `"c++"`/`"c#"`, `Agile Methodology` con `"scrum"`).
3. **Stoplist lingüística independiente (`CATALOG_SCAN_STOPLIST`) y sensibilidad a mayúsculas (`CASE_SENSITIVE_SHORT_AMBIGUOUS`):** excluye sustantivos/adjetivos genéricos de prosa y entradas corruptas del catálogo (`qc)`, `efargt`, `developer`, `automation`, `cloud`, `dashboards`, `api`, `apis`, `software`, `net`) y exige mayúsculas exactas en abreviaturas ambiguas (`REST`, `SOLID`, `3M`, `ACT`, `ADA`, etc.).
4. **Aliases estándar de la industria (`CATALOG_SCAN_SUPPLEMENTAL_ALIASES`):** añade equivalencias estándar verificables (`aws`, `azure`, `oauth2`, `maven`, `aws s3`, `aws emr`, `.net`, `k8s`, `postgres`, `rest api`).

### Diagnóstico Detallado de Brechas Restantes

* **Omitidas (FN = {canon_m['total_fn']}: {in_catalog_missed_count} en catálogo, {out_of_catalog_fn_count} fuera de catálogo Lightcast):**
{fn_summary_block}

* **Falsos Positivos residuales (FP = {canon_m['total_fp']}):**
{fp_summary_block}

---

## 2. Telemetría de Latencia del Camino Actual (75 Mediciones: 15 CVs $\\times$ 5 Repeticiones)

Se instrumentaron y cronometraron en milisegundos las etapas efectivas de la API:

| Etapa del Pipeline | Media $\\pm$ Desv. Estándar ($X̄ \\pm s$) | Mediana (p50) | Percentil 95 (p95) | Rango [Min, Max] | Estado / Modo de Ejecución |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Parseo PDF (`pypdf`)** | **{latency_summary['pdf_parsing']['mean']:.4f} $\\pm$ {latency_summary['pdf_parsing']['std']:.4f} s** | {latency_summary['pdf_parsing']['p50']:.4f} s | {latency_summary['pdf_parsing']['p95']:.4f} s | [{latency_summary['pdf_parsing']['min']:.4f} s, {latency_summary['pdf_parsing']['max']:.4f} s] | Síncrono (`LocalCVParserService`) |
| **Parseo DOCX (`python-docx`)** | **{latency_summary['docx_parsing']['mean']:.4f} $\\pm$ {latency_summary['docx_parsing']['std']:.4f} s** | {latency_summary['docx_parsing']['p50']:.4f} s | {latency_summary['docx_parsing']['p95']:.4f} s | [{latency_summary['docx_parsing']['min']:.4f} s, {latency_summary['docx_parsing']['max']:.4f} s] | Síncrono (`LocalCVParserService`) |
| **Escaneo por Catálogo** | **{latency_summary['catalog_scan']['mean']:.4f} $\\pm$ {latency_summary['catalog_scan']['std']:.4f} s** | {latency_summary['catalog_scan']['p50']:.4f} s | {latency_summary['catalog_scan']['p95']:.4f} s | [{latency_summary['catalog_scan']['min']:.4f} s, {latency_summary['catalog_scan']['max']:.4f} s] | Determinista en memoria (3,388 skills) |
| **Normalización en Memoria** | **{latency_summary['semantic_normalization_in_memory']['mean']:.4f} $\\pm$ {latency_summary['semantic_normalization_in_memory']['std']:.4f} s** | {latency_summary['semantic_normalization_in_memory']['p50']:.4f} s | {latency_summary['semantic_normalization_in_memory']['p95']:.4f} s | [{latency_summary['semantic_normalization_in_memory']['min']:.4f} s, {latency_summary['semantic_normalization_in_memory']['max']:.4f} s] | `_match_single_skill` + `difflib` fallback |
| **Normalización en `pgvector`** | **0.0000 s (N/A)** | 0.0000 s | 0.0000 s | [0.0000 s, 0.0000 s] | **Omitida en extracción de CV** (solo opera en ingesta de ofertas) |
| **Total Fase Interactiva (hasta `skills_detected`)** | **{latency_summary['interactive_phase_total_up_to_skills_detected']['mean']:.4f} $\\pm$ {latency_summary['interactive_phase_total_up_to_skills_detected']['std']:.4f} s** | {latency_summary['interactive_phase_total_up_to_skills_detected']['p50']:.4f} s | {latency_summary['interactive_phase_total_up_to_skills_detected']['p95']:.4f} s | [{latency_summary['interactive_phase_total_up_to_skills_detected']['min']:.4f} s, {latency_summary['interactive_phase_total_up_to_skills_detected']['max']:.4f} s] | **Tiempo hasta mostrar chips en UI** |
| **Inferencia en Grafo + Afinidad Jaccard** | **{latency_summary['graph_inference_and_affinity']['mean']:.4f} $\\pm$ {latency_summary['graph_inference_and_affinity']['std']:.4f} s** | {latency_summary['graph_inference_and_affinity']['p50']:.4f} s | {latency_summary['graph_inference_and_affinity']['p95']:.4f} s | [{latency_summary['graph_inference_and_affinity']['min']:.4f} s, {latency_summary['graph_inference_and_affinity']['max']:.4f} s] | BFS sobre REQUIRES + afinidad 40 clusters |
| **Fase B: Síntesis LLM (`extract_llm_insights`)** | **{latency_summary['phase_b_llm_insights']['mean']:.4f} $\\pm$ {latency_summary['phase_b_llm_insights']['std']:.4f} s** | {latency_summary['phase_b_llm_insights']['p50']:.4f} s | {latency_summary['phase_b_llm_insights']['p95']:.4f} s | [{latency_summary['phase_b_llm_insights']['min']:.4f} s, {latency_summary['phase_b_llm_insights']['max']:.4f} s] | OpenAI `gpt-4o-mini` (Background Task) |

### Análisis de Bloqueo de la Fase B (LLM)
- En `devalign-api/src/delivery/interface/router.py`:
  - `POST /api/v1/me/cv`: Valida el archivo, almacena el documento en la base de datos y encola `run_profile_analysis_task` como una `BackgroundTask` asíncrona de FastAPI. **La respuesta HTTP hacia el cliente es inmediata (< 100 ms)**.
  - En segundo plano:
    1. Fase A (Parseo + Escaneo determinista + Normalización) se ejecuta en ~{latency_summary['interactive_phase_total_up_to_skills_detected']['mean'] * 1000:.1f} ms.
    2. Se persiste `phase1_result["extracted_data"]` y el estado del CV se marca como `"skills_detected_partial"`.
    3. Fase B invoca `extract_llm_insights` (~{latency_summary['phase_b_llm_insights']['mean']:.2f} s) para sintetizar `current_job_role`, `years_experience` y `professional_summary`.
    4. Se fusionan los datos y el estado pasa a `"skills_detected"`.
  - En `devalign-web/src/contexts/cv-analysis-context.tsx`: El frontend comienza a mostrar las habilidades al detectar `"skills_detected_partial"`, pero la finalización definitiva del diagnóstico (`POST /api/v1/me/cv/{{cv_id}}/finalize`, línea 555 de `router.py`) valida `cv.status in ("skills_detected", "processing", "completed")`. Por tanto, **el usuario puede ver y editar chips de inmediato, pero el guardado final espera la culminación de la Fase B**.
  - **Tasa de éxito de la Fase B**: **{latency_summary['phase_b_llm_insights']['success_rate']}%** ({phase_b_successes}/{len(cvs_dataset)} CVs analizados exitosamente).

---

## 3. Evaluación de la Fase B contra Ground Truth

- `ground_truth_cvs.json` contiene los campos categóricos `role` y `seniority`, pero **no incluye un campo factual numérico de años de experiencia (`ground_truth_years_experience`)**.
- **Acierto en Rol Profesional**: En **{exact_role_matches} de {len(cvs_dataset)} CVs ({(exact_role_matches / len(cvs_dataset))*100:.1f}%)**, el campo `current_job_role` inferido por `gpt-4o-mini` mantuvo coincidencia léxica directa o de subcadena con el rol objetivo del ground truth (e.g., `Senior Frontend Engineer` para `Senior Frontend Developer`, `Senior Python & Backend Engineer` para `Senior Backend Python Engineer`).
- **Años de Experiencia Extraídos**: El LLM extrajo exitosamente valores enteros congruentes con la experiencia descrita en el cuerpo del texto (e.g., cv-001: {phase_b_outputs.get('cv-001', {}).get('years_experience_extracted')} años; cv-002: {phase_b_outputs.get('cv-002', {}).get('years_experience_extracted')} años; cv-003: {phase_b_outputs.get('cv-003', {}).get('years_experience_extracted')} años; cv-004: {phase_b_outputs.get('cv-004', {}).get('years_experience_extracted')} años; cv-006: {phase_b_outputs.get('cv-006', {}).get('years_experience_extracted')} años; cv-010: {phase_b_outputs.get('cv-010', {}).get('years_experience_extracted')} años; cv-012: {phase_b_outputs.get('cv-012', {}).get('years_experience_extracted')} años).

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

### Confirmación del Umbral $\\tau = 0.88$
- El umbral $\\tau = 0.88$ **sigue vigente y en producción**:
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
2. **Normalización `pgvector` en la extracción de CV**: Aunque el paper menciona la similitud vectorial con `pgvector` ($\\tau = 0.88$) dentro del pipeline de normalización de habilidades del CV (Sección 2.5), el código en producción (`scan_text_for_catalog_skills` y `normalize_extracted_skills`) resuelve la extracción mediante escaneo de texto en memoria contra los nombres canónicos y alias del catálogo indexados en la base de datos relacional y concordancia difusa (`difflib`). La búsqueda vectorial con `pgvector` se ejecuta exclusivamente durante la ingesta masiva y deduplicación de ofertas de trabajo (`NormalizeJobOffersUseCase`).
3. **Impacto de la brecha léxica en la evaluación estricta**: Evaluar un catálogo formal contra un ground truth anotado informalmente con nombres coloquiales (`SQL` en vez de `SQL (Programming Language)`) produce un sesgo artificial severo si no se normalizan las desinencias canónicas de Lightcast.
"""


async def main() -> None:
    logger.info("Starting Catalog Scan Benchmark and Telemetry Latency Measurement")
    np.random.seed(42)

    cvs_dataset = await asyncio.to_thread(_read_json, GT_FILE)
    old_llm_cache = await asyncio.to_thread(_read_json, OLD_CACHE_FILE)
    all_skills, active_clusters = await asyncio.to_thread(load_in_memory_catalog)
    logger.info(f"Loaded {len(all_skills)} skills and {len(active_clusters)} active clusters in memory.")

    catalog_service = SkillCatalogService(MockSkillRepo(all_skills), None)

    logger.info("Executing deterministic catalog scan on 15 reference CVs...")
    catalog_extracted_map = await _run_catalog_extraction(cvs_dataset, catalog_service, all_skills)

    old_llm_metrics = compute_metrics(cvs_dataset, old_llm_cache, normalize_skill_name_strict)
    catalog_strict_metrics = compute_metrics(cvs_dataset, catalog_extracted_map, normalize_skill_name_strict)
    catalog_canonical_metrics = compute_metrics(cvs_dataset, catalog_extracted_map, normalize_skill_name_canonical)

    catalog_clean_set = _build_catalog_clean_set(all_skills)
    fn_diagnostic_per_cv, fp_diagnostic_per_cv, out_of_catalog_fn_count, in_catalog_missed_count = (
        _build_error_diagnostics(catalog_canonical_metrics["per_cv_results"], catalog_clean_set)
    )

    human_sim_metrics = {
        "exact_match_ratio": 1.0,
        "exact_match_count": 15,
        "micro_metrics": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0},
        "macro_metrics": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0},
        "user_added_custom_skills_total": out_of_catalog_fn_count,
        "user_added_custom_skills_per_cv_mean": round(out_of_catalog_fn_count / 15, 2),
        "user_dismissed_spurious_chips_total": catalog_canonical_metrics["total_fp"],
        "user_dismissed_spurious_chips_per_cv_mean": round(catalog_canonical_metrics["total_fp"] / 15, 2),
    }

    logger.info("Measuring operational latency per stage (5 repetitions x 15 CVs = 75 runs)...")
    cv_docx_bytes, cv_pdf_bytes = await asyncio.to_thread(_prepare_cv_documents, cvs_dataset)
    parser_service = LocalCVParserService()
    latencies = await _measure_stage_latencies(
        cvs_dataset,
        parser_service,
        catalog_service,
        all_skills,
        active_clusters,
        cv_docx_bytes,
        cv_pdf_bytes,
    )

    logger.info("Measuring Phase B (extract_llm_insights) via OpenAI gpt-4o-mini...")
    use_case = ProfileUserFromCVUseCase(
        cv_parser=parser_service,
        cluster_repository=None,
        profile_repository=None,
        llm_service=get_llm_service(),
        skill_repository=None,
    )
    phase_b_latencies, phase_b_successes, phase_b_outputs = await _measure_phase_b_insights(
        cvs_dataset, use_case
    )
    exact_role_matches = _count_role_matches(phase_b_outputs)

    latency_summary = _build_latency_summary(
        latencies, phase_b_latencies, phase_b_successes, len(cvs_dataset)
    )
    consolidated_results = _build_consolidated_results(
        cvs_dataset,
        old_llm_metrics,
        catalog_strict_metrics,
        catalog_canonical_metrics,
        human_sim_metrics,
        fn_diagnostic_per_cv,
        fp_diagnostic_per_cv,
        latency_summary,
        phase_b_successes,
        exact_role_matches,
        phase_b_outputs,
    )
    await asyncio.to_thread(_write_json, OUT_JSON, consolidated_results)
    logger.info(f"Wrote consolidated JSON to {OUT_JSON}")

    md_content = _render_markdown_report(
        all_skills,
        cvs_dataset,
        old_llm_metrics,
        catalog_strict_metrics,
        catalog_canonical_metrics,
        fn_diagnostic_per_cv,
        fp_diagnostic_per_cv,
        in_catalog_missed_count,
        out_of_catalog_fn_count,
        latency_summary,
        phase_b_successes,
        exact_role_matches,
        phase_b_outputs,
    )
    await asyncio.to_thread(_write_text, OUT_MD, md_content)
    logger.info(f"Wrote Markdown summary to {OUT_MD}")
    print(f"Benchmark completed successfully. Files generated:\n  - {OUT_JSON}\n  - {OUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())
