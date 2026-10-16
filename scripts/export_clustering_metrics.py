"""Export Clustering Metrics (RQ1).

Evaluates the unsupervised clustering pipeline (UMAP 15d + HDBSCAN)
over the job offer dataset, computing:
1. Total offers processed by source portal.
2. Number of clusters discovered (k) and size distribution.
3. Count and percentage of noise offers (label -1).
4. Silhouette score in UMAP reduced space (15d, euclidean).
5. Silhouette score in original space (1024d, cosine).
6. Calibrated clusters distribution after quality filters and dedup.

Exports results to `validation/results/clustering_metrics.json`.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

# Add ml root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import config
from src.clustering import ClusteringPipeline
from src.data_loader import (
    GENERIC_OFFICE_SKILLS,
    get_supabase_client,
    load_raw_data,
    process_job_offer_vectors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("export_clustering_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RQ1 Clustering Metrics for Devalign Paper")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local cache and query Supabase directly.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit/sample size for offers (e.g. 1276). Default: all valid technical offers.",
    )
    parser.add_argument(
        "--portals",
        nargs="+",
        default=None,
        help="Filter specific portals (e.g. getonboard computrabajo). Default: all available portals.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=os.path.join(ROOT_DIR, "validation", "results", "clustering_metrics.json"),
        help="Output JSON file path.",
    )
    return parser.parse_args()


def load_dataset(use_cache: bool = True, portals: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Loads raw skills and offers with cascade lookup (raw_offers.json or raw_job_offers.json)."""
    offers_raw_candidates = [
        os.path.join(config.DATA_DIR, "raw_offers.json"),
        os.path.join(config.DATA_DIR, "raw_job_offers.json"),
    ]
    skills_cache_path = os.path.join(config.DATA_DIR, "raw_skills.json")

    skills_data = []
    offers_data = []

    if use_cache:
        for candidate_path in offers_raw_candidates:
            if os.path.exists(candidate_path) and os.path.exists(skills_cache_path):
                logger.info(f"Loading raw data from cache file: {candidate_path}")
                try:
                    with open(candidate_path, "r", encoding="utf-8") as f:
                        offers_data = json.load(f)
                    with open(skills_cache_path, "r", encoding="utf-8") as f:
                        skills_data = json.load(f)
                    break
                except Exception as exc:
                    logger.warning(f"Failed to read from {candidate_path}: {exc}")

    if not offers_data or not skills_data:
        logger.info("Cache miss or disabled. Attempting to load from Supabase...")
        skills_data, offers_data = load_raw_data(use_cache=False)

    if portals:
        portal_set = {p.lower().strip() for p in portals}
        offers_data = [o for o in offers_data if (o.get("portal") or "").lower().strip() in portal_set]
        logger.info(f"Filtered offers to portals {portals}: {len(offers_data)} remaining.")

    return skills_data, offers_data


def compute_metrics(
    valid_offers: list[dict],
    pipeline_results: dict[str, Any],
) -> dict[str, Any]:
    """Computes all clustering metrics for RQ1."""
    labels = np.array(pipeline_results["labels"])
    X_reduced = pipeline_results["X_reduced"]
    vectors = np.array([o["vector"] for o in valid_offers], dtype=np.float32)

    total_offers = len(valid_offers)
    noise_count = int(np.sum(labels == -1))
    noise_pct = round((noise_count / total_offers) * 100.0, 2) if total_offers > 0 else 0.0

    non_noise_mask = labels != -1
    unique_clusters = sorted([int(lbl) for lbl in set(labels) if lbl != -1])
    k_clusters = len(unique_clusters)

    # 1. Offers by source portal
    offers_by_portal = collections.Counter()
    for offer in valid_offers:
        p = offer.get("portal") or "unknown"
        offers_by_portal[p] += 1

    # 2. Size distribution per cluster
    cluster_sizes: dict[int, int] = {}
    for lbl in unique_clusters:
        cluster_sizes[lbl] = int(np.sum(labels == lbl))

    cluster_size_distribution = {
        "min_size": int(min(cluster_sizes.values())) if cluster_sizes else 0,
        "max_size": int(max(cluster_sizes.values())) if cluster_sizes else 0,
        "median_size": float(np.median(list(cluster_sizes.values()))) if cluster_sizes else 0.0,
        "mean_size": round(float(np.mean(list(cluster_sizes.values()))), 2) if cluster_sizes else 0.0,
        "per_cluster": {str(k): v for k, v in cluster_sizes.items()},
    }

    # 3. Silhouette in UMAP space (15d)
    sil_umap = 0.0
    if k_clusters >= 2 and np.sum(non_noise_mask) > k_clusters:
        try:
            sil_umap = float(silhouette_score(X_reduced[non_noise_mask], labels[non_noise_mask], metric="euclidean"))
            logger.info(f"Silhouette Score (UMAP 15d, euclidean): {sil_umap:.4f}")
        except Exception as exc:
            logger.error(f"Failed to compute UMAP silhouette: {exc}")

    # 4. Silhouette in Original space (1024d, cosine)
    sil_orig = 0.0
    if k_clusters >= 2 and np.sum(non_noise_mask) > k_clusters:
        try:
            # Subsample if dataset is large to avoid OOM in full N x N pairwise matrix
            n_clustered = int(np.sum(non_noise_mask))
            if n_clustered > 4000:
                sample_indices = np.random.RandomState(42).choice(n_clustered, size=4000, replace=False)
                sampled_vecs = vectors[non_noise_mask][sample_indices]
                sampled_labels = labels[non_noise_mask][sample_indices]
                sil_orig = float(silhouette_score(sampled_vecs, sampled_labels, metric="cosine"))
            else:
                sil_orig = float(silhouette_score(vectors[non_noise_mask], labels[non_noise_mask], metric="cosine"))
            logger.info(f"Silhouette Score (Original 1024d, cosine): {sil_orig:.4f}")
        except Exception as exc:
            logger.error(f"Failed to compute 1024d silhouette: {exc}")

    return {
        "total_offers_evaluated": total_offers,
        "offers_by_portal": dict(offers_by_portal),
        "discovered_clusters_k": k_clusters,
        "noise_offers_count": noise_count,
        "noise_percentage": noise_pct,
        "clustered_offers_count": total_offers - noise_count,
        "clustered_percentage": round(100.0 - noise_pct, 2),
        "silhouette_score_umap_15d": round(sil_umap, 4),
        "silhouette_score_original_1024d": round(sil_orig, 4),
        "cluster_size_distribution": cluster_size_distribution,
    }


def main():
    args = parse_args()
    logger.info("Initializing Clustering Metrics Export Pipeline...")

    skills_data, offers_data = load_dataset(use_cache=not args.no_cache, portals=args.portals)
    logger.info(f"Loaded {len(skills_data)} skills and {len(offers_data)} raw offers.")

    valid_offers, _ = process_job_offer_vectors(skills_data, offers_data)
    logger.info(f"Extracted {len(valid_offers)} valid technical offers.")

    if args.limit and args.limit < len(valid_offers):
        rng = np.random.RandomState(config.UMAP_RANDOM_STATE)
        indices = rng.choice(len(valid_offers), size=args.limit, replace=False)
        valid_offers = [valid_offers[i] for i in indices]
        logger.info(f"Applied limit/sample: evaluated {len(valid_offers)} offers.")

    vectors = np.array([o["vector"] for o in valid_offers], dtype=np.float32)

    pipeline = ClusteringPipeline()
    results = pipeline.run(vectors)

    metrics = compute_metrics(valid_offers, results)
    metrics["timestamp"] = datetime.utcnow().isoformat()
    metrics["hyperparameters"] = {
        "umap_n_neighbors": config.UMAP_N_NEIGHBORS,
        "umap_n_components": config.UMAP_N_COMPONENTS,
        "umap_min_dist": config.UMAP_MIN_DIST,
        "umap_metric": config.UMAP_METRIC,
        "umap_random_state": config.UMAP_RANDOM_STATE,
        "hdbscan_min_cluster_size": config.HDBSCAN_MIN_CLUSTER_SIZE,
        "hdbscan_min_samples": config.HDBSCAN_MIN_SAMPLES,
        "hdbscan_metric": config.HDBSCAN_METRIC,
        "transversal_penalty_threshold": config.TRANSVERSAL_PENALTY_THRESHOLD,
        "transversal_penalty_factor": config.TRANSVERSAL_PENALTY_FACTOR,
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Clustering metrics successfully saved to {args.output_file}")
    print("\n" + "=" * 60)
    print("RQ1 CLUSTERING METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Offers Evaluated: {metrics['total_offers_evaluated']}")
    print(f"Portals Breakdown:      {metrics['offers_by_portal']}")
    print(f"Clusters Discovered (k): {metrics['discovered_clusters_k']}")
    print(f"Noise Count / Pct:      {metrics['noise_offers_count']} ({metrics['noise_percentage']}%)")
    print(f"Silhouette (UMAP 15d):  {metrics['silhouette_score_umap_15d']}")
    print(f"Silhouette (1024d):     {metrics['silhouette_score_original_1024d']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
