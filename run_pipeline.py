import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from uuid import uuid4

# Set up logging before imports
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

from src import config
from src.data_loader import load_raw_data, process_job_offer_vectors, get_supabase_client, GENERIC_OFFICE_SKILLS
from src.clustering import ClusteringPipeline
from src.services.naming_service import get_naming_strategy
from src.utils import plot_and_save_clusters

def parse_args():
    parser = argparse.ArgumentParser(description="Devalign ML Clustering Pipeline (UMAP + HDBSCAN)")
    parser.add_argument("--deploy", action="store_true", help="Deploy/write results to Supabase database.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local JSON cache and force DB fetch.")
    parser.add_argument(
        "--portals",
        nargs="+",
        default=["computrabajo", "getonboard", "arbeitnow", "remotive", "weworkremotely"],
        help="Portals to include in clustering (default: computrabajo getonboard arbeitnow remotive weworkremotely)."
    )
    parser.add_argument("--plot-name", type=str, default="hdbscan_clusters.png", help="Filename of the saved plot.")
    return parser.parse_args()

def deploy_to_supabase(client, valid_offers, final_labels, cluster_results, all_skills):
    """Deploys clustering results safely to Supabase (in-place cosine sync, zero user data loss)."""
    logger.info("Starting safe deployment to Supabase...")

    # 1. Fetch existing clusters to match by centroid cosine similarity
    existing_clusters_resp = client.table("clusters").select("cluster_id, name, centroid_vec").execute()
    existing_clusters = existing_clusters_resp.data or []
    logger.info(f"Loaded {len(existing_clusters)} existing clusters from Supabase.")

    # 2. Get existing canonical skills mapping
    existing_skills_resp = client.table("skills").select("skill_id, name").execute()
    existing_skills_map = {s["name"]: s["skill_id"] for s in (existing_skills_resp.data or [])}

    # Ensure all canonical skills exist in the skills table
    new_skills_to_insert = []
    for s_name in all_skills:
        if s_name not in existing_skills_map:
            new_skills_to_insert.append({"name": s_name, "nature": "tech", "weight": 1.0})

    if new_skills_to_insert:
        logger.info(f"Inserting {len(new_skills_to_insert)} missing canonical skills...")
        inserted_skills = client.table("skills").upsert(new_skills_to_insert, on_conflict="name").execute()
        for s in (inserted_skills.data or []):
            existing_skills_map[s["name"]] = s["skill_id"]

    # 3. Synchronize clusters in-place based on centroid cosine similarity (threshold >= 0.88)
    used_existing_cluster_ids = set()
    cluster_id_mapping = {}

    for cr in cluster_results:
        c_vec = np.array(cr["centroid_vec"], dtype=np.float32)
        c_norm = c_vec / (np.linalg.norm(c_vec) + 1e-9)

        matched_id = None
        best_sim = -1.0

        for ec in existing_clusters:
            ec_id = ec["cluster_id"]
            if ec_id in used_existing_cluster_ids:
                continue
            ec_vec = ec.get("centroid_vec")
            if ec_vec and isinstance(ec_vec, list) and len(ec_vec) == 1024:
                k_vec = np.array(ec_vec, dtype=np.float32)
                k_norm = k_vec / (np.linalg.norm(k_vec) + 1e-9)
                sim = float(np.dot(c_norm, k_norm))
                if sim > best_sim:
                    best_sim = sim
                    if sim >= 0.88:
                        matched_id = ec_id

        cluster_payload = {
            "name": cr["name"],
            "description": cr["description"],
            "job_offer_count": cr["job_offer_count"],
            "compatible_roles": cr["compatible_roles"],
            "centroid_vec": cr["centroid_vec"],
            "market_insights": {
                "growth_percentage": cr.get("growth_percentage"),
                "market_share_percentage": cr["market_share_percentage"],
                "market_tier": cr.get("market_tier", "Tier 2 (Tech & Remoto Global)"),
                "average_salary_usd": cr.get("average_salary_usd"),
                "salary_p25_usd": cr.get("salary_p25_usd"),
                "salary_median_usd": cr.get("salary_median_usd"),
                "salary_p75_usd": cr.get("salary_p75_usd"),
                "salary_differential_percentage": cr.get("salary_differential_percentage"),
                "has_salary_benchmark": cr.get("has_salary_benchmark", False),
            }
        }

        if matched_id:
            used_existing_cluster_ids.add(matched_id)
            target_cluster_id = matched_id
            logger.info(f"Cluster '{cr['name']}' matched existing ID {target_cluster_id} (cosine={best_sim:.3f}). Updating in-place...")
            client.table("clusters").update(cluster_payload).eq("cluster_id", target_cluster_id).execute()
        else:
            target_cluster_id = str(uuid4())
            logger.info(f"Cluster '{cr['name']}' is new. Inserting with new ID {target_cluster_id}...")
            insert_data = dict(cluster_payload)
            insert_data["cluster_id"] = target_cluster_id
            client.table("clusters").insert(insert_data).execute()

        cluster_id_mapping[cr["cluster_index"]] = target_cluster_id

        # Update cluster_skills relations for this cluster
        try:
            client.table("cluster_skills").delete().eq("cluster_id", target_cluster_id).execute()
        except Exception as e:
            logger.warning(f"Error purging old skills for cluster {target_cluster_id}: {e}")

        cs_inserts = []
        for cs in cr["skills"]:
            s_id = existing_skills_map.get(cs["name"])
            if s_id:
                cs_inserts.append({
                    "cluster_id": target_cluster_id,
                    "skill_id": s_id,
                    "importance_score": cs["frequency"] * cs["weight"],
                })

        if cs_inserts:
            client.table("cluster_skills").insert(cs_inserts).execute()

    # 4. Unlink and re-link job offers
    logger.info("Unlinking outdated job offer cluster links...")
    try:
        client.table("job_offers").update({"cluster_id": None}).not_.is_("cluster_id", "null").execute()
    except Exception as e:
        logger.warning(f"Unlinking job offers note: {e}")

    logger.info("Linking job offers to updated clusters...")
    for cr in cluster_results:
        target_cluster_id = cluster_id_mapping.get(cr["cluster_index"])
        if not target_cluster_id:
            continue
        offer_ids = cr["offer_ids"]
        for i in range(0, len(offer_ids), 100):
            chunk = offer_ids[i : i + 100]
            client.table("job_offers").update({"cluster_id": target_cluster_id}).in_(
                "job_offer_id", chunk
            ).execute()

    logger.info("Deployment finished successfully with zero user data loss!")

def main():
    args = parse_args()
    
    # 1. Load Data
    try:
        skills_data, offers_data = load_raw_data(use_cache=not args.no_cache, target_portals=args.portals)
    except Exception as e:
        logger.critical(f"Data loading failed: {e}")
        sys.exit(1)

    # 2. Process Vectors
    valid_offers, alias_to_canonical = process_job_offer_vectors(skills_data, offers_data)
    if not valid_offers:
        logger.error("No valid job offers to cluster. Exiting.")
        sys.exit(1)

    vectors = np.array([o["vector"] for o in valid_offers])

    # 3. Run Clustering Pipeline
    pipeline = ClusteringPipeline()
    results = pipeline.run(vectors)

    sil_score = results["silhouette_score"]
    logger.info("=" * 60)
    logger.info(f"METRIC - Silhouette Score (excl. noise): {sil_score:.4f}")
    logger.info("=" * 60)

    # Minimum silhouette score safety gate before deployment
    if args.deploy and sil_score < 0.35:
        logger.critical(
            f"Quality gate failure: Silhouette Score ({sil_score:.4f}) is below minimum production threshold (0.35). "
            f"Aborting deployment to prevent catalog corruption."
        )
        sys.exit(1)

    # 4. Generate Visualization
    plot_path = plot_and_save_clusters(results["X_reduced"], results["final_labels"], args.plot_name)

    # 5. Extract Cluster Insights
    final_labels = results["final_labels"]
    non_noise_labels = results.get("non_noise_labels", [])

    if not non_noise_labels:
        logger.error("No clusters were found. Deployment skipped.")
        sys.exit(1)

    # Build occurrences DataFrame
    skill_counts = {}
    for offer in valid_offers:
        for s in offer["skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1
    all_skills = sorted(list(skill_counts.keys()))

    matrix_data = []
    for idx, offer in enumerate(valid_offers):
        row_dict = {
            "offer_id": offer["id"],
            "job_title": offer["job_title"],
            "cluster_label": final_labels[idx]
        }
        offer_skill_set = set(offer["skills"])
        for skill in all_skills:
            row_dict[skill] = 1 if skill in offer_skill_set else 0
        matrix_data.append(row_dict)

    df = pd.DataFrame(matrix_data)

    cluster_results_raw = []
    for k in non_noise_labels:
        cluster_df = df[df["cluster_label"] == k]
        cluster_size = len(cluster_df)

        # Compatible roles (Top 5 Job Titles)
        top_titles = cluster_df["job_title"].value_counts().head(5)
        compatible_roles = []
        for title, count in top_titles.items():
            match_level = "Alta" if count > (cluster_size * 0.1) else ("Media" if count > (cluster_size * 0.05) else "Baja")
            if str(title).strip():
                compatible_roles.append({
                    "title": str(title).strip().title(),
                    "match": match_level,
                    "frequency": int(count)
                })

        # Calculate skill frequencies (prioritizing technical skills over generic office tools)
        tech_skills_in_cluster = [s for s in all_skills if s.lower() not in GENERIC_OFFICE_SKILLS]
        frequencies = cluster_df[tech_skills_in_cluster].mean() if tech_skills_in_cluster else cluster_df[all_skills].mean()
        positive_frequencies = frequencies[frequencies > 0.0].sort_values(ascending=False)
        meaningful_skills = positive_frequencies[positive_frequencies >= 0.10]
        if len(meaningful_skills) == 0:
            meaningful_skills = positive_frequencies.head(3)
        top_skills = meaningful_skills

        top_5_names = list(top_skills.index[:5])

        skills_data_list = []
        for skill_name, freq in top_skills.items():
            weight = 3.0 if freq >= 0.6 else (2.0 if freq >= 0.3 else 1.0)
            skills_data_list.append({
                "name": skill_name,
                "frequency": float(freq),
                "weight": float(weight)
            })

        # Determine dominant market tier and salaries for this cluster
        cluster_offer_indices = [idx for idx, lbl in enumerate(final_labels) if lbl == k]
        cluster_offers_subset = [valid_offers[idx] for idx in cluster_offer_indices]

        tier_counts = {}
        for o in cluster_offers_subset:
            t = o.get("market_tier", "Tier 2 (Tech & Remoto Global)")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        dominant_tier = max(tier_counts.items(), key=lambda x: x[1])[0] if tier_counts else "Tier 2 (Tech & Remoto Global)"

        salaries = [o["salary_usd"] for o in cluster_offers_subset if o.get("salary_usd") is not None]
        has_salary_benchmark = len(salaries) >= 3
        if has_salary_benchmark:
            p25_usd = float(np.percentile(salaries, 25))
            p50_usd = float(np.percentile(salaries, 50))
            p75_usd = float(np.percentile(salaries, 75))
            avg_usd = float(np.mean(salaries))
            tier_ref_median = 5500.0 if "Tier 3" in dominant_tier else (1200.0 if "Tier 1" in dominant_tier else 2800.0)
            salary_diff_pct = round(((p50_usd / tier_ref_median) - 1.0) * 100, 1)
        else:
            p25_usd = None
            p50_usd = None
            p75_usd = None
            avg_usd = None
            salary_diff_pct = None

        cluster_results_raw.append({
            "cluster_index": k,
            "name": None,
            "description": None,
            "top_5_names": top_5_names,
            "num_meaningful_skills": len(meaningful_skills),
            "job_offer_count": cluster_size,
            "skills": skills_data_list,
            "compatible_roles": compatible_roles,
            "centroid_vec": results["centroids_1024d"][k],
            "offer_ids": cluster_df["offer_id"].tolist(),
            "market_share_percentage": round((cluster_size / len(valid_offers)) * 100, 1),
            "growth_percentage": None,
            "market_tier": dominant_tier,
            "average_salary_usd": round(avg_usd, 2) if avg_usd is not None else None,
            "salary_p25_usd": round(p25_usd, 2) if p25_usd is not None else None,
            "salary_median_usd": round(p50_usd, 2) if p50_usd is not None else None,
            "salary_p75_usd": round(p75_usd, 2) if p75_usd is not None else None,
            "salary_differential_percentage": salary_diff_pct,
            "has_salary_benchmark": has_salary_benchmark,
        })

    # 5b. Post-clustering quality filters
    # -------------------------------------------------------------------------
    before_filter = len(cluster_results_raw)
    filtered_low_skills = [
        cr for cr in cluster_results_raw
        if cr["num_meaningful_skills"] < config.MIN_SKILLS_PER_CLUSTER
    ]
    filtered_low_offers = [
        cr for cr in cluster_results_raw
        if cr["job_offer_count"] < config.MIN_OFFERS_PER_CLUSTER
        and cr["num_meaningful_skills"] >= config.MIN_SKILLS_PER_CLUSTER
    ]
    cluster_results = [
        cr for cr in cluster_results_raw
        if cr["num_meaningful_skills"] >= config.MIN_SKILLS_PER_CLUSTER
        and cr["job_offer_count"] >= config.MIN_OFFERS_PER_CLUSTER
    ]

    logger.info(
        f"Quality filter: {before_filter} raw → {len(cluster_results)} valid "
        f"(dropped {len(filtered_low_skills)} for < {config.MIN_SKILLS_PER_CLUSTER} skills, "
        f"{len(filtered_low_offers)} for < {config.MIN_OFFERS_PER_CLUSTER} offers)"
    )

    if not cluster_results:
        logger.error("All clusters were filtered out. Consider lowering MIN_SKILLS_PER_CLUSTER or MIN_OFFERS_PER_CLUSTER.")
        sys.exit(1)

    # 5c. Centroid deduplication — drop clusters whose centroids are too similar
    # -------------------------------------------------------------------------
    # Sort by offer count descending so that, when two centroids are duplicates,
    # we always keep the larger cluster.
    cluster_results.sort(key=lambda cr: cr["job_offer_count"], reverse=True)
    kept: list = []
    for cr in cluster_results:
        c_vec = np.array(cr["centroid_vec"], dtype=np.float32)
        c_norm = c_vec / (np.linalg.norm(c_vec) + 1e-9)
        is_dup = False
        for kept_cr in kept:
            k_vec = np.array(kept_cr["centroid_vec"], dtype=np.float32)
            k_norm = k_vec / (np.linalg.norm(k_vec) + 1e-9)
            cosine_sim = float(np.dot(c_norm, k_norm))
            if cosine_sim >= config.COSINE_DEDUP_THRESHOLD:
                logger.info(
                    f"Dedup: cluster {cr['cluster_index']} ({cr['job_offer_count']} offers, "
                    f"top={cr['top_5_names'][:3]}) is duplicate of cluster "
                    f"{kept_cr['cluster_index']} (cosine={cosine_sim:.3f}) — dropped."
                )
                is_dup = True
                break
        if not is_dup:
            kept.append(cr)

    dedup_dropped = len(cluster_results) - len(kept)
    cluster_results = kept
    logger.info(
        f"Centroid dedup: dropped {dedup_dropped} near-duplicate clusters "
        f"(threshold={config.COSINE_DEDUP_THRESHOLD}). Remaining: {len(cluster_results)}"
    )

    # 6. Generate Cluster Names via Strategy (Taxonomy / LLM)
    # -------------------------------------------------------------------------
    logger.info("Generating cluster names via configured naming strategy...")
    naming_strategy = get_naming_strategy()
    global_skill_freq = df[all_skills].mean()
    seen_names: dict = {}
    for cr in cluster_results:
        cluster_df = df[df["cluster_label"] == cr["cluster_index"]]
        cluster_freq = cluster_df[all_skills].mean()
        
        # Calculate distinctiveness ratio: (cluster_freq / (global_freq + 1e-4)) * cluster_freq
        distinctiveness = {}
        for s in cr["top_5_names"]:
            cf = float(cluster_freq.get(s, 0.0))
            gf = float(global_skill_freq.get(s, 0.0))
            score = (cf / (gf + 1e-4)) * (cf ** 1.5)
            distinctiveness[s] = score
        
        sorted_distinctive = dict(sorted(distinctiveness.items(), key=lambda x: x[1], reverse=True))
        name = naming_strategy.name_cluster(sorted_distinctive, cr["compatible_roles"])

        # Ensure uniqueness: if this name was already used, append a brief differentiator
        if name in seen_names:
            seen_names[name] += 1
            # Use the first "bonus" skill not already in the name as a disambiguator
            differentiator = ""
            for s in cr["top_5_names"]:
                clean = s.split("(")[0].strip().title()
                if clean.lower() not in name.lower():
                    differentiator = clean
                    break
            name = f"{name} ({differentiator})" if differentiator else f"{name} v{seen_names[name]}"
        else:
            seen_names[name] = 1

        cr["name"] = name
        cr["description"] = (
            f"Stack dominante: {', '.join([n.title() for n in cr['top_5_names']])}. "
            f"Tamaño: {cr['job_offer_count']} ofertas."
        )
        logger.info(f"Cluster {cr['cluster_index']}: {name} ({cr['job_offer_count']} ofertas, {cr['num_meaningful_skills']} skills)")

    logger.info(f"=== RESUMEN FINAL: {len(cluster_results)} clústeres válidos ===")


    # 7. Deployment
    if args.deploy:
        client = get_supabase_client()
        if client:
            deploy_to_supabase(client, valid_offers, final_labels, cluster_results, all_skills)
        else:
            logger.error("Deployment failed: Supabase client is not available.")
    else:
        logger.info("Dry-run finished. (Use --deploy flag to write results back to Supabase)")

if __name__ == "__main__":
    main()
