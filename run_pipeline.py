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
from src.utils import generate_cluster_names_batch, generate_cluster_name, plot_and_save_clusters

def parse_args():
    parser = argparse.ArgumentParser(description="Devalign ML Clustering Pipeline (UMAP + HDBSCAN)")
    parser.add_argument("--deploy", action="store_true", help="Deploy/write results to Supabase database.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local JSON cache and force DB fetch.")
    parser.add_argument("--portals", nargs="+", default=["computrabajo", "getonboard"], help="Portals to include in clustering (default: computrabajo getonboard).")
    parser.add_argument("--plot-name", type=str, default="hdbscan_clusters.png", help="Filename of the saved plot.")
    return parser.parse_args()

def deploy_to_supabase(client, valid_offers, final_labels, cluster_results, all_skills):
    """Deploys clustering results to Supabase (resets, deletes old, inserts new)."""
    logger.info("Starting deployment to Supabase...")
    
    # 1. Reset job_offers cluster_id to NULL
    logger.info("Unlinking existing job offers from clusters...")
    try:
        client.table("job_offers").update({"cluster_id": None}).not_.is_("cluster_id", "null").execute()
    except Exception as e:
        logger.warning(f"Unlinking job offers note: {e}")

    # 2. Delete existing diagnostics and clusters
    logger.info("Deleting old diagnostics and clusters...")
    old_diagnostics = client.table("diagnostics").select("diagnostic_id").execute()
    for od in old_diagnostics.data:
        client.table("diagnostics").delete().eq("diagnostic_id", od["diagnostic_id"]).execute()

    old_clusters = client.table("clusters").select("cluster_id").execute()
    for oc in old_clusters.data:
        client.table("clusters").delete().eq("cluster_id", oc["cluster_id"]).execute()

    # Get existing canonical skills mapping
    existing_skills_resp = client.table("skills").select("skill_id, name").execute()
    existing_skills_map = {s["name"]: s["skill_id"] for s in existing_skills_resp.data}

    # Ensure all canonical skills exist in the skills table
    new_skills_to_insert = []
    for s_name in all_skills:
        if s_name not in existing_skills_map:
            new_skills_to_insert.append({"name": s_name, "nature": "tech", "weight": 1.0})

    if new_skills_to_insert:
        logger.info(f"Inserting {len(new_skills_to_insert)} missing canonical skills...")
        inserted_skills = client.table("skills").upsert(new_skills_to_insert, on_conflict="name").execute()
        for s in inserted_skills.data:
            existing_skills_map[s["name"]] = s["skill_id"]

    # 3. Insert new clusters and link job offers
    logger.info("Inserting new clusters and linking job offers...")
    for cr in cluster_results:
        cluster_id = str(uuid4())
        client.table("clusters").insert(
            {
                "cluster_id": cluster_id,
                "name": cr["name"],
                "description": cr["description"],
                "job_offer_count": cr["job_offer_count"],
                "compatible_roles": cr["compatible_roles"],
                "centroid_vec": cr["centroid_vec"],
                "market_insights": {
                    "growth_percentage": cr["growth_percentage"],
                    "market_share_percentage": cr["market_share_percentage"],
                    "salary_differential": "N/A"
                }
            }
        ).execute()

        # Link job offers to this cluster in chunks of 100
        offer_ids = cr["offer_ids"]
        for i in range(0, len(offer_ids), 100):
            chunk = offer_ids[i : i + 100]
            client.table("job_offers").update({"cluster_id": cluster_id}).in_(
                "job_offer_id", chunk
            ).execute()

        # Insert Cluster Skills relations
        cs_inserts = []
        for cs in cr["skills"]:
            s_id = existing_skills_map.get(cs["name"])
            if s_id:
                cs_inserts.append(
                    {
                        "cluster_id": cluster_id,
                        "skill_id": s_id,
                        "importance_score": cs["frequency"] * cs["weight"],
                    }
                )

        if cs_inserts:
            client.table("cluster_skills").insert(cs_inserts).execute()

    logger.info("Deployment finished successfully!")

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

    cluster_results = []
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

        skills_data = []
        for skill_name, freq in top_skills.items():
            weight = 3.0 if freq >= 0.6 else (2.0 if freq >= 0.3 else 1.0)
            skills_data.append({
                "name": skill_name, 
                "frequency": float(freq), 
                "weight": float(weight)
            })

        cluster_results.append({
            "cluster_index": k,
            "name": None,
            "description": None,
            "top_5_names": top_5_names,
            "job_offer_count": cluster_size,
            "skills": skills_data,
            "compatible_roles": compatible_roles,
            "centroid_vec": results["centroids_1024d"][k],
            "offer_ids": cluster_df["offer_id"].tolist(),
            "market_share_percentage": round((cluster_size / len(valid_offers)) * 100, 1),
            "growth_percentage": round(np.random.uniform(5.0, 25.0), 1)  # Mocked growth
        })

    # 6. Generate Cluster Names via LLM
    logger.info("Generating cluster names...")
    all_clusters_data = [
        {
            "skills": cr["top_5_names"],
            "roles": [r["title"] for r in cr["compatible_roles"][:2]]
        }
        for cr in cluster_results
    ]
    batch_names = generate_cluster_names_batch(all_clusters_data)

    if batch_names and len(batch_names) == len(cluster_results):
        for idx, name in enumerate(batch_names):
            cr = cluster_results[idx]
            cr["name"] = name
            cr["description"] = f"Dominant skills: {', '.join([n.title() for n in cr['top_5_names']])}. Size: {cr['job_offer_count']} offers."
            logger.info(f"Cluster {cr['cluster_index']}: {name} ({cr['job_offer_count']} offers)")
    else:
        logger.warning("Falling back to single-naming queries.")
        for cr in cluster_results:
            top_roles = [r["title"] for r in cr["compatible_roles"][:2]]
            name = generate_cluster_name(cr["top_5_names"], top_roles=top_roles)
            cr["name"] = name
            cr["description"] = f"Dominant skills: {', '.join([n.title() for n in cr['top_5_names']])}. Size: {cr['job_offer_count']} offers."
            logger.info(f"Cluster {cr['cluster_index']}: {name} ({cr['job_offer_count']} offers)")

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
