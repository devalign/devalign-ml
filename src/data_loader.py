import json
import os
import logging
import numpy as np
import pandas as pd
from supabase import create_client
from src import config

logger = logging.getLogger(__name__)

def get_supabase_client():
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        logger.warning("Supabase credentials missing. Supabase operations will be unavailable.")
        return None
    try:
        return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

def fetch_skills_from_supabase(client):
    logger.info("Fetching skills, embeddings, and aliases from Supabase...")
    skills_resp = client.table("skills").select("name, embedding, skill_aliases(alias_name)").execute()
    return skills_resp.data if skills_resp else []

def fetch_job_offers_from_supabase(client):
    logger.info("Fetching job offers from Supabase...")
    offers = []
    limit = 1000
    offset = 0
    while True:
        resp = (
            client.table("job_offers")
            .select("job_offer_id, job_title, raw_hard_skills")
            .range(offset, offset + limit - 1)
            .execute()
        )
        if not resp.data:
            break
        offers.extend(resp.data)
        if len(resp.data) < limit:
            break
        offset += limit
    return offers

def load_raw_data(use_cache=True):
    """Loads raw skills and job offers, using local json cache if available."""
    skills_cache_path = os.path.join(config.DATA_DIR, "raw_skills.json")
    offers_cache_path = os.path.join(config.DATA_DIR, "raw_job_offers.json")
    
    skills_data = []
    offers_data = []
    
    # Try local cache first
    if use_cache and os.path.exists(skills_cache_path) and os.path.exists(offers_cache_path):
        logger.info("Loading raw data from local JSON cache...")
        try:
            with open(skills_cache_path, "r", encoding="utf-8") as f:
                skills_data = json.load(f)
            with open(offers_cache_path, "r", encoding="utf-8") as f:
                offers_data = json.load(f)
            return skills_data, offers_data
        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}. Falling back to database...")

    # Load from Supabase
    client = get_supabase_client()
    if client:
        try:
            skills_data = fetch_skills_from_supabase(client)
            offers_data = fetch_job_offers_from_supabase(client)
            
            # Save cache
            os.makedirs(config.DATA_DIR, exist_ok=True)
            with open(skills_cache_path, "w", encoding="utf-8") as f:
                json.dump(skills_data, f, ensure_ascii=False, indent=2)
            with open(offers_cache_path, "w", encoding="utf-8") as f:
                json.dump(offers_data, f, ensure_ascii=False, indent=2)
            logger.info("Fetched data from Supabase and saved to local cache.")
            return skills_data, offers_data
        except Exception as e:
            logger.error(f"Error fetching data from Supabase: {e}")
            
    # If Supabase is unavailable and no cache exists, raise exception or return empty
    if not skills_data or not offers_data:
        raise ValueError("No data found. Provide Supabase credentials or populate the data/ cache.")
        
    return skills_data, offers_data

def process_job_offer_vectors(skills_data, offers_data):
    """Calculates canonical skills mapping and represents each job offer as a dense vector."""
    alias_to_canonical = {}
    skill_to_embedding = {}

    for s in skills_data:
        canonical = s["name"].strip()
        alias_to_canonical[canonical.lower()] = canonical
        for a in s.get("skill_aliases", []):
            alias_to_canonical[a["alias_name"].lower().strip()] = canonical

        emb_str = s.get("embedding")
        if emb_str:
            try:
                if isinstance(emb_str, str):
                    if emb_str.startswith('[') and emb_str.endswith(']'):
                        emb = json.loads(emb_str)
                    else:
                        emb = [float(x) for x in emb_str.replace('[', '').replace(']', '').split(',') if x.strip()]
                else:
                    emb = list(emb_str)

                if len(emb) == 1024:
                    skill_to_embedding[canonical.lower()] = emb
            except Exception as e:
                logger.warning(f"Failed to parse embedding for skill '{canonical}': {e}")

    logger.info(f"Loaded {len(skill_to_embedding)} canonical skills with embeddings.")

    valid_offers = []
    for row in offers_data:
        raw_skills = row.get("raw_hard_skills") or []
        offer_skills = set()

        for raw_s in raw_skills:
            clean_s = raw_s.lower().strip()
            if not clean_s:
                continue

            if clean_s in alias_to_canonical:
                offer_skills.add(alias_to_canonical[clean_s])
            else:
                offer_skills.add(clean_s.replace(" ", "").replace(".", ""))

        # Pool embeddings
        embeddings = []
        for s in offer_skills:
            emb = skill_to_embedding.get(s.lower())
            if emb:
                embeddings.append(emb)

        if embeddings:
            offer_vector = np.mean(embeddings, axis=0)
            valid_offers.append({
                "id": row["job_offer_id"],
                "job_title": row.get("job_title", ""),
                "skills": list(offer_skills),
                "vector": offer_vector
            })

    logger.info(f"Offers with valid skill embeddings: {len(valid_offers)} / {len(offers_data)}.")
    return valid_offers, alias_to_canonical
