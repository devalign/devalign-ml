import json
import os
import re
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
    skills = []
    limit = 1000
    offset = 0
    while True:
        resp = (
            client.table("skills")
            .select("name, embedding, skill_aliases(alias_name)")
            .range(offset, offset + limit - 1)
            .execute()
        )
        if not resp.data:
            break
        skills.extend(resp.data)
        if len(resp.data) < limit:
            break
        offset += limit
    logger.info(f"Loaded {len(skills)} skills from Supabase.")
    return skills

def fetch_job_offers_from_supabase(client, target_portals=("computrabajo", "getonboard")):
    logger.info(f"Fetching job offers from Supabase (target_portals: {target_portals})...")
    offers = []
    limit = 1000
    offset = 0
    while True:
        query = (
            client.table("job_offers")
            .select("job_offer_id, job_title, raw_hard_skills, portal")
        )
        if target_portals:
            query = query.in_("portal", list(target_portals))

        resp = (
            query.range(offset, offset + limit - 1)
            .execute()
        )
        if not resp.data:
            break
        offers.extend(resp.data)
        if len(resp.data) < limit:
            break
        offset += limit
    logger.info(f"Loaded {len(offers)} job offers matching portals {target_portals}.")
    return offers

def load_raw_data(use_cache=True, target_portals=("computrabajo", "getonboard")):
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
            offers_data = fetch_job_offers_from_supabase(client, target_portals=target_portals)
            
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

GENERIC_OFFICE_SKILLS = {
    "dominio de hojas de cálculo",
    "excel",
    "microsoft excel",
    "excel avanzado",
    "visual basic excel",
    "excelvba",
    "excell",
    "manejodeherramientasofimáticas(wordyexcel)",
    "manejo office",
    "manejo de herramientas ofimáticas (word y excel)",
    "office",
    "microsoft office",
    "office 365",
    "utilizar software de hojas de cálculo",
    "word",
    "powerpoint"
}

DEV_ROLE_KEYWORDS = {
    "developer", "desarrollador", "engineer", "ingeniero", "architect", "arquitecto",
    "programmer", "programador", "backend", "frontend", "fullstack", "full-stack",
    "software", "devops", "cloud", "data", "qa", "tester", "sre", "machine learning",
    "ia", "ai", "mobile", "ios", "android", "sysadmin", "tech lead", "cto"
}

DEV_CONTEXT_SKILLS = {
    "backend", "frontend", "full-stack", "docker", "kubernetes", "microservices",
    "rest", "grpc", "sql", "postgresql", "mysql", "mongodb", "redis", "aws", "gcp",
    "azure", "git", "linux", "c++", "c#", "java", "python", "rust", "typescript",
    "javascript", "react", "angular", "vue", "ci/cd", "terraform"
}

def process_job_offer_vectors(skills_data, offers_data):
    """Calculates canonical skills mapping and represents each job offer as a dense vector."""
    alias_to_canonical = {}
    skill_to_embedding = {}

    for s in skills_data:
        canonical = s["name"].strip()
        canonical_lower = canonical.lower()
        alias_to_canonical[canonical_lower] = canonical
        for a in s.get("skill_aliases", []):
            alias_name = a["alias_name"].lower().strip()
            # Do NOT allow generic "go" as an unconditional alias for the Go programming language
            if alias_name == "go":
                continue
            alias_to_canonical[alias_name] = canonical

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
                    skill_to_embedding[canonical_lower] = emb
            except Exception as e:
                logger.warning(f"Failed to parse embedding for skill '{canonical}': {e}")

    logger.info(f"Loaded {len(skill_to_embedding)} canonical skills with embeddings.")

    # Calculate document frequency for smooth IDF weighting
    import math
    import collections
    doc_freq = collections.Counter()
    for row in offers_data:
        job_title = (row.get("job_title") or "").strip()
        job_title_lower = job_title.lower()
        if not job_title:
            continue

        raw_skills = row.get("raw_hard_skills") or []
        for s in set(raw_skills):
            clean_s = s.lower().strip()
            canon = alias_to_canonical.get(clean_s, clean_s).lower()
            doc_freq[canon] += 1

    N = len(offers_data)
    idf = {s: math.log(1.0 + N / (df + 1.0)) for s, df in doc_freq.items()}

    valid_offers = []
    for row in offers_data:
        job_title = (row.get("job_title") or "").strip()
        job_title_lower = job_title.lower()
        if not job_title:
            continue

        raw_skills = row.get("raw_hard_skills") or []
        offer_skills = set()
        raw_skills_lower = {rs.lower().strip() for rs in raw_skills if rs}

        for raw_s in raw_skills:
            clean_s = raw_s.lower().strip()
            if not clean_s:
                continue

            # Disambiguate "go": only consider Golang if explicitly "golang" or if job title specifically indicates a Go developer
            if clean_s in ("golang", "go-lang", "go language"):
                canonical_name = "Golang" if "golang" in skill_to_embedding else "go"
            elif clean_s == "go":
                is_go_title = bool(re.search(r'\b(golang|go\s+developer|go\s+backend|desarrollador\s+go|backend\s+go|go\s+engineer|ingeniero\s+go)\b', job_title_lower))
                is_gtm = "go-to-market" in job_title_lower or "go to market" in job_title_lower
                if is_go_title and not is_gtm:
                    canonical_name = "Golang" if "golang" in skill_to_embedding else "go"
                else:
                    # Drop the false positive 2-letter word "go" from normal English text
                    continue
            elif clean_s in alias_to_canonical:
                canonical_name = alias_to_canonical[clean_s]
            else:
                canonical_name = clean_s.replace(" ", "").replace(".", "")

            offer_skills.add(canonical_name)

        # Filter out generic office skills from defining tech core
        tech_skills = {s for s in offer_skills if s.lower() not in GENERIC_OFFICE_SKILLS}

        # If the offer contains no technical skills, do NOT cluster it (remains cluster_id = NULL)
        if not tech_skills:
            continue

        # Pool embeddings using smooth IDF weighting to preserve technology distinctiveness
        embeddings = []
        weights = []
        valid_tech_skills = []
        for s in tech_skills:
            emb = skill_to_embedding.get(s.lower())
            if emb:
                embeddings.append(emb)
                valid_tech_skills.append(s)
                w = idf.get(s.lower(), 1.0)
                weights.append(w)

        if embeddings:
            w_arr = np.array(weights, dtype=np.float32)
            w_arr = w_arr / (np.sum(w_arr) + 1e-9)
            offer_vector = np.sum(np.array(embeddings, dtype=np.float32) * w_arr[:, np.newaxis], axis=0)
            valid_offers.append({
                "id": row["job_offer_id"],
                "job_title": job_title,
                "skills": valid_tech_skills,
                "vector": offer_vector
            })

    logger.info(f"Offers with valid tech skill embeddings: {len(valid_offers)} / {len(offers_data)}.")
    return valid_offers, alias_to_canonical

