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

DEFAULT_TARGET_PORTALS = ("computrabajo", "getonboard", "arbeitnow", "remotive", "weworkremotely")

def classify_market_tier(offer: dict) -> str:
    """Classifies a job offer into Tier 1 (LATAM Local), Tier 2 (LATAM Tech & Startups Remoto), or Tier 3 (Europa / Alemania)."""
    portal = (offer.get("portal") or "").lower().strip()
    country = (offer.get("country") or "").upper().strip()
    location = (offer.get("location") or "").lower()

    if (
        portal == "arbeitnow"
        or country in ("DE", "AT", "CH", "ES", "UK", "FR", "NL", "EU")
        or any(c in location for c in ("germany", "deutschland", "berlin", "munich", "hamburg", "europe", "frankfurt"))
    ):
        return "Tier 3 (Europa / Alemania)"
    elif (
        portal in ("getonboard", "remotive", "weworkremotely")
        or country in ("US", "CA")
        or "remoto" in location
        or "remote" in location
    ):
        return "Tier 2 (Tech & Remoto Global)"
    else:
        return "Tier 1 (LATAM Local)"

def extract_salary_usd(offer: dict) -> float | None:
    """Extracts or estimates normalized monthly salary in USD from structured or raw text."""
    min_usd = offer.get("min_salary_usd")
    max_usd = offer.get("max_salary_usd")
    if min_usd is not None and max_usd is not None:
        return (float(min_usd) + float(max_usd)) / 2.0
    if min_usd is not None:
        return float(min_usd)
    if max_usd is not None:
        return float(max_usd)

    raw = str(offer.get("salary") or "")
    if not raw or "a convenir" in raw.lower() or "no especificado" in raw.lower():
        return None

    # Check for PEN (S/.)
    pen_match = re.search(r'S/\.?\s*([0-9.,]+)', raw)
    if pen_match:
        try:
            val = float(pen_match.group(1).replace(',', ''))
            if val > 500:
                return round(val / 3.75, 2)
        except Exception:
            pass

    # Check for USD ($)
    usd_match = re.search(r'\$\s*([0-9.,]+)', raw)
    if usd_match:
        try:
            val = float(usd_match.group(1).replace(',', ''))
            if val > 100:
                if val > 15000:  # Yearly
                    val = val / 12
                return round(val, 2)
        except Exception:
            pass

    # Check for EUR (€)
    eur_match = re.search(r'€\s*([0-9.,]+)|([0-9.,]+)\s*€', raw)
    if eur_match:
        try:
            val_str = (eur_match.group(1) or eur_match.group(2)).replace('.', '').replace(',', '.')
            val = float(val_str)
            if val > 15000:  # Yearly
                val = val / 12
            return round(val * 1.08, 2)
        except Exception:
            pass

    return None

def fetch_job_offers_from_supabase(client, target_portals=DEFAULT_TARGET_PORTALS):
    logger.info(f"Fetching job offers from Supabase (target_portals: {target_portals})...")
    offers = []
    limit = 1000
    offset = 0
    while True:
        query = (
            client.table("job_offers")
            .select("job_offer_id, job_title, raw_hard_skills, portal, salary, min_salary_usd, max_salary_usd, country, location")
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

def load_raw_data(use_cache=True, target_portals=DEFAULT_TARGET_PORTALS):
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

SYNONYM_REDUNDANCY_MAP = {
    "metodologías ágiles": "Agile Methodology",
    "metodologias agiles": "Agile Methodology",
    "metodología ágil": "Agile Methodology",
    "metodologia agil": "Agile Methodology",
    "agile": "Agile Methodology",
    "agile methodology": "Agile Methodology",
    "metodología agile": "Agile Methodology",
    "metodologias agile": "Agile Methodology",
    "desarrollo frontend": "Front-End Development",
    "front-end": "Front-End Development",
    "front end": "Front-End Development",
    "frontend": "Front-End Development",
    "desarrollo backend": "Back-End Web Development",
    "back-end": "Back-End Web Development",
    "back end": "Back-End Web Development",
    "backend": "Back-End Web Development",
    "full-stack": "Full Stack Development",
    "fullstack": "Full Stack Development",
    "full stack": "Full Stack Development",
    "pruebas de software": "Quality Assurance",
    "control de calidad": "Quality Assurance",
    "aseguramiento de la calidad": "Quality Assurance",
    "quality assurance": "Quality Assurance",
    "qa": "Quality Assurance",
    "testing de software": "Quality Assurance",
    "pruebas automatizadas": "Automated Testing",
    "testing automatizado": "Automated Testing",
    "rest api": "Representational State Transfer (REST)",
    "apis rest": "Representational State Transfer (REST)",
    "api rest": "Representational State Transfer (REST)",
}

def process_job_offer_vectors(skills_data, offers_data):
    """Calculates canonical skills mapping and represents each job offer as a dense vector."""
    alias_to_canonical = {}
    skill_to_embedding = {}

    for s in skills_data:
        canonical = s["name"].strip()
        canonical_lower = canonical.lower()
        target = SYNONYM_REDUNDANCY_MAP.get(canonical_lower, canonical)
        alias_to_canonical[canonical_lower] = target
        for a in s.get("skill_aliases", []):
            alias_name = a["alias_name"].lower().strip()
            # Do NOT allow generic "go" as an unconditional alias for the Go programming language
            if alias_name == "go":
                continue
            alias_to_canonical[alias_name] = SYNONYM_REDUNDANCY_MAP.get(alias_name, target)

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
    idf = {}
    for s, df in doc_freq.items():
        base = math.log(1.0 + N / (df + 1.0))
        freq_ratio = df / N
        # Apply extra penalty to transversal "comodín" skills (appear in >35% of offers).
        # Without this, SQL/Python/Agile dominate every cluster centroid and cause
        # semantically distinct roles (Java backend, Linux SysAdmin, Agile coach) to collapse
        # into the same embedding region.
        if freq_ratio > config.TRANSVERSAL_PENALTY_THRESHOLD:
            base *= config.TRANSVERSAL_PENALTY_FACTOR
        idf[s] = base


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
            elif clean_s in SYNONYM_REDUNDANCY_MAP:
                canonical_name = SYNONYM_REDUNDANCY_MAP[clean_s]
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
                "vector": offer_vector,
                "market_tier": classify_market_tier(row),
                "salary_usd": extract_salary_usd(row)
            })

    logger.info(f"Offers with valid tech skill embeddings: {len(valid_offers)} / {len(offers_data)}.")
    return valid_offers, alias_to_canonical

