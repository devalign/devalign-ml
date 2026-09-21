"""
Primary-skill-driven taxonomy naming for the Devalign ML pipeline.

Replaces the old "required set" approach with a primary-skill lookup:
the function receives a dict of {skill_name: distinctive_score} sorted
descending, reads the most distinctive skill first, and resolves a role
name from there — using the second/third skills only to disambiguate.

This avoids false positives from comodín skills (SQL, Python, Agile)
appearing as primary when the cluster is actually about a specific tech.
"""

import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Direct primary-skill → role name lookups (no secondary needed)
# ---------------------------------------------------------------------------

PRIMARY_DIRECT: Dict[str, str] = {
    # Infrastructure as Code
    "terraform": "Ingeniero DevOps & Infraestructura como Código",
    # QA specific tools
    "selenium (software)": "Ingeniero QA Automation (Selenium)",
    "cypress": "Ingeniero QA Automation (Cypress & CI/CD)",
    # Mobile-specific
    "flutter (software)": "Desarrollador Mobile Flutter & Dart",
    "dart (programming language)": "Desarrollador Mobile Flutter & Dart",
    "react native": "Desarrollador Mobile React Native",
    "kotlin": "Desarrollador Android (Kotlin)",
    "swift (programming language)": "Desarrollador iOS (Swift)",
    # Backend frameworks (very specific)
    "fastapi": "Desarrollador Backend Python & FastAPI",
    "django (web framework)": "Desarrollador Backend Python & Django",
    "flask (web framework)": "Desarrollador Backend Python & Flask",
    "spring boot": "Desarrollador Backend Java & Spring Boot",
    "laravel": "Desarrollador Backend PHP & Laravel",
    "ruby on rails": "Desarrollador Backend Ruby on Rails",
    "ruby (programming language)": "Desarrollador Backend Ruby on Rails",
    ".net development": "Desarrollador Backend .NET & C#",
    "c# (programming language)": "Desarrollador Backend .NET & C#",
    "golang": "Desarrollador Backend Golang",
    "go (programming language)": "Desarrollador Backend Golang",
    # Data / ML
    "machine learning": "Ingeniero Machine Learning & Ciencia de Datos",
    "pandas (python package)": "Científico de Datos Python",
    "tensorflow": "Ingeniero Machine Learning & Deep Learning",
    "scikit-learn (machine learning library)": "Ingeniero Machine Learning & Ciencia de Datos",
    "apache spark": "Ingeniero de Datos Big Data (Spark)",
    "apache kafka": "Ingeniero de Datos Streaming (Kafka)",
    "power bi": "Analista de Datos & Business Intelligence",
    "tableau (business intelligence software)": "Analista de Datos & Visualización (Tableau)",
    # Design
    "ux/ui": "Diseñador UX/UI & Product Design",
    "figma (design software)": "Diseñador UX/UI (Figma)",
    # DevOps tools
    "jenkins": "Ingeniero CI/CD & Pipelines (Jenkins)",
    "kubernetes": "Ingeniero Cloud & Kubernetes",
    "microservicios": "Arquitecto de Microservicios & Sistemas Distribuidos",
    # Games
    "unity (game engine)": "Desarrollador de Videojuegos & Simulación (Unity)",
    # Blockchain
    "blockchain": "Desarrollador Blockchain & Web3",
    # Architecture
    "system design": "Arquitecto de Software & System Design",
    # Generic full-stack tag
    "full stack development": "Desarrollador Full Stack Generalista",
}


# ---------------------------------------------------------------------------
# Domain-level fallback groups (used when primary skill has no direct rule)
# ---------------------------------------------------------------------------

DOMAIN_GROUPS: Dict[str, set] = {
    "Frontend": {
        "react.js", "angular (web framework)", "vue.js",
        "hypertext markup language (html)", "cascading style sheets (css)",
        "next.js", "svelte", "nuxt.js", "jest (javascript testing framework)",
    },
    "Backend Python": {"python (programming language)", "fastapi", "django (web framework)", "flask (web framework)"},
    "Backend Java": {"java (programming language)", "spring boot", "quarkus"},
    "Backend .NET": {".net development", "c# (programming language)", "blazor"},
    "Backend PHP": {"php (scripting language)", "laravel", "symfony"},
    "Backend Node": {"node.js", "express", "nestjs"},
    "DevOps Cloud": {
        "docker (software)", "kubernetes", "terraform", "ci/cd", "jenkins",
        "amazon web services", "microsoft azure", "google cloud platform (gcp)",
        "devops", "software configuration management",
    },
    "Data Engineering": {
        "apache spark", "apache kafka", "sql (programming language)",
        "power bi", "tableau (business intelligence software)",
        "mongodb", "postgresql", "mysql",
    },
    "ML / AI": {"machine learning", "pandas (python package)", "tensorflow", "pytorch", "scikit-learn (machine learning library)"},
    "Mobile": {
        "react native", "flutter (software)", "kotlin",
        "swift (programming language)", "apple ios", "android (operating system)",
    },
    "QA": {"selenium (software)", "cypress", "quality assurance", "automated testing"},
    "UX/UI": {"figma (design software)", "ux/ui"},
}

DOMAIN_FALLBACK_NAMES: Dict[str, str] = {
    "Frontend": "Desarrollador Frontend Web",
    "Backend Python": "Desarrollador Backend Python",
    "Backend Java": "Desarrollador Backend Java",
    "Backend .NET": "Desarrollador Backend .NET",
    "Backend PHP": "Desarrollador Backend PHP",
    "Backend Node": "Desarrollador Backend Node.js",
    "DevOps Cloud": "Ingeniero DevOps & Cloud",
    "Data Engineering": "Ingeniero de Datos & Analytics",
    "ML / AI": "Ingeniero Machine Learning & AI",
    "Mobile": "Desarrollador Mobile",
    "QA": "Ingeniero QA Automation",
    "UX/UI": "Diseñador UX/UI",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_cluster_name(
    skill_freq_dict: Dict[str, float],
    compatible_roles: Optional[list] = None,
) -> str:
    """
    Infers a descriptive Spanish cluster name driven by the MOST DISTINCTIVE skill.

    Args:
        skill_freq_dict: Ordered dict of {canonical_skill_name: score} where the
            score is the distinctiveness ratio (cluster_freq / global_freq * cluster_freq).
            Must be sorted descending — first key is the primary, most-defining skill.
        compatible_roles: Unused, kept for API compatibility.

    Returns:
        A human-readable Spanish job-market role name.
    """
    if not skill_freq_dict:
        return "Especialidad Técnica General"

    items = list(skill_freq_dict.items())
    all_keys = {k.lower() for k, _ in items}
    top3_keys = {items[i][0].lower() for i in range(min(3, len(items)))}
    top5_keys = {items[i][0].lower() for i in range(min(5, len(items)))}

    primary_name, primary_score = items[0]
    primary_key = primary_name.lower()

    # ── 1. Direct primary → name lookup ────────────────────────────────────
    if primary_key in PRIMARY_DIRECT:
        logger.debug("Direct match: %s → %s", primary_key, PRIMARY_DIRECT[primary_key])
        return PRIMARY_DIRECT[primary_key]

    # ── 2. Primary skills that need secondary context ───────────────────────

    # iOS / Android
    if primary_key == "apple ios":
        if "android (operating system)" in top3_keys:
            return "Ingeniero Mobile iOS & Android"
        return "Desarrollador iOS (Swift)"

    if primary_key == "android (operating system)":
        if "kotlin" in top5_keys:
            return "Desarrollador Android (Kotlin & Java)"
        return "Desarrollador Android (Kotlin)"

    # React.js — could be Frontend React, MERN, or React Native
    if primary_key == "react.js":
        if "react native" in top3_keys:
            return "Desarrollador Mobile React Native"
        if "node.js" in top3_keys:
            return "Desarrollador Full Stack MERN (React & Node.js)"
        if "typescript" in top3_keys or "next.js" in top5_keys:
            return "Desarrollador Frontend React & TypeScript"
        return "Desarrollador Frontend React"

    # Angular / Vue
    if primary_key in ("angular (web framework)", "angular"):
        return "Desarrollador Frontend Angular"

    if primary_key == "vue.js":
        return "Desarrollador Frontend Vue.js"

    # Node.js
    if primary_key == "node.js":
        if "react.js" in top3_keys:
            return "Desarrollador Full Stack Node.js & React"
        if "typescript" in top3_keys:
            return "Desarrollador Backend Node.js & TypeScript"
        return "Desarrollador Backend Node.js"

    # TypeScript — frontend or backend depends on co-occurring skills
    if primary_key == "typescript":
        if "react.js" in top3_keys or "next.js" in top3_keys:
            return "Desarrollador Frontend React & TypeScript"
        if "node.js" in top3_keys:
            return "Desarrollador Backend Node.js & TypeScript"
        return "Desarrollador Full Stack TypeScript"

    # HTML/CSS — always frontend
    if primary_key in ("hypertext markup language (html)", "cascading style sheets (css)"):
        if "javascript (programming language)" in top3_keys:
            return "Desarrollador Web Frontend (HTML, CSS & JavaScript)"
        return "Desarrollador Web Frontend"

    # JavaScript — depends on context
    if primary_key == "javascript (programming language)":
        if "react.js" in top3_keys:
            return "Desarrollador Frontend React"
        if "node.js" in top3_keys:
            return "Desarrollador Backend Node.js"
        if "hypertext markup language (html)" in top3_keys or "cascading style sheets (css)" in top3_keys:
            return "Desarrollador Web Frontend JavaScript"
        return "Desarrollador Web JavaScript"

    # Docker → cloud/devops context
    if primary_key == "docker (software)":
        if "kubernetes" in top3_keys:
            return "Ingeniero Cloud & Kubernetes"
        if "ci/cd" in top3_keys:
            return "Ingeniero DevOps CI/CD & Docker"
        return "Ingeniero DevOps & Contenedores"

    # Cloud providers
    if primary_key == "amazon web services":
        if "terraform" in top5_keys:
            return "Ingeniero Cloud AWS & Infraestructura como Código"
        if "google cloud platform (gcp)" in top3_keys or "microsoft azure" in top3_keys:
            return "Ingeniero Cloud Multi-Cloud (AWS/GCP/Azure)"
        if "machine learning" in top5_keys:
            return "Ingeniero ML Cloud (AWS)"
        return "Ingeniero Cloud AWS"

    if primary_key == "microsoft azure":
        if "ci/cd" in top3_keys or "devops" in top3_keys:
            return "Ingeniero Cloud DevOps Azure"
        if ".net development" in top5_keys or "c# (programming language)" in top5_keys:
            return "Desarrollador .NET Cloud Azure"
        if "kubernetes" in top3_keys:
            return "Ingeniero Cloud Azure & Kubernetes"
        return "Ingeniero Cloud Microsoft Azure"

    if primary_key == "google cloud platform (gcp)":
        if "amazon web services" in top3_keys or "microsoft azure" in top3_keys:
            return "Ingeniero Cloud Multi-Cloud (GCP/AWS/Azure)"
        if "apache spark" in top5_keys or "bigquery" in top5_keys:
            return "Ingeniero de Datos Cloud (GCP)"
        return "Ingeniero Cloud Google Cloud Platform"

    # CI/CD
    if primary_key == "ci/cd":
        if "docker (software)" in top3_keys:
            return "Ingeniero DevOps CI/CD & Docker"
        if "kubernetes" in top3_keys:
            return "Ingeniero DevOps & Kubernetes"
        if "microsoft azure" in top3_keys:
            return "Ingeniero Cloud DevOps Azure"
        return "Ingeniero DevOps & CI/CD"

    # DevOps
    if primary_key == "devops":
        if "microsoft azure" in top3_keys:
            return "Ingeniero Cloud DevOps Azure"
        if "kubernetes" in top3_keys:
            return "Ingeniero DevOps & Kubernetes"
        if "ci/cd" in top3_keys:
            return "Ingeniero DevOps CI/CD"
        return "Ingeniero DevOps & Site Reliability"

    # Software Configuration Management
    if primary_key == "software configuration management":
        if "blockchain" in top3_keys:
            return "Desarrollador Blockchain & Web3"
        if "ci/cd" in top3_keys:
            return "Ingeniero CI/CD & Control de Versiones"
        return "Ingeniero DevOps & Gestión de Configuración"

    # Linux as primary → SysAdmin / Infra, NOT data analyst
    if primary_key == "linux":
        if "ci/cd" in top3_keys or "docker (software)" in top3_keys:
            return "Ingeniero DevOps & Infraestructura Linux"
        if "python (programming language)" in top3_keys:
            return "Ingeniero de Infraestructura Linux & Python"
        return "Administrador de Sistemas Linux (SysAdmin)"

    # Python as primary — check for framework / specialty
    if primary_key == "python (programming language)":
        if any(k in top5_keys for k in ("machine learning", "tensorflow", "scikit-learn (machine learning library)", "pytorch")):
            return "Ingeniero Machine Learning & Ciencia de Datos"
        if "pandas (python package)" in top5_keys:
            return "Científico de Datos Python"
        if "fastapi" in top5_keys:
            return "Desarrollador Backend Python & FastAPI"
        if "django (web framework)" in top5_keys:
            return "Desarrollador Backend Python & Django"
        if "flask (web framework)" in top5_keys:
            return "Desarrollador Backend Python & Flask"
        if "amazon web services" in top3_keys or "docker (software)" in top3_keys:
            return "Ingeniero Backend Python & Cloud"
        if "power bi" in top5_keys or "tableau (business intelligence software)" in top5_keys:
            return "Analista de Datos & Business Intelligence"
        if "sql (programming language)" in top3_keys and "pandas (python package)" in all_keys:
            return "Analista de Datos SQL & Python"
        return "Desarrollador Backend Python"

    # Java as primary — check for Spring Boot vs Android vs Backend
    if primary_key == "java (programming language)":
        if "spring boot" in top5_keys:
            return "Desarrollador Backend Java & Spring Boot"
        if "kotlin" in top3_keys or "android (operating system)" in top5_keys:
            return "Desarrollador Android (Java & Kotlin)"
        if "sql (programming language)" in top3_keys:
            return "Desarrollador Backend Java"
        return "Desarrollador Backend Java"

    # PHP as primary
    if primary_key == "php (scripting language)":
        if "laravel" in top5_keys:
            return "Desarrollador Backend PHP & Laravel"
        if "symfony" in top5_keys:
            return "Desarrollador Backend PHP & Symfony"
        return "Desarrollador Backend PHP"

    # MySQL/PostgreSQL as primary → DBA unless paired with BI tools
    if primary_key in ("mysql", "postgresql"):
        if "power bi" in all_keys or "tableau (business intelligence software)" in all_keys:
            return "Analista de Datos & Business Intelligence"
        if "php (scripting language)" in top5_keys or "java (programming language)" in top5_keys:
            return "Administrador de Bases de Datos (DBA)"
        if "python (programming language)" in top3_keys:
            return "Ingeniero de Bases de Datos SQL & Python"
        return "Administrador de Bases de Datos (DBA)"

    # SQL as primary — data role or backend?
    if primary_key == "sql (programming language)":
        if "power bi" in top5_keys or "tableau (business intelligence software)" in top5_keys:
            return "Analista de Datos & Business Intelligence"
        if "python (programming language)" in top3_keys and "pandas (python package)" in all_keys:
            return "Analista de Datos SQL & Python"
        if "php (scripting language)" in top5_keys:
            return "Desarrollador Backend PHP & MySQL"
        if "java (programming language)" in top3_keys:
            return "Desarrollador Backend Java"
        return "Analista de Datos SQL"

    # Agile / Jira / Kanban as primary — ONLY if no dominant tech is present
    if primary_key in ("agile methodology", "jira", "kanban"):
        if "ci/cd" in top3_keys or "devops" in top3_keys:
            return "Líder Técnico DevOps & Agile"
        if "microsoft azure" in top5_keys or "amazon web services" in top5_keys:
            return "Gestor de Proyectos Cloud & DevOps"
        return "Líder Técnico Agile & Gestión de Proyectos"

    # Software Engineering as primary
    if primary_key == "software engineering":
        if "machine learning" in top3_keys:
            return "Ingeniero de Software ML & Algoritmos"
        if "agile methodology" in top3_keys:
            return "Ingeniero de Software Agile"
        return "Ingeniero de Software Generalista"

    # ── 3. Domain-group fallback ────────────────────────────────────────────
    domain_scores: Dict[str, int] = {}
    for domain, domain_skills in DOMAIN_GROUPS.items():
        hits = len(domain_skills & all_keys)
        if hits > 0:
            domain_scores[domain] = hits

    if domain_scores:
        top_domain = max(domain_scores, key=lambda d: domain_scores[d])
        name = DOMAIN_FALLBACK_NAMES.get(top_domain, f"Especialista {top_domain}")
        logger.debug("Domain fallback: %s | primary=%s", name, primary_name)
        return name

    # ── 4. Smart short concatenation (last resort) ──────────────────────────
    short_skills = []
    for skill_name, _ in items[:3]:
        clean = skill_name.split("(")[0].strip().title()
        short_skills.append(clean)
    return "Especialista: " + " & ".join(short_skills)
