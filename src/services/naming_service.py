"""Cluster naming services implementing ClusterNamingStrategy."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.domain.interfaces import ClusterNamingStrategy
from src import config

logger = logging.getLogger(__name__)


class TaxonomyNamingStrategy(ClusterNamingStrategy):
    """Deterministic, zero-latency taxonomy-driven naming strategy."""

    def name_cluster(
        self,
        skill_frequencies: Dict[str, float],
        compatible_roles: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        from src.naming import infer_cluster_name
        return infer_cluster_name(skill_frequencies, compatible_roles)


class LLMNamingStrategy(ClusterNamingStrategy):
    """LLM-assisted naming strategy with automatic taxonomy fallback."""

    def __init__(self, fallback: Optional[ClusterNamingStrategy] = None):
        self.fallback = fallback or TaxonomyNamingStrategy()

    def name_cluster(
        self,
        skill_frequencies: Dict[str, float],
        compatible_roles: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        import requests

        top_skills = list(skill_frequencies.keys())[:5]
        if not top_skills:
            return self.fallback.name_cluster(skill_frequencies, compatible_roles)

        api_key = config.OPENAI_API_KEY if config.LLM_PROVIDER == "openai" else config.GROQ_API_KEY
        if not api_key:
            return self.fallback.name_cluster(skill_frequencies, compatible_roles)

        url = (
            "https://api.openai.com/v1/chat/completions"
            if config.LLM_PROVIDER == "openai"
            else "https://api.groq.com/openai/v1/chat/completions"
        )
        model = config.LLM_MODEL or ("gpt-4o-mini" if config.LLM_PROVIDER == "openai" else "llama-3.3-70b-versatile")
        headers = {"Authorization": f"Bearer {api_key}"}

        roles_context = ""
        if compatible_roles:
            role_titles = [r.get("title", "") for r in compatible_roles if r.get("title")]
            if role_titles:
                roles_context = f" | Puestos frecuentes: {', '.join(role_titles[:2])}"

        prompt = (
            f"Skills: {', '.join(top_skills)}{roles_context}\n"
            "Asignar un título profesional conciso y específico (máx 5 palabras) en español que represente "
            "la especialidad técnica del rol (ej: 'Desarrollador Backend Python & FastAPI', 'Ingeniero DevOps Cloud'). "
            "Responder ÚNICAMENTE con el título."
        )

        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Eres un experto reclutador técnico en TI."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 80,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"].get("content") or ""
                cleaned = raw.strip().replace('"', '').strip()
                if len(cleaned) > 2:
                    return cleaned
        except Exception as exc:
            logger.warning(f"LLMNamingStrategy failed ({exc}). Falling back to Taxonomy.")

        return self.fallback.name_cluster(skill_frequencies, compatible_roles)


def get_naming_strategy(provider: Optional[str] = None) -> ClusterNamingStrategy:
    """Factory to retrieve the configured cluster naming strategy."""
    active_provider = (provider or config.LLM_PROVIDER or "taxonomy").lower()
    if active_provider in ("openai", "groq", "llm"):
        return LLMNamingStrategy()
    return TaxonomyNamingStrategy()
