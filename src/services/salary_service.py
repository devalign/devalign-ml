"""Salary and market tier normalization service."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


class SalaryNormalizer:
    """Encapsulates parsing, currency conversion, and tier classification for job offers."""

    DEFAULT_PEN_RATE: float = 3.75
    DEFAULT_EUR_RATE: float = 1.08

    @classmethod
    def classify_market_tier(cls, offer: Dict[str, Any]) -> str:
        """Classifies a job offer into Tier 1 (LATAM Local), Tier 2 (Tech & Remoto Global), or Tier 3 (Europa)."""
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

    @classmethod
    def extract_salary_usd(cls, offer: Dict[str, Any]) -> Optional[float]:
        """Extracts and normalizes monthly salary in USD from structured bounds or raw text."""
        min_usd = offer.get("min_salary_usd")
        max_usd = offer.get("max_salary_usd")
        if min_usd is not None and max_usd is not None:
            return round((float(min_usd) + float(max_usd)) / 2.0, 2)
        if min_usd is not None:
            return round(float(min_usd), 2)
        if max_usd is not None:
            return round(float(max_usd), 2)

        raw = str(offer.get("salary") or "").strip()
        if not raw or any(x in raw.lower() for x in ("a convenir", "no especificado", "null", "none")):
            return None

        # Check for PEN (S/.)
        pen_match = re.search(r'S/\.?\s*([0-9.,]+)', raw)
        if pen_match:
            try:
                val = float(pen_match.group(1).replace(',', ''))
                if val > 500:
                    return round(val / cls.DEFAULT_PEN_RATE, 2)
            except (ValueError, TypeError):
                pass

        # Check for USD ($)
        usd_match = re.search(r'\$\s*([0-9.,]+)', raw)
        if usd_match:
            try:
                val = float(usd_match.group(1).replace(',', ''))
                if val > 100:
                    if val > 15000:  # Yearly to monthly
                        val = val / 12.0
                    return round(val, 2)
            except (ValueError, TypeError):
                pass

        # Check for EUR (€)
        eur_match = re.search(r'€\s*([0-9.,]+)|([0-9.,]+)\s*€', raw)
        if eur_match:
            try:
                val_str = (eur_match.group(1) or eur_match.group(2)).replace('.', '').replace(',', '.')
                val = float(val_str)
                if val > 15000:  # Yearly to monthly
                    val = val / 12.0
                return round(val * cls.DEFAULT_EUR_RATE, 2)
            except (ValueError, TypeError):
                pass

        return None
