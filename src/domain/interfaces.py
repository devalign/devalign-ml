"""Domain interfaces and contracts for devalign-ml."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ClusterNamingStrategy(ABC):
    """Abstract strategy for naming technical occupational clusters."""

    @abstractmethod
    def name_cluster(
        self,
        skill_frequencies: Dict[str, float],
        compatible_roles: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Infers a professional, concise role title for a cluster.

        Args:
            skill_frequencies: Ordered mapping of {skill_name: distinctiveness_score}.
            compatible_roles: Optional list of frequent scraped job titles.

        Returns:
            A representative Spanish role title.
        """
        pass
