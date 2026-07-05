"""
Validación del diagnóstico de brechas técnicas.

Módulos:
  - alignment.py:   Tres variantes del algoritmo de alineación (Weighted Jaccard)
  - profile_generator.py: Generación de perfiles sintéticos con skill sets controlados
  - ablation_study.py:    Experimentos de ablación y comparación con baselines
  - run.py:               Punto de entrada único
"""

from . import alignment, profile_generator, ablation_study
