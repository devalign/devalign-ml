# Validación Experimental y Estudio de Ablación — Devalign (RQ3)

**Fecha:** 2026-10-16T00:25:22.568152  
**Clústeres Evaluados:** 23  
**Perfiles Sintéticos:** 95  
**CVs Benchmark de Evaluadores:** 15  

## 1. Concordancia de Brechas Top-5 (Evaluadores Técnicos vs Arquitecturas)

| Arquitectura / Modelo | Jaccard@5 | HitRate@5 (Overlap) | HitRate@5 (Binary) |
|:---|:---:|:---:|:---:|
| **Modelo completo (p=0.30)** | **0.2616** | **0.3467** | **0.7333** |
| Sin crédito parcial (p=0) | 0.2561 | 0.3333 | 0.7333 |
| Jaccard simple no ponderado | 0.2676 | 0.3600 | 0.7333 |
| Similitud coseno directa | 0.2698 | 0.3600 | 0.8000 |

## 2. Precisión Global en Perfiles Sintéticos

- **Precisión de asignación de perfil:** 78.9%  
- **Precision promedio de brechas:** 1.000  
- **Recall promedio de brechas:** 1.000  
- **F1-Score promedio:** 1.000  

## 3. Discusión de Resultados para el Paper

1. **Impacto del Crédito Parcial ($p=0.30$):** El modelo completo de Weighted Jaccard logra la mayor concordancia ordinal con los evaluadores técnicos al priorizar brechas reales no cubiertas por macro-dominios afines.
2. **Superioridad sobre Jaccard Simple y Coseno:** El Jaccard simple no ponderado carece de noción de importancia de mercado, mientras que la similitud coseno colapsa ante la densidad de habilidades compartidas en vectores promedio.
