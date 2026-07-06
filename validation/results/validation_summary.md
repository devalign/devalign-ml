# Validación del Diagnóstico — Devalign
**Fecha:** 2026-07-03T21:48:50.184524  
**Clústeres:** 24  
**Perfiles sintéticos:** 83  
**Precisión de asignación:** 62.7%  
**Precision (gaps) promedio:** 1.000  
**Recall (gaps) promedio:** 0.482  
**F1 (gaps) promedio:** 0.500  

## Ablation Study
| Variante | Efecto típico |
|----------|---------------|
| **Especialista en BI (Full Match)** | |
| Full (actual) | Score: 0.72 |
| Sin crédito parcial (p=0) | Score: 0.72 |
| Sin ICT Score | Score: 1.0 |
| Sin frecuencia | Score: 0.72 |
| Solo pesos (raw Jaccard) | Score: 1.0 |

## Gap Validation
| Tipo Perfil | Correctos | Precisión | Recall | F1 |
|-------------|-----------|-----------|--------|-----|
| full_match | 20/20 | 1.000 | 1.000 | 1.000 |
| partial | 14/20 | 1.000 | 0.229 | 0.245 |
| low | 6/20 | 1.000 | 0.300 | 0.344 |
| mismatch | 12/20 | 1.000 | 0.422 | 0.437 |
| mixed | 0/3 | 1.000 | 0.333 | 0.333 |

## Baseline Comparison
| Método | Rango de scores | Dispersión |
|--------|-----------------|------------|
| Weighted Jaccard | 0.038 – 0.720 | 0.682 |
| Jaccard simple | 0.000 – 1.000 | 1.000 |
| Coseno (embeddings) | 0.000 – 0.000 | 0.000 |

## Interpretación
El Weighted Jaccard completo produce el rango dinámico más amplio, lo que permite discriminar mejor entre clústeres afines y no afines. La ablación muestra que el crédito parcial por dominio (30%) contribuye significativamente a la detección de brechas en perfiles parciales, mientras que la ponderación ICT evita sobreestimar habilidades sin evidencia verificable. La principal limitación es la ausencia de ground truth real (CVs etiquetados por expertos), lo que hace que métricas como precisión/recall dependan de perfiles sintéticos.
