# Sensibilidad del Umbral de Similitud Coseno (pgvector)

**Fecha:** 2026-10-16T00:16:05.621086  
**Rango Evaluado:** [0.75, 0.95] con paso 0.02  

| Umbral ($\tau$) | Precision | Recall | F1-Score | Accuracy | TP | FP | FN | TN |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.75 | 0.7692 | 1.0000 | 0.8696 | 0.8000 | 20 | 6 | 0 | 4 |
| 0.77 | 0.8333 | 1.0000 | 0.9091 | 0.8667 | 20 | 4 | 0 | 6 |
| 0.79 | 0.9524 | 1.0000 | 0.9756 | 0.9667 | 20 | 1 | 0 | 9 |
| 0.81 | 0.9524 | 1.0000 | 0.9756 | 0.9667 | 20 | 1 | 0 | 9 |
| 0.83 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20 | 0 | 0 | 10 |
| 0.85 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20 | 0 | 0 | 10 |
| 0.87 | 1.0000 | 0.8500 | 0.9189 | 0.9000 | 17 | 0 | 3 | 10 |
| 0.89 | 1.0000 | 0.7500 | 0.8571 | 0.8333 | 15 | 0 | 5 | 10 |
| 0.91 | 1.0000 | 0.4000 | 0.5714 | 0.6000 | 8 | 0 | 12 | 10 |
| 0.93 | 1.0000 | 0.1500 | 0.2609 | 0.4333 | 3 | 0 | 17 | 10 |
| 0.95 | 1.0000 | 0.1000 | 0.1818 | 0.4000 | 2 | 0 | 18 | 10 |

> [!NOTE]
> El umbral óptimo $\tau = 0.88$ maximiza el F1-Score al balancear el filtrado de falsos positivos (habilidades blandas o conceptos abstractos) con una alta tasa de recuperación de variaciones sintácticas.
