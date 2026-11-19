# Estadística Inferencial y Concordancia RQ3 (`stats_confidence`)

- **Fecha (UTC):** `2026-11-19T17:40:48.967596+00:00`
- **Fuente:** `validation/results/validation_results.json` ($N = 15$ CVs)
- **Bootstrap:** 10,000 remuestreos, semilla `42`, IC 95% por percentiles `[2.5, 97.5]`
- **Nota de precisión:** Per-CV inputs in validation_results.json are stored rounded to 4 decimal places by ablation_study.py; all bootstrap and Wilcoxon statistics below are reported at full IEEE-754 floating-point precision.

---

## 1. Intervalos de Confianza 95% por Bootstrap ($N = 15$ CVs)

### Jaccard@5

| Arquitectura | Media | DE muestral | Mediana | IQR | IC 95% Inferior | IC 95% Superior | Error Est. Bootstrap |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Modelo completo (p=0.30) | 0.26164666666666664 | 0.24797530368586151 | 0.25 | 0.37305 | 0.14635316666666667 | 0.38806266666666656 | 0.0618259736512607 |
| Sin crédito parcial (p=0) | 0.2560933333333333 | 0.24918454855406913 | 0.2 | 0.37305 | 0.14110533333333333 | 0.3828251666666666 | 0.062161738859832845 |
| Jaccard simple no ponderado | 0.2676266666666667 | 0.254523345734354 | 0.25 | 0.37305 | 0.14577333333333334 | 0.3955769999999999 | 0.06363761092256645 |
| Similitud coseno directa | 0.2698466666666667 | 0.28798333748094984 | 0.1111 | 0.3175 | 0.13967999999999997 | 0.4243466666666667 | 0.07234531685322255 |

### HitRate@5 (Overlap)

| Arquitectura | Media | DE muestral | Mediana | IQR | IC 95% Inferior | IC 95% Superior | Error Est. Bootstrap |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Modelo completo (p=0.30) | 0.3466666666666667 | 0.28751811537130434 | 0.4 | 0.5 | 0.21333333333333332 | 0.4803333333333286 | 0.0716317287939678 |
| Sin crédito parcial (p=0) | 0.3333333333333333 | 0.2894987457822984 | 0.2 | 0.5 | 0.1866666666666667 | 0.4799999999999999 | 0.07221745624684757 |
| Jaccard simple no ponderado | 0.36000000000000004 | 0.2947153590452029 | 0.4 | 0.5 | 0.21333333333333335 | 0.5066666666666666 | 0.07346081796032332 |
| Similitud coseno directa | 0.36 | 0.31350552512789037 | 0.2 | 0.39999999999999997 | 0.21333333333333337 | 0.52 | 0.07870420355602335 |

### Binary HitRate@5

| Arquitectura | Media | DE muestral | Mediana | IQR | IC 95% Inferior | IC 95% Superior | Error Est. Bootstrap |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Modelo completo (p=0.30) | 0.7333333333333333 | 0.45773770821706344 | 1.0 | 0.5 | 0.4666666666666667 | 0.9333333333333333 | 0.11393939147460845 |
| Sin crédito parcial (p=0) | 0.7333333333333333 | 0.45773770821706344 | 1.0 | 0.5 | 0.4666666666666667 | 0.9333333333333333 | 0.11393939147460845 |
| Jaccard simple no ponderado | 0.7333333333333333 | 0.4577377082170634 | 1.0 | 0.5 | 0.4666666666666667 | 0.9333333333333333 | 0.11421984744644167 |
| Similitud coseno directa | 0.8 | 0.41403933560541256 | 1.0 | 0.0 | 0.6 | 1.0 | 0.10295538398850416 |

---

## 2. Prueba Pareada de Wilcoxon Bilateral (`Modelo completo (p=0.30)` vs. Baselines)

Diferencia calculada como `Modelo completo - Baseline`.

| Baseline comparado | Métrica | Dif. media | Dif. mediana | Pares no nulos (pos / neg / empates) | Estadístico W | p-valor | Significativo ($\alpha = 0.05$) |
|:---|:---|---:|---:|---:|---:|---:|:---:|
| Sin crédito parcial (p=0) | `jaccard_at_5` | 0.005553333333333335 | 0.0 | 1 (1 / 0 / 14) | 0.0 | 0.31731050786291415 | No |
| Sin crédito parcial (p=0) | `hit_rate_at_5` | 0.013333333333333334 | 0.0 | 1 (1 / 0 / 14) | 0.0 | 0.31731050786291415 | No |
| Sin crédito parcial (p=0) | `binary_hit_rate_at_5` | 0.0 | 0.0 | 0 (0 / 0 / 15) | N/A (empate total) | 1.0 | No |
| Jaccard simple no ponderado | `jaccard_at_5` | -0.00598 | 0.0 | 5 (2 / 3 / 10) | 6.0 | 0.6858304344516057 | No |
| Jaccard simple no ponderado | `hit_rate_at_5` | -0.01333333333333333 | 0.0 | 4 (1 / 3 / 11) | 4.0 | 0.7150006546880893 | No |
| Jaccard simple no ponderado | `binary_hit_rate_at_5` | 0.0 | 0.0 | 2 (1 / 1 / 13) | 1.5 | 1.0 | No |
| Similitud coseno directa | `jaccard_at_5` | -0.008199999999999994 | 0.0 | 10 (5 / 5 / 5) | 27.0 | 0.9593272374483622 | No |
| Similitud coseno directa | `hit_rate_at_5` | -0.013333333333333315 | 0.0 | 9 (4 / 5 / 6) | 21.0 | 0.8584626501256489 | No |
| Similitud coseno directa | `binary_hit_rate_at_5` | -0.06666666666666667 | 0.0 | 5 (2 / 3 / 10) | 6.0 | 0.6547208460185769 | No |

---

## 3. Acuerdo Entre Evaluadores Antes del Consenso y Elementos No Calculables

- **Estado del acuerdo inter-evaluador pre-consenso:** `not_available`
- **Motivo:** ground_truth_cvs.json stores only the single post-consensus list ('ground_truth_gaps') per CV; individual pre-consensus Top-5 lists from Evaluator 1 and Evaluator 2 are absent from the repository. Additionally, Cohen's kappa is not directly applicable to open-set Top-5 skill lists without defining a closed negative contingency universe.
- **Elementos explícitamente no calculables con los datos disponibles:**
  - Pre-consensus inter-rater Jaccard@5 (Evaluator 1 vs Evaluator 2 lists are not stored in ground_truth_cvs.json)
  - Pre-consensus Cohen's kappa (no individual rater lists stored, and Cohen's kappa requires a fixed closed contingency universe rather than open Top-5 sets)
