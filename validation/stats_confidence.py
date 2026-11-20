"""Inferential statistics and bootstrap confidence intervals for RQ3 gap concordance.

Reads per-CV concordance metrics from ``validation/results/validation_results.json``
without modifying any existing validation code or datasets, and computes:
1. 95% bootstrap confidence intervals (10,000 resamples, seed=42) for Jaccard@5,
   HitRate@5 (Overlap), and Binary HitRate@5 of each evaluated architecture.
2. Two-sided paired Wilcoxon signed-rank tests between the full model (p=0.30)
   and each of the three baselines.
3. Explicit status reporting for pre-consensus inter-rater agreement, which is not
   computable because only post-consensus ``ground_truth_gaps`` exist in the repo.

Outputs:
- validation/results/stats_confidence.json
- validation/results/stats_confidence.md
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
DATA_DIR = os.path.join(VALIDATION_DIR, "data")

INPUT_RESULTS_PATH = os.path.join(RESULTS_DIR, "validation_results.json")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth_cvs.json")
OUT_JSON_PATH = os.path.join(RESULTS_DIR, "stats_confidence.json")
OUT_MD_PATH = os.path.join(RESULTS_DIR, "stats_confidence.md")

FULL_MODEL_NAME = "Modelo completo (p=0.30)"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

METRICS = ("jaccard_at_5", "hit_rate_at_5", "binary_hit_rate_at_5")


def bootstrap_ci(
    values: list[float],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute sample summary statistics and percentile 95% bootstrap CI."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)

    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[indices].mean(axis=1)

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    q1, q3 = np.percentile(arr, [25.0, 75.0])

    return {
        "n": int(n),
        "mean": float(np.mean(arr)),
        "std_sample": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "bootstrap_ci_95_low": float(ci_low),
        "bootstrap_ci_95_high": float(ci_high),
        "bootstrap_mean": float(np.mean(boot_means)),
        "bootstrap_se": float(np.std(boot_means, ddof=1)),
    }


def paired_wilcoxon(
    full_values: list[float],
    baseline_values: list[float],
) -> dict[str, Any]:
    """Run a two-sided paired Wilcoxon signed-rank test (full - baseline)."""
    a = np.asarray(full_values, dtype=np.float64)
    b = np.asarray(baseline_values, dtype=np.float64)
    diffs = a - b
    non_zero_mask = ~np.isclose(diffs, 0.0, atol=1e-12, rtol=0.0)
    n_non_zero = int(np.sum(non_zero_mask))

    result: dict[str, Any] = {
        "n_pairs": int(len(a)),
        "n_non_zero_diffs": n_non_zero,
        "mean_diff_full_minus_baseline": float(np.mean(diffs)),
        "median_diff_full_minus_baseline": float(np.median(diffs)),
        "positive_diffs_count": int(np.sum(diffs > 0.0)),
        "negative_diffs_count": int(np.sum(diffs < 0.0)),
        "ties_count": int(np.sum(~non_zero_mask)),
    }

    if n_non_zero == 0:
        result.update(
            {
                "statistic": None,
                "p_value": 1.0,
                "significant_alpha_0_05": False,
                "note": "All paired differences are identically zero; Wilcoxon statistic undefined.",
            }
        )
        return result

    stat, p_val = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    result.update(
        {
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant_alpha_0_05": bool(p_val < 0.05),
            "note": None,
        }
    )
    return result


def check_pre_consensus_availability(ground_truth_path: str) -> dict[str, Any]:
    """Verify whether separate pre-consensus evaluator Top-5 lists exist."""
    if not os.path.exists(ground_truth_path):
        return {
            "status": "not_available",
            "reason": f"Ground truth file not found at {ground_truth_path}",
            "jaccard_inter_rater": None,
            "cohens_kappa": None,
        }

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    sample_keys = sorted(entries[0].keys()) if entries else []
    has_rater_1 = any("evaluator_1" in e or "rater_1" in e for e in entries)
    has_rater_2 = any("evaluator_2" in e or "rater_2" in e for e in entries)

    if not (has_rater_1 and has_rater_2):
        return {
            "status": "not_available",
            "reason": (
                "ground_truth_cvs.json stores only the single post-consensus list "
                "('ground_truth_gaps') per CV; individual pre-consensus Top-5 lists "
                "from Evaluator 1 and Evaluator 2 are absent from the repository. "
                "Additionally, Cohen's kappa is not directly applicable to open-set "
                "Top-5 skill lists without defining a closed negative contingency universe."
            ),
            "observed_cv_keys": sample_keys,
            "jaccard_inter_rater": None,
            "cohens_kappa": None,
        }

    return {
        "status": "available",
        "reason": "Pre-consensus lists present",
        "jaccard_inter_rater": None,
        "cohens_kappa": None,
    }


def compute_all_stats() -> dict[str, Any]:
    """Load validation_results.json and compute bootstrap CIs and Wilcoxon tests."""
    with open(INPUT_RESULTS_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    eval_conc = val_data.get("evaluator_gap_concordance", {})
    per_profile = eval_conc.get("per_profile", [])
    if not per_profile:
        raise RuntimeError("No per_profile entries found under evaluator_gap_concordance.")

    # Collect per-architecture metric vectors ordered by cv-id
    arch_order: list[str] = []
    vectors: dict[str, dict[str, list[float]]] = {}
    cv_ids: list[str] = []

    for profile in per_profile:
        cv_ids.append(profile["id"])
        for comp in profile["comparison"]:
            arch = comp["architecture"]
            if arch not in vectors:
                arch_order.append(arch)
                vectors[arch] = {m: [] for m in METRICS}
            for m in METRICS:
                vectors[arch][m].append(float(comp[m]))

    if FULL_MODEL_NAME not in vectors:
        raise RuntimeError(f"Full model '{FULL_MODEL_NAME}' not found in validation_results.json.")

    bootstrap_results: dict[str, dict[str, Any]] = {}
    for arch in arch_order:
        bootstrap_results[arch] = {
            m: bootstrap_ci(vectors[arch][m], n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            for m in METRICS
        }

    wilcoxon_results: dict[str, dict[str, Any]] = {}
    baselines = [arch for arch in arch_order if arch != FULL_MODEL_NAME]
    for base in baselines:
        wilcoxon_results[base] = {
            m: paired_wilcoxon(vectors[FULL_MODEL_NAME][m], vectors[base][m])
            for m in METRICS
        }

    inter_rater = check_pre_consensus_availability(GROUND_TRUTH_PATH)

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": "validation/results/validation_results.json",
            "n_cvs": len(per_profile),
            "cv_ids": cv_ids,
            "n_bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "note_precision": (
                "Per-CV inputs in validation_results.json are stored rounded to 4 decimal places "
                "by ablation_study.py; all bootstrap and Wilcoxon statistics below are reported "
                "at full IEEE-754 floating-point precision."
            ),
        },
        "architectures_order": arch_order,
        "per_cv_raw_metrics": vectors,
        "bootstrap_95_ci": bootstrap_results,
        "wilcoxon_paired_vs_full_model": wilcoxon_results,
        "inter_rater_pre_consensus": inter_rater,
        "not_computable_with_available_data": [
            "Pre-consensus inter-rater Jaccard@5 (Evaluator 1 vs Evaluator 2 lists are not stored in ground_truth_cvs.json)",
            "Pre-consensus Cohen's kappa (no individual rater lists stored, and Cohen's kappa requires a fixed closed contingency universe rather than open Top-5 sets)",
        ],
    }


def write_markdown_report(report: dict[str, Any], out_path: str) -> None:
    """Write a complete, unrounded Markdown summary of all inferential statistics."""
    meta = report["metadata"]
    boot = report["bootstrap_95_ci"]
    wil = report["wilcoxon_paired_vs_full_model"]
    inter = report["inter_rater_pre_consensus"]

    lines = [
        "# Estadística Inferencial y Concordancia RQ3 (`stats_confidence`)\n\n",
        f"- **Fecha (UTC):** `{meta['generated_at_utc']}`\n",
        f"- **Fuente:** `{meta['source_file']}` ($N = {meta['n_cvs']}$ CVs)\n",
        f"- **Bootstrap:** {meta['n_bootstrap_resamples']:,} remuestreos, semilla `{meta['bootstrap_seed']}`, IC 95% por percentiles `[2.5, 97.5]`\n",
        f"- **Nota de precisión:** {meta['note_precision']}\n\n",
        "---\n\n",
        "## 1. Intervalos de Confianza 95% por Bootstrap ($N = 15$ CVs)\n\n",
    ]

    for metric, label in [
        ("jaccard_at_5", "Jaccard@5"),
        ("hit_rate_at_5", "HitRate@5 (Overlap)"),
        ("binary_hit_rate_at_5", "Binary HitRate@5"),
    ]:
        lines.extend([
            f"### {label}\n\n",
            "| Arquitectura | Media | DE muestral | Mediana | IQR | IC 95% Inferior | IC 95% Superior | Error Est. Bootstrap |\n",
            "|:---|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for arch in report["architectures_order"]:
            s = boot[arch][metric]
            lines.append(
                f"| {arch} | {s['mean']} | {s['std_sample']} | {s['median']} | {s['iqr']} | "
                f"{s['bootstrap_ci_95_low']} | {s['bootstrap_ci_95_high']} | {s['bootstrap_se']} |\n"
            )
        lines.append("\n")

    lines.extend([
        "---\n\n",
        f"## 2. Prueba Pareada de Wilcoxon Bilateral (`{FULL_MODEL_NAME}` vs. Baselines)\n\n",
        "Diferencia calculada como `Modelo completo - Baseline`.\n\n",
        "| Baseline comparado | Métrica | Dif. media | Dif. mediana | Pares no nulos (pos / neg / empates) | Estadístico W | p-valor | Significativo ($\\alpha = 0.05$) |\n",
        "|:---|:---|---:|---:|---:|---:|---:|:---:|\n",
    ])

    for base, metrics_map in wil.items():
        for metric in METRICS:
            w = metrics_map[metric]
            stat_str = str(w["statistic"]) if w["statistic"] is not None else "N/A (empate total)"
            sig_str = "Sí" if w["significant_alpha_0_05"] else "No"
            lines.append(
                f"| {base} | `{metric}` | {w['mean_diff_full_minus_baseline']} | "
                f"{w['median_diff_full_minus_baseline']} | "
                f"{w['n_non_zero_diffs']} ({w['positive_diffs_count']} / {w['negative_diffs_count']} / {w['ties_count']}) | "
                f"{stat_str} | {w['p_value']} | {sig_str} |\n"
            )

    lines.extend([
        "\n---\n\n",
        "## 3. Acuerdo Entre Evaluadores Antes del Consenso y Elementos No Calculables\n\n",
        f"- **Estado del acuerdo inter-evaluador pre-consenso:** `{inter['status']}`\n",
        f"- **Motivo:** {inter['reason']}\n",
        "- **Elementos explícitamente no calculables con los datos disponibles:**\n",
    ])
    for item in report["not_computable_with_available_data"]:
        lines.append(f"  - {item}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = compute_all_stats()

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_markdown_report(report, OUT_MD_PATH)
    print(f"Generated:\n  - {OUT_JSON_PATH}\n  - {OUT_MD_PATH}")


if __name__ == "__main__":
    main()
