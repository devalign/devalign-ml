"""Unit tests for inferential statistics helpers in validation/stats_confidence.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

from validation.stats_confidence import (
    bootstrap_ci,
    check_pre_consensus_availability,
    compute_all_stats,
    paired_wilcoxon,
)


def test_bootstrap_ci_is_deterministic_and_bounded():
    """bootstrap_ci returns deterministic CIs containing the sample mean for seed=42."""
    values = [0.2, 0.4, 0.3, 0.5, 0.6, 0.4, 0.35, 0.45]

    first = bootstrap_ci(values, n_resamples=1000, seed=42)
    second = bootstrap_ci(values, n_resamples=1000, seed=42)

    assert first == second
    assert first["n"] == len(values)
    assert first["bootstrap_ci_95_low"] <= first["mean"] <= first["bootstrap_ci_95_high"]
    assert first["min"] <= first["median"] <= first["max"]
    assert first["std_sample"] > 0.0


def test_paired_wilcoxon_handles_identical_arrays():
    """paired_wilcoxon handles zero differences without calling scipy.stats.wilcoxon."""
    vals = [0.25, 0.5, 0.0, 0.75]

    res = paired_wilcoxon(vals, vals)

    assert res["n_pairs"] == len(vals)
    assert res["n_non_zero_diffs"] == 0
    assert res["ties_count"] == len(vals)
    assert res["positive_diffs_count"] == 0
    assert res["negative_diffs_count"] == 0
    assert res["statistic"] is None
    assert math.isclose(res["p_value"], 1.0)
    assert res["significant_alpha_0_05"] is False


def test_paired_wilcoxon_detects_distinct_distributions():
    """paired_wilcoxon computes valid statistic and p-value when pairs differ."""
    full_vals = [0.6, 0.7, 0.8, 0.9, 0.75, 0.85, 0.95, 0.8]
    base_vals = [0.1, 0.2, 0.3, 0.2, 0.25, 0.35, 0.4, 0.3]

    res = paired_wilcoxon(full_vals, base_vals)

    assert res["n_non_zero_diffs"] == len(full_vals)
    assert res["positive_diffs_count"] == len(full_vals)
    assert res["negative_diffs_count"] == 0
    assert res["ties_count"] == 0
    assert res["statistic"] is not None
    assert 0.0 < res["p_value"] < 0.05
    assert res["significant_alpha_0_05"] is True


def test_check_pre_consensus_availability_missing_and_consensus_only(tmp_path: Path):
    """Pre-consensus status reports not_available when file is missing or single-rater."""
    missing = check_pre_consensus_availability(str(tmp_path / "missing.json"))
    assert missing["status"] == "not_available"
    assert missing["jaccard_inter_rater"] is None

    gt_file = tmp_path / "gt.json"
    gt_file.write_text(
        json.dumps([{"id": "cv-1", "ground_truth_gaps": ["Docker", "Kubernetes"]}]),
        encoding="utf-8",
    )
    consensus_only = check_pre_consensus_availability(str(gt_file))
    assert consensus_only["status"] == "not_available"
    assert "ground_truth_gaps" in consensus_only["observed_cv_keys"]


def test_compute_all_stats_smoke():
    """compute_all_stats loads saved validation_results.json and returns expected structure."""
    report = compute_all_stats()
    assert report["metadata"]["n_cvs"] == 15
    assert "Modelo completo (p=0.30)" in report["bootstrap_95_ci"]
    assert report["inter_rater_pre_consensus"]["status"] == "not_available"
