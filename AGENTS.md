# devalign-ml — Agent Rules

Machine learning, deterministic Lightcast catalog scanning, UMAP/HDBSCAN clustering, skill-graph inference, and empirical evaluation benchmarks.

## Stack & Tooling

- **Runtime:** Python 3.12 using the local `.venv` (`pyproject.toml`, `requirements.txt`).
- **Structure:** Core modules under `src/`, tests under `tests/`, empirical benchmarks under `validation/` (with artifacts in `validation/results/`), scripts under `scripts/`.

## Quality Gates

Run inside `c:\Projects\Devalign\devalign-ml`:

- Lint: `ruff check src/ tests/` (line length 120, targets `E`, `F`, `W` per `pyproject.toml`)
- Tests: `pytest` (runs `tests/` with `-v --strict-markers` and `pythonpath = ["."]`)

## Empirical Validation & Reproducibility

- Benchmark entry points (`validation/`):
  - `validation/benchmark_catalog_scan.py`
  - `validation/benchmark_cv_extraction.py`
  - `validation/ablation_study.py`
  - `validation/stats_confidence.py`
  - `validation/run.py`
- Run full validation benchmarks only when explicitly requested or when modifying extraction, normalization, graph inference, or scoring logic.
- **Never** alter ground-truth evaluation datasets (such as `ground_truth_cvs.json`) or overwrite saved files in `validation/results/` without explicit user approval.
- Preserve deterministic reproducibility settings (`seed=42`, `10,000` bootstrap resamples in statistical checks) and record exact Git commit hashes when reporting updated metrics.
