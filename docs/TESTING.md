<!-- generated-by: gsd-doc-writer -->
# Testing

## Test Framework and Setup

- Framework: `pytest` (declared in `requirements.txt` as `pytest>=8.0.0`).
- Setup helper: `tests/conftest.py` prepends the project root to `sys.path` for module imports.

Before running tests:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

Run full suite:

```bash
pytest tests/ -v
```

Run a single test module:

```bash
pytest tests/test_vector_store.py -v
```

Run a single test case by name:

```bash
pytest tests/test_pipeline_phase1_runtime.py -k sync_mode -v
```

## Writing New Tests

Current test naming pattern:

- Directory: `tests/`
- Files: `test_*.py`
- Test functions: `test_*`

Useful patterns already in repo:

- Monkeypatch-based runtime isolation (`tests/test_pipeline_phase1_runtime.py`) for deterministic unit testing.
- Direct component tests for loader, embeddings, retriever/generator/evaluator, and vector store.

## Coverage Requirements

No explicit coverage threshold config was found (no `.coveragerc`, `pytest-cov` config, or CI-enforced threshold).

## CI Integration

`.github/workflows/ci.yml` runs formatting, linting, typing, unit tests, and security/dependency scans. Its pytest job includes API contract tests with mocked startup (no model/provider downloads). A separate deterministic retrieval job checks the versioned quality baseline. `.github/workflows/quality-regression.yml` defaults to dataset-fixture checks, not measured model quality. See the [release audit](RELEASE_AUDIT.md) for actual local results.


## Retrieval infrastructure verification

```bash
OMP_NUM_THREADS=1 python -m pytest tests/ -q
OMP_NUM_THREADS=1 python -m evaluation.experiment --config configs/experiments/regression.json --output evaluation/reports/ci-new
python -m evaluation.retrieval_gate --report evaluation/reports/ci-new/report.json --baseline evaluation/baselines/regression-v1.json
```

Ranking tests use hand calculations and exhaustive independent scikit-learn NDCG comparisons. Index tests load saved exact/HNSW/IVF artifacts and check search consistency. Other tests cover manifest integrity, missing/duplicate labels, deterministic artifacts, paired bootstrap, regression rejection, trace privacy, API auth/options, and RAGAS accounting/cache integrity. Runtime timing tests use one FAISS thread locally. Console exporter output is disabled only in unit-test setup; explicit in-memory exporter tests verify actual spans.

Generated experiment source snapshots are excluded from static analysis to avoid duplicate module discovery; application sources remain checked. See [results](SCIFACT_RESULTS.md) for final test counts and validation limits.
