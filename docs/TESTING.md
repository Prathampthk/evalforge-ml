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

`.github/workflows/ci.yml` runs formatting, linting, typing, unit tests, and security/dependency scans. Its pytest job excludes `tests/test_api.py`; run the complete suite locally. `.github/workflows/quality-regression.yml` defaults to dataset-fixture checks, not measured model quality. See the [release audit](RELEASE_AUDIT.md) for actual local results.
