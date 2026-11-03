# EvalForge resume evidence

This file links claims to implementation, tests, commands, data, actual results and limits. The project extends an inherited RAG foundation; the upstream copyright/license are preserved. New retrieval infrastructure does not imply independent authorship of inherited RAG components.

## Reproducible retrieval benchmarking

**Supported claim:** Implemented versioned retrieval experiments comparing six configurations over three real corpus sizes (18 configuration/size results) on BEIR SciFact, with 5,183 documents, 300 test queries and 339 positive judgments.

- Implementation: [dataset loader](evaluation/retrieval_data.py), [bounded preparer](evaluation/prepare_dataset.py), [strict config](evaluation/experiment_config.py), [runner](evaluation/experiment.py), [versioned metrics](evaluation/ranking.py).
- Tests: [dataset integrity](tests/test_retrieval_data.py), [independent ranking reference](tests/test_ranking_v2.py), [deterministic runner/config/persistence](tests/test_experiments.py).
- Command: `OMP_NUM_THREADS=1 python -m evaluation.experiment --config configs/experiments/scifact.json --output evaluation/reports/scifact-new` after explicit dataset/model preparation.
- Data: [origin/licenses/splits](docs/DATASETS.md); [frozen test manifest](docs/benchmarks/scifact-v1/test.manifest.json).
- Result: [complete report](docs/benchmarks/scifact-v1/report.json), [tables](docs/SCIFACT_RESULTS.md). All 18 quality and ANN-agreement results reproduced identically across two complete runs.
- Limits: one domain/model/seed; smaller subsets deliberately retain all judged positives and sample real distractors. They are not independent external datasets. Test parameters were prespecified, not tuned against held-out scores.

## Retrieval relevance improvement on the measured dataset

**Supported claim:** Dense exact Recall@10 measured 78.33%, versus 73.19% TF-IDF (+5.14 percentage points); NDCG@10 measured 0.6451 versus 0.5924.

- Implementation: [sparse lexical/FAISS adapters](evaluation/search_backends.py), [paired bootstrap](evaluation/statistics.py), [experiment config](configs/experiments/scifact.json).
- Tests: [ranking correctness](tests/test_ranking_v2.py), [paired statistics](tests/test_statistics_and_gate.py), [search consistency](tests/test_experiments.py).
- Command/data: same SciFact full-corpus experiment above; identical 300 queries, corpus and qrels, K=10.
- Result: rows `tfidf-exact-n5183` and `dense-flat-n5183` in the [raw report](docs/benchmarks/scifact-v1/report.json); [per-query dense evidence](docs/benchmarks/scifact-v1/queries/dense-flat-n5183.json). Recall difference paired 95% CI [+0.09,+10.32] percentage points using 2,000 query-resampling draws, seed 42.
- Limits: exploratory unadjusted intervals, incomplete judgments and possible query correlation. Do not claim a universal 5.14-point improvement, generated-answer accuracy, or statistical superiority across retrieval tasks. MiniLM is not a scientifically specialized or newly trained model here.

## Index quality/performance tradeoff

**Supported claim:** On the full corpus, HNSW reduced measured warm pre-encoded P95 index-query latency from 0.235 ms (flat) to 0.055 ms (about 4.25x lower), retaining 97.87% exact-neighbor Recall@10 and 78.00% qrel Recall@10.

- Implementation: [index builders/search](evaluation/search_backends.py), [performance and exact-reference evaluation](evaluation/experiment.py).
- Tests: parametrized exact/HNSW/IVF construction, serialization/reload, dimensions, tie handling and search consistency in [test_experiments.py](tests/test_experiments.py).
- Command/data: same pinned MiniLM representations, full 5,183-document SciFact corpus and 300 queries, single CPU/FAISS thread; HNSW M=16, efConstruction=100, efSearch=32.
- Result: `dense-flat-n5183` versus `dense-hnsw-n5183` [report rows and paired intervals](docs/SCIFACT_RESULTS.md). Raw per-query timing trials are retained. These exact latency values are from the predesignated second run, not a best-of-run selection.
- Limits: lookup excludes embedding generation, corpus build, network and LLM generation. Warm serial microbenchmark on one host, not end-to-end service latency, concurrent throughput, production SLA, or isolated timing hardware. Exact-to-HNSW qrel recall changed by -0.33 points (paired CI [-1.00,0.00]); do not call approximate search lossless.

## Automated regression and observability

**Supported claim:** Expanded validation to 153 passing automated tests with no skips, adding versioned deterministic retrieval gates and privacy-safe experiment-correlated stage traces.

- Implementation: [retrieval gate](evaluation/retrieval_gate.py), [frozen baseline](evaluation/baselines/regression-v1.json), [normal CI](.github/workflows/ci.yml), [separate public workflow](.github/workflows/retrieval-benchmark.yml), [telemetry](telemetry.py), [context construction](generator.py).
- Tests: [gate failure/reproducibility](tests/test_statistics_and_gate.py), [actual in-memory spans/privacy](tests/test_experiment_telemetry.py), [credential-free mocked API contracts](tests/test_api.py), plus the existing and new full suite.
- Command: `OMP_NUM_THREADS=1 python -m pytest tests/ -q`; regression command and gate in [methodology](docs/EXPERIMENTS.md).
- Result: 153 passed, 0 skipped, 20 deprecation warnings. Baseline before this upgrade was 72 passed/24 skipped; the upgrade enables those 24 tests and adds 57 test cases. The original preceding extension included 12 benchmark tests and remains preserved. Formatting, linting, typing, Bandit and dependency checks pass locally.
- Limits: tests isolate models/providers; actual provider calls and deployment are separate. GitHub Actions configuration is implemented but has not been executed from this workspace. In-memory telemetry verification is not a live observability deployment.

## Claims not currently supported

- LLM answer accuracy/faithfulness gains, human-judge agreement, false-positive rates, inference cost savings, GPU speedups, production throughput, concurrent-user capacity, or zero hallucinations.
- Newly trained dense models, implemented BM25/hybrid baselines, multiple independent benchmark domains, or multi-seed robustness.
- Treating the old fixed-score fixture evaluator as measured model quality.

`run_ragas` preserves answer evaluation with improved provenance/cost accounting; actual measured judge costs, agreement and correctness remain **[unmeasured]** until a credentialed, independently validated run is performed.

## Three resume bullets grounded in this evidence

- Engineered reproducible retrieval experiments across **300 SciFact queries and 5,183 documents**, comparing **18 configuration/scale combinations** with versioned datasets, graded ranking metrics, and paired bootstrap intervals.
- Measured a **4.25x reduction in warm pre-encoded P95 index-query latency** using HNSW versus exact FAISS search, retaining **97.87% exact-neighbor Recall@10** on the full SciFact corpus.
- Expanded regression validation to **153 passing tests** and automated dataset-aware quality gates, instrumenting encoding, index lookup, MMR and scoring with privacy-safe OpenTelemetry experiment traces.
