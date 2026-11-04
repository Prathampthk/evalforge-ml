# EvalForge resume evidence

Claims below refer to actual implementation and measurements. EvalForge extends an inherited RAG foundation; original copyright and required attribution remain. New work does not establish independent authorship of inherited components.

## Reproducible quality and indexing experiments

Implemented versioned datasets/configs, graded Recall/MRR/NDCG/Precision/Hit Rate, paired query bootstrap intervals, lexical TF-IDF and pinned MiniLM exact/HNSW/IVF comparisons. The original 18 SciFact configuration/scale rows (5,183 full-corpus documents, 300 test queries) reproduced identical quality twice and remain unchanged in [frozen SciFact evidence](docs/SCIFACT_RESULTS.md).

SciFact exact dense Recall@10 is **78.33%** versus **73.19%** TF-IDF (+5.14 percentage points), NDCG 0.6451 versus 0.5924. HNSW warm **pre-encoded** P95 index-query latency is 0.055 ms versus 0.235 ms flat (4.25x lower), with **97.87% exact-neighbor agreement** and 78.00% qrel recall. These timings exclude embedding and HTTP; do not describe them as end-to-end latency or production speedup. [Raw reports, per-query observations, paired intervals and conditions](docs/benchmarks/scifact-v1/report.json) support those numbers.

New completed [ArguAna results](docs/benchmarks/arguana-v1/report.json) compare four methods over **8,674 documents / 1,401 valid queries**: exact dense Recall@10 **79.30%**, TF-IDF **69.31%**; HNSW **79.16%** qrel recall and **99.69%** exact-neighbor agreement. Five original qrels reference absent documents and are explicitly excluded. See [cross-domain results](docs/CROSS_DOMAIN_RESULTS.md), [dataset terms/exclusions](docs/DATASETS.md), and [reproduction commands](docs/EXPERIMENTS.md). The completed [FiQA report](docs/benchmarks/fiqa-v1/report.json) adds **57,600 usable documents / 648 test queries**: exact dense Recall@10 **44.13%** versus **20.66%** TF-IDF; HNSW recall **42.50%** and exact-neighbor agreement **95.49%**. FiQA excludes 38 empty documents and one test judgment, with raw hashes and exclusions retained. These are derived benchmark conditions, not unmodified leaderboard scores.

Code: `evaluation/retrieval_data.py`, `prepare_dataset.py`, `experiment_config.py`, `experiment.py`, `search_backends.py`, `ranking.py`, `statistics.py`. Independent ranking reference tests, strict integrity checks, reload verification, self-match exclusion and deterministic fixture gate cover correctness. Limits: one pinned model/seed, incomplete labels, exploratory unadjusted intervals, no BM25/hybrid or training-contamination audit. Do not claim universal quality superiority.

## Actual end-to-end HTTP load

[36,000 real MiniLM FastAPI retrieval requests](docs/SERVING_PERFORMANCE.md), six closed-loop cases, 6,000 per case, **zero request errors**. Two workers at 16 concurrent clients measured **362.8 successful requests/s**, P50/P95/P99 **51.61 / 81.08 / 121.15 ms** on one macOS CPU host. One worker at that concurrency measured 261.3 QPS; observed throughput ratio 1.39x. Every case evaluated all 300 SciFact queries and preserved exact dense Recall@10 **78.33%**.

Code: `serving/api.py`, `serving/embedding.py`, `evaluation/http_load.py`. [Raw requests, runtime, source hashes and summary](docs/benchmarks/serving-v1/report.json) separate embedding, encoder contention, lookup, processing, server envelope and HTTP/client residual. Server sum RSS sampled every 50 ms; report retains all six peaks. Quantiles of components are not additive. Loopback short windows, one host/model, warm startup excluded, no LLM, no WAN/TLS or production traffic, not a capacity/SLA claim. Residual is not pure physical network latency; repeat requests are not independent relevance judgments.

## Multi-worker FAISS consistency and recovery

Implemented immutable hash-verified snapshots with model fingerprint, cross-process writer lock, full-copy build, fsynced atomic CURRENT activation, request-pinned readers, automatic refresh/restart restore and stale-version HTTP 409. [Contract/tradeoffs](docs/SNAPSHOT_CONSISTENCY.md). Legacy RAG defaults to one worker and requires explicit pinned shared mode for multiple workers.

Tests: `tests/test_shared_snapshots.py`, `test_serving_api.py`, `test_serving_processes.py`, `test_shared_rag_pipeline.py`. Real two-process HTTP tests perform 12 concurrent additions and 40 reads during writes, verify all 13 IDs on both workers, stale CAS, and restart. Abrupt writer exit before activation preserves the prior committed snapshot and permits the next writer to commit. Test-hash embeddings validate mechanics, not semantic quality. POSIX local filesystem only, O(n) rebuild/update, no automatic GC/deletion or distributed replication; do not claim power-loss or production fault tolerance.

## Validation and CI

Current local suite: **169 passed, zero skipped**, 16 dependency deprecation warnings. Black, flake8, mypy (37 source files) and Bandit pass. Deterministic retrieval gate retains the original dataset/config/metric contract. `tests/test_dataset_preparation.py` checks exclusions without altering raw evidence. RAGAS SDK import smoke test runs in a subprocess to prevent its global event-loop patches from contaminating unrelated async tests.

[CI](.github/workflows/ci.yml) adds a real Docker HTTP/update/volume-restart integration job using a clearly labeled synthetic encoder. Local Docker is unavailable; actual container execution passed in [GitHub CI run 41760000510](https://github.com/Prathampthk/evalforge-ml/actions/runs/41760000510) at commit `f55fd69`: all eight jobs succeeded, including **169 tests under Python 3.11.18**, the deterministic gate, and the volume-backed Docker update/restart check. [Actual report and job evidence](docs/benchmarks/ci-serving-v1/ci-summary.json), [audit](docs/PERFORMANCE_AUDIT.md). This verifies synthetic-encoder container mechanics, not real-model Docker performance. Public benchmark timing is not gated on noisy CI runners.

## Resume wording supported by current evidence

- Built a reproducible retrieval evaluation and performance engineering platform with versioned datasets, lexical/dense/ANN comparisons, and a FastAPI service backed by atomic FAISS snapshots.
- Benchmarked **36,000 end-to-end retrieval requests with zero errors**, measuring **362.8 requests/s and 81.08 ms P95** at 16 concurrent clients across two CPU workers while retaining **78.33% SciFact Recall@10**.
- Validated concurrent ingestion, cross-worker query consistency and restart/crash recovery with **169 passing tests**, adding dataset-aware regression gates and reproducible Docker integration checks.

## Unsupported claims

Production capacity, zero hallucinations, generated-answer accuracy gains, paid judge cost savings, human-judge agreement/false-positive rates, GPU speedups, new model training, long-running memory stability, multi-seed robustness, or unmeasured deployment success. Existing `run_ragas` provenance and optional cost accounting do not replace an actual credentialed and independently validated judge run.
