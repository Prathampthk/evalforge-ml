# Retrieval infrastructure upgrade — baseline audit

Audit date: 2026-11-03. Initial working tree contains the preceding uncommitted retrieval-benchmark extension; it is preserved. No applicable AGENTS.md is present.

## Verified baseline before implementation

- `OMP_NUM_THREADS=1 /private/tmp/evalforge-release-venv/bin/python -m pytest tests/ -q`: **72 passed, 24 skipped, 16 warnings**. The API tests skip without credentials. OpenTelemetry console exporters write to closed pytest streams during shutdown; these errors are not test failures but need lifecycle attention.
- `OMP_NUM_THREADS=1 ... -m evaluation.retrieval_benchmark --output /private/tmp/evalforge-upgrade-baseline.json`: 22 queries, 20 passages; similarity K=5 Recall=0.9545454545454546, MRR=0.9545454545454546, binary NDCG=0.9508963995067358. Matches historical artifact. Latencies are machine/run dependent.
- Corpus and query labels are inherited repository examples, not an independently annotated public benchmark. `ragas_eval.jsonl` has explicit expected passage IDs and one multi-positive query. The legacy `phase2_eval.jsonl` path is a different fixture contract.

## Component inventory

| Stage | Actual implementation | Gaps |
|---|---|---|
| Loading and preprocessing | `document_loader.py`, `chunking.py`: strings, UTF-8 files, token/semantic/character chunking | No versioned retrieval dataset contract |
| Embeddings | `embeddings.py`: sentence-transformers, default BGE-large, dimension checks | Existing retrieval benchmark uses corpus-fitted TF-IDF instead; no dense benchmark run |
| Indexes | `vector_store.py`: FAISS flat/IP, IVF/IP, HNSW/IP, explicit save/load | Index memory/build/serialization and ANN agreement not separately benchmarked |
| Retrieval and reranking | `retriever.py`: similarity or MMR, threshold gate; optional LLM guardrail | No BM25; six existing TF-IDF similarity/MMR configurations only |
| Generation | `generator.py`, `llm_client.py`: OpenAI-compatible model, token/context budgeting | No verified cost or judge agreement evidence |
| Evaluation | `retrieval_benchmark.py`: binary Recall/MRR/NDCG; `run_ragas.py`: cached answers and RAGAS; `run_eval.py`: fixed fixture scores/heuristic live scores | No graded qrels, Precision/Hit Rate, statistical comparison, configurable experiments or held-out data |
| Observability | `telemetry.py` and instrumentation in pipeline, embeddings, store, retriever, generator, LLM client, API | Benchmark stages lack experiment IDs; some context-manager attributes not applied; no benchmark-specific failure lifecycle |
| Tests/CI | 72 passing baseline tests, 24 skipped; Black/flake8/mypy/Bandit configured; API tests excluded from CI | No versioned retrieval quality gate; prior evaluation gate accepts fixed annotation-derived values |

No independent production-performance or LLM-accuracy claim is supported by this baseline. Historical audits/planning records are context, not a current feature inventory. No resume file is modified by this task.

## Implementation sequence

1. Versioned BEIR-compatible datasets and documented SciFact preparation; retain regression fixture.
2. A new, explicitly versioned metric contract with independently checked graded ranking metrics.
3. Configured sparse lexical and pinned CPU dense baselines; controlled FAISS exact/HNSW/IVF/MMR comparisons.
4. Real-document scaling, index serialization/memory estimates, separated build/embedding/query timings.
5. Paired bootstrap differences, trace correlation, privacy-safe stage spans.
6. Deterministic CI gates, tests, architecture/methodology, actual run artifacts, resume evidence.

See `EXPERIMENTS.md`, `DATASETS.md`, and `RESUME_EVIDENCE.md` for implemented contracts and final measured evidence. The old fixture evaluator is retained for compatibility and is not promoted as a real quality gate.

## Final verification

All requested workstreams are implemented without replacing the RAG service. The original 22-query fixture and six configurations remain reproducible; the new versioned metric contract matches the original binary results. SciFact preparation verifies both published transfer checksum and pinned SHA-256 and separates train/test roles. A real pinned MiniLM model was downloaded and run on CPU; no fake embeddings were used for published results.

Two full public experiments produced identical relevance metrics and exact-neighbor agreement across all 18 configuration/size results. Timings varied and are reported as warm serial, pre-encoded lookup measurements, separately from model encoding/index build. The predesignated second-run raw report, per-query results and exact source snapshot are retained in `docs/benchmarks/scifact-v1/`; full index artifacts remain in the ignored local report directory. Final source differs from that frozen snapshot only in read-only Git provenance command safety (resolved executable path and static-argument security annotations), not retrieval, metrics, index settings or bootstrap algorithms.

Final local checks: **153 tests passed, zero skipped, 20 deprecation warnings**; Black, flake8, mypy, Bandit (zero findings), `pip check`, dataset/config parsing and diff whitespace checks pass. API tests now isolate startup and providers; tests explicitly validate actual in-memory telemetry. Source snapshots in generated report directories are excluded from static-analysis traversal to avoid duplicate modules. The credential-free versioned retrieval quality gate passes on a fresh run; rejection of real score regressions and incomparable reports is tested.

Results and limitations: `SCIFACT_RESULTS.md`, `EXPERIMENTS.md`, `DATASETS.md`, and `../RESUME_EVIDENCE.md`. Python 3.11/GitHub workflow runtime, Docker, paid LLM judgments, human-judge agreement, false-positive rates, multi-seed/multi-domain generalization and production load remain unverified. Production console-exporter lifecycle behavior is not claimed fixed merely because unit tests isolate exporters. No upstream attribution was removed, no Git history was modified, and no commit/push was performed for this upgrade.
