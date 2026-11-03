# Changelog

## Unreleased

- Bound LangChain integrations to the RAGAS 0.2-compatible family after a fresh Linux CI install exposed dependency drift; validate caches before imports and smoke-test actual judge integrations.

- Add strict versioned BEIR-compatible corpus/query/qrel manifests and bounded checksum-verified SciFact preparation.
- Implement graded Recall/Precision/MRR/NDCG/Hit Rate with documented semantics and independent reference checks.
- Add JSON/YAML experiments for sparse TF-IDF and pinned MiniLM exact/HNSW/IVF/MMR retrieval, real corpus subsets, index artifacts and performance measurements.
- Add paired query bootstrap intervals, frozen code snapshots, privacy-safe correlated traces, versioned CI quality gates, and separate public-benchmark workflow.
- Run API contract tests without external credentials; preserve RAGAS while adding prompt/token/cost provenance and cache integrity checks.

- Add a reproducible, corpus-backed TF-IDF/FAISS retrieval benchmark comparing similarity search and MMR.
- Report Recall@K, MRR@K, NDCG@K, warm-query latency, per-query rankings, and input/environment provenance.
- Add ranking-math, judgment-validation, and end-to-end benchmark tests; record initial sample results and limitations.

## 0.1.0 — Initial EvalForge release preparation

- Establish EvalForge identity and maintainer metadata.
- Document inherited RAG capabilities, evaluation limits, and runtime configuration.
- Separate future retrieval, model evaluation, optimization, and production milestones.
- Preserve upstream MIT attribution and existing Python/API interfaces.
- Improve environment and Docker build-context hygiene.
- No major ML features, published benchmark results, or production-readiness claims.
