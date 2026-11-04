# Retrieval performance engineering audit

This upgrade focuses on search, indexing, HTTP measurement, and reproducibility. The original SciFact reports and results remain unchanged. The inherited RAG answer pipeline and judge evaluation remain available but are excluded from retrieval throughput claims.

## Review findings and changes

| Finding | Implemented change | Evidence / limit |
|---|---|---|
| Published fast lookup excluded text embedding and HTTP | Separate retrieval FastAPI app; per-request embedding/wait/lookup/processing, HTTP residual and empty control | [36,000 real MiniLM HTTP requests](SERVING_PERFORMANCE.md); loopback only |
| Serial lookup QPS was not concurrent service throughput | Closed-loop load driver with raw requests, errors, quantiles, PID observations and sampled server RSS | Six worker/concurrency cases; short bursts, no production capacity claim |
| In-memory worker indexes diverged after ingestion | Immutable versioned snapshots, cross-process writer lock, atomic pointer, per-request refresh, CAS | [Snapshot contract and crash/restart tests](SNAPSHOT_CONSISTENCY.md) |
| Legacy default launched four independent indexes | One-worker default; explicit pinned shared-mode opt-in for multiple workers | Legacy integration tests and start script guard |
| One small scientific corpus could not establish cross-domain behavior | ArguAna and FiQA preparers/configurations, strict explicit data-cleaning manifests and self-match exclusion | [Cross-domain results](CROSS_DOMAIN_RESULTS.md); no universal generalization claim |
| RAGAS import globally patched asyncio | SDK import smoke test in a subprocess | Preserves compatibility assertion without mutating unrelated test loops |
| Docker configuration existed without runtime evidence | Dedicated nonroot retrieval image; disposable image/container/volume HTTP update/restart integration job | Local Docker unavailable; GitHub execution status recorded below |
| macOS native packages bundled competing OpenMP runtimes | Scoped subprocess native-library resolution; record loaded runtime; fail on duplicate Darwin libraries | Actual serving health reports one runtime; no disabling of duplicate check |

## Validation status

Local final suite: **169 passed, zero skipped**, 16 dependency deprecation warnings. Black, flake8, mypy (37 source files), and Bandit pass. Tests use synthetic encoders/provider mocks except the separately published real MiniLM experiments and HTTP load. CI also runs the unchanged deterministic retrieval baseline; performance thresholds are deliberately not enforced on noisy shared runners.

Docker is not installed on this Mac. The new Docker integration job must complete on GitHub before container execution is considered verified. A pending configuration is not a passing gate. Real model Docker inference, cloud deployment, live OTLP dashboards, GPU inference, WAN tests, multi-seed robustness, incremental ANN updates and long duration memory/ingestion soak remain unmeasured.
