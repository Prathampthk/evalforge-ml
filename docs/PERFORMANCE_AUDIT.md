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

Docker is not installed on this Mac. Actual container execution is now verified by [GitHub CI run 41760000510](https://github.com/Prathampthk/evalforge-ml/actions/runs/41760000510) at head commit `f55fd69a51a3160ceaaa4509e8a32c31a57cbee7`: **all eight jobs passed**, including Python 3.11.18 unit tests (**169 passed, zero skipped**, one dependency warning), deterministic retrieval regression, Black, flake8, mypy, Bandit, dependency scan and Docker integration. The first attempt failed because the minimal host lacked the evaluation package's Pydantic dependency; CI installation and the image's copied dataset module were corrected before the successful run.

The [actual Docker report](benchmarks/ci-serving-v1/docker-integration.json) verifies 12 concurrent additions, 40 reads during writes, 13 final documents on both worker PIDs, stale CAS HTTP 409, and the same version/count plus a successful query on both workers after container restart using the named volume. The test uses `synthetic-test-word-hash`, not MiniLM semantic inference. [CI summary](benchmarks/ci-serving-v1/ci-summary.json) links the exact run/jobs/artifact. The final evidence-only documentation commit can be later than the code commit verified here; no runtime implementation change is implied. Real model Docker inference, cloud deployment, live OTLP dashboards, GPU inference, WAN tests, multi-seed robustness, incremental ANN updates and long duration memory/ingestion soak remain unmeasured.
