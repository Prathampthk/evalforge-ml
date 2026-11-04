# Deploying EvalForge v0.1.0 locally

The repository contains a Python 3.11 multi-stage Dockerfile and Compose configuration. This document covers local operation; cloud, GPU, and Kubernetes deployments have not been validated for this release.

## Build and start

From the repository root, copy `.env.example` to `.env`, then set valid LLM credentials/model and a unique `RAG_API_TOKEN`.

```bash
docker compose build
# One worker keeps ingestion and queries on the same in-memory index:
docker compose run --rm --service-ports -e GUNICORN_WORKERS=1 rag-api
```

The image is `evalforge-ml:0.1.0`. Existing `rag-api`, `rag-network`, `rag-pipeline-api`, and `vector_store_data` identifiers remain for compatibility. The image runs as a non-root user. The embedding model may download at startup; startup time and resource needs depend on host hardware and model availability.

For the original Compose service lifecycle:

```bash
docker compose up --build
docker compose logs -f rag-api
docker compose down
```

The legacy entrypoint now defaults to one Gunicorn worker and rejects multiple workers without shared-snapshot mode. For opt-in shared persistence, configure `RAG_SHARED_SNAPSHOT_DIR` and a pinned `RAG_EMBEDDING_REVISION`; see [the exact contract](SNAPSHOT_CONSISTENCY.md).

## Persistence

Without shared-snapshot mode, the named volume mounts `/app/vector_store_data`. Persistence requires calling `RAGPipeline.save()` and `RAGPipeline.load()` explicitly in Python. API startup does not restore an index and ingestion does not automatically save it. A named volume does not synchronize workers. Never load untrusted index artifacts.

## Configuration and access

- Host API: `http://127.0.0.1:8000`; interactive schema: `/docs`; health: `/health`.
- Supply `OPENAI_API_KEY`, `LLM_MODEL`, and optionally `LLM_BASE_URL` through environment configuration.
- Keep authentication enabled and configure `RAG_API_TOKEN`, `TRUSTED_HOSTS`, and `CORS_ORIGINS` for the intended clients.
- Embedding/index environment entries in the legacy Compose file do not change dataclass settings. Customize `EmbeddingConfig` and `VectorStoreConfig` in Python; match embedding dimensions to the model.
- The health endpoint checks embeddings, vector storage, and LLM configuration/connectivity. It is not an answer-quality benchmark.

The build context excludes environment secrets, vector data, generated reports, caches, and local virtual environments. Supply secrets at runtime; do not bake credentials into images.

## Observability

See [observability](OBSERVABILITY.md) for the optional collector, Jaeger, Prometheus, and Grafana stack. Provision a Grafana password. Existing metric names, dashboard UID, and telemetry instrumentation identifiers remain for compatibility, while visible dashboard branding uses EvalForge.

## Before production

Validate dependency versions and model licensing, index lifecycle and concurrency, authentication/rate limits, secret management, network exposure, resource budgets, measured quality gates, backup/restore, and deployment monitoring. The [roadmap](../README.md#development-roadmap) tracks production work; existing Docker and CI files do not establish production readiness.

## Retrieval-only container

`Dockerfile.retrieval` provides the separate LLM-free app with two Uvicorn workers and nonroot UID 1000. It does not require provider credentials. The real MiniLM model must already be prepared; runtime loading is offline. Build snapshots on the host using the same model fingerprint. Mount a writable trusted snapshot directory and the prepared model cache:

```bash
docker build -f Dockerfile.retrieval -t evalforge-retrieval .
docker run --rm -p 127.0.0.1:8000:8000 \
  -e RAG_API_TOKEN \
  --mount type=bind,source="$PWD/artifacts/serving-new",target=/data/snapshots \
  --mount type=bind,source="$PWD/artifacts/model-cache",target=/models,readonly \
  evalforge-retrieval
```

Ensure UID 1000 can read the cache and write the snapshot directory. Real-model Docker inference has not been benchmarked here. For credential-free container mechanics, `python -m evaluation.docker_check` builds an owned image and uses `EVALFORGE_EMBEDDER=test-hash` solely for integration validation, removing only its own container/image/volume afterward. It tests concurrent ingestion/read visibility on both workers and restart with the same volume. [Audit](PERFORMANCE_AUDIT.md) distinguishes implemented CI from verified execution.
