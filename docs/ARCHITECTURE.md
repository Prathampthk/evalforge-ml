<!-- generated-by: gsd-doc-writer -->
# EvalForge v0.1.0 architecture

## System Overview

The system is a layered Python RAG pipeline that accepts a user query plus ingested documents and returns an answer with retrieval traces, guardrail output, and factual consistency evaluation.

## Component Diagram

```mermaid
graph TD
  U[User Query] --> P[RAGPipeline main.py]
  P --> R[Retriever retriever.py]
  R --> E[EmbeddingModel embeddings.py]
  R --> V[VectorStore vector_store.py]
  P --> G[Optional GuardrailAgent guardrail_agent.py]
  P --> N[Generator generator.py]
  P --> X[Sync or deferred EvaluatorAgent evaluator_agent.py]
  G --> L[LLMClient llm_client.py]
  N --> L
  X --> L
  P --> M[Models models.py]
  P --> D[DocumentLoader document_loader.py]
```

## Data Flow

1. Documents are ingested via `RAGPipeline.ingest` or `RAGPipeline.ingest_documents`, chunked by `DocumentLoader` using configurable token-aware strategies, embedded by `EmbeddingModel`, then stored in `VectorStore`.
2. A query enters `RAGPipeline.query`, which calls `Retriever.retrieve`.
3. Retriever embeds the query and executes FAISS search (`search` or `mmr_search`) in `VectorStore`.
4. Retrieved chunks are threshold-filtered and optionally passed through `GuardrailAgent.evaluate`.
5. `Generator.generate` produces a grounded response using selected context chunks.
6. `EvaluatorAgent.evaluate` runs in sync or deferred mode and scores factual consistency claim-by-claim.
7. `PipelineResult` returns answer, reliability score, stage outputs, and total timing.

## Key Abstractions

- `RAGPipeline` in `main.py`: end-to-end orchestrator and public API.
- `PipelineConfig` and nested config dataclasses in `config.py`: centralized runtime, model, retrieval, and chunking controls.
- `LLMClient` in `llm_client.py`: OpenAI-compatible chat client with retry and JSON mode helpers.
- `EmbeddingModel` in `embeddings.py`: sentence-transformers wrapper with dimension validation.
- `VectorStore` in `vector_store.py`: FAISS index management, search/MMR, persistence.
- `Retriever` in `retriever.py`: embed-query and ranking orchestration.
- `GuardrailAgent` in `guardrail_agent.py`: relevance and safety filtering using JSON LLM output.
- `Generator` in `generator.py`: grounded answer generation with context token budgeting.
- `chunking.py`: token-aware semantic, token-window, and legacy character chunk strategies plus chunk quality metrics.
- `EvaluatorAgent` in `evaluator_agent.py`: factual consistency decomposition and scoring.
- `PipelineResult` and related pydantic models in `models.py`: typed contracts between stages.

## Directory Structure Rationale

```text
.
|- main.py                # Pipeline orchestration entry point
|- config.py              # All runtime/config dataclasses
|- models.py              # Shared typed data contracts
|- llm_client.py          # LLM transport and retries
|- embeddings.py          # Embedding model wrapper
|- vector_store.py        # FAISS storage and retrieval
|- document_loader.py     # Loading and ingestion-facing chunking API
|- chunking.py            # Token-aware chunking strategies and metrics
|- retriever.py           # Query retrieval path
|- guardrail_agent.py     # Relevance and safety stage
|- generator.py           # Answer generation stage
|- evaluator_agent.py     # Consistency evaluation stage
|- demo.py                # Interactive demo runner
|- tests/                 # Unit/runtime behavior tests
|- docs/                  # Project documentation
```

This structure keeps each pipeline concern isolated to a module while `main.py` composes those modules into a single query interface.

## Runtime boundaries

The guardrail is optional and disabled by default. Deferred evaluation returns pending fields without a result-polling endpoint; select sync mode to consume completed scores. Without shared mode, legacy FAISS indexes are process-local, save/load is explicit, and API startup does not restore a persisted index. The new `serving.api:app` and opt-in legacy shared mode publish immutable, hash-verified versions, refresh CURRENT per request, and restore automatically; [snapshot contract](SNAPSHOT_CONSISTENCY.md) defines visibility, locking and restart behavior. The README diagram describes implemented stages only; the roadmap is separate.

## Retrieval experiment path (unreleased upgrade)

```mermaid
flowchart TD
  C[Strict JSON or YAML config] --> D[Hashed BEIR-compatible dataset manifest]
  D --> V[Validate IDs, split roles and graded qrels]
  V --> S[Real-document corpus subsets]
  S --> E[Corpus-only TF-IDF or pinned CPU MiniLM]
  E --> I[Sparse exact or FAISS flat/HNSW/IVF]
  I --> R[Lookup and optional MMR]
  R --> M[Versioned ranking metrics per query]
  M --> B[Paired query bootstrap]
  I --> P[Build, artifact, RSS and warm query measurements]
  B --> A[JSON artifacts and comparison report]
  P --> A
  A --> G[Versioned deterministic fixture gate]
```

This path does not call an LLM or replace `RAGPipeline`. Dataset text and qrels stay separate from trace metadata. Experiment IDs link privacy-safe stage spans to file artifacts. See [experiments](EXPERIMENTS.md), [datasets](DATASETS.md), and [measured results](SCIFACT_RESULTS.md).
