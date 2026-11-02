# EvalForge API v0.1.0

Start from the repository root after [installation](../README.md#installation):

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Startup constructs the pipeline and downloads the embedding model if needed. Set `OPENAI_API_KEY`, the provider model/endpoint, and `RAG_API_TOKEN` in `.env`. Interactive schemas are at `/docs`, `/redoc`, and `/openapi.json`.

## Routes

| Route | Behavior | Authentication |
|---|---|---|
| `GET /` | Project name, version, documentation and health links | Public |
| `GET /health` | Embedding, vector-store, and LLM checks; 503 when not healthy | Public |
| `POST /ingest` | Chunk and embed a JSON list of texts; 10 requests/minute | Bearer token |
| `POST /query` | Generate an answer from retrieved context; 30 requests/minute | Bearer token |

There is no HTTP file-upload route. Load UTF-8 files through `DocumentLoader.load_file` in Python or send their contents to `/ingest`.

## Ingestion

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"texts":["EvalForge evaluates RAG pipelines."],"source":"example","metadata":{"type":"demo"}}'
```

Export the bearer token in your shell before using curl. The `texts` list accepts 1–1000 entries and cannot be entirely blank. `source` defaults to `api`; `metadata` is optional. The response includes `success`, `chunks_ingested`, `source`, and `message`.

## Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does EvalForge evaluate?","top_k":10,"enable_guardrail":false,"sync_evaluation":true,"stream":false}'
```

`query` is required (1–2000 characters). `top_k` defaults to 10 (1–100). Guardrail, synchronous evaluation, and streaming default to false. Responses include the answer, retrieved chunks, consistency/reliability fields, timing, evaluation status, and generation warnings.

With `sync_evaluation=false`, consistency/reliability fields are placeholders while evaluation runs in the background. No polling endpoint exposes the completed background result. Use `sync_evaluation=true` when inspecting scores.

`stream=true` returns newline-delimited JSON (`application/x-ndjson`) after the pipeline completes: one `result` record followed by `chunk` records. This is not token-by-token LLM streaming. Errors during streaming may be emitted as JSON error records.

## Authentication and errors

Keep `RAG_AUTH_DISABLED=false`. Missing/invalid bearer credentials return 401; an unconfigured server token returns a service error. Request/schema validation can return 400 or 422, rate limiting 429, and pipeline failures 500. Host restrictions can reject unknown hosts. See [configuration](CONFIGURATION.md) for `TRUSTED_HOSTS` and `CORS_ORIGINS`.

## Operational limits

Indexes are process-local. Startup does not restore persisted data, and ingestion does not automatically save it. Use one worker for local ingestion/query consistency. The API is a development foundation; no validated production deployment or quality benchmark is claimed. See [deployment](DEPLOYMENT.md) and [release audit](RELEASE_AUDIT.md).
