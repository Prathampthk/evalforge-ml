"""LLM-free retrieval API with versioned indexes and measured request stages."""

import asyncio
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from config import TelemetryConfig
from models import Chunk
from serving.embedding import Encoder
from serving.snapshots import SnapshotRepository, VersionConflict
from telemetry import configure_observability, start_span


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=10, ge=1, le=100)


class InputDocument(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=100000)


class UpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[InputDocument] = Field(min_length=1, max_length=1000)
    expected_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_observability(TelemetryConfig())
    encoder = Encoder(
        os.getenv("EVALFORGE_EMBEDDER", "minilm"), os.getenv("EVALFORGE_MODEL_CACHE", "artifacts/model-cache")
    )
    app.state.encoder = encoder
    app.state.snapshots = SnapshotRepository(
        Path(os.getenv("EVALFORGE_SNAPSHOT_DIR", "vector_store_data/shared")), encoder.identity
    )
    app.state.snapshots.latest()  # fail startup if the active snapshot is incompatible/corrupt
    yield


app = FastAPI(title="EvalForge Retrieval API", version="0.2.0-dev", lifespan=lifespan)


@app.middleware("http")
async def server_timing(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-EvalForge-Server-Ms"] = str((time.perf_counter() - started) * 1000)
    return response


async def authorize(authorization: str | None = Header(default=None)):
    token = os.getenv("RAG_API_TOKEN")
    if not token:
        raise HTTPException(503, "RAG_API_TOKEN must be configured")
    if authorization is None or not secrets.compare_digest(authorization, "Bearer " + token):
        raise HTTPException(401, "Invalid bearer token")


@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/health")
def health():
    try:
        current = app.state.snapshots.latest()
    except (ValueError, OSError) as exc:
        raise HTTPException(503, "Snapshot unavailable") from exc
    return {
        "status": "ready",
        "version": current.version if current else None,
        "count": current.manifest["count"] if current else 0,
        "pid": os.getpid(),
        "encoder": app.state.encoder.identity,
        "native_runtime": app.state.encoder.native_runtime,
    }


def retrieve_sync(payload: RetrieveRequest) -> dict:
    started = time.perf_counter()
    faiss.omp_set_num_threads(1)  # explicitly set in each request thread
    with start_span("serving.snapshot_refresh", tracer_name="evalforge.serving"):
        snapshot = app.state.snapshots.latest()
    if snapshot is None:
        raise HTTPException(503, "No snapshot has been published")
    prepared = payload.query.strip()
    if not prepared:
        raise HTTPException(422, "Query must contain text")
    before_lock = time.perf_counter()
    with app.state.encoder.lock:
        embedding_started = time.perf_counter()
        with start_span("serving.embedding", {"snapshot.version": snapshot.version}, "evalforge.serving"):
            query = app.state.encoder.encode([prepared])[0]
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
    wait_ms = (embedding_started - before_lock) * 1000
    lookup_started = time.perf_counter()
    with start_span("serving.faiss_lookup", {"snapshot.version": snapshot.version}, "evalforge.serving"):
        matches = snapshot.store.search(query, top_k=payload.k, threshold=0.0)
    lookup_ms = (time.perf_counter() - lookup_started) * 1000
    results = [{"id": match.chunk.id, "score": match.similarity_score} for match in matches]
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "snapshot_version": snapshot.version,
        "pid": os.getpid(),
        "results": results,
        "timings_ms": {
            "embedding": embedding_ms,
            "embedding_wait": wait_ms,
            "faiss_lookup": lookup_ms,
            "processing": total_ms - embedding_ms - wait_ms - lookup_ms,
            "handler_total": total_ms,
        },
    }


@app.post("/retrieve", dependencies=[Depends(authorize)])
async def retrieve(payload: RetrieveRequest):
    try:
        return await asyncio.to_thread(retrieve_sync, payload)
    except (ValueError, OSError) as exc:
        raise HTTPException(503, "Snapshot or encoder unavailable") from exc


def upsert_sync(payload: UpsertRequest) -> dict:
    with app.state.encoder.lock:
        vectors = app.state.encoder.encode([doc.text for doc in payload.documents])
    chunks = [Chunk(id=d.id, document_id=d.id, content=d.text, source="api", chunk_index=0) for d in payload.documents]
    faiss.omp_set_num_threads(1)
    snapshot = app.state.snapshots.upsert(chunks, vectors, payload.expected_version)
    return {"snapshot_version": snapshot.version, "count": snapshot.manifest["count"], "pid": os.getpid()}


@app.post("/documents", dependencies=[Depends(authorize)])
async def documents(payload: UpsertRequest):
    try:
        return await asyncio.to_thread(upsert_sync, payload)
    except VersionConflict as exc:
        raise HTTPException(409, "Snapshot version conflict") from exc
    except ValueError as exc:
        raise HTTPException(422, "Invalid documents or incompatible embeddings") from exc
