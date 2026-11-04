# Shared FAISS snapshot contract

`serving.snapshots.SnapshotRepository` implements serialized writes and immutable published versions on one POSIX local filesystem. This is index lifecycle management, not job orchestration.

## Publication and visibility

1. Embed incoming documents outside the cross-process writer lock.
2. Acquire `writer.lock` with `flock`; read the latest committed snapshot while holding it. Upsert by document ID, merging against the latest state so competing writers cannot overwrite unrelated additions.
3. Build a new exact FAISS index, ordered chunk map and float32 vectors in a temporary directory. Write a manifest with model fingerprint, dimension, count, parent version and SHA-256 for each data file.
4. Fsync files/directories, rename the complete version into `versions/<uuid>`, verify it, then atomically replace and fsync `CURRENT`.
5. Each worker checks `CURRENT` per request and verifies/loads a changed version before swapping its local reference. A request pins one version through lookup. A request already in flight may finish on the previous version; the response identifies its version. After a completed update, a new request observes that version or a later committed update.

An optional `expected_version` implements compare-and-swap and returns HTTP 409 for stale updates. Without it, ID upserts are serialized with last committed replacement winning for the same ID. Retries with identical IDs replace rows; this is not an exactly-once request protocol. Empty stores return HTTP 503 for retrieval until the first publication. Corrupt or incompatible active snapshots fail rather than serving a silently empty index.

## Validation

- Unit tests verify hash/model/dimension/count checks, ID replacement, immutable reader vectors, pinned prior versions, concurrent readers and two competing writer processes.
- A spawned writer exits abruptly immediately before pointer activation. Restart retains the previous committed version, and a later writer acquires the released OS lock and publishes successfully.
- Real HTTP tests use two Uvicorn processes and an explicitly synthetic hash encoder: 12 concurrent additions plus 40 reads during writes, all 13 IDs visible on both worker PIDs, consistent version/count pairs, stale-update rejection, stop/restart and restore on both new worker processes.
- `evaluation.docker_check` applies the same HTTP checks to an actual built container and named volume across `docker restart`. Synthetic embeddings test index/worker mechanics, not semantic quality or dense inference speed. See the CI evidence in [performance audit](PERFORMANCE_AUDIT.md).

## Tradeoffs and limits

Full-copy rebuild and verification are O(corpus size) per update. Every worker stores its own model/index; a writer can temporarily retain old/new stores. Updates block other writers, and refresh/build can delay reads on the same worker. There is no deletion API, batching scheduler, automatic compaction, snapshot garbage collector, or disk-capacity admission control. Old/orphan complete versions are retained intentionally; monitor disk and perform offline cleanup only when no readers need those versions. Hashes detect accidental corruption, not malicious artifacts from a writer who can rewrite the manifest. Keep the snapshot directory trusted.

The durability protocol covers process interruption and normal filesystem fsync/rename semantics. It is not a power-loss/hardware fault campaign, distributed consensus, cross-host replication, or a guarantee for NFS/object storage/Windows. No automatic fallback masks a corrupt CURRENT. Copy all files of an immutable version plus a consistent CURRENT for backup; do not edit published files.

## APIs and compatibility

The retrieval-only app uses `EVALFORGE_SNAPSHOT_DIR`, a fixed pinned MiniLM encoder by default, and `/documents` and `/retrieve` protected by `RAG_API_TOKEN`. It preserves one document unit per submitted ID, with the model's 256-token truncation; the benchmark joins SciFact title and abstract. `/health` and `/ping` are unauthenticated operational controls; restrict external access as needed. Ingestion is trusted-client use, not an unbounded public upload service.

The inherited `api:app` keeps its original contract. `start.sh` now defaults to **one worker** and rejects multiple legacy workers without `RAG_SHARED_SNAPSHOT_DIR`. To opt into shared persistence, set that directory and a 40-character `RAG_EMBEDDING_REVISION`; only flat indexes are supported. Its model identity includes normalization and dimension. Ingestion persists automatically and each query pins the latest version. Legacy `save()`/`load()` are disabled in this mode to protect published files. Without shared mode, legacy persistence remains explicit and in-memory indexes are not synchronized. Legacy chunk IDs/repeated-ingestion semantics are unchanged; use the retrieval API for explicit document-ID upserts. The two applications have distinct fingerprint/schema contracts and should use separate snapshot directories.
