"""POSIX local-filesystem snapshots with serialized writers and atomic activation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import VectorStoreConfig
from models import Chunk
from vector_store import VectorStore


class VersionConflict(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Snapshot:
    version: str
    manifest: dict
    store: VectorStore
    chunks: tuple[Chunk, ...]
    vectors: np.ndarray


class SnapshotRepository:
    """Read visibility is pinned at CURRENT read; requests in flight can finish on old versions.

    No published index is mutated. Writers merge the latest snapshot under a process lock.
    Old versions are retained for readers and recovery; no automatic garbage collection.
    This contract requires atomic rename/advisory locks on one POSIX local filesystem.
    """

    def __init__(self, root: Path, model: dict):
        self.root = root
        self.model = model
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)
        self._mutex = threading.RLock()
        self._cached: Snapshot | None = None

    @contextmanager
    def writer(self):
        import fcntl  # POSIX-only serving contract; not imported by legacy clients on Windows

        with self._mutex, (self.root / "writer.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def current_version(self) -> str | None:
        try:
            version = (self.root / "CURRENT").read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        if not re.fullmatch(r"[a-f0-9]{32}", version):
            raise ValueError("Invalid snapshot pointer")
        return version

    def latest(self) -> Snapshot | None:
        with self._mutex:
            version = self.current_version()
            if version is None:
                return None
            if self._cached is not None and self._cached.version == version:
                return self._cached
            loaded = self.load(version)
            self._cached = loaded  # one atomic local swap after all files verify
            return loaded

    def load(self, version: str) -> Snapshot:
        if not re.fullmatch(r"[a-f0-9]{32}", version):
            raise ValueError("Invalid snapshot version")
        directory = self.root / "versions" / version
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["version"] != version or manifest["model"] != self.model:
            raise ValueError("Snapshot version/model mismatch")
        if set(manifest["sha256"]) != {"index.faiss", "chunks.json", "embeddings.npy"}:
            raise ValueError("Invalid snapshot file inventory")
        for name, checksum in manifest["sha256"].items():
            if digest(directory / name) != checksum:
                raise ValueError("Snapshot integrity failure")
        payload = json.loads((directory / "chunks.json").read_text())
        chunks = tuple(Chunk.model_validate(c) for c in payload["chunks"])
        vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
        if vectors.shape != (len(chunks), manifest["dimension"]) or not np.all(np.isfinite(vectors)):
            raise ValueError("Snapshot vectors/count mismatch")
        if len({c.id for c in chunks}) != len(chunks) or len(chunks) != manifest["count"]:
            raise ValueError("Snapshot duplicate IDs/count mismatch")
        store = VectorStore(VectorStoreConfig(persist_dir=str(self.root / "versions")), manifest["dimension"])
        store.load(version)
        if store.size != len(chunks):
            raise ValueError("Snapshot index/count mismatch")
        vectors.setflags(write=False)
        return Snapshot(version, manifest, store, chunks, vectors)

    def _activate(self, version: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=self.root, prefix=".pointer-", delete=False) as handle:
            name = Path(handle.name)
            handle.write(version + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(name, self.root / "CURRENT")
            fsync_directory(self.root)
        finally:
            name.unlink(missing_ok=True)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, expected_version: str | None = None) -> Snapshot:
        if not chunks or len({c.id for c in chunks}) != len(chunks):
            raise ValueError("Nonempty unique input document IDs required")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] < 1:
            raise ValueError("Embedding dimension/count mismatch")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Nonfinite embeddings")
        with self.writer():
            current = self.latest()
            actual_version = current.version if current else None
            if expected_version is not None and expected_version != actual_version:
                raise VersionConflict("Expected snapshot is no longer current")
            if current and vectors.shape[1] != current.manifest["dimension"]:
                raise ValueError("Embedding dimension mismatch")
            merged = {c.id: (c, v) for c, v in zip(current.chunks, current.vectors)} if current else {}
            merged.update({c.id: (c, v) for c, v in zip(chunks, vectors)})
            ordered = [merged[key] for key in sorted(merged)]
            all_chunks = [c for c, _ in ordered]
            all_vectors = np.asarray([v for _, v in ordered], dtype=np.float32)
            version = uuid.uuid4().hex
            # Build and fsync complete files off the active path. Failure cannot replace CURRENT.
            with tempfile.TemporaryDirectory(dir=self.root, prefix=".build-") as scratch:
                store = VectorStore(VectorStoreConfig(persist_dir=scratch), all_vectors.shape[1])
                store.add(all_chunks, all_vectors)
                store.save(version)
                directory = Path(scratch) / version
                manifest = {
                    "schema_version": 1,
                    "version": version,
                    "parent": actual_version,
                    "model": self.model,
                    "dimension": all_vectors.shape[1],
                    "count": len(all_chunks),
                    "sha256": {
                        name: digest(directory / name) for name in ("index.faiss", "chunks.json", "embeddings.npy")
                    },
                }
                (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
                for file in directory.iterdir():
                    with file.open("rb") as handle:
                        os.fsync(handle.fileno())
                fsync_directory(directory)
                os.replace(directory, self.root / "versions" / version)
                fsync_directory(self.root / "versions")
            # Verify the completed version before making it visible to any worker.
            loaded = self.load(version)
            self._activate(version)
            self._cached = loaded
            return loaded
