from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import os

import faiss
import numpy as np
import pytest

from models import Chunk
from serving.snapshots import SnapshotRepository, VersionConflict

MODEL = {"name": "fixture", "revision": "v1"}


def chunk(name):
    return Chunk(id=name, document_id=name, content=name, chunk_index=0)


def publish_process(root, name, vector):
    faiss.omp_set_num_threads(1)
    SnapshotRepository(Path(root), MODEL).upsert([chunk(name)], np.asarray([vector], dtype=np.float32))


def crash_before_activation(root):
    faiss.omp_set_num_threads(1)
    repo = SnapshotRepository(Path(root), MODEL)
    repo._activate = lambda version: os._exit(23)
    repo.upsert([chunk("unpublished")], np.array([[0, 1]], dtype=np.float32))


def test_writer_process_crash_keeps_committed_version(tmp_path):
    old = SnapshotRepository(tmp_path, MODEL).upsert([chunk("a")], np.array([[1, 0]], dtype=np.float32))
    process = get_context("spawn").Process(target=crash_before_activation, args=(str(tmp_path),))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 23
    restored = SnapshotRepository(tmp_path, MODEL)
    assert restored.latest().version == old.version
    assert {c.id for c in restored.latest().chunks} == {"a"}
    # The OS released the writer lock, so a subsequent update can commit normally.
    assert restored.upsert([chunk("b")], np.array([[0, 1]], dtype=np.float32)).manifest["count"] == 2


def test_refresh_restart_and_pinned_reader(tmp_path):
    a = SnapshotRepository(tmp_path, MODEL)
    b = SnapshotRepository(tmp_path, MODEL)
    old = a.upsert([chunk("a")], np.array([[1, 0]], dtype=np.float32))
    assert b.latest().version == old.version
    new = a.upsert([chunk("b")], np.array([[0, 1]], dtype=np.float32), expected_version=old.version)
    assert old.store.size == 1
    assert b.latest().version == new.version
    assert SnapshotRepository(tmp_path, MODEL).latest().store.size == 2
    assert b.latest().store.search(np.array([0, 1], dtype=np.float32), 1)[0].chunk.id == "b"
    with pytest.raises(VersionConflict):
        b.upsert([chunk("c")], np.array([[1, 0]], dtype=np.float32), old.version)


def test_competing_processes_do_not_lose_ingestion(tmp_path):
    context = get_context("spawn")
    processes = [
        context.Process(target=publish_process, args=(str(tmp_path), name, vector))
        for name, vector in [("a", [1, 0]), ("b", [0, 1])]
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    current = SnapshotRepository(tmp_path, MODEL).latest()
    assert {c.id for c in current.chunks} == {"a", "b"}


def test_failed_activation_preserves_current(tmp_path, monkeypatch):
    repo = SnapshotRepository(tmp_path, MODEL)
    old = repo.upsert([chunk("a")], np.array([[1, 0]], dtype=np.float32))

    def fail(version):
        raise OSError("simulated interrupted activation")

    monkeypatch.setattr(repo, "_activate", fail)
    with pytest.raises(OSError):
        repo.upsert([chunk("b")], np.array([[0, 1]], dtype=np.float32))
    assert SnapshotRepository(tmp_path, MODEL).latest().version == old.version


def test_integrity_dimension_model_and_duplicates(tmp_path):
    repo = SnapshotRepository(tmp_path, MODEL)
    current = repo.upsert([chunk("a")], np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        repo.upsert([chunk("b")], np.ones((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        repo.upsert([chunk("b"), chunk("b")], np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="model"):
        SnapshotRepository(tmp_path, {"name": "other"}).latest()
    path = tmp_path / "versions" / current.version / "embeddings.npy"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        SnapshotRepository(tmp_path, MODEL).latest()


def test_replace_document_and_simultaneous_readers(tmp_path):
    repo = SnapshotRepository(tmp_path, MODEL)
    repo.upsert([chunk("a")], np.array([[1, 0]], dtype=np.float32))
    with ThreadPoolExecutor(max_workers=4) as pool:
        readers = list(pool.map(lambda _: repo.latest(), range(10)))
    repo.upsert([chunk("a")], np.array([[0, 1]], dtype=np.float32))
    assert repo.latest().manifest["count"] == 1
    assert all(np.array_equal(r.vectors, np.array([[1, 0]])) for r in readers)
