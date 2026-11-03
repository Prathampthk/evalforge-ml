import json
from pathlib import Path

import faiss
import numpy as np
import pytest
from pydantic import ValidationError
from scipy import sparse

from evaluation.experiment import execute, subset_ids
from evaluation.experiment_config import ExperimentConfig, RetrievalSpec, load_config
from evaluation.retrieval_data import load_dataset
from evaluation.search_backends import SearchIndex, finite_vectors

MANIFEST = Path("evaluation/datasets/regression-v1/manifest.json")


def small_config():
    return ExperimentConfig(
        dataset=str(MANIFEST),
        repeats=1,
        bootstrap_samples=100,
        configurations=[
            RetrievalSpec(name="lexical", representation="tfidf", index="sparse", k=5),
            RetrievalSpec(name="lexical-mmr", representation="tfidf", index="sparse", mode="mmr", k=5),
        ],
    )


def test_json_configuration_and_unknown_fields(tmp_path):
    config = load_config(Path("configs/experiments/regression.json"))
    assert len(config.configurations) == 6
    payload = config.model_dump()
    payload["unknown"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        {"name": "invalid", "representation": "tfidf", "index": "flat"},
        {"name": "invalid", "representation": "dense", "index": "flat", "k": 10, "fetch_k": 5},
        {"name": "invalid", "representation": "dense", "index": "ivf", "nprobe": 32, "nlist": 4},
    ],
)
def test_invalid_search_combinations(override):
    with pytest.raises(ValidationError):
        RetrievalSpec.model_validate(override)


def test_dense_model_requires_pinned_revision():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            dataset="x",
            model_revision="main",
            configurations=[RetrievalSpec(name="dense", representation="dense", index="flat")],
        )


def test_subset_sizes_and_positive_coverage():
    data = load_dataset(MANIFEST)
    with pytest.raises(ValueError, match="too small"):
        subset_ids(data, 1, 42)
    with pytest.raises(ValueError, match="exceeds"):
        subset_ids(data, 999, 42)
    assert subset_ids(data, 0, 42) == sorted(data.corpus)


def test_sparse_and_exact_ties_empty_query_dimension(tmp_path):
    ids = ["z", "a", "b"]
    matrix = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    spec = RetrievalSpec(name="exact", representation="dense", index="flat", k=1)
    index = SearchIndex(matrix, ids, spec, 42, tmp_path / "exact.faiss")
    assert index.retrieve(np.array([1, 0], dtype=np.float32), "test") == ["a"]
    index.index = faiss.read_index(str(index.path))
    assert index.retrieve(np.array([1, 0], dtype=np.float32), "test") == ["a"]
    assert index.performance["artifact_bytes"] > 0
    with pytest.raises(ValueError, match="dimension"):
        index.lookup(np.ones(3), 1)
    lexical = SearchIndex(
        sparse.csr_matrix(matrix),
        ids,
        RetrievalSpec(name="sparse", representation="tfidf", index="sparse", k=1),
        42,
        tmp_path / "sparse.faiss",
    )
    assert lexical.retrieve(sparse.csr_matrix([[1, 0]], dtype=np.float32), "test") == ["a"]
    assert lexical.retrieve(sparse.csr_matrix((1, 2)), "test") == []


def test_mmr_diversity_and_lambda_configuration(tmp_path):
    vectors = np.array([[1, 0], [0.999, 0.01], [0.5, 0.866]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0]
    spec = RetrievalSpec(name="mmr", representation="dense", index="flat", mode="mmr", k=2, fetch_k=3, mmr_lambda=0.1)
    index = SearchIndex(vectors, ["a", "b", "c"], spec, 42, tmp_path / "mmr.faiss")
    assert index.retrieve(query, "test") == ["a", "c"]
    assert index.retrieve(query, "test", mode="similarity") == ["a", "b"]


@pytest.mark.parametrize("matrix,rows", [(np.ones((2, 3)), 3), (np.array([[np.nan]]), 1)])
def test_invalid_embeddings(matrix, rows):
    with pytest.raises(ValueError):
        finite_vectors(matrix, rows)


def test_reproducible_artifacts_and_query_pairing(tmp_path):
    data = load_dataset(MANIFEST)
    first = execute(small_config(), data, tmp_path / "one", source={"fixture": True})
    second = execute(small_config(), data, tmp_path / "two", source={"fixture": True})
    assert first["experiment_id"] == second["experiment_id"]
    assert first["run_id"] != second["run_id"]
    assert first["status"] == second["status"] == "COMPLETE"
    assert first["comparisons"][0]["query_count"] == 22
    for a, b in zip(first["results"], second["results"]):
        assert a["quality"] == b["quality"]
        assert a["performance"]["calls"] == 22
        assert a["performance"]["p99_ms"] >= a["performance"]["p50_ms"]
        rows = json.loads((tmp_path / "one" / a["artifacts"]["per_query"]).read_text())
        assert len(rows) == 22
    with pytest.raises(ValueError, match="empty"):
        execute(small_config(), data, tmp_path / "one")


def test_failed_run_is_recorded_and_threads_restored(tmp_path, monkeypatch):
    import evaluation.experiment as module

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive query should not appear in failure metadata")

    monkeypatch.setattr(module, "encode", fail)
    previous = faiss.omp_get_max_threads()
    with pytest.raises(RuntimeError):
        execute(small_config(), load_dataset(MANIFEST), tmp_path / "failed")
    report = json.loads((tmp_path / "failed" / "report.json").read_text())
    assert report["status"] == "FAILED"
    assert report["error_type"] == "RuntimeError"
    assert "sensitive query" not in json.dumps(report)
    assert faiss.omp_get_max_threads() == previous


@pytest.mark.parametrize("kind", ["flat", "hnsw", "ivf"])
def test_dense_index_loading_and_exact_agreement(kind, tmp_path):
    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(128, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = [f"d{i:03d}" for i in range(len(vectors))]
        spec = RetrievalSpec(name="index", representation="dense", index=kind, k=5, nlist=2, nprobe=2, ef_search=128)
        index = SearchIndex(vectors, ids, spec, 42, tmp_path / "index.faiss")
        expected = [ids[i] for i in np.argsort(-(vectors @ vectors[17]))[:5]]
        assert index.retrieve(vectors[17], "test") == expected
        index.index = faiss.read_index(str(index.path))
        assert index.retrieve(vectors[17], "test") == expected
    finally:
        faiss.omp_set_num_threads(old)
