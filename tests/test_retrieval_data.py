import json
from pathlib import Path

import pytest

from evaluation.retrieval_data import load_dataset, sha256

FIXTURE = Path("evaluation/datasets/regression-v1/manifest.json")


def test_regression_identity_and_multi_relevance():
    data = load_dataset(FIXTURE)
    assert data.identity()["query_count"] == 22
    assert data.identity()["corpus_count"] == 20
    assert data.identity()["qrel_count"] == 23
    assert len(data.qrels["ragas-022"]) == 2


@pytest.fixture
def dataset(tmp_path):
    for p in FIXTURE.parent.iterdir():
        (tmp_path / p.name).write_bytes(p.read_bytes())
    return tmp_path / "manifest.json"


def mutate_data(manifest, kind, content):
    meta = json.loads(manifest.read_text())
    path = manifest.parent / meta["files"][kind]["path"]
    path.write_text(content)
    meta["files"][kind]["sha256"] = sha256(path)
    meta.pop("qrel_count", None)
    meta.pop("query_count", None)
    manifest.write_text(json.dumps(meta))


def test_checksum_failure(dataset):
    (dataset.parent / "corpus.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="checksum/version"):
        load_dataset(dataset)


@pytest.mark.parametrize(
    "kind,content",
    [
        ("corpus", '{"_id":"a","text":"one"}\n{"_id":"a","text":"two"}\n'),
        ("queries", '{"text":"query"}\n'),
        ("qrels", "query-id\tcorpus-id\tscore\nmissing\tctx-mmr\t1\n"),
        ("qrels", "query-id\tcorpus-id\tscore\nragas-001\tmissing\t1\n"),
        ("qrels", "query-id\tcorpus-id\tscore\nragas-001\tctx-mmr\t1.5\n"),
        ("qrels", "query-id\tcorpus-id\tscore\nragas-001\tctx-mmr\t-1\n"),
        ("qrels", "query-id\tcorpus-id\tscore\nragas-001\tctx-mmr\t1\nragas-001\tctx-mmr\t1\n"),
    ],
)
def test_malformed_data_rejected(dataset, kind, content):
    mutate_data(dataset, kind, content)
    with pytest.raises(ValueError):
        load_dataset(dataset)


def test_zero_positive_judgment_retained(dataset):
    mutate_data(dataset, "qrels", "query-id\tcorpus-id\tscore\nragas-001\tctx-mmr\t0\n")
    assert load_dataset(dataset).identity()["queries_without_positives"] == 1


def test_dataset_count_mismatch(dataset):
    meta = json.loads(dataset.read_text())
    meta["query_count"] = 999
    dataset.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="count/version"):
        load_dataset(dataset)


def test_test_split_cannot_be_development(dataset):
    meta = json.loads(dataset.read_text())
    meta.update(split="test", role="development")
    dataset.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Test split"):
        load_dataset(dataset)
