"""Validate ranking math, judgment integrity, and corpus-backed benchmark output."""

import json
import math
from pathlib import Path

import pytest

from evaluation.retrieval_benchmark import load_inputs, ranking_metrics, run_benchmark


def test_perfect_ranking():
    assert ranking_metrics(["a", "b"], {"a", "b"}, 2) == {"recall": 1, "mrr": 1, "ndcg": 1}


def test_partial_relevance_at_rank_two():
    result = ranking_metrics(["x", "a"], {"a", "b"}, 2)
    assert result["recall"] == 0.5
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_cutoff_excludes_late_hit():
    assert ranking_metrics(["x", "a"], {"a"}, 1) == {"recall": 0, "mrr": 0, "ndcg": 0}


def test_empty_retrieval_scores_zero():
    assert ranking_metrics([], {"a"}, 5) == {"recall": 0, "mrr": 0, "ndcg": 0}


def test_duplicate_results_rejected():
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a"}, 2)


def test_unjudged_query_rejected():
    with pytest.raises(ValueError):
        ranking_metrics(["a"], set(), 1)


def test_invalid_cutoff_rejected():
    with pytest.raises(ValueError):
        ranking_metrics(["a"], {"a"}, 0)


def inputs(tmp_path, expected=None):
    corpus = tmp_path / "corpus.json"
    dataset = tmp_path / "queries.jsonl"
    corpus.write_text(json.dumps({"passages": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]}))
    dataset.write_text(json.dumps({"id": "q", "question": "alpha", "expected_context_ids": expected or ["a"]}))
    return corpus, dataset


def test_unknown_judgment_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        load_inputs(*inputs(tmp_path, ["missing"]))


def test_duplicate_judgments_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        load_inputs(*inputs(tmp_path, ["a", "a"]))


def test_empty_corpus_rejected(tmp_path):
    corpus, dataset = inputs(tmp_path)
    corpus.write_text('{"passages": []}')
    with pytest.raises(ValueError, match="nonempty"):
        load_inputs(corpus, dataset)


def test_benchmark_retrieves_labeled_passage_and_restores_threads(tmp_path):
    import faiss

    previous = faiss.omp_get_max_threads()
    report = run_benchmark(*inputs(tmp_path), ks=(1,), repeats=2, fetch_k=2)
    assert faiss.omp_get_max_threads() == previous
    assert report["query_count"] == 1
    assert len(report["experiments"]) == 2
    for experiment in report["experiments"]:
        assert experiment["per_query"][0]["ranked_ids"] == ["a"]
        assert experiment["metrics"] == {"recall": 1, "mrr": 1, "ndcg": 1}
        assert experiment["warm_query_latency_ms"]["timed_calls"] == 2
    json.dumps(report, allow_nan=False)


def test_invalid_benchmark_config_rejected_before_io():
    with pytest.raises(ValueError):
        run_benchmark(Path("missing"), Path("missing"), ks=(5,), fetch_k=2)
