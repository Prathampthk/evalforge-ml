from copy import deepcopy

import pytest

from evaluation.retrieval_gate import evaluate_gate
from evaluation.statistics import paired_bootstrap


def test_paired_bootstrap_identity_constant_difference_and_seed():
    left = {"a": 0.1, "b": 0.3, "c": 0.6}
    assert paired_bootstrap(left, left, 500, 42)["ci95"] == [0.0, 0.0]
    right = {k: v + 0.1 for k, v in left.items()}
    constant = paired_bootstrap(left, right, 500, 42)
    assert constant["ci95"] == pytest.approx([0.1, 0.1])
    right["b"] = 0.9
    first = paired_bootstrap(left, right, 1000, 42)
    assert first == paired_bootstrap(dict(reversed(list(left.items()))), right, 1000, 42)
    assert first["ci95"][0] <= first["difference_right_minus_left"] <= first["ci95"][1]


def test_bootstrap_requires_paired_query_ids():
    with pytest.raises(ValueError):
        paired_bootstrap({"a": 1}, {"b": 1}, 100, 42)
    with pytest.raises(ValueError):
        paired_bootstrap({"a": float("nan")}, {"a": 1}, 100, 42)


def gate_fixture():
    row = {
        "id": "x",
        "corpus_count": 2,
        "query_count": 2,
        "scored_query_count": 2,
        "corpus_ids_sha256": "c",
        "query_ids_sha256": "q",
        "quality": {"recall": 0.5},
    }
    identity = {"metric_version": "v2", "config": {"k": 5}, "dataset": {"version": "v1"}}
    report = {"status": "COMPLETE", **identity, "results": [row]}
    baseline = {"schema_version": 1, "conditions": identity, "expected": {"x": deepcopy(row)}}
    return report, baseline


def test_quality_gate_fails_actual_regression_and_nan():
    report, baseline = gate_fixture()
    assert evaluate_gate(report, baseline) == []
    report["results"][0]["quality"]["recall"] = 0.4
    assert len(evaluate_gate(report, baseline)) == 1
    report["results"][0]["quality"]["recall"] = float("nan")
    assert len(evaluate_gate(report, baseline)) == 1


@pytest.mark.parametrize("change", ["version", "missing", "queries", "incomplete"])
def test_gate_rejects_incomparable_results(change):
    report, baseline = gate_fixture()
    if change == "version":
        report["dataset"]["version"] = "v2"
        # baseline identity is copied to avoid mutating both fixtures.
        baseline["conditions"] = deepcopy(baseline["conditions"])
        baseline["conditions"]["dataset"]["version"] = "v1"
    elif change == "missing":
        report["results"] = []
    elif change == "queries":
        report["results"][0]["query_ids_sha256"] = "other"
    else:
        report["status"] = "FAILED"
    with pytest.raises(ValueError):
        evaluate_gate(report, baseline)
