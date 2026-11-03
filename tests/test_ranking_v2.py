import itertools
import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from evaluation.ranking import evaluate_ranking, rank_scores


def test_hand_calculated_graded_metrics():
    result = evaluate_ranking(["wrong", "b", "a"], {"a": 3, "b": 1}, 3)
    assert result is not None
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["mrr"] == 0.5
    assert result["hit_rate"] == 1
    assert result["ndcg"] == pytest.approx((1 / math.log2(3) + 7 / 2) / (7 + 1 / math.log2(3)))


def test_precision_fixed_k_larger_than_corpus():
    assert evaluate_ranking(["a"], {"a": 1}, 10)["precision"] == 0.1


def test_empty_retrieval_and_no_positives():
    assert all(v == 0 for v in evaluate_ranking([], {"a": 1}, 2).values())
    assert evaluate_ranking(["a"], {"a": 0}, 2) is None
    assert all(v == 0 for v in evaluate_ranking(["a"], {"a": 0}, 2, "zero").values())


@pytest.mark.parametrize(
    "ranked,qrels,k",
    [
        (["a", "a"], {"a": 1}, 1),
        ([], {}, 1),
        ([], {"a": -1}, 1),
        ([], {"a": 0.5}, 1),
        ([], {"a": True}, 1),
        ([], {"a": 1}, 0),
    ],
)
def test_invalid_metric_inputs(ranked, qrels, k):
    with pytest.raises(ValueError):
        evaluate_ranking(ranked, qrels, k)


def test_ties_document_id_and_nonfinite_rejection():
    assert rank_scores({"z": 1, "a": 1, "b": 2}) == ["b", "a", "z"]
    with pytest.raises(ValueError):
        rank_scores({"a": float("nan")})


def test_all_permutations_against_independent_sklearn_ndcg():
    # sklearn uses linear gains: pass exp2 gains to independently verify our chosen convention.
    qrels = {"a": 3, "b": 1, "c": 0, "d": 2}
    ids = list(qrels)
    for ranking in itertools.permutations(ids):
        prediction = np.array([[len(ids) - ranking.index(doc) for doc in ids]])
        truth = np.array([[2 ** qrels[doc] - 1 for doc in ids]])
        previous_recall = 0.0
        for k in range(1, 6):
            scores = evaluate_ranking(ranking, qrels, k)
            assert scores["ndcg"] == pytest.approx(ndcg_score(truth, prediction, k=k))
            assert scores["recall"] >= previous_recall
            assert all(0 <= value <= 1 for value in scores.values())
            previous_recall = scores["recall"]
