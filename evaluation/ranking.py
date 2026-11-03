"""Versioned ranking semantics. See docs/EXPERIMENTS.md for exact conventions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

METRIC_VERSION = "ranking-v2-exp2-fixed-k"
METRICS = ("recall", "precision", "mrr", "ndcg", "hit_rate")


def rank_scores(scores: Mapping[str, float]) -> list[str]:
    """Descending finite score, then ascending document ID; no tie averaging."""
    if any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("Ranking scores must be finite")
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def evaluate_ranking(
    ranked: Sequence[str], qrels: Mapping[str, int], k: int, no_positive: str = "exclude"
) -> dict[str, float] | None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("K must be a positive integer")
    if no_positive not in {"exclude", "zero"}:
        raise ValueError("Unknown no-positive policy")
    if not qrels:
        raise ValueError("Missing relevance judgments")
    if any(isinstance(g, bool) or not isinstance(g, int) or not 0 <= g <= 30 for g in qrels.values()):
        raise ValueError("Judgment grades must be integers in [0, 30]")
    if len(ranked) != len(set(ranked)):
        raise ValueError("Duplicate retrieved document IDs")
    positive = sum(g > 0 for g in qrels.values())
    if not positive:
        return None if no_positive == "exclude" else dict.fromkeys(METRICS, 0.0)
    grades = [qrels.get(doc, 0) for doc in ranked[:k]]
    hits = [g > 0 for g in grades]
    dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return {
        "recall": sum(hits) / positive,
        "precision": sum(hits) / k,
        "mrr": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
        "ndcg": dcg / idcg,
        "hit_rate": float(any(hits)),
    }
