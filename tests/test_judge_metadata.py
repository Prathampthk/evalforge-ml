from types import SimpleNamespace

import pytest

from evaluation.judge_metadata import judge_accounting, judge_identity, prompt_manifest
from evaluation.run_ragas import _extract_scores, _score_coverage


def test_judge_accounting_known_rates_and_missing_usage(monkeypatch):
    monkeypatch.setenv("JUDGE_INPUT_USD_PER_MILLION", "2")
    monkeypatch.setenv("JUDGE_OUTPUT_USD_PER_MILLION", "4")
    result = SimpleNamespace(total_tokens=lambda: SimpleNamespace(input_tokens=1000, output_tokens=500))
    accounting = judge_accounting(result)
    assert accounting["estimated_judge_usd"] == pytest.approx(0.004)
    result.total_tokens = lambda: SimpleNamespace(input_tokens=0, output_tokens=0)
    assert judge_accounting(result)["estimated_judge_usd"] is None
    assert judge_accounting(result)["provider_reported_input_tokens"] is None


def test_unknown_price_is_not_zero_cost(monkeypatch):
    monkeypatch.delenv("JUDGE_INPUT_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("JUDGE_OUTPUT_USD_PER_MILLION", raising=False)
    result = SimpleNamespace(total_tokens=lambda: SimpleNamespace(input_tokens=1000, output_tokens=500))
    assert judge_accounting(result)["estimated_judge_usd"] is None
    assert judge_identity()["false_positive_rate"] is None


def test_prompt_hashes_are_stable_and_scores_reject_infinity():
    prompt = SimpleNamespace(to_string=lambda: "static criteria")
    metric = SimpleNamespace(name="faithfulness", get_prompts=lambda: {"criteria": prompt})
    assert prompt_manifest([metric]) == prompt_manifest([metric])
    result = SimpleNamespace(_repr_dict={"a": float("inf"), "b": 0.9}, scores=[{"a": float("inf"), "b": 0.9}])
    assert _extract_scores(result) == {"a": None, "b": 0.9}
    assert _score_coverage(result) == {"a": 0, "b": 1}


def test_ragas_cache_column_mismatch_fails_before_judge(tmp_path):
    from evaluation.run_ragas import run_scoring_stage

    cache = {
        "questions": ["fixture"],
        "answers": [],
        "contexts": [[]],
        "references": ["reference"],
        "latencies_ms": [1],
        "consistency": [0.5],
    }
    with pytest.raises(ValueError, match="mismatched lengths"):
        run_scoring_stage(cache, tmp_path / "result.json")


def test_resume_rejects_stale_dataset_before_scoring(tmp_path):
    import json
    from evaluation.run_ragas import run

    corpus = tmp_path / "corpus.json"
    dataset = tmp_path / "queries.jsonl"
    corpus.write_text("changed corpus")
    dataset.write_text("changed query")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"provenance": {"corpus_sha256": "old", "dataset_sha256": "old"}}))
    with pytest.raises(ValueError, match="cache corpus/dataset mismatch"):
        run(corpus, dataset, tmp_path / "output.json", None, cache, resume=True)
