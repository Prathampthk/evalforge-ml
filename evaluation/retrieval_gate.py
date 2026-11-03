"""Versioned deterministic fixture gate; no latency thresholds on shared CI workers."""

import argparse
import json
import math
from pathlib import Path


def conditions(report: dict) -> dict:
    return {"metric_version": report["metric_version"], "config": report["config"], "dataset": report["dataset"]}


def evaluate_gate(report: dict, baseline: dict) -> list[str]:
    if report.get("status") != "COMPLETE" or baseline.get("schema_version") != 1:
        raise ValueError("Incomplete report or unsupported baseline")
    if conditions(report) != baseline["conditions"]:
        raise ValueError("Baseline dataset/configuration/metric definitions differ")
    results = {r["id"]: r for r in report["results"]}
    if len(results) != len(report["results"]) or set(results) != set(baseline["expected"]):
        raise ValueError("Missing, duplicate or unexpected configurations")
    failures = []
    for name, expected in baseline["expected"].items():
        current = results[name]
        for key in ("corpus_count", "query_count", "scored_query_count", "corpus_ids_sha256", "query_ids_sha256"):
            if current[key] != expected[key]:
                raise ValueError(f"Benchmark conditions differ: {name}/{key}")
        if set(current["quality"]) != set(expected["quality"]):
            raise ValueError("Metric keys differ")
        for metric, target in expected["quality"].items():
            value = current["quality"][metric]
            if not math.isfinite(value) or value < target - 1e-12:
                failures.append(f"{name} {metric}: {value} < {target}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    failures = evaluate_gate(json.loads(args.report.read_text()), json.loads(args.baseline.read_text()))
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS: versioned deterministic retrieval quality baseline")


if __name__ == "__main__":
    main()
