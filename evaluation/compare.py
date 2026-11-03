"""Render a machine-readable experiment report as quality/performance tables."""

import argparse
import json
from pathlib import Path


def render(report: dict) -> str:
    if report.get("status") != "COMPLETE":
        raise ValueError("Only complete experiments can be compared")
    lines = [
        "# Retrieval experiment comparison",
        "",
        f"Experiment `{report['experiment_id']}`; dataset `{report['dataset']['id']}` "
        f"version `{report['dataset']['version']}`, split `{report['dataset']['split']}`.",
        "",
        "Quality scores and runtime measurements are separate. All paired differences use the same query IDs/K/corpus.",
        "",
        "## Retrieval relevance",
        "",
        "| Configuration | Documents | Queries | K | Recall | Precision | MRR | NDCG | Hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        scores = row["quality"]
        values = " | ".join(f"{scores[m]:.4f}" for m in ("recall", "precision", "mrr", "ndcg", "hit_rate"))
        lines.append(
            f"| {row['configuration']['name']} | {row['corpus_count']} | {row['scored_query_count']} "
            f"| {row['configuration']['k']} | {values} |"
        )
    lines += [
        "",
        "## Index performance (pre-encoded, warm serial queries)",
        "",
        "| Configuration | Documents | Build ms | Artifact MiB | P50 ms | P95 ms | P99 ms | Serial QPS | ANN recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        p = row["performance"]
        ann = "N/A" if row["ann_recall_at_k_vs_exact"] is None else f"{row['ann_recall_at_k_vs_exact']:.4f}"
        lines.append(
            f"| {row['configuration']['name']} | {row['corpus_count']} | {p['index_build_ms']:.2f} "
            f"| {p['artifact_bytes']/2**20:.2f} | {p['p50_ms']:.3f} | {p['p95_ms']:.3f} "
            f"| {p['p99_ms']:.3f} | {p['serial_queries_per_second']:.1f} | {ann} |"
        )
    lines += [
        "",
        "## Paired exploratory Recall/NDCG differences",
        "",
        "Differences are right minus the named predeclared lexical or exact-dense reference at each corpus size. "
        "95% percentile bootstrap resamples queries, not timing repetitions. "
        "Intervals are unadjusted for multiple comparisons and do not by themselves prove superiority.",
        "",
        "| Left | Right | Metric | Difference | 95% CI | Queries | Resamples |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for c in report["comparisons"]:
        if c["metric"] in {"recall", "ndcg"}:
            lines.append(
                f"| {c['left']} | {c['right']} | {c['metric']} "
                f"| {c['difference_right_minus_left']:.4f} | [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] "
                f"| {c['query_count']} | {c['resamples']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = render(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
