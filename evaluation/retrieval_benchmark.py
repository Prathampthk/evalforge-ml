"""Corpus-backed retrieval benchmark; TF-IDF baseline, not LLM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ranking_metrics(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Binary relevance; duplicate results are invalid, empty judgments undefined."""
    if k < 1 or not relevant or len(ranked) != len(set(ranked)):
        raise ValueError("Require k >= 1, nonempty relevant IDs, and unique ranked IDs")
    hits = [int(item in relevant) for item in ranked[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {
        "recall": sum(hits) / len(relevant),
        "mrr": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
        "ndcg": dcg / ideal,
    }


def load_inputs(corpus: Path, dataset: Path) -> tuple[list[dict], list[dict]]:
    passages = json.loads(corpus.read_text(encoding="utf-8"))["passages"]
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not passages or not rows:
        raise ValueError("Corpus and dataset must be nonempty")
    ids = [p["id"] for p in passages]
    if len(ids) != len(set(ids)) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Corpus and query IDs must be unique")
    if any(not p["text"].strip() for p in passages):
        raise ValueError("Passage text must be nonempty")
    for row in rows:
        expected = row["expected_context_ids"]
        if not row["question"].strip() or not expected or not set(expected) <= set(ids):
            raise ValueError(f"Invalid query or unknown/empty relevance labels: {row['id']}")
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate relevance labels")
    return passages, rows


def run_benchmark(
    corpus: Path,
    dataset: Path,
    ks: tuple[int, ...] = (1, 3, 5),
    repeats: int = 10,
    fetch_k: int = 10,
    mmr_lambda: float = 0.7,
    threads: int = 1,
) -> dict[str, Any]:
    if not ks or any(k < 1 for k in ks) or repeats < 1 or threads < 1 or fetch_k < max(ks):
        raise ValueError("Require positive K, repeats and threads, with fetch_k >= max(K)")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("MMR lambda must be in [0, 1]")
    passages, rows = load_inputs(corpus, dataset)

    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    from config import VectorStoreConfig
    from models import Chunk
    from vector_store import VectorStore

    # Corpus-only fitting avoids using query text or relevance judgments in IDF.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32)
    started = time.perf_counter()
    vectors = vectorizer.fit_transform([p["text"] for p in passages]).toarray()
    store = VectorStore(VectorStoreConfig(index_type="flat"), dimension=vectors.shape[1])
    chunks = [
        Chunk(id=p["id"], document_id=p["id"], content=p["text"], source="retrieval_benchmark", chunk_index=0)
        for p in passages
    ]
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        store.add(chunks, vectors)
        build_ms = (time.perf_counter() - started) * 1000

        def retrieve(question: str, mode: str, k: int) -> list[str]:
            query = vectorizer.transform([question]).toarray()[0]
            if not np.any(query):
                return []
            if mode == "similarity":
                results = store.search(query, top_k=k, threshold=0.0)
            else:
                results = store.mmr_search(query, top_k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda, threshold=0.0)
            return [r.chunk.id for r in results]

        experiments = []
        for mode in ("similarity", "mmr"):
            for k in sorted(set(ks)):
                # Warm up every query; exclude warmup and build from timings.
                for row in rows:
                    retrieve(row["question"], mode, k)
                samples = []
                timings = []
                for row in rows:
                    ranked = retrieve(row["question"], mode, k)
                    scores = ranking_metrics(ranked, set(row["expected_context_ids"]), k)
                    samples.append({"query_id": row["id"], "ranked_ids": ranked, **scores})
                    for _ in range(repeats):
                        start = time.perf_counter()
                        retrieve(row["question"], mode, k)
                        timings.append((time.perf_counter() - start) * 1000)
                ordered = sorted(timings)
                experiments.append(
                    {
                        "mode": mode,
                        "k": k,
                        "metrics": {name: statistics.fmean(s[name] for s in samples) for name in scores},
                        "warm_query_latency_ms": {
                            "mean": statistics.fmean(timings),
                            "median": statistics.median(timings),
                            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
                            "timed_calls": len(timings),
                        },
                        "per_query": samples,
                    }
                )
    finally:
        faiss.omp_set_num_threads(previous_threads)
    return {
        "benchmark": "retrieval_tfidf_faiss",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(passages),
        "query_count": len(rows),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "config": {"ks": sorted(set(ks)), "repeats": repeats, "fetch_k": fetch_k, "mmr_lambda": mmr_lambda},
        "representation": "Corpus-fitted L2-normalized word unigram/bigram TF-IDF; one vector per passage",
        "index": "FAISS IndexFlatIP",
        "vector_dimension": vectors.shape[1],
        "build_ms": build_ms,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "faiss_threads": threads,
            "versions": {name: importlib.metadata.version(name) for name in ("faiss-cpu", "numpy", "scikit-learn")},
        },
        "limitations": [
            "Repository sample is small and in-domain; not a held-out external benchmark.",
            "TF-IDF lexical baseline; does not evaluate the production sentence-transformers embeddings or LLM.",
            "Latency includes query vectorization and search/MMR; excludes build, downloads, generation and network.",
            "Repeated serial warm queries are not concurrent throughput or independent quality observations.",
        ],
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/datasets/ragas_corpus.json"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/ragas_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/retrieval-latest.json"))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    report = run_benchmark(args.corpus, args.dataset, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for experiment in report["experiments"]:
        print(experiment["mode"], "K=", experiment["k"], experiment["metrics"], experiment["warm_query_latency_ms"])
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
