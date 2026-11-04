"""Configuration-driven, corpus-backed retrieval experiments and paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess  # nosec B404 - fixed read-only Git metadata commands only
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import psutil

from config import TelemetryConfig
from evaluation.experiment_config import ExperimentConfig, load_config
from evaluation.ranking import METRICS, METRIC_VERSION, evaluate_ranking
from evaluation.retrieval_data import RetrievalDataset, load_dataset, sha256
from evaluation.search_backends import SearchIndex, encode
from evaluation.statistics import paired_bootstrap
from telemetry import configure_observability, start_span


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def subset_ids(data: RetrievalDataset, size: int, seed: int) -> list[str]:
    """Nested real-document subsets retaining all judged positives; never duplicate docs."""
    if not size or size == len(data.corpus):
        return sorted(data.corpus)
    if size > len(data.corpus):
        raise ValueError("Requested corpus size exceeds real corpus")
    positives = {doc for q in data.qrels.values() for doc, grade in q.items() if grade > 0}
    if size < len(positives):
        raise ValueError("Corpus subset too small to retain all positive judgments")
    distractors = sorted(set(data.corpus) - positives)
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(distractors).tolist()[: size - len(positives)]
    return sorted(positives | set(chosen))


def provenance() -> dict:
    paths = sorted(
        set(Path("evaluation").glob("*.py"))
        | set(Path("serving").glob("*.py"))
        | {Path("telemetry.py"), Path("config.py"), Path("main.py"), Path("embeddings.py")}
    )
    sources = {str(p): sha256(p) for p in paths}
    try:
        git = shutil.which("git")
        if git is None:
            raise OSError("Git executable unavailable")
        # No shell or user-controlled command arguments; read-only provenance.
        commit = subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip()  # nosec B603
        dirty = bool(subprocess.check_output([git, "status", "--porcelain"], text=True).strip())  # nosec B603
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    return {"commit": commit, "working_tree_dirty": dirty, "source_sha256": sources}


def latency_summary(timings: list[float]) -> dict:
    if not timings:
        raise ValueError("No query timings")
    return {
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.quantile(timings, 0.5)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "p99_ms": float(np.quantile(timings, 0.99)),
        "serial_queries_per_second": 1000 * len(timings) / sum(timings),
        "calls": len(timings),
        "scope": "pre-encoded query index lookup plus optional MMR, tracing and Python overhead; warm serial calls",
    }


def execute(cfg: ExperimentConfig, data: RetrievalDataset, output: Path, source: dict | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; experiments are never silently overwritten")
    output.mkdir(parents=True, exist_ok=True)
    code = source or provenance()
    identity = {"config": cfg.model_dump(), "dataset": data.identity(), "code": code, "metrics": METRIC_VERSION}
    experiment_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": f"{experiment_id}-{uuid.uuid4().hex[:8]}",
        "started_at": timestamp(),
        "status": "RUNNING",
        "metric_version": METRIC_VERSION,
        "dataset": data.identity(),
        "config": cfg.model_dump(),
        "code": code,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "device": "cpu",
            "threads": cfg.threads,
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "faiss-cpu", "sentence-transformers", "torch", "psutil")
            },
        },
        "results": [],
        "comparisons": [],
        "timing_trials_are_quality_samples": False,
    }
    # Preserve the exact dirty-tree implementation used by this run, not just a commit label.
    for filename, digest in code.get("source_sha256", {}).items():
        path = Path(filename)
        if sha256(path) != digest:
            raise ValueError("Source changed while taking experiment snapshot")
        snapshot = output / "source" / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    report["source_snapshot"] = "source/" if code.get("source_sha256") else None
    write_json(output / "report.json", report)
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(cfg.threads)
    previous_torch_threads = None
    attrs = {"experiment.id": experiment_id, "run.id": report["run_id"], "dataset.id": data.metadata["id"]}
    try:
        with start_span("evalforge.experiment", attrs, "evalforge.experiments"):
            dense_model = None
            if any(s.representation == "dense" for s in cfg.configurations):
                import torch
                from sentence_transformers import SentenceTransformer

                torch.manual_seed(cfg.seed)
                previous_torch_threads = torch.get_num_threads()
                torch.set_num_threads(cfg.threads)
                with start_span("evalforge.model_load", attrs, "evalforge.experiments"):
                    dense_model = SentenceTransformer(
                        cfg.model_name,
                        revision=cfg.model_revision,
                        device="cpu",
                        cache_folder=cfg.model_cache,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    dense_model.max_seq_length = cfg.max_seq_length
            for size in sorted(set(len(data.corpus) if n == 0 else n for n in cfg.corpus_sizes)):
                doc_ids = subset_ids(data, size, cfg.seed)
                qids = sorted(data.queries)
                with start_span("evalforge.query_preprocessing", attrs, "evalforge.experiments"):
                    queries = [data.queries[q] for q in qids]
                    texts = [data.corpus[d] for d in doc_ids]
                representations = {}
                for representation in sorted({s.representation for s in cfg.configurations}):
                    with start_span(
                        "evalforge.embedding_generation",
                        {**attrs, "representation": representation},
                        "evalforge.experiments",
                    ):
                        representations[representation] = encode(
                            texts, queries, cfg, dense_model if representation == "dense" else None
                        )
                for spec in cfg.configurations:
                    print(f"{size} documents: {spec.name}", flush=True)
                    vectors, query_vectors, encoding = representations[spec.representation]
                    result_id = f"{spec.name}-n{size}"
                    index_path = output / "indexes" / (result_id + ".faiss")
                    with start_span(
                        "evalforge.index_build", {**attrs, "configuration": spec.name}, "evalforge.experiments"
                    ):
                        index = SearchIndex(vectors, doc_ids, spec, cfg.seed, index_path)
                    # Check saved artifacts can be loaded with matching dimension/count.
                    if index.index is not None:
                        loaded = faiss.read_index(str(index.path))
                        if loaded.d != vectors.shape[1] or loaded.ntotal != size:
                            raise ValueError("Persisted index dimension/count mismatch")
                        index.index = loaded
                    reference = None
                    if spec.index in {"ivf", "hnsw"}:
                        reference = faiss.IndexFlatIP(vectors.shape[1])
                        reference.add(vectors)
                    rows: list[dict[str, Any]] = []
                    timings: list[float] = []
                    ann_agreement: list[float] = []
                    for i, qid in enumerate(qids):
                        query = query_vectors[i]
                        for _ in range(cfg.warmup):
                            index.retrieve(
                                query, experiment_id, exclude_id=qid if data.metadata.get("exclude_query_id") else None
                            )
                        ranked = index.retrieve(
                            query, experiment_id, exclude_id=qid if data.metadata.get("exclude_query_id") else None
                        )
                        # Runtime repetitions are stored separately from one quality observation/query.
                        for trial in range(cfg.repeats):
                            started = time.perf_counter()
                            repeated = index.retrieve(
                                query, experiment_id, exclude_id=qid if data.metadata.get("exclude_query_id") else None
                            )
                            elapsed = (time.perf_counter() - started) * 1000
                            if repeated != ranked:
                                raise ValueError("Non-deterministic ranking during timing repetitions")
                            timings.append(elapsed)
                        with start_span("evalforge.evaluation_scoring", attrs, "evalforge.experiments"):
                            query_metrics = evaluate_ranking(ranked, data.qrels[qid], spec.k, cfg.no_positive)
                        row: dict[str, Any] = {
                            "query_id": qid,
                            "ranked_ids": ranked,
                            "metrics": query_metrics,
                            "latency_ms_trials": timings[-cfg.repeats :],
                        }
                        rows.append(row)
                        if reference is not None:
                            exclude = qid if data.metadata.get("exclude_query_id") else None
                            _, exact = reference.search(
                                query.reshape(1, -1), min(spec.k + int(exclude is not None), size)
                            )
                            reference_ids = {doc_ids[int(j)] for j in exact[0] if j >= 0 and doc_ids[int(j)] != exclude}
                            # Preserve rank before truncation when removing identical query/document IDs.
                            reference_ids = set(
                                [doc_ids[int(j)] for j in exact[0] if j >= 0 and doc_ids[int(j)] != exclude][: spec.k]
                            )
                            approximate = set(
                                index.retrieve(query, experiment_id, mode="similarity", exclude_id=exclude)
                            )
                            ann_agreement.append(len(reference_ids & approximate) / len(reference_ids))
                    valid = [row for row in rows if row["metrics"] is not None]
                    if not valid:
                        raise ValueError("No evaluable queries under no-positive policy")
                    metric_means = {m: float(np.mean([r["metrics"][m] for r in valid])) for m in METRICS}
                    per_query_path = output / "queries" / (result_id + ".json")
                    write_json(per_query_path, rows)
                    result = {
                        "id": result_id,
                        "configuration": spec.model_dump(),
                        "corpus_count": size,
                        "query_count": len(qids),
                        "scored_query_count": len(valid),
                        "qrel_count": sum(map(len, data.qrels.values())),
                        "corpus_ids_sha256": hashlib.sha256(json.dumps(doc_ids).encode()).hexdigest(),
                        "query_ids_sha256": hashlib.sha256(json.dumps(qids).encode()).hexdigest(),
                        "subset_policy": (
                            "all positives plus seeded nested real distractors; full corpus when size=full"
                        ),
                        "chunking": "one document = title plus text, no overlapping chunks",
                        "preprocessing": (
                            "trim edges, join title/text; TF-IDF lowercase default token pattern, unigram/bigram"
                        ),
                        "similarity": "L2-normalized vectors, inner product/cosine",
                        "model": (
                            {"id": cfg.model_name, "revision": cfg.model_revision, "max_seq_length": cfg.max_seq_length}
                            if spec.representation == "dense"
                            else None
                        ),
                        "quality": metric_means,
                        "encoding": encoding,
                        "performance": {**index.performance, **latency_summary(timings)},
                        "ann_recall_at_k_vs_exact": float(np.mean(ann_agreement)) if ann_agreement else None,
                        "ann_scope": "raw index candidates before MMR, same representation/subset/query/K",
                        "artifacts": {
                            "index": str(index.path.relative_to(output)),
                            "per_query": str(per_query_path.relative_to(output)),
                        },
                    }
                    report["results"].append(result)
                    write_json(output / "report.json", report)
                same_size = [r for r in report["results"] if r["corpus_count"] == size]
                # Predeclared first configuration is reference, not a test-set-selected winner.
                pairs = []
                for right in same_size:
                    left = next(r for r in same_size if r["configuration"]["k"] == right["configuration"]["k"])
                    if left["id"] != right["id"]:
                        pairs.append((left, right))
                    if right["configuration"]["representation"] == "dense":
                        exact = next(
                            (
                                r
                                for r in same_size
                                if r["configuration"]["representation"] == "dense"
                                and r["configuration"]["index"] == "flat"
                                and r["configuration"]["mode"] == "similarity"
                                and r["configuration"]["k"] == right["configuration"]["k"]
                            ),
                            None,
                        )
                        if exact and exact["id"] not in {left["id"], right["id"]}:
                            pairs.append((exact, right))
                for left, right in pairs:
                    left_rows = json.loads((output / left["artifacts"]["per_query"]).read_text())
                    right_rows = json.loads((output / right["artifacts"]["per_query"]).read_text())
                    for metric in METRICS:
                        lvalues = {r["query_id"]: r["metrics"][metric] for r in left_rows if r["metrics"] is not None}
                        rvalues = {r["query_id"]: r["metrics"][metric] for r in right_rows if r["metrics"] is not None}
                        report["comparisons"].append(
                            {
                                "left": left["id"],
                                "right": right["id"],
                                "metric": metric,
                                **paired_bootstrap(lvalues, rvalues, cfg.bootstrap_samples, cfg.seed),
                            }
                        )
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__  # no raw queries, credentials or documents in failure metadata
        raise
    finally:
        faiss.omp_set_num_threads(previous_threads)
        if previous_torch_threads is not None:
            torch.set_num_threads(previous_torch_threads)
        report["completed_at"] = timestamp()
        write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-exporter", choices=["none", "console", "otlp"], default="none")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = load_dataset(args.config.parent / cfg.dataset)
    configure_observability(TelemetryConfig(telemetry_exporter=args.trace_exporter, structured_logs_enabled=False))
    report = execute(cfg, data, args.output)
    print(report["experiment_id"], report["status"], args.output / "report.json")


if __name__ == "__main__":
    main()
