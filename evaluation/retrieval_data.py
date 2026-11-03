"""Strict, hashed BEIR-compatible corpus/query/qrels manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path, title: bool = False) -> dict[str, str]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier, text = row.get("_id"), row.get("text")
        if not isinstance(identifier, str) or not identifier.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid identifier/text in {path.name}:{number}")
        if identifier in result:
            raise ValueError(f"Duplicate identifier in {path.name}:{number}")
        heading = row.get("title", "")
        if not isinstance(heading, str):
            raise ValueError("Title must be a string")
        result[identifier] = f"{heading.strip()} {text.strip()}".strip() if title else text.strip()
    if not result:
        raise ValueError(f"Empty data file: {path.name}")
    return result


@dataclass(frozen=True)
class RetrievalDataset:
    metadata: dict
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def identity(self) -> dict:
        return {
            **self.metadata,
            "corpus_count": len(self.corpus),
            "query_count": len(self.queries),
            "qrel_count": sum(map(len, self.qrels.values())),
            "positive_qrel_count": sum(g > 0 for judgments in self.qrels.values() for g in judgments.values()),
            "queries_without_positives": sum(not any(g > 0 for g in q.values()) for q in self.qrels.values()),
        }


def load_dataset(manifest_path: Path) -> RetrievalDataset:
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "id", "version", "split", "role", "origin", "license", "files"}
    if required - set(meta) or meta["schema_version"] != 1:
        raise ValueError("Invalid dataset manifest/schema version")
    if any(not isinstance(meta[key], str) or not meta[key].strip() for key in required - {"files", "schema_version"}):
        raise ValueError("Dataset identity fields must be nonempty strings")
    if meta["role"] not in {"benchmark", "development", "regression"}:
        raise ValueError("Invalid dataset role")
    if meta["split"] == "train" and meta["role"] != "development":
        raise ValueError("Training split cannot be a benchmark")
    if meta["split"] == "test" and meta["role"] != "benchmark":
        raise ValueError("Test split must be marked benchmark")
    if set(meta["files"]) != {"corpus", "queries", "qrels"}:
        raise ValueError("Manifest must specify corpus, queries and qrels")
    paths = {}
    for kind, spec in meta["files"].items():
        path = manifest_path.parent / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Dataset checksum/version mismatch: {kind}")
        paths[kind] = path
    corpus = _records(paths["corpus"], title=True)
    queries = _records(paths["queries"])
    qrels: dict[str, dict[str, int]] = {}
    with paths["qrels"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Invalid qrels header")
        for row in reader:
            qid, doc = row["query-id"], row["corpus-id"]
            if qid not in queries or doc not in corpus:
                raise ValueError("Qrel references missing query or document ID")
            try:
                grade = int(row["score"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid relevance grade") from exc
            if str(grade) != row["score"] or not 0 <= grade <= 30:
                raise ValueError("Relevance grade must be a canonical integer in [0, 30]")
            if doc in qrels.setdefault(qid, {}):
                raise ValueError("Duplicate relevance judgment")
            qrels[qid][doc] = grade
    if not qrels:
        raise ValueError("Empty qrels")
    # BEIR queries.jsonl can contain multiple splits; only qrels-listed queries are evaluated.
    selected = {qid: queries[qid] for qid in sorted(qrels)}
    for field, actual in (
        ("corpus_count", len(corpus)),
        ("query_count", len(selected)),
        ("qrel_count", sum(map(len, qrels.values()))),
    ):
        if field in meta and meta[field] != actual:
            raise ValueError(f"Dataset count/version mismatch: {field}")
    return RetrievalDataset(meta, dict(sorted(corpus.items())), selected, qrels)
