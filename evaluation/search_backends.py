"""Small sparse/FAISS adapters for controlled experiments; no pipeline replacement."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation.experiment_config import ExperimentConfig, RetrievalSpec
from telemetry import start_span


def finite_vectors(vectors, rows: int):
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] < 1:
        raise ValueError("Embedding dimensions/count mismatch")
    values = vectors.data if sparse.issparse(vectors) else vectors
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite embedding values")


def encode(corpus: list[str], queries: list[str], cfg: ExperimentConfig, dense_model=None):
    started = time.perf_counter()
    if dense_model is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32)
        docs = vectorizer.fit_transform(corpus)
        corpus_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        questions = vectorizer.transform(queries)
    else:
        docs = dense_model.encode(
            corpus, batch_size=cfg.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        corpus_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        questions = dense_model.encode(
            queries,
            batch_size=cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    query_ms = (time.perf_counter() - started) * 1000
    finite_vectors(docs, len(corpus))
    finite_vectors(questions, len(queries))
    if docs.shape[1] != questions.shape[1]:
        raise ValueError("Query embedding dimension mismatch")
    return (
        docs,
        questions,
        {
            "corpus_encoding_ms": corpus_ms,
            "query_encoding_ms": query_ms,
            "dimension": docs.shape[1],
            "query_batch_size": cfg.batch_size,
        },
    )


class SearchIndex:
    def __init__(self, vectors, ids: list[str], spec: RetrievalSpec, seed: int, path: Path):
        finite_vectors(vectors, len(ids))
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate index document IDs")
        self.vectors, self.ids, self.spec = vectors, ids, spec
        rss_before = psutil.Process().memory_info().rss
        started = time.perf_counter()
        self.index: Any = None
        if spec.index != "sparse":
            d = vectors.shape[1]
            if spec.index == "flat":
                self.index = faiss.IndexFlatIP(d)
            elif spec.index == "hnsw":
                self.index = faiss.IndexHNSWFlat(d, spec.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.rng = getattr(faiss, "RandomGenerator")(seed)
                self.index.hnsw.efConstruction = spec.ef_construction
                self.index.hnsw.efSearch = spec.ef_search
            else:
                if len(ids) < 39 * spec.nlist:
                    raise ValueError("IVF needs at least 39 training documents per centroid in this runner")
                self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, spec.nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.cp.seed = seed
                self.index.nprobe = spec.nprobe
                self.index.train(vectors)
            self.index.add(vectors)
        build_ms = (time.perf_counter() - started) * 1000
        rss_after = psutil.Process().memory_info().rss
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            sparse.save_npz(str(path.with_suffix(".npz")), vectors)
            path = path.with_suffix(".npz")
            vector_bytes = vectors.data.nbytes + vectors.indices.nbytes + vectors.indptr.nbytes
            serialized_bytes = None
        else:
            faiss.write_index(self.index, str(path))
            vector_bytes = vectors.nbytes
            serialized_bytes = faiss.serialize_index(self.index).nbytes
        self.path = path
        self.performance = {
            "index_build_ms": build_ms,
            "artifact_bytes": path.stat().st_size,
            "serialized_index_memory_estimate_bytes": serialized_bytes,
            "retained_vector_bytes": vector_bytes,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "process_rss_delta_bytes": rss_after - rss_before,
            "memory_scope": (
                "RSS delta includes allocator noise; " "serialized FAISS size is a native-memory proxy, not RSS"
            ),
        }

    def lookup(self, query, count: int) -> list[int]:
        count = min(count, len(self.ids))
        if query.shape[-1] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimension mismatch")
        if self.index is None:
            if query.nnz == 0:
                return []
            scores = (self.vectors @ query.T).toarray().ravel()
            return np.lexsort((np.array(self.ids), -scores))[:count].tolist()
        scores, positions = self.index.search(np.asarray(query).reshape(1, -1), min(count + 1, len(self.ids)))
        # Expand exact ties at the K boundary to make document-ID tie breaks independent of FAISS.
        if self.spec.index == "flat" and len(scores[0]) > count and scores[0][count] == scores[0][count - 1]:
            scores, positions = self.index.search(np.asarray(query).reshape(1, -1), len(self.ids))
        candidates = [(float(s), int(i)) for s, i in zip(scores[0], positions[0]) if i >= 0]
        return [i for _, i in sorted(candidates, key=lambda pair: (-pair[0], self.ids[pair[1]]))[:count]]

    def retrieve(self, query, experiment_id: str, mode: str | None = None) -> list[str]:
        spec = self.spec
        mode = mode or spec.mode
        attrs = {"experiment.id": experiment_id, "configuration": spec.name, "top_k": spec.k}
        with start_span("evalforge.index_lookup", attrs, "evalforge.experiments"):
            candidate_ids = self.lookup(query, spec.fetch_k if mode == "mmr" else spec.k)
        if mode == "mmr" and candidate_ids:
            with start_span("evalforge.rerank_mmr", attrs, "evalforge.experiments"):
                candidates = self.vectors[candidate_ids]
                if sparse.issparse(candidates):
                    relevance = (candidates @ query.T).toarray().ravel()
                    similarity = (candidates @ candidates.T).toarray()
                else:
                    relevance = (candidates @ query.reshape(-1)).ravel()
                    similarity = candidates @ candidates.T
                chosen: list[int] = []
                remaining = list(range(len(candidate_ids)))
                while remaining and len(chosen) < spec.k:

                    def score(i):
                        redundancy = max(similarity[i, chosen]) if chosen else 0.0
                        return spec.mmr_lambda * relevance[i] - (1 - spec.mmr_lambda) * redundancy

                    best = min(remaining, key=lambda i: (-score(i), self.ids[candidate_ids[i]]))
                    chosen.append(best)
                    remaining.remove(best)
                candidate_ids = [candidate_ids[i] for i in chosen]
        return [self.ids[i] for i in candidate_ids[: spec.k]]
