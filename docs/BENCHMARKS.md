# Retrieval baseline measurements

Measured 2026-11-02 on macOS arm64, Python 3.14.2. This is a local lexical retrieval benchmark, not production RAG quality or an LLM benchmark.

## Method

The inherited sample contains **22 questions and 20 passages**. Fit L2-normalized unigram/bigram TF-IDF on corpus text only, create one stable-ID chunk per passage, and run the existing FAISS flat inner-product similarity search and MMR implementations. No generated scores, external credentials, model downloads, or relevance-label fitting are used.

Compare K=1, 3, 5; MMR fetch_k=10 and lambda=0.7; FAISS threads=1. Warm every query before timing. Each configuration times 10 repetitions per question (220 serial calls), for 1,320 timed calls total. Quality metrics use 22 queries per configuration, not 220 independent quality observations.

## Results

| Search | K | Recall@K | MRR@K | NDCG@K | Warm median (ms) | Warm p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| similarity | 1 | 0.9318 | 0.9545 | 0.9545 | 0.144 | 0.154 |
| similarity | 3 | 0.9545 | 0.9545 | 0.9509 | 0.149 | 0.166 |
| similarity | 5 | 0.9545 | 0.9545 | 0.9509 | 0.155 | 0.166 |
| mmr | 1 | 0.9318 | 0.9545 | 0.9545 | 0.159 | 0.166 |
| mmr | 3 | 0.9318 | 0.9545 | 0.9370 | 0.197 | 0.205 |
| mmr | 5 | 0.9318 | 0.9545 | 0.9370 | 0.230 | 0.244 |

Recall@K is the fraction of labeled relevant passages retrieved, averaged across questions. MRR@K is the mean reciprocal rank of the first relevant hit within K. NDCG@K uses binary relevance and logarithmic rank discount, normalized by the ideal ranking. Empty retrievals score zero; unjudged questions and duplicate result IDs are rejected.

Similarity search reached **95.45% Recall@5** and **0.9545 MRR@5** on this sample. MMR did not improve retrieval quality here. These are baseline results, not gains over an earlier implementation. No parameters were tuned to improve the observed numbers.

## Reproduce

From the project root, after installing `requirements-dev.txt`:

```bash
OMP_NUM_THREADS=1 python -m evaluation.retrieval_benchmark
OMP_NUM_THREADS=1 python -m pytest tests/ -q
```

The benchmark writes ignored local output to `evaluation/reports/retrieval-latest.json`. This document records the initial run; reruns may have different timings. [Raw initial results](benchmarks/retrieval-tfidf-baseline.json) retain every ranked passage ID, per-query score, input SHA-256 hash, dependency version, and host information.

## Validation and limits

- Added 12 tests for known ranking outcomes, cutoff behavior, missing hits, duplicate results, invalid judgments, malformed configuration, and a real corpus-to-index benchmark with thread-setting restoration.
- Full suite: **72 passed, 24 skipped**, with `OMP_NUM_THREADS=1`. API tests skip without credentials. Benchmark module statement coverage: **83%** in this run; this is not repository-wide coverage.
- Black, flake8, and mypy passed for the changes. Existing dependency deprecations and OpenTelemetry shutdown exporter errors remain.
- Corpus/query sample is small, in-domain, and not held out. Questions overlap corpus terminology. Scores do not establish generalization, answer faithfulness, or dense embedding performance.
- Timings include TF-IDF query vectorization and retrieval/MMR, but exclude index build, corpus encoding, network, and answer generation. They are serial warm measurements on a 20-passage index, not production latency or concurrent throughput.
- The next useful evidence is a larger held-out retrieval dataset and a dense-model comparison, followed by corpus-backed answer evaluation. Do not present this baseline as a production RAG speedup.
