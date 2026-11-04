# Reproducible retrieval experiments

EvalForge studies retrieval relevance, ranking metrics, dataset integrity, index tradeoffs, and observability. It does not implement AgentEval's asynchronous job execution/regression service architecture. Existing RAG serving and RAGAS functionality remain separate from retrieval-only experiments.

## Setup and run

Use Python 3.11 for parity with CI/Docker. Local measured runs used macOS arm64, Python 3.14.2, 10 logical CPUs and 16 GiB RAM; full hardware/dependency details are in each report.

```bash
python -m pip install -r requirements-experiments.txt
# Small, credential-free deterministic regression (no downloads):
OMP_NUM_THREADS=1 python -m evaluation.experiment \
  --config configs/experiments/regression.json --output evaluation/reports/regression-new
python -m evaluation.retrieval_gate \
  --report evaluation/reports/regression-new/report.json \
  --baseline evaluation/baselines/regression-v1.json

# Public dataset and a small CPU embedding model; downloads are explicit:
python -m evaluation.prepare_dataset --output data/scifact
python -m evaluation.prepare_model
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m evaluation.experiment \
  --config configs/experiments/scifact.json --output evaluation/reports/scifact-new
python -m evaluation.compare \
  --report evaluation/reports/scifact-new/report.json \
  --output evaluation/reports/scifact-new/comparison.md
```

Choose a new empty output directory for each run; the runner refuses overwrite. JSON and YAML configurations are supported. Dataset paths are relative to the config file; `model_cache` and output paths are relative to the working directory (run from repository root). Schema and unknown-field validation are strict. `corpus_sizes: [0]` selects the complete corpus. Negative/oversized subsets, invalid index combinations, unpinned dense revisions, or inadequate IVF training sizes fail rather than silently falling back.

The local model download is approximately 87 MiB including tokenizer/config files. It is pinned to `sentence-transformers/all-MiniLM-L6-v2` commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0 according to its [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). The experiment runner uses offline-only loading, CPU, no remote model code, 384-dimensional vectors, normalized embeddings, batch size 32, and max sequence length 256. It never substitutes fake embeddings when the model is missing.

## Configurations

The regression config preserves all six TF-IDF similarity/MMR combinations at K=1,3,5. The legacy runner still supplies the original binary metric implementation for reproducibility.

The SciFact config compares six methods at K=10, across three corpus sizes (18 rows):

| Name | Representation / lookup | Reranking and parameters |
|---|---|---|
| tfidf-exact | Corpus-fitted word unigram/bigram TF-IDF, sparse cosine search | No feature cap; scikit-learn lowercase/token-pattern defaults, L2 normalization |
| tfidf-mmr | Same sparse TF-IDF | fetch=50, lambda=0.7 |
| dense-flat | Pinned MiniLM, FAISS IndexFlatIP | Exact normalized inner product |
| dense-mmr | Same MiniLM/flat | fetch=50, lambda=0.7 |
| dense-hnsw | Same MiniLM, HNSW IP | M=16, efConstruction=100, efSearch=32 |
| dense-ivf | Same MiniLM, IVF-flat IP | nlist=16, nprobe=4; fixed training seed |

FAISS and PyTorch use one thread, seed 42. HNSW RNG and IVF clustering seed are set. IVF requires >=39 corpus documents per centroid to avoid undersized clustering experiments. Ranking repeatability is checked during timing calls. This is one predeclared seed, not a multi-seed robustness study. BM25 and hybrid retrieval remain future additions; the credible lexical baseline here is TF-IDF.

## Metric contract

New metric version: `ranking-v2-exp2-fixed-k`. Relevance grade `g(q,d)` is a nonnegative integer. Let `R(q)` be the number of documents with grade >0 and `h_i=1` when the document at rank i has grade >0. Missing returned ranks contribute zero.

- **Recall@K** = sum(h_i, i<=K) / R(q).
- **Precision@K** = sum(h_i, i<=K) / K, even when fewer than K documents exist or return.
- **MRR@K** = 1 / first relevant rank within K, or 0 when none return.
- **Hit Rate@K** = 1 if any relevant document appears within K, else 0.
- **DCG@K** = sum((2^g_i - 1) / log2(i+1), i=1..K); **NDCG@K** = DCG / ideal DCG from the same judgments sorted by grade.

Report macro averages over queries. Missing judgments fail; unjudged retrieved documents count as nonrelevant. Explicit zero-positive queries are excluded by default (scored denominator is reported), or scored zero with `no_positive: zero`. Duplicate retrieved IDs fail rather than inflating metrics or silently changing K. Empty results score zero for judged positive queries. Grade >0 is binary relevance for Recall/Precision/MRR/Hit Rate, while NDCG preserves grade magnitude.

Scores sort descending, then document ID ascending for ties; no tie averaging. Exact FAISS search expands a boundary tie before applying the ID tie-break. Approximate indexes tie-break within their candidate set; global exact boundary membership is not guaranteed. These conventions are not interchangeable with tie-averaged or linear-gain NDCG. The legacy binary fixture is retained; binary gain agrees with exp2 gain for grades 0/1.

Validation includes hand calculations and exhaustive permutations against scikit-learn `ndcg_score` with exp2-transformed gains, plus range and recall-monotonicity properties. It is not just a test that mirrors the implementation formula.

## Timing and memory methodology

- Corpus encoding and all-query batch encoding have separate wall-clock timers. Batched query encoding is not a per-request latency claim.
- Index build timing starts after encoding and includes training/addition but excludes serialization. Dense exact/HNSW/IVF and sparse storage are all measured.
- Warm each query once, then record three serial repetitions; store raw timings per query. P50/P95/P99 use NumPy's default linear quantile. Quality has one observation per query, not one per timing repetition.
- Timed query execution includes pre-encoded index lookup, optional MMR, Python and tracing overhead; it excludes text encoding, model download, context construction, generation, and network. It is **not end-to-end RAG latency**.
- Serial QPS = timed call count / sum of per-call measured wall times. It excludes outside-loop overhead and is not concurrent request throughput or a service capacity claim. There is no dedicated isolated timing host; runtime noise applies.
- Each index is serialized and reloaded before dense queries; dimension/count checks fail on inconsistent persistence. Artifacts retain parameters and exact byte size.
- Record actual process RSS before/after build and delta, retained vector bytes, and serialized FAISS bytes as a native-memory estimate. RSS deltas can be negative/noisy due to allocator effects and are **not isolated native index memory**. Dense vectors remain in memory for MMR and are additional to FAISS storage. Sparse artifacts are compressed NPZ; their byte sizes are not directly equivalent to native FAISS memory.
- ANN Recall@K is intersection with exact flat-index top-K / min(K,corpus size), before MMR, over the same vectors/queries/corpus. It measures neighbor agreement, not relevance; do not confuse it with qrel-based Recall@K.

## Statistical comparisons

Prespecified comparisons use the first declared configuration at each K as a lexical reference, plus exact dense similarity as the reference for dense MMR/HNSW/IVF. All pairs require identical evaluated query IDs. Differences are right minus left per query. For each metric, resample paired query differences with replacement 2,000 times, seed 42; report mean difference and 2.5/97.5 percentiles. Do not resample timing repetitions as independent quality observations.

Intervals describe this query sample under an IID-query approximation. They are exploratory, unadjusted for multiple comparisons, and do not establish a universal ranking or production significance. Queries may be correlated by scientific topic. The 22-query fixture is too small for a strong generalization claim. Multi-seed/index-build studies remain future work; additional measured domains and their derived dataset conditions are documented in [cross-domain results](CROSS_DOMAIN_RESULTS.md). Runtime repeated trials estimate warm timing noise, not stochastic relevance uncertainty.

## Artifacts and integrity

Each run writes `report.json`, per-query rankings/scores/raw timing trials, index artifacts, and a snapshot of hashed evaluation/telemetry/config source. Reports include dataset manifest/version/counts, file hashes, corpus/query ID hashes, full retrieval/model settings, random seed, metric version, dependency versions, code commit, dirty-tree flag, source hashes, hardware, start/completion timestamps, experiment ID and unique run ID. Experiment IDs hash config/dataset/code/metric semantics; run IDs distinguish repetitions. Dirty-tree source snapshots make the code used by an uncommitted run inspectable.

The measured environment is also captured in `docs/benchmarks/scifact-v1/environment.lock.txt`; it records the actual Python 3.14/macOS environment and is not a portable replacement for the normal requirements.

A failed run writes `status: FAILED` and exception type, then raises nonzero; incomplete results cannot pass the comparison/gate. No database is introduced. Publish measurements and IDs without redistributing downloaded text or weights. Source snapshots are source code only, excluding `.env` and credentials.

## CI and quality gates

Normal CI runs small deterministic fixtures, all unit/API contract tests, and a versioned retrieval regression gate. `evaluation/baselines/regression-v1.json` locks dataset/config/metric semantics and expected quality/count/ID-hash results. No degradation is allowed on this deterministic fixture; 1e-12 is only an aggregation roundoff tolerance. Changed conditions, missing configurations, NaN or a lower metric fail. Baseline updates must be intentional, reviewed, and separately versioned; the runner does not auto-accept a regression.

Latency is not gated on noisy shared CI hosts. `.github/workflows/retrieval-benchmark.yml` runs the larger public benchmark manually or monthly and uploads machine-readable results/indexes. The separate scheduled/public benchmark workflow remains unverified; the main deterministic fixture and Docker integration passed in [CI run 41760000510](https://github.com/Prathampthk/evalforge-ml/actions/runs/41760000510). Existing annotation-derived `quality-regression.yml` is retained as a legacy harness check and must not be cited as measured retrieval quality.

## RAG answer evaluation remains distinct

`evaluation.run_ragas` still evaluates generated answers with its configured LLM judge. It records static prompt templates/hashes, RAGAS version, model alias and optional `JUDGE_MODEL_VERSION`, coverage, provider-reported token usage, and optional price-based cost estimates. Set `JUDGE_INPUT_USD_PER_MILLION` and `JUDGE_OUTPUT_USD_PER_MILLION` using your actual provider rates; missing usage/rates produce unknown costs, not fabricated zero cost. Scope excludes pipeline calls and unreported retries. New caches carry corpus/dataset hashes; stale provenance mismatches fail before rescoring. Old caches remain readable but lack verified provenance.

Retrieval relevance uses qrels; context utility/coverage uses judge-based RAGAS metrics; answer faithfulness checks context support; correctness requires separate reference validation. The existing four-metric set does not measure answer correctness, human-judge agreement, or false-positive rate. No independently human-labeled generated-answer subset is available, so disagreement/false-positive fields remain null, and no paid judge run or answer-quality gain is claimed. Provider aliases can change, temperature zero does not guarantee determinism, and self-consistency scores are not objective truth.

## Observability

`--trace-exporter console` or `otlp` enables export using existing telemetry configuration. Retrieval-only stages include preprocessing, encoding, index build, lookup, MMR, and metric scoring. Experiment/run IDs connect traces to artifacts. Existing RAG spans cover embedding, retrieval, generation/LLM and evaluation; context construction now has a span. Attributes include counts/model/config identities, not raw questions/documents. Exception events expose exception type only, not potentially sensitive messages. In-memory exporter tests verify attribute propagation and privacy. Live collector/dashboard deployment is not validated by these tests.

## End-to-end serving, load and index lifecycle

The original SciFact bundles remain byte unchanged. New [HTTP measurements](SERVING_PERFORMANCE.md) execute real MiniLM inference for every request through FastAPI and retain 36,000 raw client/stage observations. These are separate from this document's warm pre-encoded index measurements. `evaluation.http_load` reports closed-loop throughput, P50/P95/P99, errors and sampled server sum RSS, plus encoder lock wait, inference, lookup, processing and an explicitly mixed transport/client residual. Empty HTTP controls do not isolate physical network latency. No LLM call is included.

`serving.build_snapshot` prepares a hash-verified immutable exact index; [snapshot publication semantics](SNAPSHOT_CONSISTENCY.md) define update visibility, process crash recovery, model compatibility and local-filesystem limits. `evaluation.serving_checks` verifies real two-process HTTP updates/reads/restarts; `evaluation.docker_check` repeats them in a disposable container/volume. The hash encoder is explicitly integration-only, never evidence for dense semantic accuracy or model throughput.

Reproduce additional domains with explicit downloads and new output directories:

```bash
python -m evaluation.prepare_dataset --dataset arguana --output data/arguana
python -m evaluation.prepare_dataset --dataset fiqa --output data/fiqa
OMP_NUM_THREADS=1 python -m evaluation.experiment --config configs/experiments/arguana.json --output evaluation/reports/arguana-new
OMP_NUM_THREADS=1 python -m evaluation.experiment --config configs/experiments/fiqa.json --output evaluation/reports/fiqa-new
python -m evaluation.docker_check --output reports/docker-integration.json
```

See [dataset exclusions and licenses](DATASETS.md), [cross-domain measurements](CROSS_DOMAIN_RESULTS.md), and [current validation status](PERFORMANCE_AUDIT.md). Dense runs remain offline/pinned after model preparation. ArguAna/FiQA have one warm serial repetition, explicitly recorded in their configs, versus three in the preserved SciFact study. Neither implies concurrent HTTP performance. Source snapshots and hashes preserve exact dirty-tree code used for measured runs, even when later maintenance changes the repository.

Current code validation: [CI run 41760000510](https://github.com/Prathampthk/evalforge-ml/actions/runs/41760000510), all eight jobs successful, Python 3.11.18 with 169 tests and actual synthetic-encoder Docker update/restart checks. This is separate from the measured local CPU performance runs; CI did not produce the published MiniLM load timings.
