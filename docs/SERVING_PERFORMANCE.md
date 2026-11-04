# End-to-end retrieval serving measurements

Actual CPU loopback results, November 3, 2026 local time (November 4 UTC). The frozen [report](benchmarks/serving-v1/report.json), [36,000 request samples](benchmarks/serving-v1/requests.jsonl.gz), [exact source snapshot](benchmarks/serving-v1/source-snapshot.json), and [snapshot build report](benchmarks/serving-v1/build-report.json) are the evidence. Existing [SciFact index experiments](SCIFACT_RESULTS.md) are unchanged.

## Workload and environment

FastAPI `serving.api:app`, Uvicorn processes, HTTPX client, real text-to-vector MiniLM inference, normalized 384-dimensional vectors and FAISS exact inner product over all 5,183 SciFact documents. All 300 test queries cycle 20 times per case, K=10; one ranking per unique query is scored. Every case reproduces Recall@10 **0.783333**, MRR@10 **0.604725**, and NDCG@10 **0.645082**. Repeated requests are timing observations, not independent relevance labels. No answer generation or LLM judge is involved.

macOS 26.6.2 arm64, Python 3.14.2, 10 logical CPUs, 16 GiB RAM. Exact dependency versions, model revision, data hashes and native runtime are in the report; the [actual full environment freeze](benchmarks/serving-v1/environment.lock.txt) records the validation/measurement environment, not a cross-platform lockfile. It also applies to the new ArguAna/FiQA runs. Each encoder and FAISS request uses one CPU thread; inference is serialized by a per-process lock. One model and one loaded snapshot per worker. The client shares the same host. An unrelated workload-free dedicated benchmark host was not available.

Closed-loop concurrency: each client issues its next request after completing the previous one. Workers=1/2, clients=1/4/16, 6,000 timed requests per case, warmups excluded. Six cases cover **36,000 successful retrieval requests, zero errors**, plus 600 separate empty HTTP controls. Individual measured windows are only 16.5–30.1 seconds: these are repeatable short load tests, not long-running soak tests, open-loop saturation, production capacity, WAN latency, or an SLA.

## Observed client latency and throughput

All latency values below include real embedding and HTTP request/response processing. RSS is the sampled peak sum for server parent plus children, MiB. Shared pages may be counted more than once; this is neither isolated index memory nor PSS and excludes the client and startup peak.

| Workers | Clients | Requests | Success QPS | P50 ms | P95 ms | P99 ms | Errors | Peak sum RSS MiB | Window s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6000 | 228.9 | 4.05 | 6.32 | 7.75 | 0 | 719.8 | 26.2 |
| 1 | 4 | 6000 | 249.5 | 14.47 | 24.00 | 45.99 | 0 | 626.9 | 24.1 |
| 1 | 16 | 6000 | 261.3 | 57.30 | 81.57 | 127.63 | 0 | 349.6 | 23.0 |
| 2 | 1 | 6000 | 199.1 | 4.37 | 8.05 | 12.83 | 0 | 1316.7 | 30.1 |
| 2 | 4 | 6000 | 349.1 | 12.05 | 19.29 | 35.86 | 0 | 876.0 | 17.2 |
| 2 | 16 | 6000 | 362.8 | 51.61 | 81.08 | 121.15 | 0 | 842.2 | 16.5 |

Two workers at concurrency 16 delivered 362.8 successful QPS versus 261.3 with one worker (1.39x in these cases), while P95 remained about 81 ms. Two workers were slower at concurrency 1. Connection affinity means a single keepalive client need not use both workers; observed PIDs are retained. The result does not support a universal worker-scaling multiplier. RSS varies with allocator behavior and OS reclamation/compression; the falling values across some cases are not a memory optimization claim.

## Stage decomposition

Medians in milliseconds; percentiles of separate components do **not** add to the percentile of the whole request. Full P50/P95/P99 distributions are in the report and every sample retains measured stage values.

| Workers / clients | Encoder lock wait | Embedding | FAISS lookup | Processing | Handler total | Empty HTTP P50 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 0.000 | 2.810 | 0.192 | 0.077 | 3.089 | 0.615 |
| 1 / 4 | 8.784 | 3.516 | 0.556 | 0.124 | 13.026 | 1.209 |
| 1 / 16 | 45.774 | 3.468 | 0.545 | 0.269 | 50.175 | 13.011 |
| 2 / 1 | 0.000 | 2.950 | 0.210 | 0.096 | 3.269 | 0.630 |
| 2 / 4 | 5.418 | 4.190 | 0.608 | 0.136 | 10.396 | 1.192 |
| 2 / 16 | 37.890 | 4.930 | 0.745 | 0.164 | 45.372 | 13.954 |

`embedding` measures actual one-query inference. `embedding_wait` isolates encoder lock contention. `faiss_lookup` includes native search and VectorStore result conversion. `processing` is the handler remainder including snapshot refresh, preparation and result formatting. `handler_total` encloses those stages. The response header measures ASGI middleware entry to response headers, including parsing, authentication, thread dispatch and serialization; its difference from handler total is framework/queue overhead.

Client elapsed minus server-to-headers time is explicitly a **transport/framework/client residual**, including socket handling, pre-ASGI queueing, response completion, scheduling and HTTPX overhead. It cannot isolate physical network latency. A post-load `/ping` control at each concurrency measures the same-host HTTP stack without embedding/search; it is not pure network either. Raw residuals are retained rather than relabeled as network measurements. Increasing contention dominates the latency tail; an ANN-only lookup speedup would not remove the encoder queue.

RSS is sampled every 50 ms and can miss brief peaks. Successful and all-request distributions and error counts are separate; this run has no errors. Timeouts are failures rather than silently dropped samples. Warmup and server startup are excluded. No production traffic mix, TLS/WAN, alternative model, batching, sustained ingestion load, or ANN-backed HTTP service has been measured.

## Reproduce

Install `requirements-experiments.txt` and `requirements-serving.txt`, prepare the pinned model and SciFact, then use new output paths:

```bash
python -m evaluation.prepare_dataset --dataset scifact --output data/scifact
python -m evaluation.prepare_model
OMP_NUM_THREADS=1 python -m serving.build_snapshot \
  --manifest data/scifact/test.manifest.json --output artifacts/serving-new
OMP_NUM_THREADS=1 python -m evaluation.http_load \
  --manifest data/scifact/test.manifest.json --snapshots artifacts/serving-new \
  --output evaluation/reports/http-new --workers 1 2 --concurrency 1 4 16 --requests 6000
```

On the measured macOS environment, torch and FAISS bundle separate OpenMP libraries. The launcher removes an inherited `KMP_DUPLICATE_LIB_OK` and sets the subprocess `DYLD_LIBRARY_PATH` to the installed torch `lib` directory so one compatible runtime is used. Health/runtime evidence confirms one OpenMP library with one thread. Before the builder/experiment commands on this host, set `DYLD_LIBRARY_PATH` to that same installed torch `lib` directory (for example `$(python -c "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('torch').origin).parent / 'lib')")`). Do not disable the duplicate-runtime check. This is a platform/dependency workaround, not a portable performance tuning claim. Original frozen SciFact measurements are not rewritten; their environment may have inherited scikit-learn’s duplicate-runtime setting. Linux containers use their native library resolution.

Frozen source reflects the measured working tree and is hash-verified; later maintenance edits can differ. Reports record the base commit plus dirty-tree source hashes. Exact timing reproduction is not promised on other machines.
