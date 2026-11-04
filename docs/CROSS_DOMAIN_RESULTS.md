# Cross-domain retrieval measurements

These are actual qrel-based retrieval results, not generated-answer quality or HTTP throughput. Original [SciFact results](SCIFACT_RESULTS.md) remain unchanged. Additional runs use the same pinned normalized MiniLM model, CPU/FAISS one thread, seed 42, K=10, no test-score tuning. TF-IDF uses corpus-only unigram/bigram vocabulary and sparse cosine; dense uses exact flat, HNSW M=16/efSearch=32, or IVF. ArguAna uses IVF16/nprobe4; larger FiQA uses predeclared IVF128/nprobe16. ANN neighbor agreement is overlap with exact top-10, distinct from qrel recall.

## Corpus and judgment integrity

ArguAna evaluates the full 8,674-document corpus and **1,401 queries / 1,401 positive judgments**. The official archive has 1,406 judgments but five reference missing corpus IDs. The available-only derived version records every excluded row and raw hashes; those five queries cannot be evaluated against absent evidence. Query IDs matching corpus IDs are excluded from candidates before selecting K, following BEIR self-match convention. These derived conditions are not a claim of reproducing the unmodified leaderboard split.

FiQA has **57,638 raw documents**; 38 empty/whitespace documents are explicitly excluded, producing **57,600 usable documents**, with **648 test queries / 1,705 judgments** after one invalid judgment is removed. Raw files remain unchanged and excluded rows and all hashes are recorded. Train/dev are separate (5,498 / 500 usable queries) and unused for tuning. No labels are generated. These conditions differ from an unfiltered FiQA leaderboard run.

## Quality and warm index timing

Latency below is **warm pre-encoded index query latency**, excluding text embedding, HTTP, startup and LLM generation. New domain runs use one timed repetition per query after warmup (1,401 ArguAna / 648 FiQA observations per configuration); original SciFact has three. No inference-throughput claim can be derived from these serial timings.

| Domain | Configuration | Qrel Recall@10 | MRR@10 | NDCG@10 | ANN agreement@10 | Warm P95 ms | Build ms | Serialized MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arguana | tfidf-exact | 0.6931 | 0.3294 | 0.4145 | — | 6.492 | 0.0 | 10.48 |
| arguana | dense-flat | 0.7930 | 0.4163 | 0.5056 | — | 0.198 | 2.3 | 12.71 |
| arguana | dense-hnsw | 0.7916 | 0.4151 | 0.5044 | 0.9969 | 0.027 | 455.6 | 13.90 |
| arguana | dense-ivf | 0.7730 | 0.4088 | 0.4953 | 0.9690 | 0.073 | 9.2 | 12.80 |
| fiqa | tfidf-exact | 0.2066 | 0.2169 | 0.1663 | — | 32.889 | 0.0 | 62.56 |
| fiqa | dense-flat | 0.4413 | 0.4451 | 0.3687 | — | 2.611 | 3.2 | 84.38 |
| fiqa | dense-hnsw | 0.4250 | 0.4295 | 0.3538 | 0.9549 | 0.043 | 9654.6 | 92.30 |
| fiqa | dense-ivf | 0.4269 | 0.4410 | 0.3603 | 0.9667 | 0.413 | 198.6 | 85.00 |

ArguAna exact dense recall is 79.30% versus 69.31% TF-IDF; HNSW retains 99.69% exact-neighbor agreement with 79.16% qrel recall, while IVF reduces recall to 77.30%. Approximate search therefore has a measurable relevance tradeoff. MiniLM truncates long arguments/financial posts at 256 tokens, which can omit crucial evidence. This is a baseline representation, not a specialized counterargument or financial model.

FiQA exposes the larger-corpus tradeoff: HNSW warm pre-encoded P95 is 0.043 ms versus 2.611 ms flat, but qrel Recall@10 drops from 44.13% to 42.50% and serialized size grows from 84.38 to 92.30 MiB; building the graph takes 9.65 seconds versus 3.2 ms adding pre-encoded vectors to flat. This does not translate directly into HTTP speedup because real encoding/queueing is excluded. IVF is between flat and HNSW in lookup time and has 42.69% qrel recall.

Corpus encoding and all-query batch encoding are separately timed in each report. Build excludes encoding/serialization; serialized bytes are disk/native-memory proxies, while RSS deltas include allocator noise and shared representations. Sparse compressed bytes are not comparable to isolated FAISS resident memory. The sparse adapter has no separate index construction beyond the already timed TF-IDF fitting, so its near-zero build time does not imply free lexical preprocessing.

## Evidence and limitations

Each available bundle under `docs/benchmarks/arguana-v1/` and `fiqa-v1/` contains a COMPLETE report, derived test manifest, exclusion list, compressed per-query IDs/metrics/raw timing samples, and exact hashed source snapshot. No downloaded text, index or model weights are redistributed. Original on-disk index paths in reports refer to ignored local experiment output and are not bundled; reproduce using the recorded config. [Dataset provenance/licenses](DATASETS.md) and [commands/methodology](EXPERIMENTS.md#end-to-end-serving-load-and-index-lifecycle) apply. Paired query bootstrap comparisons are included in the reports, exploratory and unadjusted for multiple comparisons.

Single runs, one model and one seed per new domain; timing noise and correlated queries remain. There is no BM25/hybrid baseline, model training, multi-seed robustness, independent training-data contamination audit, or blanket cross-domain superiority claim. Larger-domain service load and Docker real-model inference are unmeasured; the HTTP benchmark uses SciFact only.
