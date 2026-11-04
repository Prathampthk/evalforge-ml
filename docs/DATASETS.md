# Versioned retrieval datasets

## Supported contract

`evaluation.retrieval_data.load_dataset` loads a schema-v1 JSON manifest pointing to three content-hashed files:

- `corpus.jsonl`: unique string `_id`, nonempty `text`, optional string `title`.
- `queries.jsonl`: unique string `_id`, nonempty `text`.
- `qrels.tsv`: exact header `query-id`, `corpus-id`, `score`; nonnegative integer grade in [0,30]. Grade > 0 means relevant; grades also determine NDCG gain.

Manifest fields: schema version, dataset ID, immutable version, split, role, origin, license, and SHA-256 for every file. Optional expected corpus/query/judgment counts are checked. Missing identifiers, duplicate IDs/judgments, invalid grades, checksum/count mismatches, and qrels pointing to nonexistent queries/documents fail the run. BEIR's queries file can contain multiple splits; only queries explicitly listed in the selected qrels file are evaluated. An evaluated query cannot silently lack judgments.

`train` must have role `development`; `test` must have role `benchmark`. Never select hyperparameters using test scores. New datasets can use the same format without a new framework. Preserve raw judgments; do not synthesize relevance from answer text or LLM scores.

## Dataset inventory

| Dataset | Version / role | Corpus | Evaluated queries | Judgments | Provenance |
|---|---|---:|---:|---:|---|
| EvalForge regression | `1.0.0`, regression only | 20 | 22 | 23 positive | Inherited repository RAGAS examples, labels not independently validated |
| BEIR SciFact train | Published BEIR archive checksum, development only | 5,183 | 809 | 919 positive | Original scientific claims/evidence adapted to BEIR |
| BEIR SciFact test | Same archive version, benchmark only | 5,183 | 300 | 339 positive | Public BEIR test split; no overlap with train query IDs |

The original `ragas_corpus.json`, `ragas_eval.jsonl`, and six-configuration `retrieval_benchmark.py` remain untouched by the infrastructure upgrade. `regression-v1/` is a frozen BEIR-compatible conversion with stable passage IDs, preserving all 23 expected-context labels. The older `phase2_eval.jsonl` is a different harness fixture with annotation-derived fixed scores, not a benchmark.

## SciFact download and licenses

```bash
python -m evaluation.prepare_dataset --output data/scifact
```

The preparer downloads only [BEIR's SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip), verifies [the published BEIR MD5](https://github.com/beir-cellar/beir/wiki/Datasets-available) plus the pinned SHA-256, then extracts four explicitly named files. Download is bounded to 50 MiB and selected extraction to 100 MiB; observed extracted data is approximately 8 MiB. No arbitrary archive paths are extracted, existing datasets are not overwritten, and development/test query intersection is checked.

Archive SHA-256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.

Data licenses differ from this repository's MIT code license. The [original AllenAI SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) identifies claims/evidence annotations as **CC BY 4.0**, and corpus abstracts (from S2ORC) as **ODC-By 1.0**. The [BEIR Hugging Face card](https://huggingface.co/datasets/BeIR/scifact/blob/main/README.md) advertises **CC BY-SA 4.0** for its distribution/card. Manifests preserve this distinction and source links. Downloads stay in ignored `data/`; this repository publishes experiment measurements and IDs, not copied SciFact text or model weights. Preserve original attribution and applicable terms when redistributing data.

## Preprocessing and split discipline

Join trimmed title and abstract with one space, preserving one indexed unit per original document. There are no invented or duplicated passages. TF-IDF fits vocabulary and IDF on the corpus only; query/qrel text is not included in fitting. Dense MiniLM uses a pinned model, L2 normalization, batch size 32 and a 256-token truncation limit. Truncation can omit evidence; this is documented, not presented as equivalent to full-document reasoning.

Test comparisons use identical query IDs, qrels, document subsets, and K across configurations. Hyperparameters are predeclared in `configs/experiments/scifact.json` and have not been tuned on the test results. The training split is supported separately and has not been used for tuning in the measured runs. MiniLM's published training mixture includes scientific text, so no claim of training-corpus independence or uncontaminated generalization is made.

## Corpus scaling policy

The 1,000- and 3,000-document subsets contain **all judged positive documents plus a seeded, nested sample of real distractors**. This intentionally keeps the same 300-query/339-positive evaluation standard while varying distractor count. It uses labels for inclusion, not parameter optimization; these reduced corpora are easier, label-aware diagnostic subsets, not standard BEIR leaderboard conditions or representative random samples. The 5,183-document run uses the complete corpus. Every result records counts and corpus/query ID hashes. No document duplication or synthetic scaling data is used.

## Judgment limitations

SciFact is one scientific-claim retrieval domain, with incomplete evidence judgments; unjudged documents are treated as nonrelevant by convention. Retrieving a judged document is distinct from supporting/refuting a claim correctly. These judgments are not labels for generated answer correctness, faithfulness, or LLM judge agreement. Zero-positive judged queries are supported through an explicit `exclude` (default) or `zero` metric policy; counts and scored denominators are reported.

## Additional domains: explicit derived versions

```bash
python -m evaluation.prepare_dataset --dataset arguana --output data/arguana
python -m evaluation.prepare_dataset --dataset fiqa --output data/fiqa
```

Both preparers verify a fixed official BEIR archive checksum plus pinned SHA-256, bound downloads to 50 MiB and selected extraction to 100 MiB, preserve raw data, refuse existing nonempty destinations, and validate the derived manifest. [BEIR dataset inventory](https://github.com/beir-cellar/beir/wiki/Datasets-available), [ArguAna original source](https://webis.de/data/arguana-counterargs.html), [ArguAna BEIR card](https://huggingface.co/datasets/BeIR/arguana), and [FiQA BEIR card](https://huggingface.co/datasets/BeIR/fiqa) identify origins and terms. The BEIR cards advertise CC-BY-SA-4.0; original argument and financial-post rights are retained. These are not MIT-licensed code assets. Raw downloads remain ignored and are not republished here.

| Derived benchmark version | Usable corpus | Test queries | Positive test judgments | Explicit exclusions |
|---|---:|---:|---:|---|
| ArguAna `beir-md5-8ad3e3c2a5867cdced806d6503f29b99-available-v1` | 8,674 | 1,401 | 1,401 | Five of 1,406 archive qrels reference absent corpus IDs; five affected queries omitted |
| FiQA `beir-md5-17918ed23cd04fb15047f73e6c3bd9d9-nonempty-v1` | 57,600 | 648 | 1,705 | 38 of 57,638 documents are empty/whitespace; one test qrel references an excluded document |

ArguAna archive SHA-256: `cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b`. FiQA: `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02`. Derived manifests include raw/filtered hashes, original/evaluated counts, exclusions, and no relabeling. FiQA train has 5,498 usable queries/14,131 judgments (35 excluded qrels); dev has 500/1,236 (two excluded). Test/dev/train query IDs are checked disjoint; neither development split was used for parameter tuning.

For these new datasets `exclude_query_id: true` filters a result whose document ID equals the query ID before selecting K, following BEIR self-match convention (essential for ArguAna arguments also present in the corpus). Extra candidates are fetched to fill K; exact-neighbor comparisons use the same filtering. This policy is recorded rather than retroactively changing SciFact conditions. One document per original nonempty text, title joined where present, normalized MiniLM truncation at 256 tokens. ArguAna has no development split here; settings were fixed in config. These available-only/nonempty versions differ from unfiltered leaderboard conditions. See [measured cross-domain comparisons](CROSS_DOMAIN_RESULTS.md); additional domains alone do not prove generalization or exclude model training contamination.
