import json

from evaluation.prepare_dataset import finalize_arguana, finalize_fiqa
from evaluation.retrieval_data import load_dataset, sha256


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_arguana_exclusions_preserve_raw_judgments(tmp_path):
    (tmp_path / "qrels").mkdir()
    write_rows(tmp_path / "corpus.jsonl", [{"_id": "d", "text": "document"}])
    write_rows(tmp_path / "queries.jsonl", [{"_id": "q", "text": "query"}, {"_id": "bad", "text": "missing"}])
    raw = "query-id\tcorpus-id\tscore\nq\td\t1\nbad\tabsent\t1\n"
    (tmp_path / "qrels/test.tsv").write_text(raw)
    manifest = {
        "schema_version": 1,
        "id": "beir/arguana",
        "version": "raw",
        "split": "test",
        "role": "benchmark",
        "origin": "fixture",
        "license": "fixture",
        "exclude_query_id": True,
        "files": {
            kind: {"path": name, "sha256": sha256(tmp_path / name)}
            for kind, name in [("corpus", "corpus.jsonl"), ("queries", "queries.jsonl")]
        },
    }
    path = finalize_arguana(tmp_path, manifest)[0]
    data = load_dataset(path)
    assert set(data.queries) == {"q"} and data.qrels == {"q": {"d": 1}}
    assert json.loads(path.read_text())["excluded_qrel_count"] == 1
    assert (tmp_path / "qrels/test.tsv").read_text() == raw


def test_fiqa_empty_document_filter_and_split_integrity(tmp_path):
    (tmp_path / "qrels").mkdir()
    write_rows(tmp_path / "corpus.jsonl", [{"_id": "d", "text": "real"}, {"_id": "empty", "text": "  "}])
    write_rows(tmp_path / "queries.jsonl", [{"_id": s, "text": s} for s in ("train", "dev", "test")])
    for split in ("train", "dev", "test"):
        (tmp_path / f"qrels/{split}.tsv").write_text(f"query-id\tcorpus-id\tscore\n{split}\td\t1\n{split}\tempty\t1\n")
    paths = finalize_fiqa(tmp_path, "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02")
    for path in paths:
        data = load_dataset(path)
        metadata = json.loads(path.read_text())
        assert metadata["excluded_empty_documents"] == metadata["excluded_qrel_count"] == 1
        assert set(data.corpus) == {"d"}
        assert data.identity()["query_count"] == data.identity()["qrel_count"] == 1
    assert len((tmp_path / "corpus.jsonl").read_text().splitlines()) == 2
