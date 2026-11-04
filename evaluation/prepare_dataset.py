"""Prepare a bounded, checksum-verified BEIR SciFact download (no auto-download on import)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from evaluation.retrieval_data import load_dataset, sha256

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


def prepare_scifact(destination: Path) -> list[Path]:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty destination; existing datasets are never overwritten")
    with urllib.request.urlopen(URL, timeout=60) as response:  # nosec B310 - fixed HTTPS dataset URL
        content = response.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("Download exceeds 50 MiB budget")
    # Published BEIR transfer checksum, not a security signature.
    if hashlib.md5(content, usedforsecurity=False).hexdigest() != MD5:
        raise ValueError("BEIR archive checksum mismatch")
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Pinned archive SHA-256 mismatch")
    archive = zipfile.ZipFile(io.BytesIO(content))
    names = ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"]
    if sum(archive.getinfo("scifact/" + name).file_size for name in names) > 100 * 1024 * 1024:
        raise ValueError("Extracted data exceeds 100 MiB budget")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = destination / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(archive.read("scifact/" + name))
    manifests = []
    for split in ("train", "test"):
        manifest = {
            "schema_version": 1,
            "id": "beir/scifact",
            "version": f"beir-md5-{MD5}",
            "archive_sha256": hashlib.sha256(content).hexdigest(),
            "split": split,
            "role": "development" if split == "train" else "benchmark",
            "origin": URL,
            "license": "Original claims/annotations CC-BY-4.0; abstracts ODC-By-1.0; BEIR card CC-BY-SA-4.0",
            "license_source": "https://github.com/allenai/scifact/blob/master/LICENSE.md",
            "preprocessing": "Join title and text with one space; preserve document units; no generated judgments",
            "files": {
                kind: {"path": name, "sha256": sha256(destination / name)}
                for kind, name in (
                    ("corpus", "corpus.jsonl"),
                    ("queries", "queries.jsonl"),
                    ("qrels", f"qrels/{split}.tsv"),
                )
            },
        }
        path = destination / f"{split}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        data = load_dataset(path)
        manifest.update({key: data.identity()[key] for key in ("corpus_count", "query_count", "qrel_count")})
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append(path)
    train, test = [load_dataset(path) for path in manifests]
    if set(train.queries) & set(test.queries):
        raise ValueError("Development/benchmark query leakage detected")
    return manifests


def prepare_arguana(destination: Path) -> list[Path]:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty destination; existing datasets are never overwritten")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip"
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        content = response.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("Download exceeds 50 MiB budget")
    if hashlib.md5(content, usedforsecurity=False).hexdigest() != "8ad3e3c2a5867cdced806d6503f29b99":
        raise ValueError("Published BEIR checksum mismatch")
    if hashlib.sha256(content).hexdigest() != "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b":
        raise ValueError("Pinned ArguAna SHA-256 mismatch")
    archive = zipfile.ZipFile(io.BytesIO(content))
    files = {"corpus": "corpus.jsonl", "queries": "queries.jsonl", "qrels": "qrels/test.tsv"}
    if sum(archive.getinfo("arguana/" + name).file_size for name in files.values()) > 100 * 1024 * 1024:
        raise ValueError("Extraction exceeds 100 MiB budget")
    destination.mkdir(parents=True)
    for name in files.values():
        path = destination / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(archive.read("arguana/" + name))
    manifest = {
        "schema_version": 1,
        "id": "beir/arguana",
        "version": "beir-md5-8ad3e3c2a5867cdced806d6503f29b99",
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "split": "test",
        "role": "benchmark",
        "origin": url,
        "license": "CC-BY-SA-4.0 per BEIR distribution card; original publisher rights retained",
        "license_source": "https://huggingface.co/datasets/BeIR/arguana",
        "exclude_query_id": True,
        "original_source": "https://webis.de/data/arguana-counterargs.html",
        "files": {kind: {"path": name, "sha256": sha256(destination / name)} for kind, name in files.items()},
    }
    path = destination / "test.manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return finalize_arguana(destination, manifest)


def finalize_arguana(destination: Path, manifest: dict) -> list[Path]:
    corpus_ids = {json.loads(line)["_id"] for line in (destination / "corpus.jsonl").read_text().splitlines()}
    query_ids = {json.loads(line)["_id"] for line in (destination / "queries.jsonl").read_text().splitlines()}
    with (destination / "qrels/test.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    excluded = [row for row in rows if row["corpus-id"] not in corpus_ids or row["query-id"] not in query_ids]
    valid = [row for row in rows if row not in excluded]
    available = destination / "qrels/test-available.tsv"
    with available.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(valid)
    (destination / "excluded-qrels.json").write_text(json.dumps(excluded, indent=2) + "\n")
    manifest.update(
        version="beir-md5-8ad3e3c2a5867cdced806d6503f29b99-available-v1",
        original_query_count=len(query_ids),
        original_qrel_count=len(rows),
        excluded_qrel_count=len(excluded),
        exclusion_reason="Published archive references five unavailable corpus IDs; derived available-only subset",
    )
    manifest["raw_qrels_sha256"] = sha256(destination / "qrels/test.tsv")
    manifest["files"]["qrels"] = {"path": "qrels/test-available.tsv", "sha256": sha256(available)}
    path = destination / "test.manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    data = load_dataset(path)
    manifest.update({key: data.identity()[key] for key in ("corpus_count", "query_count", "qrel_count")})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return [path]


def prepare_fiqa(destination: Path) -> list[Path]:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty destination; existing datasets are never overwritten")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        content = response.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("Download exceeds 50 MiB budget")
    md5 = "17918ed23cd04fb15047f73e6c3bd9d9"
    if hashlib.md5(content, usedforsecurity=False).hexdigest() != md5:
        raise ValueError("Published FiQA BEIR checksum mismatch")
    archive = zipfile.ZipFile(io.BytesIO(content))
    names = ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/dev.tsv", "qrels/test.tsv"]
    if sum(archive.getinfo("fiqa/" + name).file_size for name in names) > 100 * 1024 * 1024:
        raise ValueError("Extraction exceeds 100 MiB budget")
    destination.mkdir(parents=True)
    for name in names:
        path = destination / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(archive.read("fiqa/" + name))
    return finalize_fiqa(destination, hashlib.sha256(content).hexdigest())


def finalize_fiqa(destination: Path, archive_sha256: str) -> list[Path]:
    if archive_sha256 != "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02":
        raise ValueError("Pinned FiQA archive SHA-256 mismatch")
    corpus = [json.loads(line) for line in (destination / "corpus.jsonl").read_text().splitlines()]
    valid = [row for row in corpus if isinstance(row.get("text"), str) and row["text"].strip()]
    ids = {row["_id"] for row in valid}
    filtered = destination / "corpus-nonempty.jsonl"
    filtered.write_text("".join(json.dumps(row) + "\n" for row in valid))
    paths = []
    for split in ("train", "dev", "test"):
        raw_qrels = destination / f"qrels/{split}.tsv"
        with raw_qrels.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        available = [row for row in rows if row["corpus-id"] in ids]
        excluded = [row for row in rows if row["corpus-id"] not in ids]
        qrels = destination / f"qrels/{split}-available.tsv"
        with qrels.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(available)
        (destination / f"excluded-{split}.json").write_text(json.dumps(excluded, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "id": "beir/fiqa",
            "version": "beir-md5-17918ed23cd04fb15047f73e6c3bd9d9-nonempty-v1",
            "archive_sha256": archive_sha256,
            "split": split,
            "role": "benchmark" if split == "test" else "development",
            "origin": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
            "license": "CC-BY-SA-4.0 per BEIR distribution card; original financial post rights retained",
            "license_source": "https://huggingface.co/datasets/BeIR/fiqa",
            "original_corpus_count": len(corpus),
            "excluded_empty_documents": len(corpus) - len(valid),
            "original_query_count": len({row["query-id"] for row in rows}),
            "original_qrel_count": len(rows),
            "excluded_qrel_count": len(excluded),
            "exclude_query_id": True,
            "preprocessing": "Drop empty/whitespace documents and only judgments referencing them; no relabeling",
            "raw_corpus_sha256": sha256(destination / "corpus.jsonl"),
            "raw_qrels_sha256": sha256(raw_qrels),
            "files": {
                kind: {"path": name, "sha256": sha256(destination / name)}
                for kind, name in (
                    ("corpus", "corpus-nonempty.jsonl"),
                    ("queries", "queries.jsonl"),
                    ("qrels", f"qrels/{split}-available.tsv"),
                )
            },
        }
        path = destination / f"{split}.manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        data = load_dataset(path)
        manifest.update({key: data.identity()[key] for key in ("corpus_count", "query_count", "qrel_count")})
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        paths.append(path)
    selected = [set(load_dataset(path).queries) for path in paths]
    if any(selected[i] & selected[j] for i in range(len(selected)) for j in range(i)):
        raise ValueError("FiQA development/test query overlap")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/scifact"))
    parser.add_argument("--dataset", choices=["scifact", "arguana", "fiqa"], default="scifact")
    args = parser.parse_args()
    prepare = {"scifact": prepare_scifact, "arguana": prepare_arguana, "fiqa": prepare_fiqa}[args.dataset]
    for manifest in prepare(args.output):
        print(manifest, load_dataset(manifest).identity())


if __name__ == "__main__":
    main()
