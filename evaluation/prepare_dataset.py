"""Prepare a bounded, checksum-verified BEIR SciFact download (no auto-download on import)."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/scifact"))
    args = parser.parse_args()
    for manifest in prepare_scifact(args.output):
        print(manifest, load_dataset(manifest).identity())


if __name__ == "__main__":
    main()
