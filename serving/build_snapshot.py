"""Build an immutable serving snapshot from a versioned retrieval corpus (offline encoder)."""

import argparse
import json
import time
from pathlib import Path

import faiss

from evaluation.retrieval_data import load_dataset
from models import Chunk
from serving.embedding import Encoder
from serving.snapshots import SnapshotRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=["minilm", "test-hash"], default="minilm")
    parser.add_argument("--model-cache", default="artifacts/model-cache")
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    data = load_dataset(args.manifest)
    encoder = Encoder(args.encoder, args.model_cache)
    chunks = [
        Chunk(id=i, document_id=i, content=text, source=data.metadata["id"], chunk_index=0)
        for i, text in data.corpus.items()
    ]
    started = time.perf_counter()
    vectors = encoder.encode([c.content for c in chunks])
    encode_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    snapshot = SnapshotRepository(args.output, encoder.identity).upsert(chunks, vectors)
    report = {
        "version": snapshot.version,
        "dataset": data.identity(),
        "encoder": encoder.identity,
        "corpus_encoding_ms": encode_ms,
        "snapshot_build_ms": (time.perf_counter() - started) * 1000,
    }
    (args.output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
