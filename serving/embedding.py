"""Real pinned CPU embeddings, plus an explicitly synthetic integration-test encoder."""

import hashlib
import re
import sys
import threading

import numpy as np

from evaluation.prepare_model import MODEL, REVISION


class Encoder:
    def __init__(self, mode: str = "minilm", cache: str = "artifacts/model-cache"):
        self.lock = threading.Lock()
        self.mode = mode
        if mode == "minilm":
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            self.model = SentenceTransformer(
                MODEL,
                revision=REVISION,
                device="cpu",
                cache_folder=cache,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model.max_seq_length = 256
            from threadpoolctl import threadpool_info

            self.native_runtime = threadpool_info()
            openmp = {r["filepath"] for r in self.native_runtime if r["internal_api"] == "openmp"}
            if sys.platform == "darwin" and len(openmp) > 1:
                raise RuntimeError("Multiple OpenMP runtimes detected; configure a single compatible native runtime")
            self.identity = {"name": MODEL, "revision": REVISION, "max_seq_length": 256, "normalize": True}
        elif mode == "test-hash":
            self.native_runtime = []
            self.identity = {"name": "synthetic-test-word-hash", "revision": "sha256-v1", "dimension": 256}
        else:
            raise ValueError("Unknown embedding mode; use minilm or explicit test-hash")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mode == "minilm":
            return self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
        result = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 256
                result[row, bucket] += 1
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)
