"""Download only the pinned small CPU embedding model needed for experiments."""

from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL,
        revision=REVISION,
        cache_dir=str(Path("artifacts/model-cache")),
        allow_patterns=["*.json", "1_Pooling/*.json", "model.safetensors", "vocab.txt", "README.md"],
    )
    print(path)


if __name__ == "__main__":
    main()
