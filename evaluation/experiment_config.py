"""Strict JSON/YAML experiment configuration; unknown settings fail closed."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.prepare_model import MODEL, REVISION


class RetrievalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    representation: Literal["tfidf", "dense"]
    index: Literal["sparse", "flat", "hnsw", "ivf"]
    mode: Literal["similarity", "mmr"] = "similarity"
    k: int = Field(default=5, ge=1, le=1000, strict=True)
    fetch_k: int = Field(default=20, ge=1, strict=True)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1)
    hnsw_m: int = Field(default=16, ge=2, strict=True)
    ef_construction: int = Field(default=100, ge=1, strict=True)
    ef_search: int = Field(default=32, ge=1, strict=True)
    nlist: int = Field(default=32, ge=1, strict=True)
    nprobe: int = Field(default=4, ge=1, strict=True)

    @model_validator(mode="after")
    def compatible(self):
        if (self.representation == "tfidf") != (self.index == "sparse"):
            raise ValueError("TF-IDF requires sparse index; dense requires a FAISS index")
        if self.fetch_k < self.k or self.nprobe > self.nlist:
            raise ValueError("Require fetch_k >= K and nprobe <= nlist")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    dataset: str
    seed: int = Field(default=42, ge=0, strict=True)
    corpus_sizes: list[int] = Field(default_factory=lambda: [0])  # 0 means full corpus
    repeats: int = Field(default=3, ge=1, le=1000, strict=True)
    warmup: int = Field(default=1, ge=0, le=100, strict=True)
    threads: int = Field(default=1, ge=1, le=16, strict=True)
    bootstrap_samples: int = Field(default=2000, ge=100, le=100000, strict=True)
    no_positive: Literal["exclude", "zero"] = "exclude"
    model_name: str = MODEL
    model_revision: str = REVISION
    model_cache: str = "artifacts/model-cache"
    batch_size: int = Field(default=32, ge=1, le=256, strict=True)
    max_seq_length: int = Field(default=256, ge=1, le=512, strict=True)
    configurations: list[RetrievalSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self):
        if not self.corpus_sizes or any(type(n) is not int or n < 0 for n in self.corpus_sizes):
            raise ValueError("Corpus sizes must be nonnegative integers")
        names = [s.name for s in self.configurations]
        if len(set(names)) != len(names):
            raise ValueError("Configuration names must be unique")
        if any(s.representation == "dense" for s in self.configurations):
            if len(self.model_revision) != 40 or any(c not in "0123456789abcdef" for c in self.model_revision):
                raise ValueError("Dense model revision must be an immutable 40-character commit")
        return self


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
