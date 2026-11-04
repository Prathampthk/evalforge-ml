import pytest

from config import ChunkingConfig, EmbeddingConfig, PipelineConfig, RetrieverConfig, RuntimeConfig
from main import RAGPipeline
from serving.embedding import Encoder


class TestEmbedding:
    __test__ = False
    dimension = 256

    def __init__(self, cfg):
        self.encoder = Encoder("test-hash")

    def embed(self, texts):
        return self.encoder.encode(texts)

    def embed_query(self, text):
        return self.embed([text])[0]


class TestLLM:
    __test__ = False

    def chat(self, **kwargs):
        return "fixture answer"

    def chat_json(self, **kwargs):
        return {"overall_consistency_score": 1.0, "claims": [], "summary": "fixture"}


def test_shared_rag_queries_refresh_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_SHARED_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr("main.EmbeddingModel", TestEmbedding)
    monkeypatch.setattr("main.LLMClient", lambda cfg: TestLLM())
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="fixture", dimension=256, revision="f" * 40),
        chunking=ChunkingConfig(strategy="legacy_char", min_chunk_size=1),
        retriever=RetrieverConfig(use_mmr=False, similarity_threshold=0.0),
        runtime=RuntimeConfig(evaluator_mode="sync"),
    )
    with RAGPipeline(cfg) as first:
        assert first.ingest(["alpha document"], source="first") == 1
        with RAGPipeline(cfg) as second:
            assert len(second.query("alpha").retrieval) == 1
            first.ingest(["beta document"], source="second")
            result = second.query("beta")
            assert len(result.retrieval) == 2
            assert result.retrieval[0].chunk.content == "beta document"
            with pytest.raises(ValueError, match="restores CURRENT"):
                second.load()
            with pytest.raises(ValueError, match="persist automatically"):
                second.save()
    with RAGPipeline(cfg) as restarted:
        assert len(restarted.query("beta").retrieval) == 2


def test_shared_mode_requires_model_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_SHARED_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr("main.EmbeddingModel", TestEmbedding)
    monkeypatch.setattr("main.LLMClient", lambda cfg: TestLLM())
    with pytest.raises(ValueError, match="pinned"):
        RAGPipeline(PipelineConfig(embedding=EmbeddingConfig(revision=None)))
