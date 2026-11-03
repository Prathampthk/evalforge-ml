import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import telemetry
from evaluation.experiment import execute
from evaluation.experiment_config import ExperimentConfig, RetrievalSpec
from evaluation.retrieval_data import load_dataset


@pytest.fixture
def exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


def test_context_manager_attributes_and_private_error(exporter):
    with pytest.raises(RuntimeError):
        with telemetry.span_context_or_null("stage", {"experiment.id": "stable-id"}):
            raise RuntimeError("PRIVATE_DOCUMENT_AND_SECRET")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["experiment.id"] == "stable-id"
    assert span.status.status_code.name == "ERROR"
    assert span.status.description == "RuntimeError"
    assert "PRIVATE_DOCUMENT_AND_SECRET" not in span.to_json()
    assert span.events[0].attributes["exception.type"] == "RuntimeError"


def test_benchmark_stage_spans_and_experiment_correlation(exporter, tmp_path):
    config = ExperimentConfig(
        dataset="fixture",
        repeats=1,
        bootstrap_samples=100,
        configurations=[RetrievalSpec(name="mmr", representation="tfidf", index="sparse", mode="mmr", k=3)],
    )
    data = load_dataset(Path("evaluation/datasets/regression-v1/manifest.json"))
    report = execute(config, data, tmp_path / "run")
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {
        "evalforge.experiment",
        "evalforge.query_preprocessing",
        "evalforge.embedding_generation",
        "evalforge.index_lookup",
        "evalforge.rerank_mmr",
        "evalforge.evaluation_scoring",
    } <= names
    for span in spans:
        assert span.attributes["experiment.id"] == report["experiment_id"]
    trace_text = json.dumps([s.to_json() for s in spans])
    assert data.queries["ragas-001"] not in trace_text
    assert data.corpus["ctx-mmr"] not in trace_text
