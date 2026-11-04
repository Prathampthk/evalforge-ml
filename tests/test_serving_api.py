import pytest
from fastapi.testclient import TestClient

from serving.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_EMBEDDER", "test-hash")
    monkeypatch.setenv("EVALFORGE_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_API_TOKEN", "fixture")
    with TestClient(app) as client:
        yield client


def test_auth_ingestion_query_timings_and_conflict(client):
    assert client.post("/retrieve", json={"query": "alpha"}).status_code == 401
    headers = {"Authorization": "Bearer fixture"}
    assert client.post("/retrieve", json={"query": "alpha"}, headers=headers).status_code == 503
    created = client.post("/documents", json={"documents": [{"id": "a", "text": "alpha"}]}, headers=headers)
    assert created.status_code == 200
    version = created.json()["snapshot_version"]
    response = client.post("/retrieve", json={"query": "alpha", "k": 1}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["id"] == "a"
    assert data["snapshot_version"] == version
    timings = data["timings_ms"]
    assert all(t >= 0 for t in timings.values())
    assert sum(timings[key] for key in ("embedding", "embedding_wait", "faiss_lookup", "processing")) == pytest.approx(
        timings["handler_total"]
    )
    assert float(response.headers["X-EvalForge-Server-Ms"]) >= timings["handler_total"]
    updated = client.post(
        "/documents", json={"documents": [{"id": "b", "text": "beta"}], "expected_version": version}, headers=headers
    )
    assert updated.status_code == 200
    assert (
        client.post(
            "/documents",
            json={"documents": [{"id": "c", "text": "gamma"}], "expected_version": version},
            headers=headers,
        ).status_code
        == 409
    )
    assert client.get("/health").json()["count"] == 2


def test_invalid_duplicate_and_blank_inputs(client):
    headers = {"Authorization": "Bearer fixture"}
    duplicate = {"documents": [{"id": "a", "text": "alpha"}, {"id": "a", "text": "beta"}]}
    assert client.post("/documents", json=duplicate, headers=headers).status_code == 422
    assert client.post("/retrieve", json={"query": ""}, headers=headers).status_code == 422
