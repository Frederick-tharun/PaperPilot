from fastapi.testclient import TestClient

from paperpilot import api
from paperpilot.rag import RAGPipeline


def test_ask_endpoint(indexed, monkeypatch):
    pipeline = RAGPipeline.from_settings(indexed)
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
    client = TestClient(api.app)

    assert client.get("/health").json()["status"] == "ok"

    response = client.post("/ask", json={"question": "Why are echo state networks fast to train?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] and body["citations"][0]["paper_id"] == "paperA"
    assert len(body["sources"]) > 0


def test_rejects_empty_question(indexed, monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", lambda: RAGPipeline.from_settings(indexed))
    assert TestClient(api.app).post("/ask", json={"question": ""}).status_code == 422
