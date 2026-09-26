import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api import app, get_services
from tests.fakes import FakeVectorStore, FakeGraphStore

def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "node_count" in data

def test_api_empty_query_validation():
    client = TestClient(app)
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]

def test_api_query_execution():
    # Override services with fast mocks for unit test
    fake_vec = FakeVectorStore()
    fake_vec.add_documents([
        {"id": "c1", "text": "GraphMind hybrid RAG utilizes RRF.", "source": "guide.pdf", "page": 1}
    ])
    fake_graph = FakeGraphStore()

    fake_agent = MagicMock()
    fake_agent.answer_query.return_value = {
        "answer": "GraphMind uses RRF [1].",
        "confidence": 92,
        "raw_confidence": 95,
        "confidence_signals": {"citation_grounding": 1.0},
        "category": "HYBRID",
        "reasoning": "Needs text and graph",
        "retrieval_metadata": {"elapsed_ms": 15.0, "final_chunk_count": 1}
    }

    from src.permissions import AccessControlManager
    from src.observability import QueryAuditLogger
    mock_services = (fake_vec, None, fake_graph, fake_agent, None, AccessControlManager(), QueryAuditLogger())

    app.dependency_overrides[get_services] = lambda: mock_services
    client = TestClient(app)

    response = client.post("/query", json={"query": "How does GraphMind work?", "role": "admin"})
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["answer"] == "GraphMind uses RRF [1]."
    assert res_data["confidence"] == 92
    assert res_data["category"] == "HYBRID"
    assert "contradictions" in res_data

    app.dependency_overrides.clear()
