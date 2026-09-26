import pytest
from unittest.mock import MagicMock
from src.reranker import CrossEncoderReranker
from src.retrieval import RetrievalOrchestrator
from tests.fakes import FakeVectorStore

def test_cross_encoder_reranker_reordering():
    reranker = CrossEncoderReranker(enabled=True)
    
    # Mock the internal model to avoid heavy real model download during unit test
    mock_model = MagicMock()
    # Let candidate 2 score higher than candidate 1
    mock_model.predict.return_value = [0.12, 0.89]
    reranker._model = mock_model
    reranker._load_attempted = True

    candidates = [
        {"id": "c1", "text": "General introductory text"},
        {"id": "c2", "text": "Highly relevant specific answer"},
    ]

    reranked = reranker.rerank(query="specific answer", candidates=candidates, top_n=2)

    assert len(reranked) == 2
    assert reranked[0]["id"] == "c2"
    assert reranked[0]["reranker_score"] == 0.89
    assert reranked[1]["id"] == "c1"
    assert reranked[1]["reranker_score"] == 0.12

def test_reranker_fallback_on_error():
    reranker = CrossEncoderReranker(enabled=True)
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Prediction failure")
    reranker._model = mock_model
    reranker._load_attempted = True

    candidates = [
        {"id": "c1", "text": "Doc 1"},
        {"id": "c2", "text": "Doc 2"},
    ]

    # Should gracefully catch error and preserve input order
    results = reranker.rerank(query="test", candidates=candidates, top_n=2)
    assert len(results) == 2
    assert results[0]["id"] == "c1"
    assert results[1]["id"] == "c2"

def test_reranker_disabled_bypass():
    reranker = CrossEncoderReranker(enabled=False)
    candidates = [{"id": "c1", "text": "Doc 1"}]
    results = reranker.rerank(query="test", candidates=candidates)
    assert results == candidates

def test_orchestrator_integration_with_reranker():
    vec_store = FakeVectorStore()
    vec_store.add_documents([
        {"id": "c1", "text": "Low relevance document", "source": "d1.txt", "page": 1},
        {"id": "c2", "text": "Exact pinpoint information", "source": "d2.txt", "page": 1}
    ])

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, c, top_n: sorted(c, key=lambda x: "pinpoint" in x.get("text", ""), reverse=True)[:top_n]

    orchestrator = RetrievalOrchestrator(
        vector_store=vec_store,
        reranker=mock_reranker
    )

    res = orchestrator.retrieve("pinpoint", top_k=2)
    assert len(res.chunks) == 2
    assert res.chunks[0]["id"] == "c2"
