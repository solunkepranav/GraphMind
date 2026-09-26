import pytest
from unittest.mock import patch
from src.agents import QueryAgent
from src.retrieval import RetrievalOrchestrator
from tests.fakes import FakeVectorStore, FakeGraphStore, FakeLLM

def test_query_agent_with_orchestrator():
    vec_store = FakeVectorStore()
    vec_store.add_documents([
        {"id": "doc1", "text": "GraphMind hybrid retrieval blends dense embeddings with BM25 lexical search.", "source": "guide.pdf", "page": 1}
    ])
    
    graph_store = FakeGraphStore()
    graph_store.add_triples(
        [{"subject": "GraphMind", "relation": "features", "object": "RRF"}],
        source="guide.pdf",
        page=1
    )

    agent = QueryAgent(
        vector_store=vec_store,
        graph_store=graph_store
    )

    fake_llm = FakeLLM()

    with patch("src.llm.generate_json") as mock_json, patch("src.llm.generate_text") as mock_text:
        # First call: router classification -> HYBRID
        # Second call: QA generation -> answer + confidence
        mock_json.side_effect = [
            {"category": "HYBRID", "reasoning": "Needs both text and relations"},
            {"answer": "GraphMind uses hybrid retrieval and RRF [1].", "confidence": 95}
        ]
        
        result = agent.answer_query("How does GraphMind retrieve information?", source_filter=["guide.pdf"])

        assert result["category"] == "HYBRID"
        assert result["raw_confidence"] == 95
        assert result["confidence"] == 72
        assert "confidence_signals" in result
        assert "GraphMind uses hybrid retrieval" in result["answer"]
        assert len(result["vector_chunks"]) >= 1
        assert "retrieval_metadata" in result
        assert result["retrieval_metadata"]["source_filter"] == ["guide.pdf"]
        assert result["cached"] is False

def test_query_agent_caching():
    vec_store = FakeVectorStore()
    vec_store.add_documents([
        {"id": "doc1", "text": "Cached document content.", "source": "test.pdf", "page": 1}
    ])
    graph_store = FakeGraphStore()
    agent = QueryAgent(vector_store=vec_store, graph_store=graph_store)

    with patch("src.llm.generate_json") as mock_json:
        mock_json.side_effect = [
            {"category": "SIMPLE", "reasoning": "Direct fact"},
            {"answer": "Answer from LLM [1].", "confidence": 90}
        ]
        
        # First call
        res1 = agent.answer_query("What is cached?", source_filter=["test.pdf"])
        assert res1["cached"] is False
        assert mock_json.call_count == 2

        # Second call with same query and filter: hits cache!
        res2 = agent.answer_query("What is cached?", source_filter=["test.pdf"])
        assert res2["cached"] is True
        assert res2["answer"] == "Answer from LLM [1]."
        # LLM was not called again
        assert mock_json.call_count == 2

        # After clearing cache, LLM is called again
        agent.clear_cache()
        mock_json.side_effect = [
            {"category": "SIMPLE", "reasoning": "Direct fact"},
            {"answer": "Fresh answer [1].", "confidence": 90}
        ]
        res3 = agent.answer_query("What is cached?", source_filter=["test.pdf"])
        assert res3["cached"] is False
        assert mock_json.call_count == 4

