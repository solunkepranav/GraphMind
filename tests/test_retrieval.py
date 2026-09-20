import pytest
from src.retrieval import RetrievalOrchestrator, reciprocal_rank_fusion, RetrievalResult
from tests.fakes import FakeVectorStore, FakeGraphStore

def test_rrf_scoring_and_deduplication():
    # Chunk A appears in both dense (rank 1) and bm25 (rank 1)
    # Chunk B appears only in dense (rank 2)
    # Chunk C appears only in bm25 (rank 2)
    dense_results = [
        {"id": "doc_a", "text": "Document A text", "source": "a.txt", "page": 1},
        {"id": "doc_b", "text": "Document B text", "source": "b.txt", "page": 1},
    ]
    bm25_results = [
        {"chunk_id": "doc_a", "text": "Document A text", "source": "a.txt", "page": 1},
        {"chunk_id": "doc_c", "text": "Document C text", "source": "c.txt", "page": 1},
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    
    assert len(fused) == 3
    # doc_a must be ranked #1 because it has rank 1 in both lists: 1/61 + 1/61
    assert fused[0]["id"] == "doc_a" or fused[0].get("chunk_id") == "doc_a"
    expected_doc_a_score = (1.0 / 61.0) + (1.0 / 61.0)
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-3) == expected_doc_a_score
    assert fused[0]["dense_rank"] == 1
    assert fused[0]["bm25_rank"] == 1

def test_retrieval_orchestrator_source_filter_enforcement():
    vec_store = FakeVectorStore()
    vec_store.add_documents([
        {"id": "c1", "text": "Project roadmap details", "source": "allowed.pdf", "page": 1},
        {"id": "c2", "text": "Classified executive notes", "source": "forbidden.pdf", "page": 1},
    ])
    
    # Simple fake bm25 store
    class FakeBM25:
        def search(self, query, n_results=20, source_filter=None):
            docs = [
                {"chunk_id": "c1", "text": "Project roadmap details", "source": "allowed.pdf", "page": 1},
                {"chunk_id": "c2", "text": "Classified executive notes", "source": "forbidden.pdf", "page": 1},
            ]
            if source_filter:
                return [d for d in docs if d["source"] in source_filter]
            return docs

    orchestrator = RetrievalOrchestrator(
        vector_store=vec_store,
        bm25_store=FakeBM25(),
        graph_store=FakeGraphStore()
    )

    result = orchestrator.retrieve(
        query="roadmap",
        query_type="HYBRID",
        top_k=5,
        source_filter=["allowed.pdf"]
    )

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 1
    assert result.chunks[0]["source"] == "allowed.pdf"
    assert result.metadata["source_filter"] == ["allowed.pdf"]

def test_structural_graph_retrieval_separate_from_chunks():
    vec_store = FakeVectorStore()
    vec_store.add_documents([
        {"id": "c1", "text": "Alpha connects to Beta in system architecture.", "source": "sys.pdf", "page": 1}
    ])

    graph_store = FakeGraphStore()
    graph_store.add_triples(
        [{"subject": "Alpha", "relation": "connects_to", "object": "Beta"}],
        source="sys.pdf",
        page=1,
        chunk_id="c1",
        text="Alpha connects to Beta in system architecture."
    )

    orchestrator = RetrievalOrchestrator(
        vector_store=vec_store,
        bm25_store=None,
        graph_store=graph_store
    )

    # Complex query should trigger graph traversal
    result = orchestrator.retrieve(query="Alpha", query_type="COMPLEX", top_k=3)
    
    # Chunks are text documents
    assert len(result.chunks) >= 1
    # Graph edges are structural relations and NOT merged into chunk list
    assert len(result.graph_edges) >= 1
    assert result.graph_edges[0]["relation"] == "connects_to"
    assert result.metadata["graph_edge_count"] >= 1
