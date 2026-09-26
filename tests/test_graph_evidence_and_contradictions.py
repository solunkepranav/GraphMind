import pytest
import networkx as nx
from unittest.mock import patch
from src.graph_store import GraphStore

def test_evidence_span_tracking(tmp_path):
    store = GraphStore()
    store.graph_path = str(tmp_path / "test_graph.json")
    store.graph = nx.DiGraph()
    store.contradictions = []

    # Mock LLM extraction
    with patch("src.llm.generate_json") as mock_extract:
        mock_extract.return_value = [
            {"subject": "GraphMind", "relation": "features", "object": "Hybrid RAG"}
        ]
        chunk = {
            "text": "GraphMind features Hybrid RAG with knowledge graph integration.",
            "source": "paper.pdf",
            "page": 1,
            "chunk_id": "paper.pdf_p1_c0"
        }
        store.add_relations_from_chunk(chunk)

    assert store.graph.has_edge("Graphmind", "Hybrid Rag")
    edge = store.graph.edges["Graphmind", "Hybrid Rag"]
    assert "evidence" in edge
    assert len(edge["evidence"]) == 1
    assert edge["evidence"][0]["source"] == "paper.pdf"
    assert edge["evidence"][0]["chunk_id"] == "paper.pdf_p1_c0"
    assert "GraphMind features" in edge["evidence"][0]["text"]
    assert edge["support_count"] == 1
    assert edge["source_count"] == 1

def test_contradiction_detection_for_one_cardinality(tmp_path):
    store = GraphStore()
    store.graph_path = str(tmp_path / "test_graph.json")
    store.graph = nx.DiGraph()
    store.contradictions = []
    
    with patch("src.llm.generate_json") as mock_extract:
        # Document 1 claims headquartered in Seattle
        mock_extract.return_value = [
            {"subject": "TechCorp", "relation": "headquartered_in", "object": "Seattle"}
        ]
        chunk1 = {"text": "TechCorp was headquartered in Seattle.", "source": "filing_2020.pdf", "page": 1}
        store.add_relations_from_chunk(chunk1)

        # Document 2 claims headquartered in Austin
        mock_extract.return_value = [
            {"subject": "TechCorp", "relation": "headquartered_in", "object": "Austin"}
        ]
        chunk2 = {"text": "TechCorp is headquartered in Austin.", "source": "filing_2024.pdf", "page": 1}
        store.add_relations_from_chunk(chunk2)

    contradictions = store.get_contradictions()
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c["subject"] == "Techcorp"
    assert c["relation"] == "headquartered_in"
    assert c["existing_object"] == "Seattle"
    assert c["new_object"] == "Austin"
    assert "filing_2020.pdf" in c["existing_sources"]
    assert c["new_source"] == "filing_2024.pdf"

def test_no_contradiction_for_many_cardinality(tmp_path):
    store = GraphStore()
    store.graph_path = str(tmp_path / "test_graph.json")
    store.graph = nx.DiGraph()
    store.contradictions = []
    
    with patch("src.llm.generate_json") as mock_extract:
        # Document 1 claims author wrote book A
        mock_extract.return_value = [
            {"subject": "AuthorX", "relation": "author_of", "object": "Book One"}
        ]
        store.add_relations_from_chunk({"text": "AuthorX wrote Book One", "source": "d1.pdf", "page": 1})

        # Document 2 claims author wrote book B (valid for "many" cardinality)
        mock_extract.return_value = [
            {"subject": "AuthorX", "relation": "author_of", "object": "Book Two"}
        ]
        store.add_relations_from_chunk({"text": "AuthorX wrote Book Two", "source": "d2.pdf", "page": 1})

    # Should have 0 contradictions because an author can write many books
    contradictions = store.get_contradictions()
    assert len(contradictions) == 0

def test_evidence_and_contradiction_cleanup_on_delete_source(tmp_path):
    store = GraphStore()
    store.graph_path = str(tmp_path / "test_graph.json")
    store.graph = nx.DiGraph()
    store.contradictions = []
    
    with patch("src.llm.generate_json") as mock_extract:
        # doc1 creates edge with evidence
        mock_extract.return_value = [
            {"subject": "CompanyA", "relation": "founded_in", "object": "2010"}
        ]
        store.add_relations_from_chunk({"text": "CompanyA founded in 2010", "source": "src1.pdf", "page": 1})

        # doc2 supports same edge with second evidence
        mock_extract.return_value = [
            {"subject": "CompanyA", "relation": "founded_in", "object": "2010"}
        ]
        store.add_relations_from_chunk({"text": "CompanyA was established in 2010", "source": "src2.pdf", "page": 1})

        # doc3 creates contradiction
        mock_extract.return_value = [
            {"subject": "CompanyA", "relation": "founded_in", "object": "2015"}
        ]
        store.add_relations_from_chunk({"text": "CompanyA founded in 2015", "source": "src3.pdf", "page": 1})

    edge = store.graph.edges["Companya", "2010"]
    assert len(edge["evidence"]) == 2
    assert edge["support_count"] == 2
    assert edge["source_count"] == 2
    assert len(store.get_contradictions()) == 1

    # Delete src1.pdf: edge should still exist via src2.pdf, but evidence from src1.pdf removed
    store.delete_by_source("src1.pdf")
    edge = store.graph.edges["Companya", "2010"]
    assert len(edge["evidence"]) == 1
    assert edge["evidence"][0]["source"] == "src2.pdf"
    assert edge["support_count"] == 1
    assert edge["source_count"] == 1
    assert "src1.pdf" not in edge["sources"]

    # Delete src3.pdf: contradiction referencing src3.pdf should be removed
    store.delete_by_source("src3.pdf")
    assert len(store.get_contradictions()) == 0

def test_contradiction_persistence_save_load(tmp_path):
    store = GraphStore()
    store.graph_path = str(tmp_path / "test_graph.json")
    store.graph = nx.DiGraph()
    store.contradictions = []

    with patch("src.llm.generate_json") as mock_extract:
        mock_extract.return_value = [
            {"subject": "EntityX", "relation": "ceo_of", "object": "Alice"}
        ]
        store.add_relations_from_chunk({"text": "Alice is CEO of EntityX", "source": "d1.pdf", "page": 1})

        mock_extract.return_value = [
            {"subject": "EntityX", "relation": "ceo_of", "object": "Bob"}
        ]
        store.add_relations_from_chunk({"text": "Bob is CEO of EntityX", "source": "d2.pdf", "page": 1})

    assert len(store.get_contradictions()) == 1
    store.save()

    # Load in a fresh store instance pointing to same file
    store2 = GraphStore()
    store2.graph_path = str(tmp_path / "test_graph.json")
    store2.load()

    assert len(store2.get_contradictions()) == 1
    c = store2.get_contradictions()[0]
    assert c["subject"] == "Entityx"
    assert c["existing_object"] == "Alice"
    assert c["new_object"] == "Bob"


