import os
import tempfile
import pytest
from src.bm25_store import BM25Store

def test_bm25_basic_indexing_and_scoring():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "bm25_test.pkl")
        store = BM25Store(index_path=index_path)

        docs = [
            {"text": "Python is a popular programming language.", "source": "python.txt", "page": 1, "chunk_id": "c1"},
            {"text": "GraphMind utilizes hybrid retrieval with graphs.", "source": "graphmind.txt", "page": 1, "chunk_id": "c2"},
            {"text": "ChromaDB stores dense vector embeddings.", "source": "chroma.txt", "page": 1, "chunk_id": "c3"},
        ]
        store.add_documents(docs)

        results = store.search("hybrid retrieval", n_results=2)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c2"
        assert results[0]["source"] == "graphmind.txt"
        assert results[0]["score"] > 0.0

def test_bm25_exact_code_matching():
    """Verify BM25 succeeds on exact code/ID matches that dense embeddings often drift on."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "bm25_test.pkl")
        store = BM25Store(index_path=index_path)

        docs = [
            {"text": "The service failed with error ERR_503_SERVICE_UNAVAILABLE on node 4.", "source": "logs.txt", "page": 1, "chunk_id": "c_err503"},
            {"text": "The request encountered ERR_404_NOT_FOUND when accessing resource.", "source": "logs.txt", "page": 2, "chunk_id": "c_err404"},
            {"text": "Generic failure occured during operation execution.", "source": "logs.txt", "page": 3, "chunk_id": "c_generic"},
        ]
        store.add_documents(docs)

        results = store.search("ERR_404_NOT_FOUND", n_results=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c_err404"
        assert "ERR_404_NOT_FOUND" in results[0]["text"]

def test_bm25_source_filtering():
    """Verify source_filter strictly isolates retrieval to permitted sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "bm25_test.pkl")
        store = BM25Store(index_path=index_path)

        docs = [
            {"text": "Confidential financial report for Q3 with revenue growth.", "source": "financials.pdf", "page": 1, "chunk_id": "c_fin"},
            {"text": "Public engineering roadmap for Q3 with platform features.", "source": "public_roadmap.pdf", "page": 1, "chunk_id": "c_pub"},
        ]
        store.add_documents(docs)

        # Filter for public_roadmap.pdf only
        results = store.search("revenue Q3", n_results=5, source_filter=["public_roadmap.pdf"])
        assert len(results) == 1
        assert results[0]["source"] == "public_roadmap.pdf"

        # Filter for financials.pdf only
        results_fin = store.search("revenue Q3", n_results=5, source_filter=["financials.pdf"])
        assert len(results_fin) == 1
        assert results_fin[0]["source"] == "financials.pdf"

def test_bm25_persistence_and_reload():
    """Verify state persists to disk and reloads accurately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "bm25_test.pkl")
        store = BM25Store(index_path=index_path)

        docs = [
            {"text": "Knowledge graphs represent entities as nodes and relationships as edges.", "source": "kg.md", "page": 1, "chunk_id": "c_kg"},
        ]
        store.add_documents(docs)

        # Create new store instance pointing to same file
        reloaded_store = BM25Store(index_path=index_path)
        assert len(reloaded_store.corpus) == 1

        results = reloaded_store.search("entities relationships", n_results=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c_kg"

def test_bm25_delete_source():
    """Verify deleting a source removes its chunks and keeps other chunks indexed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "bm25_test.pkl")
        store = BM25Store(index_path=index_path)

        docs = [
            {"text": "Document Alpha text about algorithms.", "source": "alpha.pdf", "page": 1, "chunk_id": "c_a"},
            {"text": "Document Beta text about databases.", "source": "beta.pdf", "page": 1, "chunk_id": "c_b"},
        ]
        store.add_documents(docs)
        assert len(store.corpus) == 2

        store.delete_source("alpha.pdf")
        assert len(store.corpus) == 1
        assert store.corpus[0]["source"] == "beta.pdf"

        # Search for Alpha content should return empty
        results_alpha = store.search("algorithms", n_results=5)
        assert len(results_alpha) == 0

        # Search for Beta content should return result
        results_beta = store.search("databases", n_results=5)
        assert len(results_beta) == 1
        assert results_beta[0]["chunk_id"] == "c_b"
