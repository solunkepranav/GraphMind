import pytest
from src.ingestion import enrich_chunks_with_metadata

def test_enrich_chunks_with_metadata():
    raw_chunks = [
        {"text": "# Introduction\nGraphMind is a hybrid RAG system.", "page": 1},
        {"text": "It indexes documents with dense vectors and BM25.", "page": 1},
        {"text": "## Architecture\nGraph traversal complements lexical search.", "page": 2},
    ]

    enriched = enrich_chunks_with_metadata(raw_chunks, "test_doc.pdf")

    assert len(enriched) == 3

    # Check first chunk
    c0 = enriched[0]
    assert c0["chunk_id"] == "test_doc.pdf_p1_c0"
    assert c0["section_header"] == "Introduction"
    assert c0["prev_chunk_id"] is None
    assert c0["next_chunk_id"] == "test_doc.pdf_p1_c1"
    assert c0["token_count"] > 0
    assert len(c0["content_hash"]) == 64  # SHA256 hex length

    # Check second chunk inherits header
    c1 = enriched[1]
    assert c1["chunk_id"] == "test_doc.pdf_p1_c1"
    assert c1["section_header"] == "Introduction"
    assert c1["prev_chunk_id"] == "test_doc.pdf_p1_c0"
    assert c1["next_chunk_id"] == "test_doc.pdf_p2_c2"

    # Check third chunk updates header
    c2 = enriched[2]
    assert c2["chunk_id"] == "test_doc.pdf_p2_c2"
    assert c2["section_header"] == "Architecture"
    assert c2["prev_chunk_id"] == "test_doc.pdf_p1_c1"
    assert c2["next_chunk_id"] is None
