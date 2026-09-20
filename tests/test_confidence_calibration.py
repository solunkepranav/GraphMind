import pytest
from src.mistake_ledger import calibrate_confidence

def test_calibrate_confidence_well_grounded():
    # Answer with 2 citations, 4 chunks, 2 graph relations, 90 raw LLM
    answer = "GraphMind uses RRF [1] and knowledge graphs [2] for multi-hop retrieval."
    chunks = [{"source": "d1.pdf"}, {"source": "d2.pdf"}, {"source": "d3.pdf"}]
    relations = [{"relation": "features"}]

    res = calibrate_confidence(
        raw_llm_score=90,
        answer=answer,
        retrieved_chunks=chunks,
        graph_relations=relations
    )

    assert res["calibrated_score"] >= 85
    assert res["signals"]["citation_grounding"] == 1.0
    assert res["signals"]["retrieval_coverage"] == 1.0

def test_calibrate_confidence_penalizes_overconfident_uncited():
    # Model claims 99% confidence, but provides no citations and only 1 chunk
    answer = "The system is absolutely capable of quantum teleportation."
    chunks = [{"source": "spec.pdf"}]
    relations = []

    res = calibrate_confidence(
        raw_llm_score=99,
        answer=answer,
        retrieved_chunks=chunks,
        graph_relations=relations
    )

    # Must be penalized heavily due to no citations (c_cite=0.2) and low coverage
    assert res["calibrated_score"] < 50
    assert res["signals"]["citation_grounding"] == 0.2

def test_calibrate_confidence_admitted_ignorance():
    # Model admits it cannot find the answer
    answer = "I cannot find the answer in the provided documents."
    chunks = []
    relations = []

    res = calibrate_confidence(
        raw_llm_score=20,
        answer=answer,
        retrieved_chunks=chunks,
        graph_relations=relations
    )

    # Admitting ignorance gets full citation credit (not hallucinating)
    assert res["signals"]["citation_grounding"] == 1.0
