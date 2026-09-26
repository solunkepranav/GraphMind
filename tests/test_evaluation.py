import pytest
from src.evaluation import calculate_chunk_relevance, compute_retrieval_metrics

def test_chunk_relevance_scoring():
    gold_sources = ["report.pdf"]
    gold_evidence = ["revenue rose 15%"]

    # Both match
    c_both = {"source": "report.pdf", "text": "In Q3, revenue rose 15% year over year."}
    assert calculate_chunk_relevance(c_both, gold_sources, gold_evidence) == 2

    # Only source matches
    c_source = {"source": "report.pdf", "text": "Unrelated boilerplate text."}
    assert calculate_chunk_relevance(c_source, gold_sources, gold_evidence) == 1

    # Only evidence matches
    c_evidence = {"source": "other.txt", "text": "revenue rose 15% according to notes."}
    assert calculate_chunk_relevance(c_evidence, gold_sources, gold_evidence) == 1

    # Neither matches
    c_none = {"source": "irrelevant.pdf", "text": "Completely unrelated content."}
    assert calculate_chunk_relevance(c_none, gold_sources, gold_evidence) == 0

def test_retrieval_metrics_perfect_ranking():
    chunks = [
        {"source": "doc1.pdf", "text": "Alpha beta gamma key phrase here."},
        {"source": "doc2.pdf", "text": "Secondary documentation text."},
    ]
    gold_sources = ["doc1.pdf"]
    gold_evidence = ["key phrase"]

    metrics = compute_retrieval_metrics(chunks, gold_sources, gold_evidence, k_list=[1, 3, 5])

    assert metrics["mrr"] == 1.0
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0

def test_retrieval_metrics_second_rank_match():
    chunks = [
        {"source": "noise.pdf", "text": "Irrelevant noise passage."},
        {"source": "doc1.pdf", "text": "Gold evidence is present here."},
    ]
    gold_sources = ["doc1.pdf"]
    gold_evidence = ["gold evidence"]

    metrics = compute_retrieval_metrics(chunks, gold_sources, gold_evidence, k_list=[1, 2])

    assert metrics["mrr"] == 0.5  # 1/2
    assert metrics["hit_rate@1"] == 0.0
    assert metrics["hit_rate@2"] == 1.0
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 1.0

def test_retrieval_metrics_partial_recall():
    chunks = [
        {"source": "doc1.pdf", "text": "Alpha content"},
        {"source": "doc3.pdf", "text": "Gamma content"},
    ]
    gold_sources = ["doc1.pdf", "doc2.pdf"]

    metrics = compute_retrieval_metrics(chunks, gold_sources, k_list=[2])
    # Only doc1.pdf is retrieved out of [doc1.pdf, doc2.pdf]
    assert metrics["recall@2"] == 0.5

def test_load_benchmark_questions():
    from src.evaluation import load_benchmark_questions, BENCHMARK_QUESTIONS
    questions = load_benchmark_questions()
    assert len(questions) >= 1
    assert "question" in questions[0]
    assert "ground_truth" in questions[0]
    assert len(BENCHMARK_QUESTIONS) >= 1

