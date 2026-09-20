import os
import json
import argparse
from src.retrieval import RetrievalOrchestrator
from src.bm25_store import BM25Store
from src.vector_store import VectorStore
from src.evaluation import compute_retrieval_metrics

def load_benchmark_questions(filepath: str = "evaluation/questions.jsonl") -> list[dict]:
    questions = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def evaluate_retrieval_pipeline(
    questions: list[dict],
    orchestrator: RetrievalOrchestrator,
    k_list: list[int] = None
) -> dict:
    if k_list is None:
        k_list = [3, 5, 10]

    all_metrics = []

    for q in questions:
        query = q["question"]
        qtype = q.get("type", "hybrid").upper()
        gold_sources = q.get("gold_sources", [])
        gold_evidence = q.get("gold_evidence", [])

        # Retrieve top 10 candidates
        result = orchestrator.retrieve(query=query, query_type=qtype, top_k=max(k_list))
        q_metrics = compute_retrieval_metrics(
            retrieved_chunks=result.chunks,
            gold_sources=gold_sources,
            gold_evidence=gold_evidence,
            k_list=k_list
        )
        all_metrics.append(q_metrics)

    # Compute averages across all questions
    num_q = max(1, len(all_metrics))
    summary = {}
    
    summary["total_questions"] = len(questions)
    summary["mrr"] = round(sum(m["mrr"] for m in all_metrics) / num_q, 4)

    for k in k_list:
        summary[f"recall@{k}"] = round(sum(m[f"recall@{k}"] for m in all_metrics) / num_q, 4)
        summary[f"hit_rate@{k}"] = round(sum(m[f"hit_rate@{k}"] for m in all_metrics) / num_q, 4)
        summary[f"ndcg@{k}"] = round(sum(m[f"ndcg@{k}"] for m in all_metrics) / num_q, 4)

    return summary

def main():
    parser = argparse.ArgumentParser(description="GraphMind Quantitative Retrieval Benchmark")
    parser.add_argument("--questions", default="evaluation/questions.jsonl", help="Path to questions JSONL")
    parser.add_argument("--output", default="evaluation/benchmark_results.json", help="Path to save output metrics")
    args = parser.parse_args()

    questions = load_benchmark_questions(args.questions)
    print(f"Loaded {len(questions)} evaluation questions from {args.questions}.")

    vec_store = VectorStore()
    bm25_store = BM25Store()
    orchestrator = RetrievalOrchestrator(vector_store=vec_store, bm25_store=bm25_store)

    results = evaluate_retrieval_pipeline(questions, orchestrator)

    print("\n--- Retrieval Benchmark Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
