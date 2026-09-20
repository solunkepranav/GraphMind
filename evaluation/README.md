# GraphMind Retrieval Evaluation Benchmark

This directory contains the automated, quantitative evaluation framework for measuring retrieval performance across architectural upgrades.

## 1. Ground Truth Dataset Schema (`evaluation/questions.jsonl`)

The ground truth dataset uses an evidence-and-source schema rather than brittle chunk IDs:

```json
{
  "id": "q01",
  "question": "What is GraphMind's core architectural approach to document retrieval?",
  "type": "hybrid",
  "gold_sources": ["architecture.md"],
  "gold_pages": [1],
  "gold_evidence": ["hybrid retrieval", "knowledge graph", "dense embeddings"],
  "gold_answer": "GraphMind combines dense vector search with lexical BM25 and knowledge graph traversal using Reciprocal Rank Fusion."
}
```

### Query Types
- `simple`: Direct factual inquiry, single document target.
- `keyword`: Technical codes, IDs, or exact phrases (e.g. error codes).
- `complex`: Multi-hop relation reasoning across entities/documents.
- `global`: Broad corpus-level summaries or thematic questions.
- `hybrid`: Requiring both textual factual evidence and relational graph context.

---

## 2. Quantitative Evaluation Metrics

All metrics are calculated automatically without LLM-judge bias:
- **Recall@K (K=3, 5, 10)**: Fraction of required gold sources retrieved in top K.
- **HitRate@K (K=3, 5, 10)**: Binary indicator whether any gold source or evidence appeared in top K.
- **MRR (Mean Reciprocal Rank)**: $1 / \text{rank}$ of the highest-ranked relevant chunk.
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Graded relevance score accounting for exact rank positions.

---

## 3. Running the Benchmark

Execute the automated evaluation suite:
```bash
python evaluation/run_eval.py --questions evaluation/questions.jsonl --output evaluation/benchmark_results.json
```
