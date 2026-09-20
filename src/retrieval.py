import time
import hashlib
from dataclasses import dataclass, field
from src import config

@dataclass
class RetrievalResult:
    """Structured result returned by the RetrievalOrchestrator."""
    chunks: list[dict] = field(default_factory=list)
    graph_edges: list[dict] = field(default_factory=list)
    query_type: str = "HYBRID"
    metadata: dict = field(default_factory=dict)


def get_chunk_key(chunk: dict) -> str:
    """Generate a consistent chunk identifier for deduplication."""
    cid = chunk.get("id") or chunk.get("chunk_id")
    if cid:
        return str(cid)
    source = chunk.get("source", "unknown")
    page = chunk.get("page", 1)
    text_snippet = (chunk.get("text") or "").strip()[:64]
    text_hash = hashlib.md5(text_snippet.encode("utf-8")).hexdigest()[:8]
    return f"{source}_p{page}_{text_hash}"


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """Combines dense and lexical BM25 results using Reciprocal Rank Fusion (RRF).
    Formula: RRF(d) = sum_{m in M} 1 / (k + rank_m(d)) where rank_m is 1-indexed.
    """
    merged: dict[str, dict] = {}

    # Score dense results
    for rank, doc in enumerate(dense_results, start=1):
        key = get_chunk_key(doc)
        if key not in merged:
            merged[key] = {
                "chunk": dict(doc),
                "rrf_score": 0.0,
                "dense_rank": rank,
                "bm25_rank": None
            }
        else:
            merged[key]["dense_rank"] = rank
        merged[key]["rrf_score"] += 1.0 / (k + rank)

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        key = get_chunk_key(doc)
        if key not in merged:
            merged[key] = {
                "chunk": dict(doc),
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": rank
            }
        else:
            merged[key]["bm25_rank"] = rank
        merged[key]["rrf_score"] += 1.0 / (k + rank)

    # Convert to list and sort descending by rrf_score
    fused_list = []
    for item in merged.values():
        chunk_data = dict(item["chunk"])
        chunk_data["rrf_score"] = float(item["rrf_score"])
        chunk_data["dense_rank"] = item["dense_rank"]
        chunk_data["bm25_rank"] = item["bm25_rank"]
        fused_list.append(chunk_data)

    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_list


from src.reranker import CrossEncoderReranker

class RetrievalOrchestrator:
    """Coordinates hybrid retrieval: Dense embeddings + BM25 -> RRF -> Reranker -> Graph Subgraph."""

    def __init__(
        self,
        vector_store,
        bm25_store=None,
        graph_store=None,
        reranker=None,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.graph_store = graph_store
        if reranker is not None:
            self.reranker = reranker
        elif getattr(config, "ENABLE_RERANKER", False):
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        query_type: str = "HYBRID",
        top_k: int = 5,
        source_filter: list[str] = None
    ) -> RetrievalResult:
        """Executes multi-stage retrieval matching the query type.
        Graph traversal is handled as a separate structural retrieval path.
        """
        start_time = time.perf_counter()
        
        # 1. Parallel dense and BM25 retrieval
        dense_top_k = getattr(config, "DENSE_TOP_K", 20)
        bm25_top_k = getattr(config, "BM25_TOP_K", 20)

        dense_results = []
        if self.vector_store:
            try:
                dense_results = self.vector_store.search(
                    query,
                    top_k=dense_top_k,
                    source_filter=source_filter
                )
            except Exception as e:
                print(f"Warning: Dense search failed: {e}")

        bm25_results = []
        if self.bm25_store:
            try:
                bm25_results = self.bm25_store.search(
                    query,
                    n_results=bm25_top_k,
                    source_filter=source_filter
                )
            except Exception as e:
                print(f"Warning: BM25 search failed: {e}")

        # 2. Reciprocal Rank Fusion
        fused_chunks = reciprocal_rank_fusion(dense_results, bm25_results, k=self.rrf_k)

        # 3. Optional Reranking (Phase 4 integration)
        if self.reranker and fused_chunks:
            try:
                # Take top RRF candidates for reranking
                candidate_pool = fused_chunks[:top_k * 2]
                ranked_chunks = self.reranker.rerank(query, candidate_pool, top_n=top_k)
            except Exception as e:
                print(f"Warning: Reranker failed, falling back to RRF ordering: {e}")
                ranked_chunks = fused_chunks[:top_k]
        else:
            ranked_chunks = fused_chunks[:top_k]

        # 4. Structural Knowledge Graph Retrieval (Separate path)
        graph_relations = []
        if self.graph_store and query_type in ["COMPLEX", "GLOBAL", "HYBRID"]:
            try:
                if query_type == "GLOBAL":
                    # Global: high-degree hub nodes
                    if hasattr(self.graph_store, "graph") and self.graph_store.graph:
                        degrees = self.graph_store.graph.degree()
                        hub_nodes = [node for node, deg in sorted(degrees, key=lambda x: x[1], reverse=True)[:5]]
                        graph_relations = self.graph_store.traverse_subgraph(
                            hub_nodes,
                            max_depth=1,
                            source_filter=source_filter
                        )
                else:
                    # Complex / Hybrid: seed-based traversal
                    seeds = []
                    if hasattr(self.graph_store, "find_seeds_in_query"):
                        seeds = self.graph_store.find_seeds_in_query(query)
                    
                    # If no seeds directly in query, extract seeds from top retrieved chunks
                    if not seeds and ranked_chunks and hasattr(self.graph_store, "find_seeds_in_query"):
                        for chunk in ranked_chunks[:2]:
                            chunk_seeds = self.graph_store.find_seeds_in_query(chunk.get("text", ""))
                            seeds.extend(chunk_seeds)
                        seeds = list(set(seeds))
                    
                    if seeds and hasattr(self.graph_store, "traverse_subgraph"):
                        graph_relations = self.graph_store.traverse_subgraph(
                            seeds,
                            max_depth=2,
                            source_filter=source_filter
                        )
            except Exception as e:
                print(f"Warning: Graph traversal failed: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        metadata = {
            "elapsed_ms": round(elapsed_ms, 2),
            "dense_retrieved_count": len(dense_results),
            "bm25_retrieved_count": len(bm25_results),
            "fused_count": len(fused_chunks),
            "final_chunk_count": len(ranked_chunks),
            "graph_edge_count": len(graph_relations),
            "source_filter": source_filter
        }

        return RetrievalResult(
            chunks=ranked_chunks,
            graph_edges=graph_relations,
            query_type=query_type,
            metadata=metadata
        )
