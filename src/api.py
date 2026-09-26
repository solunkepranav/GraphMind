import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.graph_store import GraphStore
from src.agents import QueryAgent
from src.source_registry import SourceRegistry
from src.permissions import AccessControlManager
from src.observability import QueryAuditLogger
from src import config
from src import llm

app = FastAPI(
    title="GraphMind Enterprise API",
    description="Headless REST API for Hybrid RAG with Knowledge Graph and Reciprocal Rank Fusion",
    version="0.2.0"
)

# Global singleton stores for the API service
_vector_store = None
_bm25_store = None
_graph_store = None
_agent = None
_registry = None
_acl = None
_audit_logger = None

def get_services():
    global _vector_store, _bm25_store, _graph_store, _agent, _registry, _acl, _audit_logger
    if _vector_store is None:
        _vector_store = VectorStore()
    if _bm25_store is None:
        _bm25_store = BM25Store()
    if _graph_store is None:
        _graph_store = GraphStore()
    if _agent is None:
        _agent = QueryAgent(_vector_store, _graph_store, bm25_store=_bm25_store)
    if _registry is None:
        _registry = SourceRegistry()
    if _acl is None:
        _acl = AccessControlManager()
    if _audit_logger is None:
        _audit_logger = QueryAuditLogger()
    return _vector_store, _bm25_store, _graph_store, _agent, _registry, _acl, _audit_logger

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[list[str]] = None
    role: Optional[str] = "admin"

class QueryResponse(BaseModel):
    answer: str
    confidence: int
    raw_confidence: int
    confidence_signals: dict
    category: str
    reasoning: str
    retrieval_metadata: dict
    contradictions: list[dict]

@app.get("/health")
def health_check(services = Depends(get_services)):
    vec, bm25, graph, agent, reg, acl, audit = services
    cfg = llm.get_active_config()
    
    return {
        "status": "ok",
        "provider": cfg["provider"],
        "node_count": graph.graph.number_of_nodes() if graph and graph.graph else 0,
        "edge_count": graph.graph.number_of_edges() if graph and graph.graph else 0,
        "bm25_corpus_size": len(bm25.corpus) if bm25 else 0,
        "registered_sources": len(reg.list_sources()) if reg else 0
    }

@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest, services = Depends(get_services)):
    vec, bm25, graph, agent, reg, acl, audit = services

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Enforce Source-Level Access Control (ACL)
    all_sources = [s["filename"] for s in reg.list_sources()] if reg else []
    effective_sources = acl.get_effective_sources(
        role=req.role or "general",
        all_sources=all_sources,
        user_selected_sources=req.source_filter
    )

    # 2. Execute Query through QueryAgent
    res = agent.answer_query(query=req.query, source_filter=effective_sources)

    # 3. Check for any contradictions in the knowledge graph
    contradictions = graph.get_contradictions() if graph else []

    # 4. Log audit trail
    meta = res.get("retrieval_metadata", {})
    audit.log_query_execution(
        query=req.query,
        category=res.get("category", "HYBRID"),
        latency_ms=meta.get("elapsed_ms", 0.0),
        chunk_count=meta.get("final_chunk_count", 0),
        edge_count=meta.get("graph_edge_count", 0),
        confidence=res.get("confidence", 0),
        sources_used=res.get("retrieval_metadata", {}).get("source_filter") or effective_sources
    )

    return QueryResponse(
        answer=res.get("answer", ""),
        confidence=res.get("confidence", 0),
        raw_confidence=res.get("raw_confidence", 0),
        confidence_signals=res.get("confidence_signals", {}),
        category=res.get("category", "HYBRID"),
        reasoning=res.get("reasoning", ""),
        retrieval_metadata=meta,
        contradictions=contradictions
    )

@app.get("/sources")
def list_sources(services = Depends(get_services)):
    _, _, _, _, reg, _, _ = services
    return {"sources": reg.list_sources() if reg else []}

@app.get("/audit")
def get_audit_trail(limit: int = 25, services = Depends(get_services)):
    _, _, _, _, _, _, audit = services
    return {"records": audit.read_recent_records(limit=limit) if audit else []}

@app.get("/contradictions")
def get_contradictions(services = Depends(get_services)):
    _, _, graph, _, _, _, _ = services
    return {"contradictions": graph.get_contradictions() if graph else []}
