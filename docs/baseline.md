# GraphMind Safety Baseline & Architecture Documentation (Phase 0)

*Date: 2026-09-20*  
*Status: Established Baseline (Pre-Upgrade)*

This document captures the exact operational state, architectural workflows, configuration parameters, data persistence layouts, and known limitations of the GraphMind codebase prior to the 18-phase enterprise upgrade.

---

## 1. System Architecture & Current Retrieval Flow

The existing system operates as a hybrid retrieval prototype integrating ChromaDB vector search and NetworkX knowledge graph traversal, coordinated through Streamlit.

### 1.1 Query Routing & Processing Flow
1. **User Query Input**: Submitted via the Streamlit interface (`app.py`).
2. **Intent Classification / Routing (`src/agents.py:QueryAgent.route_query`)**:
   - Classifies query into one of four categories via `src.config.ROUTER_PROMPT`:
     - `SIMPLE`: Direct factual inquiry, single document target.
     - `COMPLEX`: Multi-hop relation reasoning across entities/documents.
     - `GLOBAL`: High-level summarization or corpus-wide theme extraction.
     - `HYBRID`: Combined factual retrieval and relationship analysis.
3. **Retrieval Execution (`src/agents.py:QueryAgent.answer_query`)**:
   - **`SIMPLE`**: Calls `vector_store.search(query, n_results=5, source_filter=sources)`. Graph traversal is omitted (`graph_context = "No graph traversal needed for simple factual queries."`).
   - **`COMPLEX`**: Extracts named entities from the query using heuristic regex / prompt keywords. For each entity, executes `graph_store.traverse_subgraph(entity, depth=2)`. Also executes vector search (`n_results=3`).
   - **`GLOBAL`**: Retrieves top central nodes and community summaries from `graph_store.get_global_summary()`. Also calls `vector_store.search(query, n_results=5)`.
   - **`HYBRID`**: Executes both `vector_store.search(query, n_results=5)` and entity-based `graph_store.traverse_subgraph(entity, depth=2)` with up to 15 edges.
4. **Context Assembly**:
   - Vector context is formatted as:
     `"[Doc: {source}, Page: {page}]\n{text}\n---"`
   - Graph context is formatted as:
     `"[Relation: {u} -> {relation} -> {v}] (Sources: {sources})"`
5. **Generation & Verification**:
   - `QA_PROMPT` synthesizes the response using `src.llm.generate_text(prompt, task="reasoning")`.
   - `MistakeLedger.evaluate_answer()` provides an LLM-as-a-judge factuality and citation check, assigning a confidence score (0.0 - 1.0).

---

## 2. Configuration & Environment Variables

All settings are configured in `src/config.py` and loaded from environment variables or `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `ollama` | Active provider (`ollama` or `gemini`) |
| `OLLAMA_BASE_URL`| `http://localhost:11434` | Ollama HTTP server endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Base Ollama model name fallback |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama text embedding model |
| `GEMINI_API_KEY` | `""` (empty) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini generation model |
| `GEMINI_EMBED_MODEL`| `text-embedding-004` | Gemini embedding model |
| `CHUNK_SIZE` | `500` | Target chunk size (characters/tokens) |
| `CHUNK_OVERLAP`| `100` | Chunk overlap (characters/tokens) |

---

## 3. Model Calls & Task Tiering (`src/llm.py`)

The application routes requests across provider-specific task tiers:

### Ollama Task Tiers
- `fast`: `gemma3:1b` (Routing, JSON formatting)
- `validation`: `gemma3:4b` (Mistake ledger validation)
- `reasoning`: `gemma3:4b` (Triple extraction, QA synthesis)
- `vision`: `moondream:latest` (Image & visual ingestion)

### Gemini Task Tiers
- `fast`, `validation`, `reasoning`, `vision`: `gemini-2.5-flash`
- Embedding: `text-embedding-004`

---

## 4. Ingestion & Chunking Parameters (`src/ingestion.py`)

- **Chunking Engine**: Hand-rolled `RecursiveCharacterTextSplitter` splitting on `["\n\n", "\n", " ", ""]`.
- **Chunk Size / Overlap**: `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 100`.
- **Chunk Metadata Schema**:
  ```python
  {
      "text": str,
      "source": str,  # filename
      "page": int     # 1-based page index
  }
  ```
- **Chunk IDs**: Deterministic format: `{source}_p{page}_c{idx}`.
- **Document Support**: PDF (`pypdf` / `pymupdf`), DOCX (`python-docx`), PPTX (`python-pptx`), TXT/MD, Images (`PIL` + vision model OCR).

---

## 5. Persistence Layout & Paths

All persistent data resides under the `data/` directory:

```text
data/
├── chromadb/                  # Persistent ChromaDB vector database files
├── graphs/
│   └── knowledge_graph.json   # Serialized NetworkX graph (node_link_data JSON)
├── mistake_ledger.json        # Evaluation mistakes & audit history
└── source_registry.json       # Document registration & metadata tracking
```

Additionally:
- `output/`: Temporary graph visualizations (`graph.html`).

---

## 6. Known Limitations (Baseline Gaps)

1. **No Lexical / BM25 Search**: Search is strictly dense vector embeddings; exact keywords, codes, or domain terms suffer from embedding drift.
2. **No Reranker**: Top chunks from dense search are passed directly to the LLM context without cross-encoder reranking.
3. **No Retrieval Orchestrator**: `src/agents.py` directly coordinates ChromaDB and NetworkX; no unified multi-stage pipeline abstraction exists.
4. **Minimal Chunk Metadata**: Chunks lack token count, section headers, previous/next chunk references, or hash digests for deduplication.
5. **No Graph Evidence Spans**: Graph edges only store `sources`, `pages`, and hit `count`, lacking verbatim sentence evidence spans, chunk IDs, and confidence scores.
6. **No Structural Retrieval Path Separation**: Graph seeds and traversal are ad-hoc without formal path scoring or cardinality-aware contradiction detection.
7. **No Automated Test Suite**: The repository lacks a `tests/` directory; no unit, integration, or regression tests currently exist.
8. **No Standard Benchmark Dataset**: Evaluation relies on a small fixed list in `src/evaluation.py` evaluated solely by LLM-as-a-judge without automated retrieval metrics (Recall@K, MRR, NDCG).
9. **No Security / Injection Defenses**: Chunks are interpolated directly into prompts without boundary guards, isolation, or indirect prompt injection checks.
