# 🧠 GraphMind 2.0: Enterprise Hybrid RAG & Knowledge Graph Engine

GraphMind is an advanced local Hybrid RAG (Retrieval-Augmented Generation) and Knowledge Graph reasoning system. Inspired by **NotebookLM**, it combines lexical search, dense vector embeddings, cross-encoder neural reranking, graph traversal, and prompt injection defense into a unified offline-first architecture.

Complete architectural details and data-flow diagrams are documented in [docs/architecture.md](docs/architecture.md).

---

## 🌟 Key Features

### 1. 🔍 Dual-Path Hybrid Retrieval & Reranking
* **Lexical Search (BM25Plus)**: Exact keyword and token-preserved code/identifier matching using strictly positive IDF weighting.
* **Semantic Vectors (ChromaDB + Nomic)**: Deep semantic retrieval via `nomic-embed-text`.
* **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense candidates using standard RRF ($k=60$).
* **Cross-Encoder Neural Reranking**: CPU-offloaded `cross-encoder/ms-marco-MiniLM-L6-v2` reranks top candidates for precision while protecting GPU VRAM.
* **Structural Graph Retrieval**: Separated from chunk scoring; traverses the Knowledge Graph via depth-decay path ranking (`score = (support * 2) / (depth + 1)`) to pull relational context.

### 2. 🕸️ Knowledge Graph & Contradiction Detection
* **Verbatim Evidence Spans**: Graph edges track exact text quotes, source file, page number, and chunk ID.
* **Cardinality Constraints**: Enforces domain relationships (e.g. `FOUNDED_IN` has `one` cardinality).
* **Contradiction Alerts**: Automatically identifies conflicting triples across ingested sources and alerts the user in both the UI and REST API.

### 3. 🎯 Calibrated Confidence Scoring
* Replaces uncalibrated LLM self-scoring with a deterministic 4-signal weighted formula:
  $$\text{Confidence} = 0.35 \times \text{Citation Grounding} + 0.25 \times \text{Retrieval Coverage} + 0.20 \times \text{Graph Evidence} + 0.20 \times \text{Model Self-Score}$$
* Penalizes hallucinated citations or ungrounded assertions while rewarding factual source alignment.

### 4. 🛡️ Security & Access Control
* **Prompt Injection Detection**: Scans retrieved chunks for injection patterns without mutating or stripping source text.
* **XML Context Boundary Delimiters**: Isolates untrusted text inside `<untrusted_document>` and `<retrieved_documents>` boundary tags.
* **Strict ACL Enforcement**: Ensures access rules are applied as `effective_sources = ACL_allowed ∩ user_selected`.

### 5. ⚡ Production-Ready Observability & Caching
* **Structured Query Audit Logging**: Appends timestamped JSONL records to `data/audit_log.jsonl` tracking queries, latency, sources, and calibrated confidence.
* **Thread-Safe LRU Cache**: Memory-efficient query caching with configurable TTL expiration.
* **Headless REST API**: High-performance FastAPI backend with endpoints for `/health`, `/query`, `/sources`, `/audit`, and `/contradictions`.
* **Standard Evaluation Harness**: Evaluates retrieval accuracy (`Recall@K`, `HitRate@K`, `MRR`, `NDCG@K`) over ground-truth benchmark questions.

---

## 🏗️ Project Structure

```
GraphMind/
├── docs/
│   ├── architecture.md     # Enterprise architecture specification & Mermaid diagrams
│   └── baseline.md         # Pre-upgrade safety and capabilities baseline
├── evaluation/
│   ├── README.md           # Evaluation benchmark documentation
│   ├── questions.jsonl     # 20 ground-truth questions with evidence schemas
│   └── run_eval.py         # Automated retrieval evaluation runner
├── src/
│   ├── __init__.py         # Package initializer
│   ├── agents.py           # QueryAgent & Synthesizer with calibrated confidence
│   ├── api.py              # FastAPI REST endpoints
│   ├── bm25_store.py       # BM25Plus sparse index & persistence
│   ├── cache.py            # Thread-safe LRU cache with TTL
│   ├── config.py           # Model settings, paths, and retrieval hyperparameters
│   ├── evaluation.py       # Metrics calculator (Recall, HitRate, MRR, NDCG)
│   ├── graph_store.py      # NetworkX store, path ranking, and contradiction engine
│   ├── ingestion.py        # Multi-format parsing, chunk enrichment, and hashing
│   ├── llm.py              # LLM connectors (Ollama & Gemini API)
│   ├── mistake_ledger.py   # Calibrated confidence & error ledger
│   ├── observability.py    # QueryAuditLogger JSONL recorder
│   ├── permissions.py      # AccessControlManager for source filtering
│   ├── reranker.py         # CPU CrossEncoder neural reranker
│   ├── retrieval.py        # RetrievalOrchestrator (RRF + Graph + Rerank)
│   ├── security.py         # PromptInjectionDetector & safe context formatting
│   ├── source_registry.py  # File tracking, SHA256 hashes, and deduplication
│   └── vector_store.py     # ChromaDB dense vector store
├── tests/                  # 39 model-free unit tests
├── app.py                  # Streamlit dashboard interface
├── requirements.txt        # Python library dependencies
├── pyproject.toml          # Pytest and tool configurations
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.13)
* **[Ollama](https://ollama.com/)** for local model execution.

Pull the recommended local models:
```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull moondream
ollama pull nomic-embed-text
```

### 2. Installation
Clone the repository and install dependencies in a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the configuration template:
```bash
cp .env.example .env
```
Default configuration targets local Ollama:
```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4. Running the Applications

#### Option A: Interactive Streamlit UI
```bash
streamlit run app.py
```
* Access the notebook at `http://localhost:8501`.
* Manage documents in the left sidebar, inspect active knowledge graphs, chat with cited answers, view confidence signal breakdowns, and monitor contradiction warnings.

#### Option B: Headless FastAPI Backend
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger documentation at `http://localhost:8000/docs`.
* Endpoints:
  - `GET /health` - Health status and active component readiness.
  - `POST /query` - Execute queries with source filtering, reranking, and audit logging.
  - `GET /sources` - List all registered documents and metadata.
  - `GET /audit` - Query the structured execution audit log.
  - `GET /contradictions` - List detected Knowledge Graph contradictions.

---

## 🧪 Testing & Evaluation

### Run Unit Tests
The test suite contains 39 isolated, model-free unit tests covering BM25 indexing, Cross-Encoder reranking, RRF fusion, graph traversal, confidence calibration, prompt injection defense, access control, audit logging, and FastAPI endpoints:
```bash
pytest
```

### Run Retrieval Evaluation Benchmark
Run retrieval metrics (`Recall@K`, `HitRate@K`, `MRR`, `NDCG@K`) against the 20-question ground-truth dataset:
```bash
python evaluation/run_eval.py
```

---

## ⚡ Parallel Ingestion Workers — Hardware Guide

GraphMind uses a **parallel dual-stream extraction engine** to build the Knowledge Graph efficiently without exhausting GPU VRAM:

| GPU | VRAM | Recommended Workers | Peak VRAM (est.) | VRAM Headroom |
|---|---|---|---|---|
| Intel / AMD Integrated | Shared RAM | **1 (CPU only)** | — | System RAM |
| NVIDIA GTX 1060 / RX 580 | 6 GB | **2** | ~2.6 GB | ~3.4 GB |
| NVIDIA RTX 3050 (Laptop) | 4 GB | **2** *(default)* | ~2.6 GB | ~1.4 GB |
| NVIDIA RTX 3060 / 4060 | 8–12 GB | **4** | ~4.8 GB | ~3.2 GB+ |
| NVIDIA RTX 3080 / 4080 / 4090 | 10–24 GB | **4–6** | ~5.6–6.4 GB | ~5–18 GB |
| Apple Silicon (M1/M2/M3) | Unified | **3–4** | ~3.5–4.8 GB | System Memory |

> **Note:** The Cross-Encoder neural reranker is explicitly configured to run on **CPU** (`RERANKER_DEVICE="cpu"`) so that GPU VRAM remains dedicated to local LLMs and embeddings.