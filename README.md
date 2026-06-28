# 🧠 GraphMind: Knowledge Notebook

GraphMind is an advanced local Hybrid RAG (Retrieval-Augmented Generation) and Knowledge Graph Q&A system. It leverages large language models (Ollama or Gemini) coupled with vector database retrieval (ChromaDB) and graph-based memory storage (NetworkX/Pyvis) to provide deep, contextual answers with complete source citations.

---

## 🌟 Key Features

1. **Agentic Query Routing**:
   Intelligently routes user queries into four distinct categories for optimal retrieval:
   * **SIMPLE**: Fetches precise details directly from document vector chunks (ChromaDB).
   * **COMPLEX**: Navigates multi-hop entity relationships in the Knowledge Graph.
   * **GLOBAL**: Summarizes cross-document themes and high-level structure.
   * **HYBRID**: Blends semantic text searches with relationship graph traversals.

2. **Automated Knowledge Graph Extraction**:
   Extracts entities and semantic relationships directly from uploaded documents, structuring them as triples `(Subject, Relation, Object)` and storing them dynamically.

3. **Interactive Visualizations**:
   Embeds an interactive Pyvis visual interface in the Streamlit UI to let you browse, zoom, and inspect your extracted knowledge graph network.

4. **Extensive Evaluation Dashboard**:
   Includes a benchmark dashboard to evaluate retrieval accuracy, latency, and answer quality.

---

## 🏗️ Project Structure

```
GraphMind/
├── src/
│   ├── __init__.py         # Package initializer
│   ├── agents.py           # Query routing & retrieval coordination
│   ├── config.py           # Settings, paths, and model prompt templates
│   ├── evaluation.py       # Metrics and benchmarking pipelines
│   ├── graph_store.py      # NetworkX store and pyvis visualization engine
│   ├── ingestion.py        # Document text extraction and splitting
│   ├── llm.py              # LLM connectors (Ollama & Gemini API)
│   └── vector_store.py     # ChromaDB client & vector embeddings manager
├── app.py                  # Main Streamlit dashboard interface
├── requirements.txt        # Python library dependencies
├── .env.example            # Environment variables configuration template
└── .gitignore              # Files excluded from version control
```

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.10+ installed on your system.

If you plan to run models locally, download and install **[Ollama](https://ollama.com/)** and pull your model of choice (e.g., `llama3.2` and `nomic-embed-text`):
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the configuration template to create your `.env` file:
```bash
cp .env.example .env
```
Open the `.env` file and configure your preferred provider (`ollama` or `gemini`). If you use Gemini, make sure to add your API key:
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

---

## ⚙️ Development Configs
You can fine-tune text chunking sizes, model targets, and extraction prompts in the [src/config.py](src/config.py) module.