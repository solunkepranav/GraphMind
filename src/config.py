import os
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "chromadb")
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MISTAKE_LEDGER_PATH = os.path.join(DATA_DIR, "mistake_ledger.json")
SOURCE_REGISTRY_PATH = os.path.join(DATA_DIR, "source_registry.json")

# Ensure directories exist
for d in [DATA_DIR, DB_DIR, GRAPH_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Default API and Model Config
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # 'ollama' or 'gemini'
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# Chunking Config — reduced for parallel dual-stream VRAM safety (2x gemma3:1b @ 4GB RTX 3050)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Prompts
ENTITY_EXTRACTION_PROMPT = """You are an expert knowledge engineer. Your task is to extract semantic entities and their relationships from the given text chunk.
Extract the triples in the form of (subject, relation/predicate, object).

Guidelines:
1. Subject and Object should be specific entities (e.g. names, places, organizations, concepts, dates).
2. Relation/Predicate should describe how the subject and object are connected (e.g. "teaches", "located_in", "author_of", "contradicts", "supports", "associated_with"). Keep relationships simple and clear.
3. Only extract relationships that are explicitly mentioned or strongly implied by the text.
4. Output your response ONLY as a valid JSON array of objects. Do not include any other text, markdown formatting (like ```json), or explanation.

Format:
[
  {{"subject": "Entity A", "relation": "relationship type", "object": "Entity B"}},
  ...
]

Text to analyze:
{text}
"""

ROUTER_PROMPT = """You are an intelligent query router for a document retrieval system.
Classify the user's query into one of the following categories:
- SIMPLE: The query asks for basic facts, direct information, or summaries located in a single document chunk (e.g. "What is the author's name?", "When was company X founded?").
- COMPLEX: The query requires connecting information, analyzing relationships between multiple entities, or multi-hop reasoning across different documents or sections (e.g. "How does project A relate to company B?", "What is the connection between person X and person Y?").
- GLOBAL: The query asks for high-level summaries, main themes, or global sensemaking of the entire document collection (e.g. "Summarize the main themes of all documents", "What are the common issues discussed?").
- HYBRID: The query requires both factual details (vector search) and relationship analysis (graph traversal) to answer comprehensively.

Output your response ONLY as a JSON object with two fields: "category" (one of SIMPLE, COMPLEX, GLOBAL, HYBRID) and "reasoning" (a short explanation). Do not use markdown wrappers.

Query: {query}
"""

QA_PROMPT = """You are GraphMind, an advanced Q&A system. Answer the user's question using the provided context, which includes both relevant text chunks and knowledge graph relations.
You MUST provide source citations for your answers.

Context Chunks:
{vector_context}

Knowledge Graph Relations:
{graph_context}

Question: {question}

Instructions:
1. Rely ONLY on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
2. Cite your sources.
   - For text chunks, cite them like this: [Doc: DocumentName, Page: PageNum]
   - For graph relations, cite them like this: [Relation: Subject -> predicate -> Object]
3. Synthesize a coherent, professional answer in markdown.
4. Be precise and truthful. Do not hallucinate.
"""
