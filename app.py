import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="GraphMind - Knowledge Notebook",
    page_icon="○",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist Developer & Systems Engineering Interface
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background: Pitch black #080808 with subtle technical matrix grid */
.stApp {
    background-color: #080808;
    background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 0);
    background-size: 24px 24px;
    color: #e5e5e5;
}

/* Sidebar: Deep charcoal #0d0d0d with crisp hairline border */
[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #262626 !important;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #737373;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

/* Headings: Ultra-bold grotesque with tight tracking */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: -0.035em !important;
}

/* Technical Eyebrows / Stamped Metadata */
.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    background-color: #141414;
    border: 1px solid #262626;
    color: #a3a3a3;
    border-radius: 2px;
}

.tech-mono-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #737373;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Bento Cards: Pitch black panels with 1px border & corner crosshair (+) */
.premium-card {
    position: relative;
    background-color: #111111;
    border: 1px solid #262626;
    border-radius: 4px;
    padding: 1.35rem;
    margin-bottom: 1rem;
    box-shadow: none !important;
    transition: border-color 0.15s ease;
}

.premium-card:hover {
    border-color: #737373;
}

.premium-card::after {
    content: "+";
    position: absolute;
    top: 6px;
    right: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #525252;
}

/* High-contrast solid CTA buttons (White on Black with hover inversion) */
.stButton>button {
    background-color: #ffffff !important;
    color: #080808 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border: 1px solid #ffffff !important;
    border-radius: 2px !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}

.stButton>button:hover {
    background-color: #080808 !important;
    color: #ffffff !important;
    border-color: #ffffff !important;
}

.stButton>button:active {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

/* Secondary Button Style */
.stButton>button[kind="secondary"], .stButton>button:has(div:contains("Clear")), .stButton>button:has(div:contains("Delete")), .stButton>button:has(div:contains("Reset")) {
    background-color: #141414 !important;
    color: #737373 !important;
    border: 1px solid #262626 !important;
}

.stButton>button[kind="secondary"]:hover, .stButton>button:has(div:contains("Clear")):hover, .stButton>button:has(div:contains("Delete")):hover, .stButton>button:has(div:contains("Reset")):hover {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    border-color: #737373 !important;
}

/* Metric Strips: Crisp key-value blocks */
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', 'JetBrains Mono', monospace !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.85rem !important;
    letter-spacing: -0.03em !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #737373 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Tabs: Minimalist Terminal IDE styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #262626;
    padding: 0;
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #737373 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.65rem 1.4rem !important;
    border-radius: 0 !important;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
    background: transparent !important;
}

/* Chat Messages: Clean console panels */
[data-testid="stChatMessage"] {
    background-color: #111111 !important;
    border: 1px solid #262626 !important;
    border-radius: 2px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
    box-shadow: none !important;
}

/* Input / Selectbox styling */
.stTextInput>div>div>input, .stSelectbox>div>div, .stTextArea>div>div>textarea {
    background-color: #111111 !important;
    border: 1px solid #262626 !important;
    color: #ffffff !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: #737373 !important;
    box-shadow: none !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #111111 !important;
    border: 1px solid #262626 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #a3a3a3 !important;
}

/* Progress bar: Stark solid white on dark rail */
.stProgress>div>div>div>div {
    background-color: #ffffff !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

.stProgress>div>div>div {
    background-color: #1e1e1e !important;
    border-radius: 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 4px;
    height: 4px;
}
::-webkit-scrollbar-track {
    background: #080808;
}
::-webkit-scrollbar-thumb {
    background: #262626;
}
::-webkit-scrollbar-thumb:hover {
    background: #525252;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #111111 !important;
    border: 1px dashed #262626 !important;
    border-radius: 2px !important;
}

/* Alert / Toast */
.stAlert {
    background-color: #111111 !important;
    border: 1px solid #262626 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Status container */
[data-testid="stStatusWidget"] {
    background-color: #111111 !important;
    border: 1px solid #262626 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Checkbox visual styling */
div[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
    border-radius: 2px !important;
}

div[data-testid="stCheckbox"] svg {
    stroke: #080808 !important;
    fill: #080808 !important;
}
</style>
""", unsafe_allow_html=True)

import hashlib
import time

# Imports from src
from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.graph_store import GraphStore
from src.agents import QueryAgent
from src import ingestion
from src import llm
from src.source_registry import SourceRegistry
from src.observability import QueryAuditLogger

# Initialize session state for DB & Graph connections
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "bm25_store" not in st.session_state:
    st.session_state.bm25_store = BM25Store()
if "graph_store" not in st.session_state:
    st.session_state.graph_store = GraphStore()
if "query_agent" not in st.session_state:
    st.session_state.query_agent = QueryAgent(
        st.session_state.vector_store,
        st.session_state.graph_store,
        bm25_store=st.session_state.bm25_store
    )
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = QueryAuditLogger()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
registry = SourceRegistry()

# Sidebar Layout - NotebookLM style (sources first)
st.sidebar.markdown("### Sources")

# NotebookLM Style: "+ Add Source" in the sidebar at the very top!
st.sidebar.markdown("### + Add Source")
uploaded_files = st.sidebar.file_uploader(
    "Upload files to your notebook",
    type=["pdf", "docx", "pptx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="sidebar_uploader",
    label_visibility="collapsed"
)

if st.sidebar.button("Ingest & Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one document.")
    else:
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        total_files = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files):
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            duplicate = registry.is_duplicate_file(file_hash)
            if duplicate:
                st.sidebar.warning(f"Skipping '{uploaded_file.name}': identical file already ingested as '{duplicate['filename']}'.")
                continue

            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            file_header = f"[{idx+1}/{total_files}] {uploaded_file.name}"
            status_text.markdown(f"📄 **Parsing text & layout:** `{file_header}`...")
            progress_bar.progress(0.05)
            
            try:
                # 1. Ingest text chunks
                chunks = ingestion.ingest_file(file_path)
                total_chunks = len(chunks)
                progress_bar.progress(0.12)
                
                # 2. Dense Embeddings
                status_text.markdown(f"🔢 **Computing vector embeddings:** `{file_header}` ({total_chunks} chunks)...")
                st.session_state.vector_store.add_chunks(chunks)
                progress_bar.progress(0.18)
                
                # 3. BM25 Lexical Index
                status_text.markdown(f"🔤 **Indexing lexical keywords:** `{file_header}`...")
                st.session_state.bm25_store.add_documents(chunks)
                progress_bar.progress(0.22)
                
                # 4. Knowledge Graph with live chunk ETA callback
                workers = st.session_state.get("max_workers", 2)
                def on_kg_progress(completed, total, elapsed):
                    pct_complete = completed / max(1, total)
                    scaled_progress = 0.22 + (pct_complete * 0.76)
                    progress_bar.progress(min(0.98, scaled_progress))
                    
                    rate = completed / max(0.1, elapsed)
                    remaining = max(0, int((total - completed) / max(0.01, rate)))
                    mins, secs = divmod(remaining, 60)
                    eta_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
                    status_text.markdown(
                        f"🕸️ **Knowledge Graph:** `{file_header}`\n\n"
                        f"`{completed}/{total}` chunks ({int(pct_complete*100)}%) • **ETA:** ~{eta_str} ({rate:.1f} chunks/s, {workers} streams)"
                    )

                status_text.markdown(f"🕸️ **Extracting Knowledge Graph:** `{file_header}` ({total_chunks} chunks, {workers} streams)...")
                st.session_state.graph_store.add_relations_from_chunks_parallel(
                    chunks,
                    max_workers=workers,
                    progress_callback=on_kg_progress
                )
                progress_bar.progress(1.0)
                
                # Register source in registry
                page_count = max([c.get("page", 1) for c in chunks]) if chunks else 1
                registry.register_source(
                    filename=uploaded_file.name,
                    file_type=uploaded_file.name.split('.')[-1].lower(),
                    chunk_count=len(chunks),
                    page_count=page_count,
                    size_bytes=uploaded_file.size,
                    file_hash=file_hash
                )
            except Exception as e:
                st.sidebar.error(f"Error ingesting {uploaded_file.name}: {e}")
            
            try:
                os.remove(file_path)
            except Exception:
                pass
                
        st.session_state.graph_store.save()
        status_text.markdown("✅ **Ingestion complete!**")
        progress_bar.progress(1.0)
        st.sidebar.success("Ingestion complete!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Active Sources")

sources = registry.list_sources()
selected_sources = []

if not sources:
    st.sidebar.info("No sources ingested yet.")
else:
    select_all = st.sidebar.checkbox("Select All Sources", value=True)
    for src in sources:
        is_selected = st.sidebar.checkbox(
            src["filename"],
            value=select_all,
            key=f"select_{src['id']}"
        )
        if is_selected:
            selected_sources.append(src["filename"])
            
        with st.sidebar.expander(f"Source: {src['filename']}", expanded=False):
            st.write(f"**Chunks:** {src['chunk_count']}")
            st.write(f"**Pages/Slides:** {src['page_count']}")
            st.write(f"**Size:** {src['size_bytes'] / 1024:.1f} KB")
            st.write(f"**Ingested:** {src['ingested_at'][:10]}")
            
            # Show preview
            preview = st.session_state.vector_store.get_source_preview(src["filename"])
            st.text_area("Preview Content", preview, height=120, disabled=True, key=f"prev_{src['id']}")
            
            if st.button("Delete Source", key=f"del_{src['id']}", type="secondary"):
                filename = registry.delete_source(src["id"])
                if filename:
                    st.session_state.vector_store.delete_by_source(filename)
                    st.session_state.graph_store.delete_by_source(filename)
                    st.session_state.bm25_store.delete_source(filename)
                    st.toast(f"Deleted source: {filename}")
                    st.rerun()

st.sidebar.markdown("---")

# Collapsed Settings Panel at the bottom
with st.sidebar.expander("Settings", expanded=False):
    # Select LLM Provider
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["Ollama", "Gemini API"],
        index=0 if os.getenv("LLM_PROVIDER", "ollama") == "ollama" else 1
    )

    st.session_state.llm_provider = "ollama" if llm_provider == "Ollama" else "gemini"

    if st.session_state.llm_provider == "gemini":
        gemini_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Generate a free key at Google AI Studio"
        )
        st.session_state.gemini_key = gemini_key
        st.session_state.gemini_model = "gemini-2.5-flash"
        
        st.info(
            "Auto-Routing:\n"
            "- Fast (JSON, Router): gemini-2.5-flash\n"
            "- Validation (OKF Critic): gemini-2.5-flash\n"
            "- Reasoning (Q&A): gemini-2.5-flash\n"
            "- Vision (Multimodal): gemini-2.5-flash\n"
            "- Embedding (Vectors): text-embedding-004"
        )
    else:
        ollama_url = st.text_input("Ollama Endpoint", value="http://localhost:11434")
        st.session_state.ollama_url = ollama_url
        st.session_state.ollama_model = "gemma3:4b"
        
        st.info(
            "Auto-Routing:\n"
            "- Fast (JSON, Router): gemma3:1b\n"
            "- Validation (OKF Critic): gemma3:4b\n"
            "- Reasoning (Q&A): gemma3:4b\n"
            "- Vision (Image OCR): moondream:latest\n"
            "- Embedding (Vectors): nomic-embed-text"
        )

    if st.button("Test LLM Connection"):
        with st.spinner("Connecting..."):
            success, msg = llm.test_connection()
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("#### Parallel Ingestion Workers")
    max_workers = st.slider(
        "Extraction Workers",
        min_value=1,
        max_value=6,
        value=st.session_state.get("max_workers", 2),
        help=(
            "Number of parallel streams used to extract the knowledge graph from your documents.\n"
            "Each worker loads one gemma3:1b model instance.\n"
            "Increase this if your GPU has more than 4 GB of VRAM.\n"
            "See the README Hardware Guide for recommended values per GPU."
        )
    )
    st.session_state.max_workers = max_workers
                
    st.markdown("---")
    st.markdown("#### System Statistics")
    try:
        num_nodes = st.session_state.graph_store.graph.number_of_nodes()
        num_edges = st.session_state.graph_store.graph.number_of_edges()
        num_chunks = len(st.session_state.vector_store.collection.get().get("ids", []))
    except Exception:
        num_nodes = 0
        num_edges = 0
        num_chunks = 0

    st.metric("Document Chunks", num_chunks)
    st.metric("KG Entities (Nodes)", num_nodes)
    st.metric("KG Relations (Edges)", num_edges)

    if st.button("Reset Database & Graph", type="secondary"):
        st.session_state.vector_store.reset()
        st.session_state.graph_store.reset()
        st.session_state.bm25_store.clear()
        if hasattr(st.session_state.query_agent, "clear_cache"):
            st.session_state.query_agent.clear_cache()
        registry.reset()
        st.toast("Database, Graph, BM25 Index, and Source Registry reset successfully.")
        st.rerun()
# Main Layout
st.markdown("""
<div style="border-bottom: 1px solid #262626; padding-bottom: 1.25rem; margin-bottom: 1.75rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
        <div class="tech-badge">// 01 • SYSTEMS ARCHITECTURE</div>
        <div class="tech-mono-meta">LOC: LOCALHOST • RUNTIME: CUDA • STACK: HYBRID RRF</div>
    </div>
    <h1 style="font-size: 2.7rem; font-weight: 800; letter-spacing: -0.04em; text-transform: uppercase; margin: 0; color: #ffffff;">
        GRAPHMIND <span style="color: #525252;">//</span> CORE
    </h1>
    <div style="margin-top: 0.5rem; color: #737373; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; letter-spacing: 0.04em;">
        [ ENGINE: DENSE VECTORS • BM25 PLUS • KNOWLEDGE GRAPH TRAVERSAL • OKF CRITIC ]
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation tabs
tab_qa, tab_guide, tab_graph, tab_eval = st.tabs([
    "Chat", 
    "Notebook Guide", 
    "Knowledge Graph", 
    "Evaluation"
])

# Tab 1: Q&A Engine
with tab_qa:
    st.markdown("### Chat with Sources")
    st.markdown("Ask questions about your selected documents. The agent will retrieve relevant vector chunks and trace knowledge graph connections to give a sourced answer.")
    
    # Add a clear chat button
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear History", type="secondary"):
            st.session_state.chat_history = []
            st.toast("Chat history cleared.")
            st.rerun()
        
    st.markdown("---")
    
    # Display previous chat messages
    for msg_idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If it's an assistant message and has extra details, show them in expanders
            if message["role"] == "assistant" and "category" in message:
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("Routed Category", message["category"])
                with col_met2:
                    st.metric("Calibrated Confidence", f"{message['confidence']}%")
                with col_met3:
                    latency = message.get("retrieval_metadata", {}).get("elapsed_ms", 0.0)
                    st.metric("Retrieval Latency", f"{latency:.1f} ms")
                
                # Expanders for tracing
                with st.expander("Show Routing & Confidence Breakdown", expanded=False):
                    st.markdown(f"**Routed Category:** `{message['category']}`")
                    st.markdown(f"**Classification Reasoning:** *{message['reasoning']}*")
                    signals = message.get("confidence_signals", {})
                    if signals:
                        st.markdown("**Calibration Signals:**")
                        st.write(f"- Citation Grounding: {signals.get('citation_grounding', 0) * 100:.0f}%")
                        st.write(f"- Retrieval Coverage: {signals.get('retrieval_coverage', 0) * 100:.0f}%")
                        st.write(f"- Graph Evidence: {signals.get('graph_evidence', 0) * 100:.0f}%")
                        st.write(f"- Model Self-Score: {signals.get('raw_llm', 0) * 100:.0f}%")
                    st.markdown(f"**Chunks Retrieved (RRF):** {message['num_chunks']}")
                    st.markdown(f"**Relations Discovered:** {message['num_relations']}")
                
                # Show contradiction warning if any
                contradictions = st.session_state.graph_store.get_contradictions()
                if contradictions:
                    with st.expander("⚠️ Knowledge Graph Contradictions Detected", expanded=False):
                        for c in contradictions:
                            st.warning(
                                f"**{c['subject']}** has conflicting **{c['relation']}**: "
                                f"`{c['existing_object']}` (from {', '.join(c['existing_sources'])}) vs "
                                f"`{c['new_object']}` (from {c['new_source']})"
                            )
                
                with st.expander("Show Retrieved Document Chunks (RRF + Reranker)", expanded=False):
                    for idx, chunk in enumerate(message["vector_chunks"]):
                        st.markdown(f"**Chunk {idx+1} (Source: {chunk['source']}, Page: {chunk['page']})**")
                        st.info(chunk["text"])
                        
                with st.expander("Show Traversed Subgraph Relations (KG)", expanded=False):
                    if not message["graph_relations"]:
                        st.write("No relation triples traversed for this query.")
                    else:
                        for idx, rel in enumerate(message["graph_relations"]):
                            st.write(f"- **({rel['subject']})** --`{rel['relation']}`--> **({rel['object']})** (Sources: {', '.join(rel['sources'])})")
                            
                # Download button for this specific answer
                st.download_button(
                    label="Export Answer (Markdown)",
                    data=message["content"],
                    file_name=f"graphmind_answer_{msg_idx}.md",
                    mime="text/markdown",
                    key=f"export_{msg_idx}"
                )
                
    # Chat Input
    query = st.chat_input("Ask a question about your sources...")
    
    if query:
        # Check source requirements
        if not selected_sources:
            st.warning("Please select at least one source document in the sidebar to ask a question.")
        else:
            # Display user message instantly
            with st.chat_message("user"):
                st.markdown(query)
                
            # Append user message to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Generate response
            with st.chat_message("assistant"):
                with st.status("🧠 GraphMind Agent is analyzing...", expanded=True) as status_box:
                    try:
                        st.write("🔍 Searching Dense Vectors, BM25 Lexical Index & Knowledge Graph...")
                        start_time = time.time()
                        result = st.session_state.query_agent.answer_query(query, source_filter=selected_sources)
                        elapsed_sec = round(time.time() - start_time, 2)
                        elapsed_ms = elapsed_sec * 1000.0

                        st.session_state.audit_logger.log_query_execution(
                            query=query,
                            category=result["category"],
                            latency_ms=elapsed_ms,
                            chunk_count=len(result["vector_chunks"]),
                            edge_count=len(result["graph_relations"]),
                            confidence=result["confidence"],
                            sources_used=selected_sources
                        )

                        if result.get("cached"):
                            st.write("⚡ Cache Hit (instant sub-millisecond return)")
                            status_box.update(label=f"⚡ Answer retrieved from cache in {elapsed_sec}s", state="complete", expanded=False)
                        else:
                            cat = result.get("category", "HYBRID")
                            num_c = len(result["vector_chunks"])
                            num_r = len(result["graph_relations"])
                            st.write(f"📊 Routed to **{cat}** • Retrieved {num_c} passages & {num_r} relations")
                            st.write("✨ Synthesizing verified answer with citations & confidence calibration...")
                            status_box.update(label=f"✅ Response generated in {elapsed_sec}s ({cat})", state="complete", expanded=False)

                        st.markdown(result["answer"])
                        
                        # Append assistant message to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "category": result["category"],
                            "confidence": result["confidence"],
                            "confidence_signals": result.get("confidence_signals", {}),
                            "reasoning": result["reasoning"],
                            "num_chunks": len(result["vector_chunks"]),
                            "num_relations": len(result["graph_relations"]),
                            "vector_chunks": result["vector_chunks"],
                            "graph_relations": result["graph_relations"],
                            "retrieval_metadata": result.get("retrieval_metadata", {})
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error answering query: {e}")

# Tab 2: Notebook Guide
with tab_guide:
    st.markdown("### Notebook Guide")
    st.markdown("Generate key study guides and summaries from your active sources.")
    
    st.write("")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        <div class="premium-card">
            <h4>Summary Brief</h4>
            <p style="color: #737373; font-size: 0.9rem; margin-top: 0.5rem;">An executive overview and high-level briefing of your active sources.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate Summary", type="secondary"):
            st.info("Notebook Guide features will be fully unlocked in Phase B.")
            
        st.markdown("""
        <div class="premium-card" style="margin-top: 1.5rem;">
            <h4>Timeline Chronology</h4>
            <p style="color: #737373; font-size: 0.9rem; margin-top: 0.5rem;">Key events, history, or development phases tracked in chronological order.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate Timeline", type="secondary"):
            st.info("Notebook Guide features will be fully unlocked in Phase B.")
            
    with col_g2:
        st.markdown("""
        <div class="premium-card">
            <h4>Frequently Asked Questions</h4>
            <p style="color: #737373; font-size: 0.9rem; margin-top: 0.5rem;">The most critical questions and detailed answers extracted automatically.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate FAQ Guide", type="secondary"):
            st.info("Notebook Guide features will be fully unlocked in Phase B.")
            
        st.markdown("""
        <div class="premium-card" style="margin-top: 1.5rem;">
            <h4>Study Guide & Concepts</h4>
            <p style="color: #737373; font-size: 0.9rem; margin-top: 0.5rem;">Definitions of core terminology, concepts, and study flashcards.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate Study Guide", type="secondary"):
            st.info("Notebook Guide features will be fully unlocked in Phase B.")

# Tab 3: Knowledge Graph Visualizer
with tab_graph:
    st.markdown("### Interactive Knowledge Graph")
    st.markdown("Explore the graph memory constructed from your uploaded files. Click and drag nodes, zoom, and highlight connection paths.")
    
    if num_nodes == 0:
        st.info("No nodes in the graph to visualize. Please ingest documents to populate the knowledge graph.")
    else:
        # Render minimalist 3-column topology metrics
        st.markdown("#### Network Topology Metrics")
        graph = st.session_state.graph_store.graph
        density = nx.density(graph)
        components_count = nx.number_weakly_connected_components(graph)
        
        met_col1, met_col2, met_col3 = st.columns(3)
        met_col1.metric("Graph Density", f"{density:.4f}")
        met_col2.metric("Weakly Connected Components", str(components_count))
        met_col3.metric("Total Relations (Edges)", str(graph.number_of_edges()))
        
        st.markdown("---")
        
        with st.spinner("Generating interactive graph network..."):
            html_path = st.session_state.graph_store.generate_visualization_html()
            
            # Load HTML and embed
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                # Render using iframe
                components.html(html_content, height=600)
            else:
                st.error("Failed to generate graph HTML file.")
        
        st.markdown("---")
        
        # Display horizontal bar chart of the top 10 central hubs
        st.markdown("### Top 10 Central Hubs (Node Degree)")
        nodes_sorted = sorted(
            [(node, st.session_state.graph_store.graph.degree(node)) for node in st.session_state.graph_store.graph.nodes],
            key=lambda x: x[1],
            reverse=True
        )
        top_10 = nodes_sorted[:10]
        if top_10:
            df = pd.DataFrame(top_10, columns=["Entity", "Degree"])
            st.bar_chart(
                data=df,
                x="Entity",
                y="Degree",
                color="#a3a3a3",
                horizontal=True
            )
        else:
            st.info("No nodes available to plot.")

# Tab 4: Evaluation Benchmark
with tab_eval:
    st.markdown("### Evaluation Benchmark (GraphRAG vs Vector RAG)")
    st.markdown("Compare the performance of our GraphMind Hybrid engine against standard Vector-only RAG.")
    
    # We load evaluation page
    st.markdown("""
    <div class="premium-card">
        <h4>Benchmark metrics target: 25-40% improvement on multi-hop questions</h4>
        <p>A multi-hop question (e.g. "How does the founder of Company X relate to Project Y?") requires connecting information from different pages. Vector search retrieves isolated chunks and fails to establish links, whereas the Knowledge Graph traces relationships directly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Run Simulation Benchmark"):
        with st.spinner("Running evaluation benchmark on sample multi-hop questions..."):
            # We can import and run evaluation
            from src import evaluation
            eval_results = evaluation.run_comparison(
                st.session_state.vector_store,
                st.session_state.graph_store
            )
            
            # Display results
            st.markdown("### Evaluation Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Vector-only RAG Accuracy", f"{eval_results['vector_accuracy']}%")
            col2.metric("GraphMind Hybrid Accuracy", f"{eval_results['hybrid_accuracy']}%")
            col3.metric("Improvement Margin", f"+{eval_results['improvement']}%")
            
            st.markdown("### Detailed Comparison Results")
            st.table(eval_results["details"])
