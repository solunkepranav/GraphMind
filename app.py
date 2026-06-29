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

# Custom premium CSS - Bento Monochrome Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base application styling with bento monochrome grid */
.stApp {
    background-color: #050505;
    background-image: radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 0);
    background-size: 24px 24px;
    color: #e5e5e5;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #050505;
    border-right: 1px solid #1f1f1f;
}

/* Text color for sidebar */
[data-testid="stSidebar"] .stMarkdown {
    color: #a3a3a3;
}

/* Styled headers */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 600;
    margin-bottom: 0.5rem;
    letter-spacing: -0.015em;
}

/* Premium card wrappers - Bento Monochrome styled */
.premium-card {
    background-color: rgba(23, 23, 23, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease;
}
.premium-card:hover {
    border-color: rgba(255, 255, 255, 0.2);
}

/* Button stylings - Monochrome */
.stButton>button {
    background-color: transparent;
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton>button:hover {
    background-color: #ffffff;
    color: #050505;
    border: 1px solid #ffffff;
}

.stButton>button:active {
    transform: translateY(0);
}

.stButton>button:focus {
    outline: 2px solid #ffffff;
    outline-offset: 2px;
}

/* Status logs */
.log-box {
    background-color: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 0.75rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: #a3a3a3;
    margin-bottom: 0.5rem;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] {
    color: #737373 !important;
}

/* Tab styling - Monochrome */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #262626;
}

.stTabs [data-baseweb="tab"] {
    color: #737373;
    border-bottom: 2px solid transparent;
    padding: 0.75rem 1.25rem;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #ffffff !important;
}

/* Input / selectbox styling */
.stTextInput>div>div>input, .stSelectbox>div>div {
    background-color: #171717;
    border: 1px solid #262626;
    color: #ffffff;
    border-radius: 8px;
}

.stTextInput>div>div>input:focus {
    border-color: #ffffff;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.15);
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    color: #a3a3a3;
}

/* Progress bar accent */
.stProgress>div>div>div>div {
    background-color: #ffffff;
}

/* Selection highlight */
::selection {
    background-color: rgba(255, 255, 255, 0.15);
    color: #ffffff;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #050505;
}
::-webkit-scrollbar-thumb {
    background: #262626;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #52525b;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #171717;
    border: 1px dashed #262626;
    border-radius: 8px;
}

/* Toast / alerts */
.stAlert {
    background-color: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
}

/* Checkbox visual styling for visible ticks */
div[data-testid="stCheckbox"] svg {
    stroke: #050505 !important;
    fill: #050505 !important;
}
div[data-testid="stCheckbox"] [role="checkbox"] {
    border-color: #262626 !important;
}
div[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.vector_store import VectorStore
from src.graph_store import GraphStore
from src.agents import QueryAgent
from src import ingestion
from src import llm
from src.source_registry import SourceRegistry

# Initialize session state for DB & Graph connections
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "graph_store" not in st.session_state:
    st.session_state.graph_store = GraphStore()
if "query_agent" not in st.session_state:
    st.session_state.query_agent = QueryAgent(
        st.session_state.vector_store,
        st.session_state.graph_store
    )
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
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status_text.text(f"Ingesting: {uploaded_file.name}")
            try:
                chunks = ingestion.ingest_file(file_path)
                st.session_state.vector_store.add_chunks(chunks)
                
                total_chunks = len(chunks)
                for c_idx, chunk in enumerate(chunks):
                    status_text.text(f"Extracting {c_idx+1}/{total_chunks}...")
                    st.session_state.graph_store.add_relations_from_chunk(chunk)
                
                # Register source in registry
                page_count = max([c.get("page", 1) for c in chunks]) if chunks else 1
                registry.register_source(
                    filename=uploaded_file.name,
                    file_type=uploaded_file.name.split('.')[-1].lower(),
                    chunk_count=len(chunks),
                    page_count=page_count,
                    size_bytes=uploaded_file.size
                )
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
            
            try:
                os.remove(file_path)
            except Exception:
                pass
                
            progress_bar.progress(int((idx + 1) / len(uploaded_files) * 100))
            
        st.session_state.graph_store.save()
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
        registry.reset()
        st.toast("Database, Graph, and Source Registry reset successfully.")
        st.rerun()
# Main Layout
st.markdown("# GraphMind: Knowledge Notebook")
st.markdown("##### *An Intelligent Document Q&A Assistant with Graph-Based Memory*")

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
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("Routed Category", message["category"])
                with col_met2:
                    st.metric("Confidence Score", f"{message['confidence']}%")
                
                # Expanders for tracing
                with st.expander("Show Routing & Extraction Logs", expanded=False):
                    st.markdown(f"**Routed Category:** `{message['category']}`")
                    st.markdown(f"**Classification Reasoning:** *{message['reasoning']}*")
                    st.markdown(f"**Chunks Retrieved:** {message['num_chunks']}")
                    st.markdown(f"**Relations Discovered:** {message['num_relations']}")
                
                with st.expander("Show Retrieved Document Chunks (Vector DB)", expanded=False):
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
                with st.spinner("Retrieving context and synthesizing answer..."):
                    try:
                        result = st.session_state.query_agent.answer_query(query, source_filter=selected_sources)
                        st.markdown(result["answer"])
                        
                        # Append assistant message to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "category": result["category"],
                            "confidence": result["confidence"],
                            "reasoning": result["reasoning"],
                            "num_chunks": len(result["vector_chunks"]),
                            "num_relations": len(result["graph_relations"]),
                            "vector_chunks": result["vector_chunks"],
                            "graph_relations": result["graph_relations"]
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
