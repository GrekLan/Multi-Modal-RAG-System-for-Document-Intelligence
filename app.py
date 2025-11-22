import streamlit as st
import os
import json
import time
from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config

st.set_page_config(
    page_title="Multi-Modal RAG System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .citation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'queries': 0, 'total_time': 0}

# Load data if not already loaded
if not st.session_state.loaded:
    # Check multiple possible locations for FAISS index
    faiss_file = f"{config.VECTOR_STORE_PATH}.faiss"
    faiss_dir = os.path.join(config.VECTOR_STORE_PATH, "index.faiss")
    
    if os.path.exists(os.path.join(config.VECTOR_STORE_PATH, "index.faiss")):
        with st.spinner("🔄 Loading vector database..."):
            try:
                vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
                vector_store.load(config.VECTOR_STORE_PATH)
                st.session_state.vector_store = vector_store
                
                try:
                    qa_system = LLMQA(model_name=config.LLM_MODEL)
                    st.session_state.qa_system = qa_system
                    st.session_state.llm_type = "FLAN-T5"
                except:
                    st.session_state.qa_system = SimpleQA()
                    st.session_state.llm_type = "Simple QA"
                
                st.session_state.loaded = True
                
            except Exception as e:
                st.error(f"❌ Error loading system: {e}")
                st.session_state.loaded = False

# Sidebar
with st.sidebar:
    st.title("🤖 System Control")
    
    if st.session_state.loaded:
        st.success("✅ System Ready")
        
        st.markdown("---")
        st.subheader("📊 System Information")
        
        if st.session_state.vector_store:
            total = len(st.session_state.vector_store.chunks)
            text_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'text')
            table_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'table')
            image_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'image')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", total)
                st.metric("Text", text_count)
            with col2:
                st.metric("Tables", table_count)
                st.metric("Images", image_count)
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        k = st.slider("Results to retrieve", 1, 10, 5)
        st.session_state.retrieval_k = k
        
        st.info(f"**LLM:** {st.session_state.llm_type}")
        
        st.markdown("---")
        st.subheader("📈 Session Stats")
        st.metric("Queries", st.session_state.stats['queries'])
        if st.session_state.stats['queries'] > 0:
            avg_time = st.session_state.stats['total_time'] / st.session_state.stats['queries']
            st.metric("Avg Time", f"{avg_time:.2f}s")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.chat_history:
                chat_export = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    "Download JSON",
                    chat_export,
                    "chat_history.json",
                    "application/json"
                )
    else:
        st.error("❌ System Not Ready")
        st.markdown("---")
        st.subheader("🔧 Setup Required")
        st.markdown("""
        **Steps:**
        
        1. Place PDF:
        ```
        data/raw/qatar_test_doc.pdf
        ```
        
        2. Run:
        ```bash
        python run_pipeline.py
        ```
        
        3. Restart app
        """)

# Main content
if st.session_state.loaded:
    st.markdown('<div class="main-header">🤖 Multi-Modal RAG Q&A System</div>', unsafe_allow_html=True)
    st.markdown("*Ask questions about your document*")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown("---")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "citations" in message and message["citations"]:
                    with st.expander("📚 View Citations"):
                        for i, cite in enumerate(message["citations"], 1):
                            st.markdown(f"""
                            <div class="citation-box">
                                <b>Citation {i}</b><br>
                                <b>Source:</b> {cite['source']}<br>
                                <b>Type:</b> {cite['type']}<br>
                                <b>Page:</b> {cite['page']}<br>
                                <b>Relevance:</b> {cite['relevance_score']:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                
                if "metrics" in message:
                    with st.expander("⚡ Performance"):
                        metrics = message["metrics"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieval", f"{metrics['retrieval_time']:.3f}s")
                        with col2:
                            st.metric("Generation", f"{metrics['generation_time']:.3f}s")
                        with col3:
                            st.metric("Total", f"{metrics['total_time']:.3f}s")
        
        # Chat input
        query = st.chat_input("💭 Ask a question...")
        
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching and generating answer..."):
                    start_time = time.time()
                    
                    retrieval_start = time.time()
                    search_results = st.session_state.vector_store.search(
                        query, 
                        k=st.session_state.retrieval_k
                    )
                    retrieval_time = time.time() - retrieval_start
                    
                    generation_start = time.time()
                    result = st.session_state.qa_system.generate_answer_with_citations(
                        query, search_results
                    )
                    generation_time = time.time() - generation_start
                    
                    total_time = time.time() - start_time
                    
                    st.markdown(result['answer'])
                    
                    if result['citations']:
                        with st.expander("📚 View Citations"):
                            for i, cite in enumerate(result['citations'], 1):
                                st.markdown(f"""
                                <div class="citation-box">
                                    <b>Citation {i}</b><br>
                                    <b>Source:</b> {cite['source']}<br>
                                    <b>Type:</b> {cite['type']}<br>
                                    <b>Page:</b> {cite['page']}<br>
                                    <b>Relevance:</b> {cite['relevance_score']:.3f}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    metrics = {
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'total_time': total_time
                    }
                    
                    with st.expander("⚡ Performance"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieval", f"{retrieval_time:.3f}s")
                        with col2:
                            st.metric("Generation", f"{generation_time:.3f}s")
                        with col3:
                            st.metric("Total", f"{total_time:.3f}s")
                    
                    st.session_state.stats['queries'] += 1
                    st.session_state.stats['total_time'] += total_time
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "citations": result['citations'],
                        "metrics": metrics
                    })
    
    with tab2:
        st.header("📊 System Analytics")
        
        if st.session_state.chat_history:
            assistant_messages = [m for m in st.session_state.chat_history if m['role'] == 'assistant']
            
            if assistant_messages:
                times = [m['metrics']['total_time'] for m in assistant_messages if 'metrics' in m]
                
                if times:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Queries", len(times))
                    with col2:
                        st.metric("Avg Response Time", f"{sum(times)/len(times):.3f}s")
                    with col3:
                        st.metric("Max Response Time", f"{max(times):.3f}s")
                
                all_citations = []
                for m in assistant_messages:
                    if 'citations' in m:
                        all_citations.extend(m['citations'])
                
                if all_citations:
                    modality_counts = {}
                    for cite in all_citations:
                        mod = cite['type']
                        modality_counts[mod] = modality_counts.get(mod, 0) + 1
                    
                    st.subheader("Citation Modality Distribution")
                    for mod, count in modality_counts.items():
                        st.write(f"**{mod.capitalize()}:** {count}")
        else:
            st.info("📝 No queries yet. Start chatting to see analytics!")
    
    with tab3:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### 🎯 Multi-Modal RAG System
        
        This system implements **Retrieval-Augmented Generation (RAG)** for multi-modal documents.
        
        #### 🏗️ Architecture
        
        1. **Document Processing**
           - Text extraction with semantic chunking
           - Table detection and extraction
           - Image extraction with OCR
        
        2. **Embedding & Indexing**
           - Unified embedding space (Sentence Transformers)
           - FAISS vector index
        
        3. **Retrieval & Generation**
           - Semantic search across modalities
           - Context-grounded responses (FLAN-T5)
           - Citation-backed answers
        
        #### 🔧 Technical Stack
        
        - **Processing:** PyMuPDF, Pytesseract
        - **Embeddings:** HuggingFace Sentence Transformers
        - **Vector Store:** FAISS
        - **LLM:** Google FLAN-T5
        - **Framework:** LangChain
        - **UI:** Streamlit
        
        #### 📊 Evaluation
        
        Run: `python evaluate.py`
        """)

else:
    st.markdown('<div class="main-header">🤖 Multi-Modal RAG System</div>', unsafe_allow_html=True)
    st.error("⚠️ System not initialized. Follow setup instructions in sidebar.")