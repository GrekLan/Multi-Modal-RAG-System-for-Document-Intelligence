# 🤖 Multi-Modal RAG System for Document Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A production-ready Retrieval-Augmented Generation system for querying multi-modal documents**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Multi-Modal Retrieval-Augmented Generation (RAG)** system designed to process and query complex documents containing text, tables, charts, and images. Built specifically for IMF Article IV reports and similar policy documents, the system provides accurate, citation-backed answers by combining semantic search with context-grounded LLM generation.

### Key Capabilities

✅ **Multi-Modal Processing:** Handles text, tables, and images (with OCR)  
✅ **Intelligent Retrieval:** Semantic search across unified embedding space  
✅ **Citation-Backed Answers:** Every response includes source attribution  
✅ **Interactive Interface:** User-friendly Streamlit chatbot  
✅ **Comprehensive Evaluation:** Built-in benchmarking suite

---

## ✨ Features

### Document Processing
- 📄 **Text Extraction** with semantic chunking
- 📊 **Table Detection** using layout analysis
- 🖼️ **Image Processing** with OCR (Tesseract)
- 🔍 **Metadata Preservation** (page numbers, source tracking)

### Retrieval & Generation
- 🧠 **Semantic Search** using Sentence Transformers
- ⚡ **FAISS Vector Store** for fast similarity search
- 🤖 **FLAN-T5 LLM** for answer generation
- 📚 **Source Attribution** with relevance scoring

### User Experience
- 💬 **Chat Interface** with conversation history
- 📈 **Analytics Dashboard** with performance metrics
- 💾 **Export Options** for chat history
- 🎨 **Modern UI** with visualizations

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│      Document Processor                     │
│  ┌────────┐  ┌────────┐  ┌──────────────┐  │
│  │  Text  │  │ Tables │  │ Images (OCR) │  │
│  └────────┘  └────────┘  └──────────────┘  │
└────────────────────┬────────────────────────┘
                     │ Chunks
                     ▼
         ┌───────────────────────┐
         │   Embedding Model     │
         │  (Sentence Transform) │
         └───────────┬───────────┘
                     │ Vectors
                     ▼
         ┌───────────────────────┐
         │    FAISS Vector DB    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │      User Query       │
         └───────────┬───────────┘
                     │ Semantic Search
                     ▼
         ┌───────────────────────┐
         │   Retrieved Context   │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │      FLAN-T5 LLM      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Answer + Citations   │
         └───────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed
- 4GB RAM minimum
- (Optional) GPU for faster inference

### System Dependencies

#### Linux/Ubuntu
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

#### Windows
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### Python Dependencies

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-modal-rag.git
cd multi-modal-rag
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Automated Pipeline

```bash
# Run the complete pipeline (recommended)
python run_pipeline.py
```

This will:
1. ✅ Create necessary directories
2. ✅ Process your PDF document
3. ✅ Extract text, tables, and images
4. ✅ Create embeddings and vector index

### Option 2: Manual Steps

```bash
# Step 1: Setup directories
python config.py

# Step 2: Place your PDF
# Copy your PDF to: data/raw/qatar_test_doc.pdf

# Step 3: Process document
python process_document.py

# Step 4: Create embeddings
python create_embeddings.py
```

### Launch the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1. Processing a New Document

1. Place your PDF in `data/raw/qatar_test_doc.pdf`
2. Run the pipeline: `python run_pipeline.py`
3. Wait for processing to complete (~1-2 minutes)

### 2. Using the Chat Interface

1. Launch the app: `streamlit run app.py`
2. Type your question in the chat input
3. View the answer with citations
4. Expand sections to see:
   - 📚 Source citations
   - ⚡ Performance metrics
   - 🔍 Retrieved context

### 3. Example Queries

**Simple Factual:**
```
What is Qatar's GDP growth rate?
```

**Table-based:**
```
What are the key economic indicators in the tables?
```

**Image-based:**
```
What trends are shown in the charts?
```

**Complex:**
```
Compare Qatar's fiscal performance across different years
```

### 4. Running Evaluation

```bash
# Run comprehensive evaluation suite
python evaluate.py
```

This generates:
- Performance metrics (retrieval time, accuracy)
- Multi-modal coverage analysis
- Results saved to `evaluation_results.json`

---

## 📊 Evaluation

The system includes a comprehensive evaluation suite that tests:

### Metrics Evaluated

| Metric | Description | Target |
|--------|-------------|--------|
| **Retrieval Time** | Time to find relevant chunks | < 0.2s |
| **Generation Time** | Time to generate answer | < 2.0s |
| **Relevance Score** | Average similarity score | > 7.0/10 |
| **Modality Accuracy** | Correct modality retrieval | > 80% |
| **Citation Coverage** | Answers with citations | > 90% |

### Test Queries

The suite includes 8 diverse queries covering:
- ✅ Simple factual questions
- ✅ Multi-part questions
- ✅ Analytical queries
- ✅ Comparison tasks
- ✅ Summarization requests

### Running Custom Evaluation

```python
from evaluate import RAGEvaluator
from vector_store import VectorStore
from llm_qa import LLMQA

# Load system
vector_store = VectorStore()
vector_store.load('data/vector_store/faiss_index')
qa_system = LLMQA()

# Create evaluator
evaluator = RAGEvaluator(vector_store, qa_system)

# Run evaluation
results = evaluator.run_full_evaluation(k=5)
evaluator.save_results('my_evaluation.json')
```

---

## 📁 Project Structure

```
multi-modal-rag/
│
├── data/                          # Data directory
│   ├── raw/                       # Input PDFs
│   ├── processed/                 # Extracted chunks
│   ├── vector_store/              # FAISS index
│   └── images/                    # Extracted images
│
├── app.py                         # Main Streamlit application
├── config.py                      # Configuration management
├── document_processor.py          # PDF processing module
├── vector_store.py                # Vector store management
├── llm_qa.py                      # QA system module
├── evaluate.py                    # Evaluation suite
│
├── process_document.py            # Step 1: Extract data
├── create_embeddings.py           # Step 2: Create vectors
├── run_pipeline.py                # Automated pipeline runner
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── TECHNICAL_REPORT.md            # Detailed technical report
```

---

## 🔧 Technical Details

### Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| **Embeddings** | `all-MiniLM-L6-v2` | 80MB | Text encoding |
| **LLM** | `google/flan-t5-base` | 248M params | Answer generation |
| **OCR** | Tesseract | - | Image text extraction |

### Configuration

Edit `config.py` to customize:

```python
# Paths
PDF_PATH = 'data/raw/your_document.pdf'
VECTOR_STORE_PATH = 'data/vector_store/faiss_index'

# Models
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base'

# Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### Performance Optimization

**For faster processing:**
```python
# Use GPU if available
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**For larger documents:**
```python
# Adjust chunk size
CHUNK_SIZE = 500  # Smaller chunks for more granular retrieval
```

---

## 🐛 Troubleshooting

### Issue: "PDF not found"

**Solution:**
```bash
# Ensure PDF is in correct location
cp your_document.pdf data/raw/qatar_test_doc.pdf
```

### Issue: "Tesseract not installed"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows - Download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: "Out of memory"

**Solution:**
```python
# In config.py, reduce chunk size
CHUNK_SIZE = 500  # Instead of 1000
```

### Issue: "Slow retrieval"

**Solution:**
```python
# Reduce number of results
k = 3  # Instead of 5
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
black .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** for RAG framework
- **HuggingFace** for embedding models
- **Streamlit** for UI framework
- **Big AIR Lab** for the assignment

---

## 📞 Support

For questions or issues:
- 📧 Email: aryansh085@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/GrekLan/multi-modal-rag/issues)
- 📖 Docs: [Technical Report](TECHNICAL_REPORT.md)

---

<div align="center">

**Made with ❤️ for the Big AIR Lab Assignment**

⭐ Star this repo if you find it helpful!

</div>
