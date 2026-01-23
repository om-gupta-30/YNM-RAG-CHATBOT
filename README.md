# 🤖 RAG PDF Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Retrieval-Augmented Generation (RAG)** application for intelligent question-answering over technical PDF documents. This system combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** to provide accurate, source-cited answers with support for figures, tables, pages, and confidence scoring.

## ✨ Features

- 🔍 **Intelligent Q&A**: Ask natural-language questions about indexed PDF documents
- 🎯 **Intent-Aware Retrieval**: Automatically classifies queries (figure, table, page, section, general, comparison) and uses optimal retrieval strategies
- 📊 **Structured Answers**: Returns formatted paragraphs and lists with source citations
- 🎨 **Modern UI**: React-based chat interface with dark/light theme, multi-chat support, and PDF export
- 📈 **Confidence Scoring**: Each answer includes High/Medium/Low confidence indicators
- 🔗 **Source Citations**: Answers include page numbers, text excerpts, and optional page images
- 🚀 **Fast Retrieval**: FAISS-based semantic search for efficient document querying

## 🎯 What This Application Does

This RAG system enables users to:

- **Query PDF Documents**: Ask questions in natural language about the indexed document (default: IRC:67-2022 Code of Practice for Road Signs)
- **Smart Retrieval**: The system intelligently chooses between:
  - **Exact matching** for specific references (e.g., "Fig 3.3", "Table 6.2", "Page 27")
  - **Semantic search** using FAISS for general questions
- **Structured Responses**: Answers are returned as structured JSON with paragraphs and lists
- **Context Expansion**: View surrounding chunks without additional API calls
- **Export Capabilities**: Export chat conversations to PDF

## 🏗️ Architecture Overview

### Core Concepts

#### 1. **Retrieval-Augmented Generation (RAG)**

Instead of relying solely on the LLM's training data, this system:

1. **Retrieves** relevant text chunks from the corpus using the user's question
2. **Augments** the LLM prompt with retrieved chunks as context
3. **Generates** answers grounded in the provided context

This approach reduces hallucinations and ensures answers are tied to specific document pages and chunks.

#### 2. **Vector Embeddings and FAISS**

- **Embeddings**: Questions and document chunks are converted to dense vectors using Google's `embedding-001` model
- **FAISS**: Fast approximate nearest-neighbor search for semantic similarity
- **Process**: Query → Embed → Search → Retrieve chunks → Pass to LLM

#### 3. **Intent Classification**

The system classifies each question before retrieval:

| Intent | Example | Retrieval Strategy |
|--------|---------|-------------------|
| `FIGURE_QUERY` | "What does Fig 3.3 show?" | Exact match on "Fig 3.3" in metadata |
| `TABLE_QUERY` | "Table 6.2 guidelines" | Exact match on "Table 6.2" |
| `PAGE_QUERY` | "What is on page 27?" | All chunks with `page_number == 27` |
| `SECTION_QUERY` | "Section on mandatory signs" | Semantic/FAISS search |
| `GENERAL_QUERY` | "Size of STOP sign?" | FAISS semantic search (k=6) |
| `COMPARISON_QUERY` | "Compare Fig 3.1 and Fig 3.2" | Retrieves both referenced items |

#### 4. **Confidence Scoring**

Each answer block includes a confidence score (High/Medium/Low) based on:
- Source type quality (OCR/IMAGE vs TEXT)
- Number of supporting chunks
- Semantic match score
- Intent type
- Verbatim presence in retrieved text

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Vector DB** | FAISS (CPU), NumPy |
| **Embeddings** | Google `embedding-001` |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **PDF Processing** | PyMuPDF (fitz), pdfplumber, Pillow |
| **Frontend** | React 19, Vite 7 |
| **Export** | jsPDF |
| **Deployment** | Docker |

## 📁 Project Structure

```
rag-chatbot/
├── app.py                  # FastAPI backend application
├── intent_classifier.py    # Query intent classification module
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── .gitignore            # Git ignore rules
├── .env                   # Environment variables (create locally)
│
├── faiss.index           # FAISS vector index (generated, not in git)
├── metadata.json         # Chunk catalog with text, source, page_number
├── vision_captions.json  # Optional: Gemini vision captions cache
├── images/               # Page images (page_{n}_img_1.png)
│
└── ui/                   # React frontend
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── src/
    │   ├── main.jsx      # React entry point
    │   ├── App.jsx       # Main chat UI component
    │   ├── App.css       # Component styles
    │   ├── index.css     # Global styles
    │   └── api.js        # API client functions
    └── public/           # Static assets
```

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18 or higher
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **FAISS Index & Metadata**: The application requires `faiss.index` and `metadata.json` files (typically generated by a separate indexing pipeline)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. **Set up Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Set up frontend**
   ```bash
   cd ui
   npm install
   cd ..
   ```

### Running the Application

#### Development Mode

**Backend** (from project root):
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Frontend** (in a separate terminal):
```bash
cd ui
npm run dev
```

- **API**: `http://127.0.0.1:8000`
- **API Docs**: `http://127.0.0.1:8000/docs`
- **Frontend**: `http://localhost:5173` (or the port Vite displays)

#### Production Build

**Frontend**:
```bash
cd ui
npm run build
```

The built files will be in `ui/dist/`. Serve this directory along with the FastAPI backend.

#### Docker Deployment

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_key \
  -v "$(pwd)/faiss.index:/app/faiss.index" \
  -v "$(pwd)/metadata.json:/app/metadata.json" \
  -v "$(pwd)/images:/app/images" \
  rag-chatbot
```

#### Deploy to GCP Cloud Run

Deploy to Google Cloud Run (service `rag-pdf-chatbot`, region `asia-south1`, project `gen-lang-client-0473608308`):

```bash
# Ensure GEMINI_API_KEY is in .env or export it, then:
./deploy-gcp.sh
```

Optional overrides: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_SERVICE_NAME`. The script enables required APIs, builds from source (Dockerfile), and deploys. The service URL is printed after deploy.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `POST` | `/classify-intent` | Classify query intent (no FAISS/LLM) |
| `POST` | `/expand-context` | Get surrounding chunks (no FAISS/LLM) |
| `POST` | `/generate-chat-title` | Generate chat title from question (Gemini, server-side only) |
| `POST` | `/ask` | Main Q&A endpoint with full RAG pipeline |

### Example API Usage

**Classify Intent**:
```bash
curl -X POST http://localhost:8000/classify-intent \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Fig 3.3 show?"}'
```

**Ask Question**:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the dimensions of a STOP sign?"}'
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM and embeddings |

**Security**: Keep `GEMINI_API_KEY` only in `.env` (or env vars). Never put it in frontend code or `VITE_*` vars—the key would be exposed in built JS. Chat title generation and all Gemini calls use the backend. Verify the key: `python test_gemini_key.py`.

### Required Files

The application expects these files in the project root:

- `faiss.index`: Pre-built FAISS vector index
- `metadata.json`: Chunk catalog mapping chunk IDs to text, source, and page numbers
- `images/`: Directory containing page images (optional but recommended)

> **Note**: These files are typically generated by a separate indexing pipeline and are not included in this repository.

## 📖 How It Works

### Query Processing Flow

1. **Intent Classification**: The query is analyzed to determine intent (figure, table, page, section, general, or comparison)

2. **Retrieval**:
   - For specific references (figures, tables, pages): Exact matching in metadata
   - For general queries: FAISS semantic search

3. **Context Building**: Retrieved chunks are assembled into a context string

4. **LLM Generation**: Context is sent to Gemini with a structured prompt

5. **Response Processing**: 
   - JSON parsing and validation
   - Confidence scoring
   - Source citation enrichment

6. **Response**: Structured answer with blocks, sources, and confidence scores

### Intent Classification Rules

- **Figure**: Matches patterns like "fig 3.3", "figure 3.3", "fig. 3.3"
- **Table**: Matches "table 6.2", "tab 6.2"
- **Page**: Matches "page 27", "pg 27", "p 27"
- **Section**: Matches keywords like "section", "chapter", "heading"
- **Comparison**: Detects "compare", "vs", "versus" with multiple references
- **General**: Default fallback for all other queries

## 🎨 Frontend Features

- **Multi-Chat Support**: Create and manage multiple conversation threads
- **Persistent Storage**: Chats saved in `localStorage`
- **Theme Toggle**: Light/dark mode support
- **Source Highlighting**: Inline highlighting of relevant phrases in source excerpts
- **PDF Export**: Export conversations using jsPDF
- **Context Expansion**: View surrounding chunks for better understanding

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FAISS** by Facebook Research for efficient similarity search
- **Google Gemini** for embeddings and LLM capabilities
- **FastAPI** for the excellent web framework
- **React** and **Vite** for the modern frontend stack

## 📚 Additional Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Google Gemini API](https://ai.google.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

**Note**: This application requires pre-indexed documents. The indexing pipeline is separate from this repository. Ensure you have `faiss.index` and `metadata.json` files before running the application.
