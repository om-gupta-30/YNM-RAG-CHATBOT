# RAG PDF Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A **Retrieval-Augmented Generation (RAG)** application for intelligent question-answering over technical PDF documents. This system combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** to provide accurate, source-cited answers with support for figures, tables, pages, and confidence scoring.

**Created by Om Gupta**

---

## Features

- **Intelligent Q&A**: Ask natural-language questions about indexed PDF documents
- **Intent-Aware Retrieval**: Automatically classifies queries (figure, table, page, section, general, comparison) and uses optimal retrieval strategies
- **Structured Answers**: Returns formatted paragraphs and lists with source citations
- **Modern UI**: React-based chat interface with dark/light theme, multi-chat support, and PDF export
- **Confidence Scoring**: Each answer includes High/Medium/Low confidence indicators
- **Source Citations**: Answers include page numbers, text excerpts, and page images
- **Fast Retrieval**: FAISS-based semantic search for efficient document querying

---

## Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## How It Works

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on the LLM's training data, this system:

1. **Retrieves** relevant text chunks from the corpus using the user's question
2. **Augments** the LLM prompt with retrieved chunks as context
3. **Generates** answers grounded in the provided context

This approach reduces hallucinations and ensures answers are tied to specific document pages and chunks.

### Query Processing Flow

```
User Question → Intent Classification → Retrieval → Context Building → LLM Generation → Response
```

1. **Intent Classification**: Analyzes the query to determine intent (figure, table, page, section, general, or comparison)
2. **Retrieval**: Uses exact matching for specific references or FAISS semantic search for general queries
3. **Context Building**: Assembles retrieved chunks into a context string
4. **LLM Generation**: Sends context to Gemini with a structured prompt
5. **Response Processing**: JSON parsing, confidence scoring, and source citation enrichment

### Intent Classification

| Intent | Example | Retrieval Strategy |
|--------|---------|-------------------|
| `FIGURE_QUERY` | "What does Fig 3.3 show?" | Exact match on figure reference |
| `TABLE_QUERY` | "Table 6.2 guidelines" | Exact match on table reference |
| `PAGE_QUERY` | "What is on page 27?" | All chunks with matching page number |
| `SECTION_QUERY` | "Section on mandatory signs" | Semantic/FAISS search |
| `GENERAL_QUERY` | "Size of STOP sign?" | FAISS semantic search |
| `COMPARISON_QUERY` | "Compare Fig 3.1 and Fig 3.2" | Retrieves both referenced items |

### Confidence Scoring

Each answer block includes a confidence score (High/Medium/Low) based on:
- Source type quality (OCR/IMAGE vs TEXT)
- Number of supporting chunks
- Semantic match score
- Verbatim presence in retrieved text

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Vector DB** | FAISS (CPU), NumPy |
| **Embeddings** | Google `embedding-001` |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **PDF Processing** | PyMuPDF (fitz), pdfplumber, Pillow |
| **Frontend** | React 19, Vite 7 |
| **Export** | jsPDF |
| **Deployment** | Docker, GCP Cloud Run |

---

## Project Structure

```
rag-chatbot/
├── app.py                  # FastAPI backend (main application)
├── intent_classifier.py    # Query intent classification module
├── requirements.txt        # Python dependencies
│
├── metadata.json           # Chunk metadata (text, source, page numbers)
├── vision_captions.json    # Gemini vision captions cache
├── images/                 # Page images (page_{n}_img_1.png)
│
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── App.jsx         # Main chat UI component
│   │   ├── App.css         # Component styles
│   │   ├── api.js          # API client functions
│   │   ├── main.jsx        # React entry point
│   │   └── index.css       # Global styles
│   ├── index.html          # HTML template
│   ├── package.json        # NPM dependencies
│   └── vite.config.js      # Vite configuration
│
├── .env.example            # Example environment file
├── .gitignore              # Git ignore rules
├── .dockerignore           # Docker ignore rules
├── .gcloudignore           # GCP ignore rules
├── Dockerfile              # Docker configuration
├── deploy-gcp.sh           # GCP Cloud Run deployment script
└── README.md               # This file

# Generated files (not in git, required at runtime):
# └── faiss.index           # FAISS vector index (generate from your PDF)
```

---

## Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18 or higher
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/app/apikey))

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
   cd frontend
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
cd frontend
npm run dev
```

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:5173

#### Production Build

```bash
cd frontend
npm run build
cd ..
uvicorn app:app --host 0.0.0.0 --port 8000
```

The built frontend files in `frontend/dist/` will be served automatically by the backend.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `POST` | `/ask` | Main Q&A endpoint with full RAG pipeline |
| `POST` | `/classify-intent` | Classify query intent |
| `POST` | `/expand-context` | Get surrounding chunks for context |
| `POST` | `/generate-chat-title` | Generate chat title from question |

### Example Usage

**Ask a Question**:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the dimensions of a STOP sign?"}'
```

**Classify Intent**:
```bash
curl -X POST http://localhost:8000/classify-intent \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Fig 3.3 show?"}'
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM and embeddings |

### Required Data Files

The application requires these data files at runtime:

| File | Description | In Git? |
|------|-------------|---------|
| `faiss.index` | FAISS vector index for semantic search | No (generate from your PDF) |
| `metadata.json` | Chunk metadata mapping IDs to text and page numbers | Yes |
| `vision_captions.json` | Gemini vision captions cache | Yes |
| `images/` | Directory containing page images | Yes |

> **Note**: The `faiss.index` file must be generated from your PDF document using the embedding pipeline. It is excluded from git due to its binary nature and size.

---

## Deployment

### Docker

```bash
# Build the image
docker build -t rag-chatbot .

# Run the container
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_key \
  rag-chatbot
```

### GCP Cloud Run

```bash
# Set required environment variables
export GCP_PROJECT_ID=your-project-id
export GEMINI_API_KEY=your_api_key

# Optional configuration
export GCP_REGION=asia-south1        # Default: asia-south1
export GCP_SERVICE_NAME=rag-pdf-chatbot  # Default: rag-pdf-chatbot

# Deploy to Cloud Run
./deploy-gcp.sh
```

The script handles building, pushing, and deploying to Cloud Run.

| Variable | Required | Description |
|----------|----------|-------------|
| `GCP_PROJECT_ID` | Yes | Your Google Cloud project ID |
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `GCP_REGION` | No | GCP region (default: asia-south1) |
| `GCP_SERVICE_NAME` | No | Cloud Run service name (default: rag-pdf-chatbot) |

---

## Security

### API Key Protection

Your Gemini API key is protected by multiple layers:

| Protection | File | What it does |
|------------|------|--------------|
| Git | `.gitignore` | Excludes `.env` and all `.env.*` files from commits |
| Docker | `.dockerignore` | Excludes `.env` files from Docker image builds |
| GCP | `.gcloudignore` | Excludes `.env` files from Cloud Run deployments |
| Architecture | `app.py` | API key used server-side only, never exposed to frontend |

### Security Checklist

Before pushing to GitHub/Vercel/GCP:

- [ ] Ensure `.env` file exists locally but is NOT tracked by git
- [ ] Verify `git status` does not show any `.env` files
- [ ] Use environment variables for all secrets in production
- [ ] Never hardcode API keys in source code

**For GCP Cloud Run**: Set `GEMINI_API_KEY` via the `--set-env-vars` flag (handled by `deploy-gcp.sh`) or through the Cloud Run console.

---

## Frontend Features

- **Multi-Chat Support**: Create and manage multiple conversation threads
- **Persistent Storage**: Chats saved in browser localStorage
- **Theme Toggle**: Light and dark mode support
- **Source Highlighting**: Inline highlighting of relevant phrases
- **PDF Export**: Export conversations to PDF
- **Context Expansion**: View surrounding chunks for better understanding

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research for efficient similarity search
- [Google Gemini](https://ai.google.dev/) for embeddings and LLM capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [React](https://react.dev/) and [Vite](https://vitejs.dev/) for the modern frontend stack

---

**Created by Om Gupta**
