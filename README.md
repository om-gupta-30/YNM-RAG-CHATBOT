# RAG PDF Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A **Retrieval-Augmented Generation (RAG)** application for intelligent question-answering over technical PDF documents. This system combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** to provide accurate, source-cited answers with support for figures, tables, pages, and confidence scoring.

**Created by Om Gupta**

---

## ✨ Features

- **🤖 Intelligent Q&A**: Ask natural-language questions about indexed PDF documents
- **🎯 Intent-Aware Retrieval**: Automatically classifies queries (figure, table, page, section, general, comparison) and uses optimal retrieval strategies
- **📝 Structured Answers**: Returns formatted paragraphs and lists with source citations
- **🎨 Modern UI**: React-based chat interface with dark/light theme, multi-chat support, and PDF export
- **📊 Confidence Scoring**: Each answer includes High/Medium/Low confidence indicators
- **📚 Source Citations**: Answers include page numbers, text excerpts, and page images
- **⚡ Fast Retrieval**: FAISS-based semantic search for efficient document querying
- **🔒 Secure**: Environment-based API key management with no hardcoded secrets

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Security](#security)
- [License](#license)

---

## 🔍 How It Works

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

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Vector DB** | FAISS (CPU), NumPy |
| **Embeddings** | Google Gemini Embedding Model |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **PDF Processing** | PyMuPDF (fitz), pdfplumber, Pillow |
| **Frontend** | React 19, Vite 7 |
| **Export** | jsPDF |
| **Deployment** | Docker, Google Cloud Run |

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                  # FastAPI backend (main application)
├── intent_classifier.py    # Query intent classification module
├── requirements.txt        # Python dependencies
│
├── metadata.json           # Chunk metadata (text, source, page numbers)
├── vision_captions.json    # Gemini vision captions cache
├── faiss.index            # FAISS vector index (generated, not in git)
├── images/                # Page images (page_{n}_img_1.png)
│
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── App.jsx        # Main chat UI component
│   │   ├── App.css        # Component styles
│   │   ├── api.js         # API client functions
│   │   ├── main.jsx       # React entry point
│   │   └── index.css      # Global styles
│   ├── index.html         # HTML template
│   ├── package.json       # NPM dependencies
│   └── vite.config.js     # Vite configuration
│
├── .env.example           # Example environment file (copy to .env)
├── .gitignore             # Git ignore rules
├── .dockerignore          # Docker ignore rules
├── .gcloudignore          # GCP ignore rules
├── Dockerfile             # Multi-stage Docker build
├── deploy-gcp.sh          # GCP Cloud Run deployment script
└── README.md              # This file
```

**Important**: The `faiss.index` file must be generated from your PDF document and is excluded from git due to its binary nature and size.

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18 or higher (for frontend development)
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Set up Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Copy the example file and add your API key:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Set up frontend** (for development)
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

Access the application:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Interactive Swagger UI)
- **Frontend**: http://localhost:5173

#### Production Build

Build the frontend and serve everything through the backend:

```bash
cd frontend
npm run build
cd ..
uvicorn app:app --host 0.0.0.0 --port 8000
```

The built frontend files in `frontend/dist/` will be served automatically by the FastAPI backend.

---

## 🌐 API Endpoints

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

**Health Check**:
```bash
curl http://localhost:8000/health
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM and embeddings |

### Required Data Files

The application requires these data files at runtime:

| File | Description | In Git? |
|------|-------------|---------|
| `faiss.index` | FAISS vector index for semantic search | ❌ No (generate from your PDF) |
| `metadata.json` | Chunk metadata mapping IDs to text and page numbers | ✅ Yes |
| `vision_captions.json` | Gemini vision captions cache | ✅ Yes |
| `images/` | Directory containing page images | ✅ Yes |

> **Note**: The `faiss.index` file must be generated from your PDF document using an embedding pipeline. It is excluded from git due to its binary nature and size.

---

## 🚢 Deployment

### Docker

Build and run using Docker:

```bash
# Build the image
docker build -t rag-chatbot .

# Run the container
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_api_key_here \
  rag-chatbot
```

Access the application at http://localhost:8080

### Google Cloud Run

Deploy to Google Cloud Run using the provided script:

```bash
# Set required environment variables
export GCP_PROJECT_ID=your-gcp-project-id
export GEMINI_API_KEY=your_gemini_api_key

# Optional configuration
export GCP_REGION=asia-south1              # Default: asia-south1
export GCP_SERVICE_NAME=rag-pdf-chatbot    # Default: rag-pdf-chatbot

# Make script executable and deploy
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

The script will:
1. Enable required GCP APIs
2. Build the Docker image using Cloud Build
3. Deploy to Cloud Run with your environment variables
4. Verify the deployment and health endpoint

**Required GCP Environment Variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `GCP_PROJECT_ID` | Yes | Your Google Cloud project ID |
| `GEMINI_API_KEY` | Yes | Google Gemini API key (set as Cloud Run env var) |
| `GCP_REGION` | No | GCP region (default: asia-south1) |
| `GCP_SERVICE_NAME` | No | Cloud Run service name (default: rag-pdf-chatbot) |

### Vercel

For Vercel deployment, you'll need to:
1. Set the `GEMINI_API_KEY` environment variable in your Vercel project settings
2. Configure build settings for Python backend and React frontend
3. Use Vercel's serverless functions for the FastAPI backend

---

## 🔒 Security

### API Key Protection

Your Gemini API key is protected by multiple layers:

| Protection | File | What it does |
|------------|------|--------------|
| **Git** | `.gitignore` | Excludes `.env` and all `.env.*` files from commits |
| **Docker** | `.dockerignore` | Excludes `.env` files from Docker image builds |
| **GCP** | `.gcloudignore` | Excludes `.env` files from Cloud Run deployments |
| **Architecture** | `app.py` | API key used server-side only, never exposed to frontend |

### Security Checklist

Before pushing to GitHub/Vercel/GCP:

- ✅ `.env` file exists locally but is NOT tracked by git
- ✅ `git status` does not show any `.env` files
- ✅ Use environment variables for all secrets in production
- ✅ Never hardcode API keys in source code
- ✅ All ignore files (`.gitignore`, `.dockerignore`, `.gcloudignore`) properly exclude secrets
- ✅ Review git history to ensure no secrets were previously committed

**For Production Deployments**:
- **Google Cloud Run**: Set `GEMINI_API_KEY` via `--set-env-vars` (handled by `deploy-gcp.sh`)
- **Vercel**: Set environment variables in project settings dashboard
- **Docker**: Pass via `-e` flag or Docker Compose environment section

### Verifying Security

Run these commands before pushing to GitHub:

```bash
# Check if .env is ignored
git check-ignore -v .env

# Verify no .env files are tracked
git ls-files | grep "\.env"

# Check for any hardcoded secrets in code
grep -r "AIzaSy" --include="*.py" --include="*.js" --include="*.jsx" .
```

All commands should return no results or confirm .env is ignored.

---

## 🎨 Frontend Features

- **💬 Multi-Chat Support**: Create and manage multiple conversation threads
- **💾 Persistent Storage**: Chats saved in browser localStorage
- **🌓 Theme Toggle**: Light and dark mode support
- **🔍 Source Highlighting**: Inline highlighting of relevant phrases
- **📄 PDF Export**: Export conversations to PDF
- **🔎 Context Expansion**: View surrounding chunks for better understanding
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

---

## 📝 License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 Om Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Meta Research for efficient similarity search
- [Google Gemini](https://ai.google.dev/) for embeddings and LLM capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent Python web framework
- [React](https://react.dev/) and [Vite](https://vitejs.dev/) for the modern frontend stack
- [PyMuPDF](https://pymupdf.readthedocs.io/) and [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF processing

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

---

## 📧 Contact

**Om Gupta** - Creator and Maintainer

For questions, issues, or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
