# RAG PDF Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Retrieval-Augmented Generation (RAG)** application for intelligent question-answering over technical PDF documents. It combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** for accurate, source-cited answers with support for figures, tables, pages, and confidence scoring.

**Author:** Om Gupta

---

## Features

- **Intelligent Q&A** — Ask natural-language questions about indexed PDF content
- **Intent-aware retrieval** — Classifies queries (figure, table, page, section, general, comparison) and uses the right retrieval strategy
- **Structured answers** — Paragraphs and lists with source citations
- **Modern UI** — React chat interface with dark/light theme, multi-chat, and PDF export
- **Confidence scoring** — High/Medium/Low per answer
- **Secure** — API key only on server; never exposed to the frontend

---

## Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Security](#security)
- [API Endpoints](#api-endpoints)
- [License](#license)

---

## How It Works

1. **Intent classification** — Query is classified (figure, table, page, section, general, comparison).
2. **Retrieval** — Exact match for specific refs (e.g. "Fig 3.3") or FAISS semantic search for general questions.
3. **Context building** — Retrieved chunks are assembled into context.
4. **LLM generation** — Context is sent to Gemini with a structured prompt.
5. **Response** — JSON is parsed, confidence is scored, and sources are attached.

| Intent           | Example                    | Strategy         |
|------------------|----------------------------|------------------|
| FIGURE_QUERY     | "What does Fig 3.3 show?"  | Exact figure ref |
| TABLE_QUERY      | "Table 6.2 guidelines"     | Exact table ref  |
| PAGE_QUERY       | "What is on page 27?"      | Page number      |
| GENERAL_QUERY    | "Size of STOP sign?"       | FAISS semantic   |
| COMPARISON_QUERY | "Compare Fig 3.1 and 3.2"  | Both refs        |

---

## Tech Stack

| Layer             | Technology                                        |
|-------------------|---------------------------------------------------|
| Backend           | Python 3.10+, FastAPI, Uvicorn                    |
| Vector Search     | FAISS (CPU), NumPy                                |
| Embeddings / LLM  | Google Gemini (`gemini-embedding-001`, `gemini-2.5-flash`) |
| PDF Processing    | PyMuPDF, pdfplumber, Pillow                       |
| Frontend          | React 19, Vite 7                                  |
| Deployment        | Docker, Google Cloud Run, Vercel (frontend)       |

---

## Project Structure

```
rag-chatbot/
├── app.py                  # FastAPI backend (RAG pipeline, all endpoints)
├── intent_classifier.py    # Query intent classification
├── rebuild_index.py        # Rebuild FAISS index with improved chunking
├── requirements.txt        # Python dependencies
├── Makefile                # Dev commands (install, dev, build, docker, clean)
│
├── metadata.json           # Chunk metadata (tracked)
├── vision_captions.json    # Vision captions cache (tracked)
├── faiss.index             # FAISS vector index (generated, not tracked)
├── images/                 # Page images (tracked)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── api.js          # API client
│   │   ├── main.jsx        # Entry point
│   │   ├── App.css         # Component styles
│   │   └── index.css       # Global styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── eslint.config.js
│
├── .env.example            # Env template (no secrets)
├── .gitignore
├── .dockerignore
├── .gcloudignore
├── Dockerfile              # Multi-stage Docker build
├── deploy-gcp.sh           # Cloud Run deployment script
├── README.md
├── SECURITY.md
└── LICENSE
```

> **Note:** `faiss.index` is generated from your PDF and listed in `.gitignore`. You must generate it before the app can run (see [Quick Start](#quick-start)).

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

**Backend:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key
```

**Frontend:**

```bash
cd frontend && npm install && cd ..
```

**Or use the Makefile:**

```bash
make install               # Install backend + frontend deps
```

### Generate the FAISS Index

If `faiss.index` does not exist, generate it:

```bash
make rebuild-index         # or: python3 rebuild_index.py
```

### Run

**Development (recommended):**

```bash
make dev                   # Starts backend + frontend concurrently
```

Or run them separately:

```bash
make dev-backend           # Backend: http://localhost:8000
make dev-frontend          # Frontend: http://localhost:5173
```

**Production (single process):**

```bash
cd frontend && npm run build && cd ..
uvicorn app:app --host 0.0.0.0 --port 8000
```

The built frontend in `frontend/dist/` is served by FastAPI at http://localhost:8000.

---

## Configuration

### Environment Variables

| Variable         | Required | Description                                              |
|------------------|----------|----------------------------------------------------------|
| `GEMINI_API_KEY` | Yes      | Google Gemini API key (LLM + embeddings)                 |
| `VITE_API_URL`   | No       | Backend URL for frontend (default: `http://127.0.0.1:8000`) |
| `PORT`           | No       | Server port (default 8000; Cloud Run uses 8080)          |

- **Local:** Copy `.env.example` to `.env` and fill in your key. Never commit `.env`.
- **Vercel:** Set env vars in Project Settings → Environment Variables.
- **GCP Cloud Run:** Set via `deploy-gcp.sh` or Cloud Console.

### Required Data Files

| File / Directory       | In Repo | Notes                              |
|------------------------|---------|------------------------------------|
| `metadata.json`        | Yes     | Chunk metadata                     |
| `vision_captions.json` | Yes     | Vision caption cache               |
| `images/`              | Yes     | Page images                        |
| `faiss.index`          | No      | Generate with `make rebuild-index` |

---

## Deployment

### Docker

```bash
docker build -t rag-chatbot .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key rag-chatbot
```

### Google Cloud Run

```bash
export GCP_PROJECT_ID=your-project-id
export GEMINI_API_KEY=your_gemini_key
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

Optional env vars: `GCP_REGION` (default: `asia-south1`), `GCP_SERVICE_NAME` (default: `rag-pdf-chatbot`). The script sets `GEMINI_API_KEY` as a Cloud Run runtime env var — it is never baked into the image.

### Vercel (Frontend Only)

Deploy the `frontend/` directory to Vercel and set:

- `VITE_API_URL` — URL of your backend (e.g. your Cloud Run service URL)

The backend must be deployed separately. Do **not** put `GEMINI_API_KEY` in the frontend — it is only used on the backend.

---

## Security

- **No secrets in the repo** — `.env` and all `.env.*` variants are in `.gitignore`. Only `.env.example` (with no real values) is committed.
- **Before you push:** Run `git status` and ensure `.env` is not staged. Run `git check-ignore -v .env` to confirm.
- **Health endpoint** — Returns only `"GEMINI_API_KEY": "set"` or `"missing"`, never the actual key.
- **Backend only** — The Gemini API key is read from the environment in `app.py` and never sent to the client.

See [SECURITY.md](SECURITY.md) for more details and what to do if you accidentally commit a secret.

---

## API Endpoints

| Method | Endpoint               | Description            |
|--------|------------------------|------------------------|
| GET    | `/health`              | Health and env check   |
| POST   | `/ask`                 | Main RAG Q&A           |
| POST   | `/classify-intent`     | Query intent           |
| POST   | `/expand-context`      | Surrounding chunks     |
| POST   | `/generate-chat-title` | Title from question    |

Interactive docs available at `/docs` (Swagger UI) when running.

---

## Makefile Commands

```
make install          Install all dependencies (backend + frontend)
make dev              Run backend + frontend concurrently
make dev-backend      Run FastAPI backend only
make dev-frontend     Run Vite dev server only
make build            Build frontend for production
make lint             Lint frontend code
make docker-build     Build Docker image
make docker-run       Run Docker container
make docker-stop      Stop Docker container
make kill             Kill processes on dev ports
make rebuild-index    Rebuild FAISS index from metadata
make clean            Remove build artifacts and caches
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) (Meta)
- [Google Gemini](https://ai.google.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/) & [Vite](https://vitejs.dev/)
- [PyMuPDF](https://pymupdf.readthedocs.io/) & [pdfplumber](https://github.com/jsvine/pdfplumber)
