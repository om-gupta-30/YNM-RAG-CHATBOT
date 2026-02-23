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

| Layer             | Technology                                                    |
|-------------------|---------------------------------------------------------------|
| Backend           | Python 3.10+, FastAPI, Uvicorn                                |
| Vector Search     | FAISS (CPU), NumPy                                            |
| Embeddings / LLM  | Google Gemini (`gemini-embedding-001`, `gemini-2.5-flash`)   |
| Vision            | Pillow (page image captions via Gemini)                       |
| Frontend          | React 19, Vite 7                                              |

---

## Project Structure

```
rag-chatbot/
├── app.py                  # FastAPI backend (RAG pipeline, all endpoints)
├── intent_classifier.py    # Query intent classification
├── rebuild_index.py        # Rebuild FAISS index from PDF
├── requirements.txt        # Python dependencies
├── Makefile                # Dev commands (install, dev, build, clean)
│
├── metadata.json           # Chunk metadata
├── vision_captions.json    # Vision captions cache
├── faiss.index             # FAISS vector index (generated, gitignored)
├── images/                 # Page images
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
├── README.md
└── LICENSE
```

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

**Or just:**

```bash
make install
```

### Generate the FAISS Index

If `faiss.index` does not exist, generate it:

```bash
make rebuild-index
```

### Run

```bash
make dev
```

This starts the backend at http://localhost:8000 and the frontend at http://localhost:5173.

Or run them separately:

```bash
make dev-backend           # http://localhost:8000
make dev-frontend          # http://localhost:5173
```

---

## Environment Variables

| Variable         | Required | Description                                  |
|------------------|----------|----------------------------------------------|
| `GEMINI_API_KEY` | Yes      | Google Gemini API key (LLM + embeddings)     |

Copy `.env.example` to `.env` and add your key. `.env` is gitignored and will never be pushed.

---

## API Endpoints

| Method | Endpoint               | Description            |
|--------|------------------------|------------------------|
| GET    | `/health`              | Health and env check   |
| POST   | `/ask`                 | Main RAG Q&A           |
| POST   | `/classify-intent`     | Query intent           |
| POST   | `/expand-context`      | Surrounding chunks     |
| POST   | `/generate-chat-title` | Title from question    |

Interactive docs at `/docs` (Swagger UI) when running.

---

## Makefile Commands

```
make install          Install all dependencies (backend + frontend)
make dev              Run backend + frontend concurrently
make dev-backend      Run FastAPI backend only
make dev-frontend     Run Vite dev server only
make build            Build frontend for production
make lint             Lint frontend code
make kill             Kill processes on dev ports
make rebuild-index    Rebuild FAISS index
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
