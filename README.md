# RAG PDF Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/om-gupta-30/YNM-RAG-CHATBOT/workflows/CI/badge.svg)](https://github.com/om-gupta-30/YNM-RAG-CHATBOT/actions)

A **Retrieval-Augmented Generation (RAG)** chatbot for intelligent question-answering over technical PDF documents. Combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** for accurate, source-cited answers with support for figures, tables, pages, and confidence scoring.

> **Security:** This project uses API keys loaded from environment variables. Never commit `.env` files. See [SECURITY.md](SECURITY.md) for details.

---

## Features

- **Intelligent Q&A** — Ask natural-language questions about indexed PDF content
- **Intent-aware retrieval** — Classifies queries (figure, table, page, section, general, comparison) and applies the right retrieval strategy
- **Structured answers** — Paragraphs and lists with source citations
- **Confidence scoring** — High / Medium / Low per answer
- **Modern UI** — React chat interface with dark/light theme, multi-chat, and PDF export

---

## How It Works

1. **Intent classification** — The query is classified (figure, table, page, section, general, comparison).
2. **Retrieval** — Exact match for specific references (e.g. "Fig 3.3") or FAISS semantic search for general questions.
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

| Layer             | Technology                                                  |
|-------------------|-------------------------------------------------------------|
| Backend           | Python 3.10+, FastAPI, Uvicorn                              |
| Vector Search     | FAISS (CPU), NumPy                                          |
| Embeddings / LLM  | Google Gemini (`gemini-embedding-001`, `gemini-2.5-flash`) |
| Vision            | Pillow (page image captions via Gemini)                     |
| Frontend          | React 19, Vite 7                                            |

---

## Project Structure

```
rag-chatbot/
├── app.py                  # FastAPI backend (RAG pipeline, all endpoints)
├── intent_classifier.py    # Query intent classification
├── rebuild_index.py        # Rebuild FAISS index from PDF
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Development dependencies (optional)
├── Makefile                # Dev commands (install, dev, build, clean)
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
├── images/                 # Page images (required at runtime)
├── scripts/
│   └── verify-deployment.sh
│
├── .github/workflows/
│   └── ci.yml              # CI/CD: lint, build, security checks
│
├── Dockerfile              # Multi-stage Docker build
├── vercel.json             # Vercel deployment config
├── .env.example            # Environment variable template
├── .gitignore
├── .dockerignore
├── .gcloudignore
├── .pre-commit-config.yaml
│
├── SETUP.md                # Detailed setup guide
├── DEPLOYMENT.md           # Deployment guide (Vercel, GCP, Railway, Render, Docker)
├── CONTRIBUTING.md         # Contribution guidelines
├── SECURITY.md             # Security policy
├── CHANGELOG.md            # Version history
└── LICENSE                 # MIT
```

**Gitignored data files** (generated locally, not in the repository):
- `faiss.index` — FAISS vector index
- `metadata.json` — Chunk metadata
- `vision_captions.json` — Vision caption cache

---

## Quick Start

> For detailed setup instructions see [SETUP.md](SETUP.md).

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone and configure

```bash
git clone https://github.com/om-gupta-30/YNM-RAG-CHATBOT.git
cd YNM-RAG-CHATBOT

make setup-env          # creates .env from .env.example
# edit .env and add your GEMINI_API_KEY
```

### 2. Install dependencies

```bash
make install            # installs both Python and Node dependencies
```

Using a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

### 3. Generate the FAISS index

If `faiss.index` and `metadata.json` do not exist yet:

```bash
make rebuild-index
```

### 4. Run

```bash
make dev
```

Backend: http://localhost:8000 | Frontend: http://localhost:5173

```bash
make health             # verify backend is responding
make status             # show running processes
```

---

## Environment Variables

| Variable         | Required | Description                              |
|------------------|----------|------------------------------------------|
| `GEMINI_API_KEY` | Yes      | Google Gemini API key (LLM + embeddings) |
| `PORT`           | No       | Server port (default: `8000`)            |
| `HOST`           | No       | Server host (default: `0.0.0.0`)         |

Copy `.env.example` to `.env` and fill in your key. `.env` is gitignored and will never be pushed.

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

```bash
# Setup
make install            # Install all dependencies
make setup-env          # Create .env from template
make check-env          # Verify environment variables

# Development
make dev                # Run backend + frontend
make dev-backend        # Backend only
make dev-frontend       # Frontend only
make health             # Check backend health
make status             # Show running processes

# Build & Quality
make build              # Production frontend build
make lint               # Lint frontend
make lint-backend       # Lint Python (requires black/flake8)

# Utilities
make kill               # Kill dev server processes
make rebuild-index      # Rebuild FAISS index
make clean              # Remove build artifacts
make clean-all          # Deep clean (node_modules, venv, indexes)
make verify-deploy      # Pre-deployment security check
```

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.

| Platform         | Complexity | Notes                     |
|------------------|------------|---------------------------|
| **Vercel**       | Easy       | Full-stack serverless     |
| **GCP Cloud Run**| Medium     | Scalable, containerized   |
| **Railway**      | Easy       | Git-push deploy           |
| **Render**       | Easy       | Git-push deploy           |
| **Docker**       | Medium     | Any container platform    |

### Pre-deployment checklist

```bash
make verify-deploy      # automated security check
make check-env          # verify environment
make build              # ensure frontend builds
```

---

## Security

- `.env` is gitignored and was **never** committed to git history
- No hardcoded API keys in source code
- CI/CD runs secret scanning on every push
- Pre-commit hooks available for local secret detection

See [SECURITY.md](SECURITY.md) for the full security policy.

---

## Troubleshooting

| Problem                  | Fix                                             |
|--------------------------|-------------------------------------------------|
| Backend won't start      | `make check-env` then `make kill && make dev`   |
| Frontend build fails     | `make clean && make install-frontend && make build` |
| FAISS index missing      | `make rebuild-index`                            |
| Port already in use      | `make kill`                                     |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork → clone → branch → code → push → PR
git checkout -b feature/your-feature
# make changes
git commit -m "Add: description"
git push origin feature/your-feature
```

---

## License

[MIT](LICENSE)

---

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) (Meta) — Vector similarity search
- [Google Gemini](https://ai.google.dev/) — LLM and embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
- [React](https://react.dev/) — UI library
- [Vite](https://vitejs.dev/) — Frontend build tool
