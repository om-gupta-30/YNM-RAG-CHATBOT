# RAG PDF Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/om-gupta-30/YNM-RAG-CHATBOT/workflows/CI/badge.svg)](https://github.com/om-gupta-30/YNM-RAG-CHATBOT/actions)

A **Retrieval-Augmented Generation (RAG)** application for intelligent question-answering over technical PDF documents. It combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** for accurate, source-cited answers with support for figures, tables, pages, and confidence scoring.

**Author:** Om Gupta  
**Repository:** [github.com/om-gupta-30/YNM-RAG-CHATBOT](https://github.com/om-gupta-30/YNM-RAG-CHATBOT)

> **🔒 Security Notice:** This project uses API keys. Never commit `.env` files. See [SECURITY.md](SECURITY.md) for details.

---

## Features

- **Intelligent Q&A** — Ask natural-language questions about indexed PDF content
- **Intent-aware retrieval** — Classifies queries (figure, table, page, section, general, comparison) and uses the right retrieval strategy
- **Structured answers** — Paragraphs and lists with source citations
- **Modern UI** — React chat interface with dark/light theme, multi-chat, and PDF export
- **Confidence scoring** — High/Medium/Low per answer

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Main documentation (you are here) |
| [SETUP.md](SETUP.md) | Detailed setup instructions |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Platform-specific deployment guides |
| [SECURITY.md](SECURITY.md) | Security policy and best practices |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes |

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
├── requirements-dev.txt    # Development dependencies (optional)
├── Makefile                # Dev commands (install, dev, build, clean)
│
├── metadata.json           # Chunk metadata (gitignored)
├── vision_captions.json    # Vision captions cache (gitignored)
├── faiss.index             # FAISS vector index (gitignored)
├── images/                 # Page images (27MB, required for deployment)
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
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI/CD
│
├── .env.example            # Environment template (safe to commit)
├── .gitignore              # Git ignore rules
├── .dockerignore           # Docker ignore rules
├── .gcloudignore           # GCP ignore rules
├── Dockerfile              # Docker configuration
├── vercel.json             # Vercel deployment config
│
├── README.md               # This file
├── DEPLOYMENT.md           # Deployment guide
├── CONTRIBUTING.md         # Contribution guidelines
├── SECURITY.md             # Security policy
└── LICENSE                 # MIT License
```

**Note:** Files marked as "gitignored" are excluded from version control but required for running the application.

---

## Quick Start

> **📖 Detailed setup instructions:** See [SETUP.md](SETUP.md) for comprehensive setup guide.

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Setup

**1. Clone the repository:**

```bash
git clone https://github.com/om-gupta-30/YNM-RAG-CHATBOT.git
cd YNM-RAG-CHATBOT
```

**2. Set up environment variables:**

```bash
make setup-env
# This creates .env from .env.example
# Edit .env and add your GEMINI_API_KEY
```

Get your Gemini API key from: https://aistudio.google.com/app/apikey

**3. Install dependencies:**

```bash
make install
```

This installs both backend (Python) and frontend (Node.js) dependencies.

**Optional - Use a virtual environment (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-backend
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

**Verify everything is working:**

```bash
make health           # Check backend health
make status           # Show running processes
```

Or run backend and frontend separately:

```bash
make dev-backend      # http://localhost:8000
make dev-frontend     # http://localhost:5173
```

---

## Environment Variables

| Variable         | Required | Description                                  |
|------------------|----------|----------------------------------------------|
| `GEMINI_API_KEY` | Yes      | Google Gemini API key (LLM + embeddings)     |

Copy `.env.example` to `.env` and add your key. `.env` is gitignored and will never be pushed.

**Security Note:** See [SECURITY.md](SECURITY.md) for detailed security guidelines and best practices.

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

### Setup & Installation
```bash
make install          # Install all dependencies (backend + frontend)
make install-backend  # Install Python dependencies only
make install-frontend # Install Node dependencies only
make setup-env        # Create .env from .env.example
make check-env        # Verify environment variables are set
```

### Development
```bash
make dev              # Run backend + frontend concurrently (recommended)
make dev-backend      # Run FastAPI backend only (http://localhost:8000)
make dev-frontend     # Run Vite dev server only (http://localhost:5173)
make health           # Check backend health endpoint
make status           # Show running processes on dev ports
```

### Build & Quality
```bash
make build            # Build frontend for production
make lint             # Lint frontend code
make lint-backend     # Lint Python code (requires black/flake8)
make test             # Run tests (placeholder)
```

### Utilities
```bash
make kill             # Kill all processes on dev ports (8000, 5173, 5174)
make rebuild-index    # Rebuild FAISS index from PDF
make clean            # Remove build artifacts and caches
make clean-all        # Deep clean (includes node_modules, venv, indexes)
make verify-deploy    # Verify project is safe to deploy (security check)
```

---

## Deployment

This application can be deployed to various platforms including Vercel, Google Cloud Platform, Railway, and Render.

**📖 See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment instructions.**

### Quick Deploy Options

| Platform | Best For | Complexity |
|----------|----------|------------|
| **Vercel** | Full-stack serverless | Easy |
| **GCP Cloud Run** | Scalable production | Medium |
| **Railway** | Quick deployment | Easy |
| **Render** | Simple hosting | Easy |
| **Docker** | Custom infrastructure | Medium |

### Pre-Deployment Checklist

- [ ] Run `make verify-deploy` to check for security issues
- [ ] Run `make check-env` to verify environment setup
- [ ] Ensure `.env` is **NOT** committed (check with `git status`)
- [ ] Test locally with `make dev`
- [ ] Build frontend successfully with `make build`
- [ ] Review [SECURITY.md](SECURITY.md) for security best practices

**Quick verification:**
```bash
make verify-deploy    # Automated security check
```

---

## Security & Best Practices

### Environment Variables

- **NEVER commit `.env` files** — Always use `.env.example` as a template
- **Use environment variables** for all secrets (API keys, credentials)
- **Rotate exposed keys immediately** — If you accidentally commit a key, revoke it and generate a new one

### Git Safety

The `.gitignore` is configured to prevent committing:
- `.env` and all `.env.*` files (except `.env.example`)
- API keys, credentials, certificates (`.key`, `.pem`, `credentials.json`)
- Build artifacts (`faiss.index`, `vision_captions.json`, `metadata.json`)
- Cache directories (`__pycache__`, `node_modules`, `.vite`)

### Deployment Checklist

- [ ] Remove or rotate any API keys from `.env`
- [ ] Verify `.gitignore` is comprehensive
- [ ] Set environment variables in deployment platform
- [ ] Test with `make check-env` before deploying
- [ ] Review `git status` to ensure no secrets are staged
- [ ] Use `git log --all --full-history -- .env` to check if `.env` was ever committed

---

## Troubleshooting

### Backend won't start
```bash
make check-env          # Verify environment variables
make kill               # Kill any stuck processes
make dev-backend        # Start backend only
```

### Frontend build fails
```bash
make clean              # Clean build artifacts
make install-frontend   # Reinstall dependencies
make build              # Rebuild
```

### FAISS index missing
```bash
make rebuild-index      # Regenerate from PDF
```

### Port already in use
```bash
make kill               # Kill processes on dev ports
make status             # Check what's running
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick steps:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add: amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) (Meta) — Vector similarity search
- [Google Gemini](https://ai.google.dev/) — LLM and embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [React](https://react.dev/) — UI library
- [Vite](https://vitejs.dev/) — Frontend build tool

---

## Project Stats

- **Lines of Code:** ~1,900 (Python) + ~1,200 (JavaScript/React)
- **Dependencies:** 8 Python packages, 6 Node packages
- **Documentation:** 7 comprehensive guides
- **CI/CD:** GitHub Actions with linting and security checks
- **Deployment:** Ready for Vercel, GCP, Railway, Render, Docker

---

## Quick Links

- 📖 [Setup Guide](SETUP.md)
- 🚀 [Deployment Guide](DEPLOYMENT.md)
- 🔒 [Security Policy](SECURITY.md)
- 🤝 [Contributing](CONTRIBUTING.md)
- 📝 [Changelog](CHANGELOG.md)
- ✅ [Project Status](PROJECT_STATUS.md)
