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
- [License](#license)

---

## How It Works

1. **Intent classification** — Query is classified (figure, table, page, section, general, comparison).
2. **Retrieval** — Exact match for specific refs (e.g. “Fig 3.3”) or FAISS semantic search for general questions.
3. **Context building** — Retrieved chunks are assembled into context.
4. **LLM generation** — Context is sent to Gemini with a structured prompt.
5. **Response** — JSON is parsed, confidence is scored, and sources are attached.

| Intent          | Example                 | Strategy              |
|----------------|-------------------------|------------------------|
| FIGURE_QUERY   | "What does Fig 3.3 show?" | Exact figure ref     |
| TABLE_QUERY    | "Table 6.2 guidelines"   | Exact table ref        |
| PAGE_QUERY     | "What is on page 27?"   | Page number            |
| GENERAL_QUERY  | "Size of STOP sign?"    | FAISS semantic         |
| COMPARISON_QUERY | "Compare Fig 3.1 and 3.2" | Both refs            |

---

## Tech Stack

| Layer     | Technology                          |
|----------|--------------------------------------|
| Backend  | Python 3.10+, FastAPI, Uvicorn      |
| Vector   | FAISS (CPU), NumPy                  |
| Embeddings / LLM | Google Gemini (embedding + `gemini-2.5-flash`) |
| PDF      | PyMuPDF, pdfplumber, Pillow         |
| Frontend | React 19, Vite 7                     |
| Deploy   | Docker, Google Cloud Run             |

---

## Project Structure

```
rag-chatbot/
├── app.py                 # FastAPI backend
├── intent_classifier.py   # Query intent classification
├── requirements.txt       # Python dependencies
├── metadata.json          # Chunk metadata (in repo)
├── vision_captions.json   # Vision captions cache (in repo)
├── images/                # Page images (in repo)
├── faiss.index            # FAISS index (not in repo; generate from PDF)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main UI
│   │   ├── api.js         # API client
│   │   ├── main.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── eslint.config.js
│
├── .env.example           # Env template (no secrets)
├── .gitignore
├── .dockerignore
├── .gcloudignore
├── Dockerfile
├── deploy-gcp.sh          # Cloud Run deploy script
├── README.md
├── LICENSE
└── SECURITY.md
```

**Note:** `faiss.index` is generated from your PDF and is in `.gitignore` (binary/large). You must generate it for the app to run.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
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

### Run

**Development**

- Backend (from repo root): `uvicorn app:app --reload --host 0.0.0.0 --port 8000`
- Frontend: `cd frontend && npm run dev`
- API: http://localhost:8000 — Docs: http://localhost:8000/docs  
- UI: http://localhost:5173

**Production (single process)**

```bash
cd frontend && npm run build && cd ..
uvicorn app:app --host 0.0.0.0 --port 8000
```

Serve the app on port 8000; the built frontend in `frontend/dist/` is served by FastAPI.

---

## Configuration

### Environment Variables

| Variable         | Required | Description                                      |
|------------------|----------|--------------------------------------------------|
| `GEMINI_API_KEY` | Yes      | Google Gemini API key (LLM + embeddings)         |
| `VITE_API_URL`   | No (frontend) | Backend URL for frontend (default: `http://127.0.0.1:8000`) |
| `PORT`           | No       | Server port (default 8000; Cloud Run uses 8080)  |

- **Local:** Use `.env` (copy from `.env.example`). Never commit `.env`.
- **Vercel:** Set env vars in Project Settings → Environment Variables.
- **GCP Cloud Run:** Set via `deploy-gcp.sh` or Cloud Console (e.g. `GEMINI_API_KEY`).

### Required Data Files

| File / dir          | In repo | Notes                    |
|---------------------|--------|---------------------------|
| `metadata.json`     | Yes    | Chunk metadata            |
| `vision_captions.json` | Yes  | Vision cache              |
| `images/`           | Yes    | Page images               |
| `faiss.index`       | No     | Generate from your PDF    |

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
export GEMINI_API_KEY=your_gemini_key   # Or use .env
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

Optional: `GCP_REGION`, `GCP_SERVICE_NAME`. The script sets `GEMINI_API_KEY` as a Cloud Run env var (not in the image).

### Vercel (frontend only)

For a frontend-only deploy, set:

- `VITE_API_URL` — URL of your backend (e.g. Cloud Run or another host).

Backend must be deployed separately (e.g. Cloud Run, Railway, or another host). Do not put `GEMINI_API_KEY` in the frontend; it is only used on the backend.

---

## Security

- **No secrets in repo** — `.env` and `.env.*` are in `.gitignore`. Only `.env.example` (no real values) is committed.
- **Before you push:** Run `git status` and ensure `.env` (and any `*.pem`, `secrets.*`, etc.) are not staged. Run `git check-ignore -v .env` to confirm `.env` is ignored.
- **Health endpoint** — Returns only `"GEMINI_API_KEY": "set"` or `"missing"`, never the key.
- **Backend only** — The Gemini API key is read in `app.py` from the environment and never sent to the client.

See [SECURITY.md](SECURITY.md) for reporting issues and what to do if you accidentally commit a secret.

---

## API Endpoints

| Method | Endpoint               | Description           |
|--------|------------------------|------------------------|
| GET    | `/health`              | Health and env check   |
| POST   | `/ask`                 | Main RAG Q&A            |
| POST   | `/classify-intent`     | Query intent            |
| POST   | `/expand-context`      | Surrounding chunks      |
| POST   | `/generate-chat-title` | Title from question     |

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
