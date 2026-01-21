# RAG PDF Chatbot

A **Retrieval-Augmented Generation (RAG)** application for question-answering over a technical PDF corpus. It combines **FAISS** vector search, **intent-based retrieval**, and **Google Gemini** to answer questions from the document with cited sources, support for figures/tables/pages, and confidence scoring.

---

## What This Application Does

- **PDF Q&A**: Ask natural-language questions about the indexed document. Answers are grounded in retrieved chunks and returned as structured **paragraphs** and **lists**, with **sources** (chunk ID, page, text excerpt, optional page image).
- **Intent-aware retrieval**: The system classifies each question (figure, table, page, section, general, or comparison) and uses different retrieval strategies—**exact reference matching** for “Fig 3.3” / “Table 6.2” / “Page 27”, and **semantic (FAISS) search** for broad questions.
- **Structured answers**: The LLM returns JSON (`blocks` with `paragraph` or `list`), which is enriched with **confidence scores** (High/Medium/Low) and source citations.
- **Context expansion**: The UI can request “surrounding” chunks (previous/next by chunk ID) for richer context, without calling the LLM or FAISS.
- **Chat UI**: Multi-chat React frontend with persistence in `localStorage`, export to PDF (jsPDF), theme (light/dark), and source highlighting.

The default corpus in this repo is **IRC:67-2022** (Code of Practice for Road Signs).

---

## Concepts Used

### 1. **Retrieval-Augmented Generation (RAG)**

Instead of relying only on the LLM’s weights, we:

1. **Retrieve** relevant text chunks from the corpus using the user’s question.
2. **Augment** the LLM prompt with those chunks as “context”.
3. **Generate** an answer that must stay within that context.

This reduces hallucination and ties answers to specific pages and chunks.

---

### 2. **Vector Embeddings and FAISS**

- **Embeddings**: The question (and, at index-build time, each chunk) is turned into a **dense vector** via Google’s `embedding-001` model. Semantically similar text gets similar vectors.
- **FAISS** (Facebook AI Similarity Search): A fast approximate nearest-neighbor index over chunk vectors. For **general** and **section** intents, we:
  - Embed the question.
  - Run `index.search(query_embedding, k=6)` to get the closest chunks.
  - Map FAISS indices back to `metadata.json` for text, `page_number`, `source`, and optional `images/page_{n}_img_1.png`.

So: **vector similarity → retrieve chunks → pass to LLM as context**.

---

### 3. **Intent Classification**

Before any retrieval, the question is **classified** by `intent_classifier.py`:

| Intent | Example | Retrieval |
|--------|---------|-----------|
| `FIGURE_QUERY` | “What does Fig 3.3 show?” | Exact match on “Fig 3.3” in `metadata`; prefer `image_ocr`/`vision` chunks; exclude “List of Figures”. |
| `TABLE_QUERY` | “Table 6.2 guidelines” | Exact match on “Table 6.2”; prefer chunks with table-like structure. |
| `PAGE_QUERY` | “What is on page 27?” | All chunks with `page_number == 27` (up to 3). |
| `SECTION_QUERY` | “Section on mandatory signs” | Treated like `GENERAL_QUERY` (semantic/FAISS). |
| `GENERAL_QUERY` | “Size of STOP sign?” | FAISS semantic search, k=6. |
| `COMPARISON_QUERY` | “Compare Fig 3.1 and Fig 3.2” | Retrieves **both** referenced items exactly, then one prompt for comparison. |

Rules are regex-based (e.g. `fig(?:ure)?\.?\s*(\d+(?:\.\d+)*)`, `(table\|tab)`, `(page\|pg\|p)\.?\s*(\d+)`, “compare”/“vs” with ≥2 refs). **Intent drives whether we use FAISS or metadata-only exact match.**

---

### 4. **Deduplication and Source Priority**

- **Chunk deduplication**: For semantic retrieval, chunks with >85% text similarity (e.g. `SequenceMatcher`) are merged to avoid near-duplicates in the context.
- **Source priority**: When several chunks match (e.g. figure), we prefer `image_ocr` > `vision` > `text` (or `text` > `vision` for tables) so the model sees the most relevant representation.

---

### 5. **Structured LLM Output and Confidence**

- The prompt forces **JSON only** with a fixed schema:
  - `blocks`: `{ type: "paragraph", text }` or `{ type: "list", items: [...] }`.
- **`clean_llm_json`** strips markdown code fences if the model wraps JSON in ` ``` `.
- **`safe_parse_blocks`** falls back to a single `paragraph` if parsing fails.
- **Confidence** is computed per block from:
  - Source types (OCR/IMAGE lower than TEXT),
  - Number of chunks used,
  - Semantic match score (for `GENERAL_QUERY`: `1/(1+d)` from FAISS distance),
  - Intent (e.g. GENERAL gets a small penalty),
  - Verbatim presence of the block in the retrieved text.
- Scores are clamped to `[0.30, 0.95]` and labeled **High / Medium / Low**. Below 0.55, a short warning is prepended to the block.

---

### 6. **Metadata and Static Assets**

- **`metadata.json`**: `chunk_id -> { text, source, page_number [, figure_id, table_id ] }`. `source` is `text` | `image_ocr` | `vision`. This is the “chunk catalog” used by both exact-match and FAISS (via `chunk_id = faiss_index + 1`).
- **`faiss.index`**: Pre-built FAISS index over chunk vectors (same order as `metadata` keys). **Must exist at runtime** (built by a separate indexing pipeline; not in this repo).
- **`images/`**: `page_{n}_img_1.png`—one image per page, served at `/images/` and attached to sources when `page_number` is present.
- **`vision_captions.json`**: Optional cache of per-page Gemini vision captions (e.g. for figures). If missing, it’s `{}`; the app can populate it on demand (when `GEMINI_API_KEY` is set and a caption is requested; in the main `/ask` flow, vision captions are not attached to sources).

---

### 7. **Context Expansion (No LLM, No FAISS)**

`/expand-context` takes `chunk_ids`, `intent`, `target_id` and returns **contextual_sources**: for each core chunk, the previous and next chunk by ordering in `metadata`. Used by the UI to show “surrounding” context without extra retrieval or generation.

---

### 8. **Frontend (React + Vite)**

- **Chat**: Multiple chats, `localStorage` persistence, auto-titles from the first reply.
- **API usage**: `classifyIntent` → `askQuestion`; optional `expandContext` for “Show more context”.
- **Rendering**: `blocks` as paragraphs or lists; `sources` with `text`, `page_number`, `image_path`, `highlight_phrases`; inline highlighting of phrases in source excerpts.
- **Export**: jsPDF to export the current thread.
- **Theme**: `data-theme` (light/dark) with `localStorage`.

---

## Tech Stack

| Layer | Tech |
|-------|------|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **Vector DB** | FAISS (CPU), NumPy |
| **Embeddings** | Google `embedding-001` (via `google-generativeai`) |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **PDF/Images** | PyMuPDF (fitz), pdfplumber, Pillow (for vision; PDF Q&A does not re-parse PDFs at runtime) |
| **Frontend** | React 19, Vite 7 |
| **Export** | jsPDF |
| **Env** | `python-dotenv`, `GEMINI_API_KEY` |

---

## Project Layout

```
.
├── app.py              # FastAPI app: /health, /classify-intent, /expand-context, /ask; mounts /images
├── intent_classifier.py # classify_query_intent() → { intent, target_id }
├── requirements.txt
├── Dockerfile          # Python 3.10, uvicorn on 8080
├── .env                # GEMINI_API_KEY (create locally; not in repo)
├── .gitignore          # .env, __pycache__, .pyc, faiss.index
│
├── faiss.index         # Pre-built FAISS index (must exist; not in git)
├── metadata.json       # Chunk catalog: id -> { text, source, page_number }
├── vision_captions.json# Optional: page -> vision caption
├── images/             # page_{n}_img_1.png, served at /images
│
└── ui/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── public/vite.svg
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx     # Chat UI, themes, export, source rendering
    │   ├── App.css
    │   ├── index.css
    │   └── api.js      # classifyIntent, askQuestion, expandContext; API_URL
    └── dist/           # Production build (npm run build)
```

---

## API Endpoints

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Liveness |
| `POST` | `/classify-intent` | `{ question }` → `{ intent, target_id [, targets ] }`; no FAISS, no LLM |
| `POST` | `/expand-context` | `{ chunk_ids, intent, target_id }` → `{ contextual_sources }`; no FAISS, no LLM |
| `POST` | `/ask` | `{ question, include_context? }` → `{ blocks, sources, contextual_sources? }` or `{ error }` |

`/ask` flow: **classify intent → retrieve by intent (exact or FAISS) → build context string → Gemini → parse JSON blocks → enrich confidence → return `StructuredAnswerResponse`.**

---

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (for the UI)
- **GEMINI_API_KEY** (Google AI / Gemini)
- **faiss.index** and **metadata.json** (and optionally **images/**, **vision_captions.json**) in the project root. The app expects these to exist; `faiss.index` is typically produced by a separate indexing script (not included here).

---

## Setup and Run

### Backend

```bash
cd "/Users/omg/Desktop/rag chatbot"
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Run from the **project root** (not from `ui/`):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# or: python app.py   (if app.py starts uvicorn)
```

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd ui
npm install
npm run dev
```

- UI: `http://localhost:5173` (or the port Vite prints). Set `API_URL` in `ui/src/api.js` if the backend runs elsewhere.

### Production build (UI)

```bash
cd ui
npm run build
```

Serves `ui/dist`. To run in production, serve `dist` (e.g. via nginx or by mounting it in the FastAPI app) and run the backend (or use the Dockerfile for the API only).

### Docker (backend only)

```bash
docker build -t rag-chatbot .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key -v "$(pwd)/faiss.index:/app/faiss.index" -v "$(pwd)/metadata.json:/app/metadata.json" -v "$(pwd)/images:/app/images" rag-chatbot
```

Ensure `faiss.index`, `metadata.json`, and `images/` are available inside the container (e.g. via volumes or by copying them in at build time if you choose to).

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for `/ask`) | Google Gemini (and embedding) API key. Without it, `/ask` returns an error. |

---

## Summary

This project is a **RAG-based PDF Q&A system** that:

- Uses **intent classification** to choose between **exact reference lookup** (figures, tables, pages, comparisons) and **FAISS semantic search** (general/section).
- Embeds queries with **Gemini embedding-001** and retrieves from a **FAISS** index over a chunk catalog in **metadata.json**.
- Sends retrieved text (and optional page images) as **context** to **Gemini** and parses **structured JSON** (paragraphs/lists).
- Adds **confidence scores** and **source citations** and exposes a **React chat UI** with export and context expansion.

The main concepts are: **RAG**, **vector embeddings**, **FAISS**, **intent-based retrieval**, **structured LLM output**, and **confidence-aware answers** tied to document chunks and pages.
