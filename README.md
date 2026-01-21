## RAG PDF System (Backend + UI)

This is a local **FastAPI + React** app for:
- **PDF Q&A (RAG)**: FAISS retrieval + Gemini generation
- **Drawing analysis (Vision)**: Gemini Vision extracts **STRICT components JSON** (“dumb but honest”)
- **BOM generation**: deterministic mapping from extracted components → **BOM rows** (no guessing)

---

## Current BOM contract (IMPORTANT)

### Vision extraction is strict

`POST /analyze-drawing` returns `components[]` where each component is **not merged** and uses only explicit data:
- `type`: `MS_PIPE | ISA | PLATE | ACP_SHEET | REFLECTIVE_SHEET | BOLT | OTHER`
- `diameter_mm | thickness_mm | width_mm | height_mm | length_mm`: `number | null`
- `quantity`: `integer | null` (null when unclear)
- `standard_reference`: `string | null`
- `drawing_note_reference`: `string | null`
- `confidence`: `0.0–1.0`

### BOM is deterministic (no LLM in BOM step)

`POST /generate-bom` maps **explicit drawing signals** → merged BOM items using rules:
- Merge when **same**: `type + dimensions + material spec`
- Sum quantities
- **Never guess quantities**: if `quantity` is unclear (`null`) → omit item
- If any items were omitted due to unclear quantity, this is reflected in `bom_metadata.notes`

**Critical**: BOM generation does **not** depend on a successful vision analysis step. Even if the drawing analysis JSON is partial/incomplete/empty, `/generate-bom` still attempts component detection from:
- visible dimensions
- repeated members
- schedules (tables)
- notes
- callouts

Response format:
- `{ success: true, bom: { bom_metadata, items, summary } }`
- `bom.items[]` columns (order): `item_no`, `description`, `material_specification`, `part_drawing_no`, `qty`

---

## Prerequisites
- **Python** 3.10+
- **Node.js** 18+
- **Gemini API key**: `GEMINI_API_KEY`

---

## Setup

### Backend

```bash
cd "/Users/omg/Desktop/RAG PDF"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

### Frontend

```bash
cd "/Users/omg/Desktop/RAG PDF/ui"
npm install
```

---

## Run

### Backend

```bash
cd "/Users/omg/Desktop/RAG PDF"
python3 app.py
```

Backend: `http://127.0.0.1:8000`

### Frontend

```bash
cd "/Users/omg/Desktop/RAG PDF/ui"
npm run dev
```

Frontend: `http://localhost:5173`

---

## API

### `GET /health`
Health check.

### `POST /ask`
PDF Q&A against the pre-indexed corpus.

- Body: `{ "question": "..." }`
- Response: `{ blocks: [...], sources: [...] }`

### `POST /analyze-drawing`
Gemini Vision analysis (first page of PDFs at 200 DPI).

- Body: `multipart/form-data` with `file`
- Response: `{ drawing_type, summary, components[], foundation, standards[], uncertain_items[] }`

### `POST /ask-drawing`
Follow-up Q&A using only the returned drawing JSON.

- Body: `{ "question": "...", "drawing_analysis": { ... } }`
- Response: `{ blocks: [...], sources: [] }`

### `POST /generate-bom`
Deterministic BOM from the extracted components.

- Body: `multipart/form-data` with `file`
- Caching: `bom_cache/{sha256}.json`
- Response:
  - `success: true`
  - `bom.bom_metadata`: `{ drawing_reference, created_at, notes }`
  - `bom.items[]`: `{ item_no, description, material_specification, part_drawing_no, qty }`
  - `bom.summary.total_items`

---

## Required runtime files

### Backend runtime
- `app.py`
- `requirements.txt`
- `faiss.index`
- `metadata.json`
- `images/` (optional but recommended for UI previews)

### Frontend runtime
- `ui/` (Vite app)


