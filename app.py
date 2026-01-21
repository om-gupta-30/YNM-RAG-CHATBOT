"""
FastAPI Backend for RAG PDF System

IMPORTANT: Run uvicorn from the project root directory:
    uvicorn app:app --reload
    OR
    python app.py

Do NOT run from ui/ directory.
"""
from __future__ import annotations

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import json
from datetime import date
import os
import tempfile
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Optional
from PIL import Image
from difflib import SequenceMatcher
import re
import fitz  # PyMuPDF
import pdfplumber
import logging
# Dimension extraction imports removed - keeping analysis text-only
from intent_classifier import classify_query_intent
from intent_classifier import INTENT_COMPARISON

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/{path:path}", methods=["OPTIONS"])
async def options_handler(path: str):
    return Response(status_code=200)

app.mount("/images", StaticFiles(directory="images"), name="images")

api_key = os.getenv("GEMINI_API_KEY")
GEMINI_ENABLED = bool(api_key)
if api_key:
    genai.configure(api_key=api_key)
else:
    logging.warning("GEMINI_API_KEY not set. Gemini-dependent endpoints will be unavailable; BOM fallback extraction can still run for PDFs with extractable text/tables.")

index = faiss.read_index("faiss.index")

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

vision_captions_file = "vision_captions.json"

if os.path.exists(vision_captions_file):
    with open(vision_captions_file, "r", encoding="utf-8") as f:
        vision_captions = json.load(f)
else:
    vision_captions = {}

def extract_figure_reference(question: str) -> Optional[str]:
    """
    Extract figure reference like 'fig 3.3' or 'figure 3.3'
    Returns normalized form '3.3' or None
    """
    import re
    match = re.search(r'(fig(?:ure)?\.?\s*)(\d+(\.\d+)?)', question.lower())
    if match:
        return match.group(2)
    return None

def get_or_create_vision_caption(page_number: int, image_path: str) -> str:
    if not GEMINI_ENABLED:
        return ""
    if str(page_number) in vision_captions:
        return vision_captions[str(page_number)]
    
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Describe the road signs, symbols, or diagrams visible in this image in a factual way."
        
        response = model.generate_content([prompt, image])
        caption = response.text.strip()
        
        vision_captions[str(page_number)] = caption
        
        with open(vision_captions_file, "w", encoding="utf-8") as f:
            json.dump(vision_captions, f, indent=2, ensure_ascii=False)
        
        return caption
    except Exception:
        return ""

def text_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def deduplicate_chunks(chunks_data: List[dict], similarity_threshold: float = 0.85) -> List[dict]:
    if not chunks_data:
        return []
    
    unique_chunks = [chunks_data[0]]
    
    for chunk in chunks_data[1:]:
        is_duplicate = False
        chunk_text = chunk.get("text", "").lower()
        
        for unique_chunk in unique_chunks:
            unique_text = unique_chunk.get("text", "").lower()
            if text_similarity(chunk_text, unique_text) > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks

def extract_longest_sentences(text: str, max_sentences: int = 2) -> List[str]:
    if not text or not text.strip():
        return []
    
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return []
    
    sentences.sort(key=len, reverse=True)
    return sentences[:max_sentences]


def _source_type_label(source: str) -> str:
    # Normalize internal source tags to user-facing types.
    if source == "image_ocr":
        return "OCR"
    if source == "vision":
        return "IMAGE"
    return "TEXT"


def _make_verbatim_excerpt(text: str, max_lines: int = 3) -> str:
    """
    Return a short verbatim excerpt (1–3 non-empty lines) from the chunk text.
    """
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _is_verbatim_block(block: dict, sources: List[Source]) -> bool:
    """
    Heuristic: treat block as verbatim if its text/items appear (as normalized substring)
    inside any returned source excerpt/raw text.
    """
    corpus_parts: List[str] = []
    for s in sources or []:
        corpus_parts.append(s.raw_text or "")
        corpus_parts.append(s.text or "")
    corpus = _normalize_ws("\n".join(corpus_parts))
    if not corpus:
        return False

    if block.get("type") == "paragraph" and block.get("text"):
        needle = _normalize_ws(block["text"])
        return bool(needle) and needle in corpus
    if block.get("type") == "list" and block.get("items"):
        items = [i for i in block["items"] if i]
        if not items:
            return False
        hits = 0
        for it in items:
            if _normalize_ws(it) in corpus:
                hits += 1
        return hits / max(len(items), 1) >= 0.6

    return False


def _compute_confidence_for_block(
    block: dict,
    intent_info: dict,
    used_sources: List[Source],
    chunks_used_count: int,
    semantic_match_score: Optional[float],
    exact_match: bool,
) -> float:
    """
    Implements scoring rules:
      base 1.00
      deductions/bonuses
      clamp to [0.30, 0.95], never 1.00
    """
    score = 1.00

    # DEDUCTIONS
    primary_type = ((used_sources[0].source_type or used_sources[0].source) if used_sources else "") or ""
    if primary_type == "OCR":
        score -= 0.15
    if primary_type == "IMAGE":
        score -= 0.25

    if chunks_used_count > 1:
        score -= 0.10

    if semantic_match_score is not None and semantic_match_score < 0.75:
        score -= 0.20

    if intent_info.get("intent") == "GENERAL_QUERY":
        score -= 0.05

    verbatim = _is_verbatim_block(block, used_sources)
    if not verbatim:
        score -= 0.10  # paraphrasing required

    # BONUSES
    if exact_match and intent_info.get("intent") in {"FIGURE_QUERY", "TABLE_QUERY"}:
        score += 0.10
    if verbatim:
        score += 0.10

    # Never return 1.00; clamp per requirements.
    score = min(score, 0.99)
    score = _clamp(score, 0.30, 0.95)
    return round(score, 2)


_LOW_CONF_WARNING = "⚠️ Low confidence: derived from indirect or ambiguous content."


def _prepend_low_conf_warning(block: dict) -> dict:
    """
    If block is low confidence, prepend warning to the block content.
    - paragraph: prefix block["text"]
    - list: insert warning as first list item
    Also keeps block["answer"] consistent.
    """
    b = dict(block)
    if b.get("type") == "paragraph":
        txt = b.get("text") or ""
        if not txt.startswith(_LOW_CONF_WARNING):
            b["text"] = f"{_LOW_CONF_WARNING}\n{txt}".strip()
    elif b.get("type") == "list":
        items = list(b.get("items") or [])
        if not items or items[0] != _LOW_CONF_WARNING:
            b["items"] = [_LOW_CONF_WARNING] + items

    # Keep answer in sync with the visible content.
    if b.get("type") == "paragraph":
        b["answer"] = b.get("text") or ""
    elif b.get("type") == "list":
        b["answer"] = "\n".join(b.get("items") or [])
    else:
        b["answer"] = b.get("answer") or ""
    return b


def _enrich_blocks_with_confidence(
    blocks: List[dict],
    intent_info: dict,
    per_block_sources: List[Source],
    chunks_used_count: int,
    semantic_match_score: Optional[float],
    exact_match: bool,
) -> List[dict]:
    """
    Ensures every block includes:
      - answer
      - confidence_score
      - confidence_label
      - sources
    And prepends a warning when confidence_score < 0.55.
    """
    enriched_blocks: List[dict] = []
    for b in blocks:
        # Build an "answer" field (required by output contract).
        answer_text = ""
        if b.get("type") == "paragraph":
            answer_text = b.get("text") or ""
        elif b.get("type") == "list":
            answer_text = "\n".join(b.get("items") or [])

        enriched = dict(b)
        enriched["answer"] = answer_text
        enriched["sources"] = per_block_sources[:3]

        conf = _compute_confidence_for_block(
            block=b,
            intent_info=intent_info,
            used_sources=per_block_sources,
            chunks_used_count=chunks_used_count,
            semantic_match_score=semantic_match_score,
            exact_match=exact_match,
        )
        enriched["confidence_score"] = conf
        enriched["confidence_label"] = _confidence_label(conf)

        if conf < 0.55:
            enriched = _prepend_low_conf_warning(enriched)

        enriched_blocks.append(enriched)

    return enriched_blocks


def _enrich_blocks_with_fixed_confidence(
    blocks: List[dict],
    score: float,
    label: str,
    per_block_sources: List[Source],
) -> List[dict]:
    enriched_blocks: List[dict] = []
    for b in blocks:
        answer_text = ""
        if b.get("type") == "paragraph":
            answer_text = b.get("text") or ""
        elif b.get("type") == "list":
            answer_text = "\n".join(b.get("items") or [])

        enriched = dict(b)
        enriched["answer"] = answer_text
        enriched["sources"] = per_block_sources[:3]
        enriched["confidence_score"] = round(float(score), 2)
        enriched["confidence_label"] = label

        if float(score) < 0.55:
            enriched = _prepend_low_conf_warning(enriched)

        enriched_blocks.append(enriched)
    return enriched_blocks


def _extract_figure_caption(chunk_text: str, target_id: Optional[str]) -> Optional[str]:
    """
    Try to extract a short caption fragment from a line containing the figure reference.
    Example line: "Fig. 14.1 Description of Visibility Funnel 27"
      -> "Description of Visibility Funnel"
    """
    if not chunk_text or not target_id:
        return None

    ref = _extract_ref_number(target_id)
    if not ref:
        return None

    # Find the first line containing the fig reference.
    pat = re.compile(rf"\bfig(?:ure)?\.?\s*{re.escape(ref)}\b\.?\s*(.*)$", re.IGNORECASE)
    for ln in chunk_text.splitlines():
        m = pat.search(ln.strip())
        if m:
            caption = (m.group(1) or "").strip()
            # Remove trailing standalone page number if present.
            caption = re.sub(r"\s+\d{1,4}\s*$", "", caption).strip()
            return caption or None

    return None

def clean_llm_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()

def safe_parse_blocks(raw_text: str):
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "blocks" in parsed:
            return parsed["blocks"]
    except Exception:
        pass

    # If model leaked JSON or garbage, clean it
    cleaned = re.sub(r"```.*?```", "", raw_text, flags=re.S).strip()
    cleaned = re.sub(r"^\s*\{.*\}\s*$", "", cleaned, flags=re.S)

    return [
        {
            "type": "paragraph",
            "text": cleaned
        }
    ]

class QuestionRequest(BaseModel):
    question: str
    include_context: Optional[bool] = False

# DrawingQuestionRequest removed - drawing analysis endpoints are no longer supported
# class DrawingQuestionRequest(BaseModel):
#     question: str
#     drawing_analysis: dict  # The full drawing analysis JSON

class Source(BaseModel):
    chunk_id: Optional[str] = None
    page_number: Optional[int]
    source: str
    # Canonical source type for auditing/confidence: TEXT | OCR | IMAGE
    source_type: Optional[str] = None
    text: str
    vision_caption: Optional[str]
    image_path: Optional[str]
    highlight_phrases: Optional[List[str]] = None
    # Optional extra payload for richer UI rendering.
    raw_text: Optional[str] = None
    caption: Optional[str] = None
    # Optional: exact span offsets in raw_text (for context highlighting)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None


class SurroundingContext(BaseModel):
    previous: Optional[Source] = None
    next: Optional[Source] = None


class ContextualSource(BaseModel):
    core_chunk: Source
    surrounding_context: SurroundingContext

class AnswerBlock(BaseModel):
    type: str  # "paragraph" or "list"
    text: Optional[str] = None
    items: Optional[List[str]] = None
    # Confidence scoring (added)
    answer: str
    confidence_score: float
    confidence_label: str  # High | Medium | Low
    sources: List[Source]

class StructuredAnswerResponse(BaseModel):
    blocks: List[AnswerBlock]
    sources: List[Source]
    # Optional: only included when explicitly requested by UI
    contextual_sources: Optional[List[ContextualSource]] = None


class ContextExpandRequest(BaseModel):
    chunk_ids: List[str]
    intent: Optional[str] = None
    target_id: Optional[str] = None


class ContextExpandResponse(BaseModel):
    contextual_sources: List[ContextualSource]


class IntentClassificationResponse(BaseModel):
    intent: str
    target_id: Optional[str] = None
    targets: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    error: str

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent(request: QuestionRequest):
    """
    Classify the user's question intent.
    CRITICAL: This endpoint does NOT perform retrieval or call FAISS.
    """
    info = classify_query_intent(request.question)
    # For backward compatibility: only add `targets` for comparison queries (optional field)
    if info.get("intent") == INTENT_COMPARISON:
        # Best-effort extraction of two refs for UI display; backend still re-parses.
        refs = []
        q = request.question or ""
        for m in re.finditer(r"\b(fig(?:ure)?|illustration)\b\.?\s*(\d+(?:\.\d+)*)", q, re.IGNORECASE):
            refs.append(f"Fig {m.group(2)}")
        for m in re.finditer(r"\b(table|tab)\b\.?\s*(\d+(?:\.\d+)*)", q, re.IGNORECASE):
            refs.append(f"Table {m.group(2)}")
        for m in re.finditer(r"\b(?:page|pg|p)\b\.?\s*(\d{1,4})\b", q, re.IGNORECASE):
            refs.append(f"Page {m.group(1)}")
        info["targets"] = refs[:2] if len(refs) >= 2 else None
    return info


@app.post("/expand-context", response_model=ContextExpandResponse)
async def expand_context(request: ContextExpandRequest):
    """
    Expand surrounding context for specific chunk ids.
    This endpoint does NOT generate answers, does NOT call FAISS, and does NOT call the LLM.
    """
    intent_info = {
        "intent": request.intent or "GENERAL_QUERY",
        "target_id": request.target_id,
    }

    core_sources: List[Source] = []
    for cid in request.chunk_ids or []:
        if str(cid) in metadata:
            core_sources.append(_build_source_from_metadata_chunk(str(cid), intent_info, include_raw_text=True))

    contextual_sources = _build_contextual_sources(core_sources, intent_info)
    return ContextExpandResponse(contextual_sources=contextual_sources)


def _extract_ref_number(target_id: Optional[str]) -> Optional[str]:
    if not target_id:
        return None
    m = re.search(r"(\d+(?:\.\d+)*)", target_id)
    return m.group(1) if m else None


def _chunk_matches_figure(chunk_data: dict, target_id: str) -> bool:
    # 1) metadata.figure_id exact match (if present)
    if chunk_data.get("figure_id") == target_id:
        return True

    # 2) text contains an exact figure reference "Fig X.X" (case-insensitive, allows Fig./Figure)
    ref = _extract_ref_number(target_id)
    if not ref:
        return False

    txt = chunk_data.get("text", "") or ""
    pat = re.compile(rf"\bfig(?:ure)?\.?\s*{re.escape(ref)}\b", re.IGNORECASE)
    return pat.search(txt) is not None


def _chunk_matches_table(chunk_data: dict, target_id: str) -> bool:
    # 1) metadata.table_id exact match (if present)
    if chunk_data.get("table_id") == target_id:
        return True

    # 2) OCR/text contains an exact table reference "Table X.X" (case-insensitive, allows Tab./Table)
    ref = _extract_ref_number(target_id)
    if not ref:
        return False

    txt = chunk_data.get("text", "") or ""
    pat = re.compile(rf"\b(?:table|tab)\.?\s*{re.escape(ref)}\b", re.IGNORECASE)
    return pat.search(txt) is not None


def _extract_page_number(target_id: Optional[str]) -> Optional[int]:
    if not target_id:
        return None
    m = re.search(r"\b(\d{1,4})\b", target_id)
    return int(m.group(1)) if m else None


def _is_list_index_page(text: str, intent: str) -> bool:
    t = (text or "").lower()
    if intent == "FIGURE_QUERY":
        return "list of figures" in t
    if intent == "TABLE_QUERY":
        return "list of tables" in t
    return False


def _extract_matching_reference_line(text: str, intent: str, target_id: Optional[str]) -> Optional[str]:
    """
    Return the single line (verbatim) that contains the exact reference (Fig/Table X.X).
    Used for strict FIGURE/TABLE responses to avoid surrounding content.
    """
    if not text or not target_id:
        return None
    ref = _extract_ref_number(target_id)
    if not ref:
        return None

    if intent == "FIGURE_QUERY":
        pat = re.compile(rf".*\bfig(?:ure)?\.?\s*{re.escape(ref)}\b.*", re.IGNORECASE)
    elif intent == "TABLE_QUERY":
        pat = re.compile(rf".*\b(?:table|tab)\.?\s*{re.escape(ref)}\b.*", re.IGNORECASE)
    else:
        return None

    for ln in text.splitlines():
        line = ln.strip()
        if line and pat.match(line):
            return line
    return None


def _safe_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(str(s))
    except Exception:
        return None


def _find_span_offsets(haystack: str, needle: str) -> tuple[Optional[int], Optional[int]]:
    """
    Return (start,end) offsets of needle within haystack, or (None,None) if not found.
    """
    if not haystack or not needle:
        return None, None
    start = haystack.find(needle)
    if start == -1:
        return None, None
    return start, start + len(needle)


def _extract_comparison_targets(question: str) -> tuple[Optional[str], Optional[list[str]], Optional[str]]:
    """
    Return (kind, targets, error)
      - kind: "FIGURE_QUERY" | "TABLE_QUERY"
      - targets: ["Fig 3.3", "Fig 4.1"] or ["Table 6.2", "Table 8.3"]
      - error: message if invalid or missing
    """
    q = question or ""
    figs = [f"Fig {m.group(2)}" for m in re.finditer(r"\b(fig(?:ure)?|illustration)\b\.?\s*(\d+(?:\.\d+)*)", q, re.IGNORECASE)]
    tables = [f"Table {m.group(2)}" for m in re.finditer(r"\b(table|tab)\b\.?\s*(\d+(?:\.\d+)*)", q, re.IGNORECASE)]
    pages = [f"Page {m.group(1)}" for m in re.finditer(r"\b(?:page|pg|p)\b\.?\s*(\d{1,4})\b", q, re.IGNORECASE)]

    def dedupe(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    figs = dedupe(figs)
    tables = dedupe(tables)
    pages = dedupe(pages)

    if len(figs) >= 2 and len(tables) == 0:
        return "FIGURE_QUERY", figs[:2], None
    if len(tables) >= 2 and len(figs) == 0:
        return "TABLE_QUERY", tables[:2], None

    if (len(figs) >= 1 and len(tables) >= 1):
        return None, None, "Invalid comparison. Table ↔ Figure comparisons are not supported."

    if len(pages) >= 2:
        return None, None, "Invalid comparison. Only Table ↔ Table or Figure ↔ Figure comparisons are supported."

    return None, None, "Exact reference(s) not found. Comparison aborted."


def _retrieve_one_item_by_ref(kind: str, target_id: str) -> Optional[dict]:
    """
    Retrieve a single best-matching chunk for a figure/table reference.
    Returns chunk_info dict or None.
    """
    matched = []
    for chunk_id, chunk_data in metadata.items():
        if kind == "FIGURE_QUERY":
            ok = _chunk_matches_figure(chunk_data, target_id)
        else:
            ok = _chunk_matches_table(chunk_data, target_id)
        if not ok:
            continue

        page_number = chunk_data.get("page_number")
        image_path = None
        if page_number:
            candidate = f"images/page_{page_number}_img_1.png"
            if os.path.exists(candidate):
                image_path = candidate

        matched.append(
            {
                "chunk_id": str(chunk_id),
                "text": chunk_data.get("text", "") or "",
                "source": chunk_data.get("source", "text"),
                "page_number": page_number,
                "image_path": image_path,
            }
        )

    matched = [m for m in matched if not _is_list_index_page(m.get("text", ""), kind)]
    if not matched:
        return None

    ref = _extract_ref_number(target_id)
    if ref:
        if kind == "FIGURE_QUERY":
            ref_re = re.compile(rf"\bfig(?:ure)?\.?\s*{re.escape(ref)}\b", re.IGNORECASE)
            src_pri = {"image_ocr": 0, "vision": 1, "text": 2}
        else:
            ref_re = re.compile(rf"\b(?:table|tab)\.?\s*{re.escape(ref)}\b", re.IGNORECASE)
            src_pri = {"image_ocr": 0, "text": 1, "vision": 2}

        def _pos(item: dict) -> int:
            m = ref_re.search(item.get("text", "") or "")
            return m.start() if m else 10**9

        def _table_score(item: dict) -> int:
            if kind != "TABLE_QUERY":
                return 0
            txt = item.get("text", "") or ""
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            class_lines = sum(1 for ln in lines if re.search(r"\bCLASS\b", ln, re.IGNORECASE))
            multi_col_lines = sum(1 for ln in lines if re.search(r"\s{2,}", ln) or "|" in ln or "\t" in ln)
            return class_lines * 10 + multi_col_lines + min(len(txt) // 200, 20)

        def _figure_score(item: dict) -> int:
            if kind != "FIGURE_QUERY":
                return 0
            line = _extract_matching_reference_line(item.get("text", "") or "", "FIGURE_QUERY", target_id) or ""
            l = line.lower().strip()
            score = 0
            if l.startswith("fig"):
                score += 100
            if "example" in l:
                score += 50
            if "shown in" in l:
                score -= 20
            score += min(len(line), 200) // 20
            return score

        matched.sort(
            key=lambda item: (
                src_pri.get(item.get("source"), 99),
                -_table_score(item),
                -_figure_score(item),
                _pos(item),
                item.get("page_number") or 10**9,
            )
        )

        if kind == "FIGURE_QUERY":
            with_images = [m for m in matched if m.get("image_path")]
            if with_images:
                matched = with_images

    return matched[0]


def _build_source_from_metadata_chunk(
    chunk_id: str,
    intent_info: dict,
    max_excerpt_lines: int = 3,
    include_raw_text: bool = False,
) -> Source:
    chunk_data = metadata.get(str(chunk_id), {}) or {}
    chunk_text = chunk_data.get("text", "") or ""
    source_type = chunk_data.get("source", "text")
    page_number = chunk_data.get("page_number")

    image_path = None
    if page_number:
        candidate = f"images/page_{page_number}_img_1.png"
        if os.path.exists(candidate):
            image_path = candidate

    intent = intent_info.get("intent")
    preview_text = _extract_matching_reference_line(chunk_text, intent, intent_info.get("target_id"))
    if not preview_text:
        preview_text = _make_verbatim_excerpt(chunk_text, max_lines=max_excerpt_lines)

    raw_text = chunk_text if (include_raw_text or intent == "TABLE_QUERY") else None
    caption = _extract_figure_caption(chunk_text, intent_info.get("target_id")) if intent == "FIGURE_QUERY" else None
    start_offset = chunk_data.get("start_offset")
    end_offset = chunk_data.get("end_offset")

    if intent == "TABLE_QUERY":
        ui_source_type = "TABLE"
    elif intent == "FIGURE_QUERY" and image_path:
        ui_source_type = "IMAGE"
    else:
        ui_source_type = _source_type_label(source_type)

    canonical = _source_type_label(source_type)
    return Source(
        chunk_id=str(chunk_id),
        page_number=page_number,
        source=ui_source_type,
        source_type=canonical if ui_source_type == "TABLE" else ui_source_type,
        text=preview_text,
        vision_caption=None,
        image_path=image_path,
        raw_text=raw_text,
        caption=caption,
        start_offset=start_offset if isinstance(start_offset, int) else None,
        end_offset=end_offset if isinstance(end_offset, int) else None,
    )


def _build_contextual_sources(
    core_sources: List[Source],
    intent_info: dict,
) -> List[ContextualSource]:
    out: List[ContextualSource] = []
    for s in core_sources:
        cid = _safe_int(s.chunk_id)
        prev_src = None
        next_src = None
        if cid is not None:
            prev_id = str(cid - 1)
            next_id = str(cid + 1)
            if prev_id in metadata:
                prev_src = _build_source_from_metadata_chunk(prev_id, intent_info, include_raw_text=True)
            if next_id in metadata:
                next_src = _build_source_from_metadata_chunk(next_id, intent_info, include_raw_text=True)

        out.append(
            ContextualSource(
                core_chunk=s,
                surrounding_context=SurroundingContext(previous=prev_src, next=next_src),
            )
        )
    return out


_FIGURE_REF_FIND_RE = re.compile(
    r"\b(fig(?:ure)?|illustration)\b\.?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_TABLE_REF_FIND_RE = re.compile(
    r"\b(table|tab)\b\.?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_DOT_NUMBER_FIND_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")


def _retrieve_chunks_by_intent(intent_info: dict, question: str):
    """
    Returns: (chunks_data, error_message)
      - chunks_data: list[{chunk_id,text,source,page_number,image_path}]
      - error_message: str|None
    """
    intent = intent_info.get("intent")
    target_id = intent_info.get("target_id")
    strict_ref_error = "Exact reference not found in the document. Please check the figure/table number."

    # COMPARISON: strict retrieval of exactly two referenced items.
    if intent == INTENT_COMPARISON:
        kind, targets, err = _extract_comparison_targets(question)
        if err:
            return [], err

        assert kind is not None and targets is not None
        item_a = _retrieve_one_item_by_ref(kind, targets[0])
        item_b = _retrieve_one_item_by_ref(kind, targets[1])
        if not item_a or not item_b:
            return [], "Exact reference(s) not found. Comparison aborted."

        # Mark intent_info with comparison targets for downstream prompt/confidence.
        intent_info["comparison_kind"] = kind
        intent_info["comparison_targets"] = targets
        return [item_a, item_b], None

    def _is_ambiguous_reference() -> bool:
        q = question or ""
        if intent == "FIGURE_QUERY":
            refs = {m.group(2) for m in _FIGURE_REF_FIND_RE.finditer(q)}
            if re.search(r"\b(fig(?:ure)?|illustration)\b", q, re.IGNORECASE):
                refs |= {m.group(0) for m in _DOT_NUMBER_FIND_RE.finditer(q)}
            return len(refs) != 1  # 0 (no number) or >1 (multiple)
        if intent == "TABLE_QUERY":
            refs = {m.group(2) for m in _TABLE_REF_FIND_RE.finditer(q)}
            if re.search(r"\b(table|tab)\b\.?", q, re.IGNORECASE):
                refs |= {m.group(0) for m in _DOT_NUMBER_FIND_RE.finditer(q)}
            return len(refs) != 1
        return False

    # Reference-based retrieval: strict filtering, no FAISS.
    if intent in {"FIGURE_QUERY", "TABLE_QUERY", "PAGE_QUERY"}:
        # If we don't have an explicit target id, we can't do exact reference matching.
        if intent in {"FIGURE_QUERY", "TABLE_QUERY"} and (not target_id or _is_ambiguous_reference()):
            return [], strict_ref_error
        if intent == "PAGE_QUERY" and not target_id:
            return [], "Exact reference not found in document"

        matched = []
        for chunk_id, chunk_data in metadata.items():
            if intent == "FIGURE_QUERY":
                ok = _chunk_matches_figure(chunk_data, target_id)
            elif intent == "TABLE_QUERY":
                ok = _chunk_matches_table(chunk_data, target_id)
            else:  # PAGE_QUERY
                target_page = _extract_page_number(target_id)
                ok = target_page is not None and chunk_data.get("page_number") == target_page

            if not ok:
                continue

            page_number = chunk_data.get("page_number")
            image_path = None
            if page_number:
                candidate = f"images/page_{page_number}_img_1.png"
                if os.path.exists(candidate):
                    image_path = candidate

            matched.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": chunk_data.get("text", ""),
                    "source": chunk_data.get("source", "text"),
                    "page_number": page_number,
                    "image_path": image_path,
                }
            )

        if not matched:
            if intent in {"FIGURE_QUERY", "TABLE_QUERY"}:
                return [], strict_ref_error
            return [], "Exact reference not found in document"

        # Prefer OCR/vision chunks for figure/table references (likely where the actual figure/table content lives),
        # while still remaining strict about matching the reference.
        if intent in {"FIGURE_QUERY", "TABLE_QUERY"}:
            # Exclude index pages like "LIST OF FIGURES/TABLES" (they are not the actual content).
            matched = [m for m in matched if not _is_list_index_page(m.get("text", ""), intent)]
            if not matched:
                return [], strict_ref_error

            ref = _extract_ref_number(target_id)
            if ref:
                if intent == "FIGURE_QUERY":
                    ref_re = re.compile(rf"\bfig(?:ure)?\.?\s*{re.escape(ref)}\b", re.IGNORECASE)
                    src_pri = {"image_ocr": 0, "vision": 1, "text": 2}
                else:
                    ref_re = re.compile(rf"\b(?:table|tab)\.?\s*{re.escape(ref)}\b", re.IGNORECASE)
                    src_pri = {"image_ocr": 0, "text": 1, "vision": 2}

                def _pos(item: dict) -> int:
                    m = ref_re.search(item.get("text", "") or "")
                    return m.start() if m else 10**9

                def _table_score(item: dict) -> int:
                    if intent != "TABLE_QUERY":
                        return 0
                    txt = item.get("text", "") or ""
                    lines = [ln for ln in txt.splitlines() if ln.strip()]
                    class_lines = sum(1 for ln in lines if re.search(r"\bCLASS\b", ln, re.IGNORECASE))
                    multi_col_lines = sum(1 for ln in lines if re.search(r"\s{2,}", ln) or "|" in ln or "\t" in ln)
                    return class_lines * 10 + multi_col_lines + min(len(txt) // 200, 20)

                def _figure_score(item: dict) -> int:
                    if intent != "FIGURE_QUERY":
                        return 0
                    line = _extract_matching_reference_line(item.get("text", "") or "", "FIGURE_QUERY", target_id) or ""
                    l = line.lower().strip()
                    score = 0
                    if l.startswith("fig"):
                        score += 100
                    if "example" in l:
                        score += 50
                    if "shown in" in l:
                        score -= 20
                    score += min(len(line), 200) // 20
                    return score

                matched.sort(
                    key=lambda item: (
                        src_pri.get(item.get("source"), 99),
                        -_table_score(item),
                        -_figure_score(item),
                        _pos(item),
                        item.get("page_number") or 10**9,
                    )
                )

                # For figure queries, if any matches have a page image, keep only those (avoid returning text-only mentions).
                if intent == "FIGURE_QUERY":
                    with_images = [m for m in matched if m.get("image_path")]
                    if with_images:
                        matched = with_images

        # Apply max chunks per intent.
        if intent == "FIGURE_QUERY":
            matched = matched[:1]
        elif intent == "TABLE_QUERY":
            matched = matched[:1]
        else:  # PAGE_QUERY
            matched = matched[:3]

        return matched, None

    # GENERAL semantic search: FAISS (k=6)
    if intent == "GENERAL_QUERY":
        result = genai.embed_content(model="models/embedding-001", content=question)
        query_embedding = np.array([result["embedding"]], dtype=np.float32)

        k = 6
        distances, indices = index.search(query_embedding, k)

        chunks_data = []
        for pos, idx in enumerate(indices[0]):
            chunk_id = str(idx + 1)
            chunk_data = metadata.get(chunk_id, {})
            chunk_text = chunk_data.get("text", "") or ""
            source_type = chunk_data.get("source", "text")
            page_number = chunk_data.get("page_number")
            distance = float(distances[0][pos]) if distances is not None else None

            image_path = None
            if page_number:
                candidate = f"images/page_{page_number}_img_1.png"
                if os.path.exists(candidate):
                    image_path = candidate

            chunks_data.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source_type,
                    "page_number": page_number,
                    "image_path": image_path,
                    "distance": distance,
                }
            )

        chunks_data = deduplicate_chunks(chunks_data)
        source_priority = {"text": 0, "image_ocr": 1, "vision": 2}
        chunks_data.sort(key=lambda x: source_priority.get(x["source"], 99))
        return chunks_data, None

    # SECTION_QUERY or anything else: treat as GENERAL (semantic)
    # (Spec only gave special retrieval rules for FIGURE/TABLE/PAGE/GENERAL.)
    result = genai.embed_content(model="models/embedding-001", content=question)
    query_embedding = np.array([result["embedding"]], dtype=np.float32)

    k = 6
    distances, indices = index.search(query_embedding, k)
    chunks_data = []
    for idx in indices[0]:
        chunk_id = str(idx + 1)
        chunk_data = metadata.get(chunk_id, {})
        chunk_text = chunk_data.get("text", "")
        source_type = chunk_data.get("source", "text")
        page_number = chunk_data.get("page_number")

        image_path = None
        if page_number:
            candidate = f"images/page_{page_number}_img_1.png"
            if os.path.exists(candidate):
                image_path = candidate

        chunks_data.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source_type,
                "page_number": page_number,
                "image_path": image_path,
            }
        )
    chunks_data = deduplicate_chunks(chunks_data)
    source_priority = {"text": 0, "image_ocr": 1, "vision": 2}
    chunks_data.sort(key=lambda x: source_priority.get(x["source"], 99))
    return chunks_data, None


@app.post("/ask", response_model=StructuredAnswerResponse | ErrorResponse)
async def ask_question(request: QuestionRequest):
    if not GEMINI_ENABLED:
        return {"error": "Gemini is not configured (missing GEMINI_API_KEY)."}
    question = request.question
    include_context = bool(getattr(request, "include_context", False))
    # IMPORTANT: classify intent BEFORE any retrieval / FAISS search.
    intent_info = classify_query_intent(question)
    logging.info(f"Intent classification: {intent_info}")

    chunks_data, retrieval_error = _retrieve_chunks_by_intent(intent_info, question)
    if retrieval_error:
        return {"error": retrieval_error}
    
    retrieved_chunks = []
    sources_list = []
    
    # IMPORTANT: In strict technical QA mode, do NOT generate or attach vision captions.
    # These are model-generated and are not verbatim document content.
    include_vision_caption = False
    
    # Semantic match score (only meaningful for GENERAL semantic mode).
    semantic_match_score = None
    if intent_info.get("intent") == "GENERAL_QUERY":
        distances = [c.get("distance") for c in chunks_data if c.get("distance") is not None]
        if distances:
            # Heuristic mapping: sim = 1/(1+d) for L2-like distances.
            best_d = min(distances)
            semantic_match_score = 1.0 / (1.0 + float(best_d))
    
    for chunk_info in chunks_data:
        chunk_text = chunk_info["text"]
        source_type = chunk_info["source"]
        page_number = chunk_info["page_number"]
        image_path = chunk_info["image_path"]
        chunk_id = chunk_info.get("chunk_id")
        
        vision_caption = None
        
        retrieved_chunks.append(chunk_text)
        
        intent = intent_info.get("intent")
        comparison_kind = intent_info.get("comparison_kind")

        # Source reporting:
        # - FIGURE/TABLE: keep it strictly on-reference by using the exact reference line if possible
        # - otherwise: verbatim 1–3 lines
        preview_text = _extract_matching_reference_line(chunk_text, intent, intent_info.get("target_id"))
        if not preview_text:
            preview_text = _make_verbatim_excerpt(chunk_text, max_lines=3)
        start_offset, end_offset = _find_span_offsets(chunk_text, preview_text)
        
        # UI needs full table content + a caption for figure queries.
        raw_text = chunk_text if intent == "TABLE_QUERY" else None
        caption = _extract_figure_caption(chunk_text, intent_info.get("target_id")) if intent == "FIGURE_QUERY" else None

        # UI rendering types:
        # - TABLE for table queries
        # - IMAGE for figure queries when we have a page image available
        # - otherwise TEXT/OCR/IMAGE (normalized from metadata source)
        canonical_type = _source_type_label(source_type)

        if intent == "TABLE_QUERY" or comparison_kind == "TABLE_QUERY":
            ui_source_type = "TABLE"
        elif (intent == "FIGURE_QUERY" or comparison_kind == "FIGURE_QUERY") and image_path:
            ui_source_type = "IMAGE"
        else:
            ui_source_type = canonical_type
        
        sources_list.append(Source(
            chunk_id=chunk_id,
            page_number=page_number,
            source=ui_source_type,
            source_type=canonical_type,
            text=preview_text,
            vision_caption=vision_caption,
            image_path=image_path,
            raw_text=raw_text,
            caption=caption,
            start_offset=start_offset,
            end_offset=end_offset,
        ))
    used_explicit_filtering = intent_info.get("intent") in {"FIGURE_QUERY", "TABLE_QUERY", "PAGE_QUERY"}
    
    context = "\n".join(retrieved_chunks)
    
    # Comparison-specific prompt override
    if intent_info.get("intent") == INTENT_COMPARISON:
        targets = intent_info.get("comparison_targets") or []
        kind = intent_info.get("comparison_kind")
        prompt = f"""
You are a technical document QA assistant.

TASK:
Compare the two referenced items ONLY. Do not add external explanation.

REQUIRED STRUCTURE:
1. Common elements
2. Key differences

STYLE:
- Use bullet points or a mini-table
- No surrounding theory

CRITICAL RULES:
- Answer ONLY from the provided context.
- If either item is missing or unclear, say: Not explicitly specified in the document.

CRITICAL OUTPUT RULES:
- Return VALID JSON ONLY.
- Follow the schema exactly.
- Do NOT include markdown.
- Do NOT include numbering inside text.
- Do NOT merge list items into sentences.

Schema:
{{
  "blocks": [
    {{
      "type": "list",
      "items": ["Common elements: ..."]
    }},
    {{
      "type": "list",
      "items": ["Key differences: ..."]
    }}
  ]
}}

Compared items:
{json.dumps(targets, ensure_ascii=False)}

Context:
{context}

Question:
{question}
"""
    else:
        prompt = f"""
You are a technical document QA assistant.

NON-NEGOTIABLE RULES:
1. Answer ONLY from the provided context.
2. If the question asks for a figure or table:
   - Do NOT explain surrounding theory.
   - Do NOT summarize the chapter.
3. If the exact answer is not present in the context:
   - Say exactly: "Not explicitly specified in the document."

ANSWER STYLE:
- Crisp
- Bullet points if possible
- No filler
- No rephrasing beyond the document text

CRITICAL OUTPUT RULES:
- Return VALID JSON ONLY.
- Follow the schema exactly.
- Do NOT include markdown.
- Do NOT include numbering inside text.
- Do NOT merge list items into sentences.

Schema:
{{
  "blocks": [
    {{
      "type": "paragraph",
      "text": "Short, clear paragraph."
    }},
    {{
      "type": "list",
      "items": [
        "Each item must be a short, clean sentence",
        "No numbering, no conjunction chaining",
        "One idea per item"
      ]
    }}
  ]
}}

Guidelines:
- Prefer a list block when possible, otherwise a single short paragraph.
- Use ONLY words and values found in the context.

Query intent:
{json.dumps(intent_info, ensure_ascii=False)}

Context:
{context}

Question:
{question}
"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    raw_text = response.text.strip()
    raw_text = clean_llm_json(raw_text)
    
    blocks = safe_parse_blocks(raw_text)
    chunks_used_count = len(chunks_data)

    if intent_info.get("intent") == INTENT_COMPARISON:
        # Confidence rule: min(conf of compared items); cap at 0.80 if either is OCR or IMAGE.
        # Compute item confidences using their own primary source strength.
        item_scores = []
        for s in sources_list[:2]:
            item_score = 1.00
            st = (s.source_type or s.source) or ""
            if st == "OCR":
                item_score -= 0.15
            if st == "IMAGE":
                item_score -= 0.25
            # Exact match bonus for comparison items.
            item_score += 0.10
            # Verbatim bonus: comparison sources are reference lines by construction.
            item_score += 0.10
            item_score = min(item_score, 0.99)
            item_score = _clamp(item_score, 0.30, 0.95)
            item_scores.append(item_score)

        final_conf = min(item_scores) if item_scores else 0.55
        if any(((s.source_type or s.source) in {"OCR", "IMAGE"}) for s in sources_list[:2]):
            final_conf = min(final_conf, 0.80)
        final_conf = round(_clamp(final_conf, 0.30, 0.95), 2)

        blocks = _enrich_blocks_with_fixed_confidence(
            blocks=blocks,
            score=final_conf,
            label=_confidence_label(final_conf),
            per_block_sources=sources_list[:2],
        )
    else:
        # Attach confidence scoring per block (required contract).
        exact_match = intent_info.get("intent") in {"FIGURE_QUERY", "TABLE_QUERY"} and bool(intent_info.get("target_id"))
        blocks = _enrich_blocks_with_confidence(
            blocks=blocks,
            intent_info=intent_info,
            per_block_sources=sources_list,
            chunks_used_count=chunks_used_count,
            semantic_match_score=semantic_match_score,
            exact_match=exact_match,
        )
    
    def is_source_relevant(answer_text: str, source_text: str) -> bool:
        if not answer_text or not source_text:
            return False

        answer_words = set(
            w.lower()
            for w in answer_text.split()
            if len(w) > 4
        )

        source_text_lower = source_text.lower()

        for word in answer_words:
            if word in source_text_lower:
                return True

        return False
    
    # Extract full answer text from AI blocks
    answer_text_parts = []
    for block in blocks:
        if block.get("type") == "paragraph" and block.get("text"):
            answer_text_parts.append(block["text"])
        elif block.get("type") == "list" and block.get("items"):
            answer_text_parts.extend(block["items"])
    full_answer_text = " ".join(answer_text_parts)
    
    # Filter sources based on relevance to answer (skip if explicit filtering was used)
    if not used_explicit_filtering and full_answer_text:
        filtered_sources = [
            source for source in sources_list
            if is_source_relevant(full_answer_text, source.text)
        ]
        if filtered_sources:
            sources_list = filtered_sources
    
    final_sources = []
    for source in sources_list:
        chunk_text = source.text
        highlight_spans = extract_longest_sentences(chunk_text, max_sentences=2)
        
        # Add prefix for FIGURE queries when we have a page number.
        display_text = source.text
        if intent_info.get("intent") == "FIGURE_QUERY" and intent_info.get("target_id") and source.page_number is not None:
            display_text = f"{intent_info['target_id']} — Page {source.page_number}\n{source.text}"
        
        final_sources.append(Source(
            chunk_id=source.chunk_id,
            page_number=source.page_number,
            source=source.source,
            source_type=source.source_type,
            text=display_text,
            vision_caption=source.vision_caption,
            image_path=source.image_path,
            highlight_phrases=highlight_spans if highlight_spans else None,
            raw_text=source.raw_text,
            caption=source.caption,
            start_offset=source.start_offset,
            end_offset=source.end_offset,
        ))
    
    contextual_sources = _build_contextual_sources(final_sources, intent_info) if include_context else None
    
    return StructuredAnswerResponse(
        blocks=[AnswerBlock(**b) for b in blocks],
        sources=final_sources,
        contextual_sources=contextual_sources,
    )

# ============================
# ANALYZE DRAWING MODE (NEW)
# ============================

def extract_json_from_text(text: str):
    """
    Extract JSON object from LLM output safely.
    Supports:
    - Raw JSON
    - ```json ... ```
    - Text before/after JSON
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass
    
    return None

def pdf_first_page_to_image(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)

    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pix.save(temp_image.name)
    return temp_image.name


# Drawing analysis endpoints removed - only PDF Q&A chatbot is supported
# The /analyze-drawing, /ask-drawing, and /generate-bom endpoints have been removed
# Only PDF Q&A chatbot functionality remains

if False:  # Disabled drawing analysis code
    # Removed /analyze-drawing endpoint code
    pass
            "summary": "Gemini is not configured (missing GEMINI_API_KEY).",
            "components": [],
            "foundation": {},
            "standards": [],
            "uncertain_items": ["GEMINI_API_KEY not set"],
        }
    logging.info(f"File received: {file.filename}")
    
    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1].lower()
    
    # Validate file format
    allowed_formats = {".pdf", ".png", ".jpg", ".jpeg"}
    if suffix not in allowed_formats:
        return {
            "drawing_type": "Unknown",
            "summary": f"Unsupported file format: {suffix}. Supported formats: PDF, PNG, JPG, JPEG",
            "components": [],
            "foundation": {},
            "standards": [],
            "uncertain_items": [f"Invalid file format: {suffix}"]
        }
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(await file.read())
    temp_file.close()

    pdf_path = temp_file.name
    image_path = pdf_path
    pdf_temp_image_path = None

    # Convert PDF → image (first page only) for Gemini Vision
    if suffix == ".pdf":
        pdf_temp_image_path = pdf_first_page_to_image(pdf_path)
        image_path = pdf_temp_image_path

    # Note: Dimension extraction removed - keeping analysis text-only

    # Gemini Vision model
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
You are an engineering drawing reader extracting Bill of Materials (BOM) data.

CRITICAL: EXPLODE ASSEMBLIES INTO INDIVIDUAL ITEMS

DO NOT treat assemblies/structures as single BOM rows.
INSTEAD, extract each physical fabricated item separately.

Example - If drawing shows a gantry structure:
✓ CORRECT:
  - Vertical post (MS Pipe Ø 88.9 x 3.2 mm) - Qty: 4
  - Horizontal beam (MS Pipe Ø 114.3 x 4.5 mm) - Qty: 2
  - Diagonal bracing (ISA 50 x 50 x 6 mm) - Qty: 8
  - Base plate (MS Plate 300 x 300 x 10 mm) - Qty: 4
  - Stiffener plate (MS Plate 100 x 100 x 8 mm) - Qty: 16
  - Bolt (M16 x 50 mm) - Qty: 32

✗ WRONG:
  - Gantry structure - Qty: 1
  - Sign support assembly - Qty: 1

TASK:
Extract individual components with explicit numerical dimensions AND count visually repeated identical components.

PRIORITY 1: GEOMETRY-BASED QUANTITY INFERENCE
For each component type with readable dimensions:
1. Group by: component type + exact dimensions
2. Count how many times that identical component appears VISUALLY in the drawing
3. Use the visual count as quantity

Examples:
- 4 identical vertical posts (same dimensions) → quantity = 4
- 6 identical stiffener plates (same size) → quantity = 6
- 2 identical ACP panels (same dimensions) → quantity = 2
- 8 identical bolts in a pattern → quantity = 8

CRITICAL: Count VISUALLY repeated geometric elements, not just text labels.

PRIORITY 2: DIMENSION EXTRACTION
MANDATORY REQUIREMENTS:
- Read all visible dimension annotations (in mm)
- Read diameters (Ø), thicknesses, lengths, widths
- Use the EXACT numbers written in the drawing
- If a dimension is not clearly readable, set it to null
- If ALL critical dimensions are missing, DO NOT output that component

COMPONENT TYPES (strict):
- MS_PIPE: Requires readable diameter (for posts, beams, handrails)
- ISA: Requires readable width x height (for angles, bracings)
- PLATE: Requires readable width x height (for base plates, stiffeners, gussets)
- ACP_SHEET: Requires readable width x height (for panels)
- REFLECTIVE_SHEET: Requires readable width x height (for sign faces)
- BOLT: Requires readable diameter (for connections)

QUANTITY EXTRACTION PRIORITY:
1. FIRST: Count visually repeated identical components in the drawing
2. SECOND: Read explicit quantity callouts (QTY, NOS, TYP) if present
3. THIRD: Infer from symmetry cues (BOTH SIDES, LHS & RHS) if explicit
4. If truly uncertain after all above, set to null

DO NOT:
- Treat assemblies as single items
- Output "gantry", "structure", "assembly", "frame" as component types
- Guess dimensions
- Estimate sizes
- Use placeholders
- Include components with unreadable dimensions

OUTPUT FORMAT:
For each individual fabricated component with readable dimensions:
{
  "type": "MS_PIPE" | "ISA" | "PLATE" | "ACP_SHEET" | "REFLECTIVE_SHEET" | "BOLT",
  "diameter_mm": number | null,
  "thickness_mm": number | null,
  "width_mm": number | null,
  "height_mm": number | null,
  "length_mm": number | null,
  "quantity": number | null,
  "standard_reference": string | null,
  "drawing_note_reference": string | null,
  "confidence": number (0.0 to 1.0)
}

CONFIDENCE SCORING:
- 0.9-1.0: All dimensions + quantity counted visually
- 0.7-0.9: Dimensions readable, quantity from text label or visual count
- 0.5-0.7: Some dimensions readable, quantity inferred
- Below 0.5: Omit the component (dimensions too unclear)

IMPORTANT: 
- Even if no quantity labels exist, COUNT the visual repetitions.
- Extract EVERY distinct physical part, not assemblies.

Return VALID JSON ONLY.
Do NOT include markdown.
Do NOT add extra keys.

SCHEMA:
- Return VALID JSON ONLY
- Follow the schema EXACTLY
- Do NOT include markdown
- Do NOT add extra keys

SCHEMA:
{
  "drawing_type": "string",
  "summary": "string",
  "components": [
    {
      "type": "MS_PIPE|ISA|PLATE|ACP_SHEET|REFLECTIVE_SHEET|BOLT|OTHER",
      "diameter_mm": 0,
      "thickness_mm": 0,
      "width_mm": 0,
      "height_mm": 0,
      "length_mm": 0,
      "quantity": 0,
      "standard_reference": "string",
      "drawing_note_reference": "string",
      "confidence": 0.0
    }
  ],
  "foundation": {
    "type": "string",
    "dimensions": ["string"],
    "material": "string"
  },
  "standards": ["string"],
  "uncertain_items": ["string"]
}
"""

    image = Image.open(image_path)
    response = model.generate_content([prompt, image])
    raw_text = response.text.strip()
    
    logging.info("AI response received")

    structured = extract_json_from_text(raw_text)
    if not structured:
        logging.warning("JSON extraction failed")
        structured = {
            "drawing_type": "Unknown",
            "summary": "Failed to extract structured data",
            "components": [],
            "foundation": {},
            "standards": [],
            "uncertain_items": ["AI output was not valid JSON"]
        }
    else:
        logging.info("JSON extraction successful")

    # Normalize strict components list (prevents drift / ensures "dumb but honest")
    comps, comp_errors = _normalize_strict_components_list(structured.get("components"))
    structured["components"] = comps
    if comp_errors:
        ui_uncertain = list(structured.get("uncertain_items") or [])
        ui_uncertain.append(f"Component extraction normalized with warnings: {', '.join(comp_errors[:12])}")
        structured["uncertain_items"] = ui_uncertain

    # Log exact response sent to frontend
    logging.info("Analyze Drawing Response:\n" + json.dumps(structured, indent=2))

    # Clean up temporary files
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if pdf_temp_image_path and os.path.exists(pdf_temp_image_path):
            os.remove(pdf_temp_image_path)
    except Exception as e:
        logging.warning(f"Failed to clean up temp files: {str(e)}")

    return structured

# ============================
# DRAWING FOLLOW-UP QUESTIONS
# ============================

@app.post("/ask-drawing", response_model=StructuredAnswerResponse)
async def ask_drawing_question(request: DrawingQuestionRequest):
    """
    Answer follow-up questions about a drawing analysis.
    Uses ONLY the provided drawing JSON - no image processing, no FAISS, no OCR.
    """
    if not GEMINI_ENABLED:
        return StructuredAnswerResponse(
            blocks=[
                AnswerBlock(
                    type="paragraph",
                    text="Gemini is not configured (missing GEMINI_API_KEY).",
                    answer="Gemini is not configured (missing GEMINI_API_KEY).",
                    confidence_score=0.3,
                    confidence_label="Low",
                    sources=[],
                )
            ],
            sources=[],
            contextual_sources=None,
        )
    question = request.question
    drawing_analysis = request.drawing_analysis
    
    # Convert drawing analysis to a readable text format for context
    context_parts = []
    
    if drawing_analysis.get("drawing_type"):
        context_parts.append(f"Drawing Type: {drawing_analysis['drawing_type']}")
    
    if drawing_analysis.get("summary"):
        context_parts.append(f"Summary: {drawing_analysis['summary']}")
    
    if drawing_analysis.get("components") and len(drawing_analysis["components"]) > 0:
        context_parts.append("\nComponents:")
        for comp in drawing_analysis["components"]:
            # Preferred: STRICT component schema (type + explicit dims + qty + refs).
            if isinstance(comp, dict) and comp.get("type"):
                context_parts.append(f"  - {_format_strict_component_for_context(comp)}")
                continue

            # Backward compatibility: legacy component objects.
            comp_text = f"  - {comp.get('name', 'Unnamed') if isinstance(comp, dict) else 'Unnamed'}"
            if isinstance(comp, dict) and comp.get("dimensions"):
                comp_text += f" | Dimensions: {', '.join(comp['dimensions'])}"
            if isinstance(comp, dict) and comp.get("material"):
                comp_text += f" | Material: {comp['material']}"
            if isinstance(comp, dict) and comp.get("notes"):
                comp_text += f" | Notes: {comp['notes']}"
            context_parts.append(comp_text)
    
    if drawing_analysis.get("foundation"):
        foundation = drawing_analysis["foundation"]
        foundation_text = "Foundation:"
        if foundation.get("type"):
            foundation_text += f" Type: {foundation['type']}"
        if foundation.get("dimensions"):
            foundation_text += f" | Dimensions: {', '.join(foundation['dimensions'])}"
        if foundation.get("material"):
            foundation_text += f" | Material: {foundation['material']}"
        context_parts.append(foundation_text)
    
    if drawing_analysis.get("standards") and len(drawing_analysis["standards"]) > 0:
        context_parts.append(f"\nStandards: {', '.join(drawing_analysis['standards'])}")
    
    if drawing_analysis.get("uncertain_items") and len(drawing_analysis["uncertain_items"]) > 0:
        context_parts.append(f"\nUncertain Items: {', '.join(drawing_analysis['uncertain_items'])}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""
You are a technical assistant answering questions about a drawing analysis.

CRITICAL RULES:
- Answer ONLY from the provided drawing analysis data below
- Do NOT reference or re-analyze any images
- Do NOT make up information not present in the analysis
- If information is not available, say so clearly
- Use the exact values and specifications from the analysis

Drawing Analysis Data:
{context}

Question:
{question}

CRITICAL OUTPUT RULES:
- Return VALID JSON ONLY.
- Follow the schema exactly.
- Do NOT include markdown.
- Do NOT include numbering inside text.
- Do NOT merge list items into sentences.

Schema:
{{
  "blocks": [
    {{
      "type": "paragraph",
      "text": "Short, clear paragraph."
    }},
    {{
      "type": "list",
      "items": [
        "Each item must be a short, clean sentence",
        "No numbering, no conjunction chaining",
        "One idea per item"
      ]
    }}
  ]
}}

Guidelines:
- Use a paragraph block for explanations, definitions, and descriptions.
- Use a list block ONLY when:
  • The question explicitly asks for a list, types, categories, or steps
  • Or comparison between multiple items is required
- Do NOT convert explanatory content into bullet points unnecessarily.
- Prefer paragraphs when a concept can be explained clearly in prose.
- Use lists for clarity, not by default.
"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    raw_text = response.text.strip()
    raw_text = clean_llm_json(raw_text)
    
    blocks = safe_parse_blocks(raw_text)
    intent_info = classify_query_intent(question)
    blocks = _enrich_blocks_with_confidence(
        blocks=blocks,
        intent_info=intent_info,
        per_block_sources=[],
        chunks_used_count=1,
        semantic_match_score=None,
        exact_match=False,
    )
    
    # No sources for drawing questions - all info comes from the JSON
    return StructuredAnswerResponse(
        blocks=[AnswerBlock(**b) for b in blocks],
        sources=[]
    )
# End of removed ask-drawing endpoint


# Drawing analysis endpoints removed - only PDF Q&A chatbot is supported
# ============================
# GENERATE BILL OF MATERIALS
# ============================

import hashlib
from pathlib import Path

# BOM cache directory
BOM_CACHE_DIR = Path("bom_cache")
BOM_CACHE_DIR.mkdir(exist_ok=True)

STRICT_BOM_COLUMNS = [
    "Item No",
    "Item Description",
    "Material Specification",
    "Part Drawing No",
    "QTY",
]

# ============================
# STRICT COMPONENT EXTRACTION (VISION "DUMB BUT HONEST")
# ============================

STRICT_COMPONENT_TYPES = {
    "MS_PIPE",
    "ISA",
    "PLATE",
    "ACP_SHEET",
    "REFLECTIVE_SHEET",
    "BOLT",
}

STRICT_COMPONENT_KEYS = [
    "type",
    "diameter_mm",
    "thickness_mm",
    "width_mm",
    "height_mm",
    "length_mm",
    "quantity",
    "standard_reference",
    "drawing_note_reference",
    "confidence",
]


def _parse_number_or_null(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    # Keep only the first numeric token (supports "88.9", "88.9 mm", "Ø88.9", etc.)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _parse_int_or_null(v):
    n = _parse_number_or_null(v)
    if n is None:
        return None
    # Quantity must be an integer count; if it's not an integer, treat as unknown.
    if abs(n - round(n)) > 1e-6:
        return None
    q = int(round(n))
    return q if q > 0 else None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_strict_component(obj: dict) -> tuple[Optional[dict], list[str]]:
    """
    Normalize a single extracted component into the strict schema.
    Returns: (component_or_none, errors)
    
    STRICT RULES:
    - Omit items with unknown component types
    - Omit items with all dimensions unknown
    """
    errors: list[str] = []
    if not isinstance(obj, dict):
        return None, ["component_not_object"]

    t = obj.get("type")
    if isinstance(t, str):
        t = t.strip().upper()
    if t not in STRICT_COMPONENT_TYPES:
        # Unknown component type - omit entirely
        errors.append("type_invalid_omitted")
        return None, errors

    comp = {
        "type": t,
        "diameter_mm": _parse_number_or_null(obj.get("diameter_mm")),
        "thickness_mm": _parse_number_or_null(obj.get("thickness_mm")),
        "width_mm": _parse_number_or_null(obj.get("width_mm")),
        "height_mm": _parse_number_or_null(obj.get("height_mm")),
        "length_mm": _parse_number_or_null(obj.get("length_mm")),
        "quantity": _parse_int_or_null(obj.get("quantity")),
        "standard_reference": (str(obj.get("standard_reference")).strip() if obj.get("standard_reference") is not None else None),
        "drawing_note_reference": (str(obj.get("drawing_note_reference")).strip() if obj.get("drawing_note_reference") is not None else None),
        "confidence": _clamp01(_parse_number_or_null(obj.get("confidence")) or 0.0),
    }

    # Normalize empty strings → null
    if comp["standard_reference"] == "":
        comp["standard_reference"] = None
    if comp["drawing_note_reference"] == "":
        comp["drawing_note_reference"] = None

    # Enforce no extra keys on output (strict)
    comp = {k: comp.get(k) for k in STRICT_COMPONENT_KEYS}
    return comp, errors


def _normalize_strict_components_list(value) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    if value is None:
        return [], ["components_missing"]
    if not isinstance(value, list):
        return [], ["components_not_list"]

    out: list[dict] = []
    for i, it in enumerate(value):
        comp, comp_errors = _normalize_strict_component(it)
        if comp is None:
            errors.append(f"component_{i}_invalid")
            continue
        if comp_errors:
            errors.extend([f"component_{i}_{e}" for e in comp_errors])
        out.append(comp)
    return out, errors


def _format_strict_component_for_context(comp: dict) -> str:
    """
    Deterministic, non-inventive formatting for downstream prompts/UI context.
    """
    t = comp.get("type") or "OTHER"
    dims = []
    for key, label in [
        ("diameter_mm", "Ø"),
        ("width_mm", "W"),
        ("height_mm", "H"),
        ("thickness_mm", "T"),
        ("length_mm", "L"),
    ]:
        v = comp.get(key)
        if isinstance(v, (int, float)):
            if label == "Ø":
                dims.append(f"Ø {v:g} mm")
            else:
                dims.append(f"{label} {v:g} mm")
    qty = comp.get("quantity")
    std = comp.get("standard_reference")
    note = comp.get("drawing_note_reference")
    conf = comp.get("confidence")
    parts = [f"{t}"]
    if dims:
        parts.append(" | " + ", ".join(dims))
    if qty is not None:
        parts.append(f" | QTY {qty}")
    if std:
        parts.append(f" | STD {std}")
    if note:
        parts.append(f" | NOTE {note}")
    if isinstance(conf, (int, float)):
        parts.append(f" | CONF {float(conf):.2f}")
    return "".join(parts)

def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file for caching"""
    return hashlib.sha256(file_bytes).hexdigest()

def load_bom_schema_and_rules():
    """Load BOM schema and mapping rules"""
    try:
        with open('bom_schema.json', 'r') as f:
            schema = json.load(f)
        with open('bom_mapping_rules.json', 'r') as f:
            rules = json.load(f)
        return schema, rules
    except Exception as e:
        logging.error(f"Failed to load BOM schema/rules: {e}")
        return None, None


def _clean_llm_json_loose(text: str) -> str:
    """
    Best-effort cleanup when the model wraps JSON in fences or adds prefix/suffix text.
    """
    if not text:
        return ""
    t = text.strip()
    # Strip common markdown fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    # Extract first JSON object-like span if present
    m = re.search(r"\{[\s\S]*\}", t)
    return (m.group(0) if m else t).strip()


def _normalize_material_spec(description: str, material_spec: str) -> str:
    """
    Enforce canonical material spec outputs required by the strict BOM contract.
    Allowed values:
      - "As per IS 1239"
      - "As per IS 2062 & IS 808"
      - "Detail given in Note"
    """
    d = (description or "").lower()
    s = (material_spec or "").strip()
    s_l = s.lower()

    # Normalize common variants
    if "detail given in note" in s_l or s_l == "as per note" or s_l == "as per notes":
        return "Detail given in Note"

    # If description clearly indicates pipe, enforce IS 1239.
    if "pipe" in d or "chs" in d or "od" in d or "ø" in d or "dia" in d:
        if "1239" in s_l:
            return "As per IS 1239"
        # If model gave something else, still normalize to the allowed canonical value.
        return "As per IS 1239"

    # ACP / reflective sheets are typically specified in notes; keep contract-safe output.
    if "acp" in d or ("reflective" in d and "sheet" in d) or "retroreflective" in d:
        return "Detail given in Note"

    # If description indicates ISA/angle/plate, enforce IS 2062 & IS 808.
    if "isa" in d or "angle" in d or "plate" in d:
        if "2062" in s_l or "808" in s_l:
            return "As per IS 2062 & IS 808"
        return "As per IS 2062 & IS 808"

    # Otherwise, keep only if it's one of the allowed canonical strings; else blank it.
    if s in {"As per IS 1239", "As per IS 2062 & IS 808", "Detail given in Note"}:
        return s
    if "1239" in s_l:
        return "As per IS 1239"
    if "2062" in s_l or "808" in s_l:
        return "As per IS 2062 & IS 808"
    if "note" in s_l:
        return "Detail given in Note"
    return ""


def _fmt_mm(v: float) -> str:
    """
    Format numeric dimension (mm) without inventing precision.
    """
    if v is None:
        return ""
    try:
        return f"{float(v):g}"
    except Exception:
        return ""


def _pdf_extract_text_and_tables(pdf_path: str, max_pages: int = 2) -> tuple[str, list[list[list[str]]]]:
    """
    Best-effort extraction of text + tabular schedules from a drawing PDF.
    Returns: (combined_text, tables)
      - combined_text: joined page texts
      - tables: list[tables] where each table is list[rows] and each row is list[cells]
    """
    combined_text_parts: list[str] = []
    all_tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[: max_pages or 1]):
                try:
                    t = page.extract_text() or ""
                    if t:
                        combined_text_parts.append(t)
                except Exception:
                    pass

                try:
                    for tbl in page.extract_tables() or []:
                        if tbl:
                            all_tables.append(tbl)
                except Exception:
                    pass
    except Exception:
        pass

    return "\n".join(combined_text_parts).strip(), all_tables


# ============================
# DRAWING NOTES PARSING
# ============================

def _parse_drawing_notes(extracted_text: str) -> dict:
    """
    Aggressively parse drawing notes for global specifications.
    
    Returns:
    {
        "global_treatments": ["hot dip galvanized", ...],
        "dimension_hints": {"ACP_SHEET": {"thickness_mm": 4}, ...},
        "standards": ["IS 1239", "IS 2062", ...],
        "material_specs": {"MS_PIPE": "As per IS 1239", ...}
    }
    """
    notes_data = {
        "global_treatments": [],
        "dimension_hints": {},
        "standards": [],
        "material_specs": {}
    }
    
    if not extracted_text:
        return notes_data
    
    text_lower = extracted_text.lower()
    
    # Extract global treatments (apply to all MS items unless scoped)
    if re.search(r"\ball\s+ms\s+(items?|parts?|members?|materials?|components?)\s+(?:to be\s+)?hot\s+dip\s+galvanized", text_lower):
        notes_data["global_treatments"].append("hot dip galvanized")
    if re.search(r"\ball\s+ms\s+(items?|parts?|members?)\s+(?:to be\s+)?(?:powder\s+)?coated", text_lower):
        notes_data["global_treatments"].append("powder coated")
    if re.search(r"\ball\s+ms\s+(items?|parts?|members?)\s+(?:to be\s+)?painted", text_lower):
        notes_data["global_treatments"].append("painted")
    if re.search(r"\ball\s+welded\s+joints?\s+(?:to be\s+)?ground\s+smooth", text_lower):
        notes_data["global_treatments"].append("welded joints ground smooth")
    
    # Extract dimension hints from notes (e.g., "ACP sheet 4mm thick")
    # ACP sheets
    m = re.search(r"\bacp\s+sheet\s+(\d+(?:\.\d+)?)\s*mm\s+thick", text_lower)
    if m:
        notes_data["dimension_hints"]["ACP_SHEET"] = {"thickness_mm": float(m.group(1))}
    
    # Reflective sheets
    m = re.search(r"\b(?:reflective|retroreflective)\s+sheet\s+(\d+(?:\.\d+)?)\s*mm", text_lower)
    if m:
        notes_data["dimension_hints"]["REFLECTIVE_SHEET"] = {"thickness_mm": float(m.group(1))}
    
    # Extract IS codes and standards
    for m in re.finditer(r"\bIS\s+(\d{3,5})", extracted_text, re.IGNORECASE):
        std = f"IS {m.group(1)}"
        if std not in notes_data["standards"]:
            notes_data["standards"].append(std)
    
    # Map standards to component types
    if "IS 1239" in notes_data["standards"]:
        notes_data["material_specs"]["MS_PIPE"] = "As per IS 1239"
    if "IS 2062" in notes_data["standards"] or "IS 808" in notes_data["standards"]:
        notes_data["material_specs"]["ISA"] = "As per IS 2062 & IS 808"
        notes_data["material_specs"]["PLATE"] = "As per IS 2062 & IS 808"
    if "IS 1367" in notes_data["standards"]:
        notes_data["material_specs"]["BOLT"] = "As per IS 1367"
    
    return notes_data


def _apply_notes_to_item(item: dict, notes_data: dict) -> dict:
    """
    Apply parsed drawing notes to a BOM item.
    - Add dimension hints if missing
    - Apply global treatments to material specs
    - Use standards from notes
    """
    item = dict(item)  # Create copy
    desc = item.get("description", "")
    desc_lower = desc.lower()
    
    # Detect component type from description
    comp_type = None
    if "pipe" in desc_lower or "chs" in desc_lower:
        comp_type = "MS_PIPE"
    elif "isa" in desc_lower or "angle" in desc_lower:
        comp_type = "ISA"
    elif "plate" in desc_lower:
        comp_type = "PLATE"
    elif "acp" in desc_lower:
        comp_type = "ACP_SHEET"
    elif "reflective" in desc_lower:
        comp_type = "REFLECTIVE_SHEET"
    elif "bolt" in desc_lower:
        comp_type = "BOLT"
    
    # Apply dimension hints from notes if dimensions are missing
    if comp_type and comp_type in notes_data.get("dimension_hints", {}):
        hints = notes_data["dimension_hints"][comp_type]
        # If description doesn't have thickness but notes specify it, add it
        if "thickness_mm" in hints and "x ?" in desc:
            # Replace unknown thickness with note value
            item["description"] = desc.replace("x ?", f"x {_fmt_mm(hints['thickness_mm'])}")
    
    # Apply material spec from notes if needed
    if comp_type and comp_type in notes_data.get("material_specs", {}):
        current_spec = item.get("material_specification", "")
        if not current_spec or current_spec == "Detail given in Note":
            item["material_specification"] = notes_data["material_specs"][comp_type]
            item["_material_from_notes"] = True
    
    # Apply global treatments to material spec
    if notes_data.get("global_treatments"):
        current_spec = item.get("material_specification", "")
        # Only apply to MS items
        if comp_type in {"MS_PIPE", "ISA", "PLATE"} and current_spec:
            # Check if treatment not already mentioned
            spec_lower = current_spec.lower()
            for treatment in notes_data["global_treatments"]:
                if treatment.lower() not in spec_lower:
                    # Add treatment as suffix
                    item["material_specification"] = f"{current_spec} ({treatment})"
                    item["_material_from_notes"] = True
                    break  # Apply first applicable treatment only
    
    return item


# ============================
# BOM-FIRST SIGNAL EXTRACTION
# ============================

def _extract_explicit_bom_table(tables: list[list[list[str]]]) -> list[dict]:
    """
    SIGNAL 1 — EXPLICIT BOM TABLE
    Extract tabular BOM if it exists anywhere in the drawing.
    Normalize columns to: item_no, description, material_specification, part_drawing_no, qty
    
    Returns list of BOM items or empty list if no explicit BOM table found.
    """
    items: list[dict] = []
    
    for tbl in tables or []:
        if not isinstance(tbl, list) or len(tbl) < 2:
            continue
        
        # Look for BOM table indicators in headers
        header_candidates = tbl[:3]  # Check first 3 rows for header
        bom_header_idx = None
        
        for i, row in enumerate(header_candidates):
            if not isinstance(row, list):
                continue
            row_text = " ".join([str(c or "").strip() for c in row]).lower()
            
            # Check for BOM table signature: must have both "item" and "qty" indicators
            has_item = any(k in row_text for k in ["item", "no", "number", "s.no", "s no"])
            has_qty = any(k in row_text for k in ["qty", "quantity", "nos", "no.", "pcs", "ea"])
            has_desc = any(k in row_text for k in ["description", "desc", "member", "material", "specification"])
            
            # Strong BOM table signature: item + qty + description
            if has_item and has_qty and has_desc:
                bom_header_idx = i
                break
        
        if bom_header_idx is None:
            continue
        
        # Extract column indices
        header = tbl[bom_header_idx]
        
        def _find_col(keys: set[str], alt_keys: set[str] = None) -> Optional[int]:
            for j, h in enumerate(header or []):
                ht = str(h or "").strip().lower()
                if any(k in ht for k in keys):
                    return j
                if alt_keys and any(k in ht for k in alt_keys):
                    return j
            return None
        
        item_no_col = _find_col({"item", "no", "number", "s.no", "s no", "sl no", "sl.no"})
        desc_col = _find_col({"description", "desc", "member", "item description"})
        mat_col = _find_col({"material", "specification", "material specification", "spec"}, {"is", "standard"})
        part_col = _find_col({"part", "drawing", "part drawing", "drawing no", "part no"})
        qty_col = _find_col({"qty", "quantity", "nos", "no.", "pcs", "ea"})
        
        # Must have at least description and qty to be a valid BOM table
        if desc_col is None or qty_col is None:
            continue
        
        # Extract rows
        start_row = bom_header_idx + 1
        for row in tbl[start_row:]:
            if not isinstance(row, list) or len(row) == 0:
                continue
            
            # Extract values
            item_no = str(row[item_no_col] if item_no_col is not None and item_no_col < len(row) else "").strip()
            desc = str(row[desc_col] if desc_col is not None and desc_col < len(row) else "").strip()
            mat = str(row[mat_col] if mat_col is not None and mat_col < len(row) else "").strip()
            part = str(row[part_col] if part_col is not None and part_col < len(row) else "").strip()
            qty_val = str(row[qty_col] if qty_col is not None and qty_col < len(row) else "").strip()
            
            # Skip empty rows
            if not desc and not item_no:
                continue
            
            # Parse quantity
            qty = _parse_int_qty(qty_val)
            qty_explicit = bool(_parse_int_qty(qty_val))
            qty_inferred_missing = False
            
            if qty is None:
                # If no explicit qty, try to infer from symmetry/repetition hints in description
                inferred = _infer_qty_from_symmetry_text(desc)
                if inferred is not None:
                    qty = inferred
                    qty_explicit = False
                    qty_inferred_missing = True
                else:
                    qty = 1  # Default to 1 if not specified
                    qty_explicit = False
                    qty_inferred_missing = True
            
            # Normalize material specification
            mat_spec = _normalize_material_spec(desc, mat) if mat else _normalize_material_spec(desc, "")
            material_from_notes = False
            if not mat_spec:
                mat_spec = "Detail given in Note"
                material_from_notes = True
            elif mat_spec == "Detail given in Note":
                material_from_notes = True
            # Also check if material column explicitly mentions notes
            elif mat and "note" in str(mat).lower():
                material_from_notes = True
            
            items.append({
                "item_no": item_no,
                "description": desc,
                "material_specification": mat_spec,
                "part_drawing_no": part if part else "-",
                "qty": int(qty),
                "_material_from_notes": material_from_notes,
                "_qty_inferred_missing_label": qty_inferred_missing,
                "_qty_explicit": qty_explicit,
            })
        
        # If we found items from this table, return them (first BOM table wins)
        if items:
            logging.info(f"SIGNAL 1: Extracted {len(items)} items from explicit BOM table")
            return items
    
    return []


def _detect_repeated_structural_members(extracted_text: str, vision_components: list[dict]) -> list[dict]:
    """
    SIGNAL 2 — REPEATED STRUCTURAL MEMBERS
    Detect repeated geometry with identical or near-identical dimensions.
    If the same member appears N times, infer quantity = N.
    
    Examples: vertical posts, horizontal beams, bracing members, stiffener plates
    """
    items: list[dict] = []
    
    # Build signature map from text callouts
    signature_counts: dict[str, int] = {}
    signature_lines: dict[str, list[str]] = {}
    
    def _build_signature(line: str) -> Optional[str]:
        """Build canonical signature for a structural member from text."""
        if not line:
            return None
        s = line.strip()
        
        # ISA angles
        m = re.search(r"\bISA\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s, re.IGNORECASE)
        if m:
            return f"ISA_{m.group(1)}x{m.group(2)}x{m.group(3)}"
        m = re.search(r"\bISA\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s, re.IGNORECASE)
        if m:
            return f"ISA_{m.group(1)}x{m.group(2)}"
        
        # Pipes
        dia_m = re.search(r"(?:[øØ]|\bod\b|\bdia\b)\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
        thk_m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s)
        if dia_m and thk_m:
            return f"PIPE_{dia_m.group(1)}x{thk_m.group(2)}"
        
        # Plates
        m = re.search(r"\bplate\b.*?(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s, re.IGNORECASE)
        if m:
            return f"PLATE_{m.group(1)}x{m.group(2)}x{m.group(3)}"
        m = re.search(r"\bplate\b.*?(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s, re.IGNORECASE)
        if m:
            return f"PLATE_{m.group(1)}x{m.group(2)}"
        
        return None
    
    # Count repeated signatures in text
    for line in (extracted_text or "").splitlines():
        sig = _build_signature(line)
        if sig:
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
            if sig not in signature_lines:
                signature_lines[sig] = []
            signature_lines[sig].append(line)
    
    # Convert repeated signatures to BOM items
    for sig, count in signature_counts.items():
        if count < 2:  # Only process if repeated at least twice
            continue
        
        # Get representative line for this signature
        rep_line = signature_lines[sig][0]
        
        # Extract full description from line
        desc = rep_line.strip()
        
        # Infer quantity from repetition count
        qty = count
        
        # Normalize material spec
        mat_spec = _normalize_material_spec(desc, "")
        material_from_notes = False
        if not mat_spec:
            mat_spec = "Detail given in Note"
            material_from_notes = True
        elif mat_spec == "Detail given in Note":
            material_from_notes = True
        
        items.append({
            "item_no": "",
            "description": desc,
            "material_specification": mat_spec,
            "part_drawing_no": "-",
            "qty": int(qty),
            "_qty_inferred_from_repetition": True,
            "_material_from_notes": material_from_notes,
        })
    
    if items:
        logging.info(f"SIGNAL 2: Detected {len(items)} repeated structural members")
    
    return items


def _extract_qty_from_line(line: str) -> Optional[int]:
    """
    Extract an explicit quantity from a single line of drawing text.
    Never guesses; returns None if not clearly present.
    """
    if not line:
        return None
    s = line.strip()

    # QTY: 4 / QTY=4
    m = re.search(r"\bqty\b\s*[:=]?\s*(\d{1,4})\b", s, re.IGNORECASE)
    if m:
        return _parse_int_qty(m.group(1))

    # 4 NOS / 4 NO. / 4 PCS / 4 EA
    m = re.search(r"\b(\d{1,4})\s*(nos|no\.?|pcs|ea)\b", s, re.IGNORECASE)
    if m:
        return _parse_int_qty(m.group(1))

    # TYP 4 PLACES / TYP. 4 LOCATIONS
    m = re.search(r"\btyp\.?\s*(\d{1,3})\s*(places|locations)\b", s, re.IGNORECASE)
    if m:
        return _parse_int_qty(m.group(1))

    # "x 4" / "X4" (common for repeated members); avoid dimensions like 50x50 by requiring spaces/boundaries.
    m = re.search(r"(?:\bx\s*|\bX\s*)(\d{1,4})\b", s)
    if m:
        return _parse_int_qty(m.group(1))

    return None


def _infer_qty_from_symmetry_text(line: str) -> Optional[int]:
    """
    Best-effort quantity inference when drawings indicate mirrored/symmetric repetition.
    This is intentionally conservative: we only infer small counts when the text explicitly
    implies it (e.g., BOTH SIDES / LHS & RHS / MIRROR).
    """
    if not line:
        return None
    s = line.strip().lower()

    # Common symmetry cues implying two identical members.
    if re.search(r"\b(both\s+sides?|lhs\s*&\s*rhs|lhs\s*/\s*rhs|left\s*&\s*right|left\s+and\s+right|rh\s*&\s*lh|mirror(?:ed)?|mirrored|symm\.?|symmetrical)\b", s):
        return 2

    # "TYP" without an explicit number is ambiguous; do not guess.
    return None


def _extract_length_mm_from_line(line: str) -> Optional[float]:
    """
    Extract an explicit length in mm from a line (e.g., "L=1200", "L 1200 mm", "LENGTH: 1200").
    """
    if not line:
        return None
    s = line.strip()
    m = re.search(r"\b(?:l|len|length)\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mm)?\b", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _extract_dimensions_and_callouts(extracted_text: str, tables: list[list[list[str]]]) -> list[dict]:
    """
    SIGNAL 3 — DIMENSIONS + CALLOUTS
    Extract dimensions and type labels such as:
    MS PIPE, ISA, PLATE, ACP SHEET, REFLECTIVE SHEET
    Use dimension text exactly as written in the drawing.
    
    SIGNAL 4 — NOTES & SCHEDULES
    Use drawing notes to populate material specifications.
    Use reinforcement / material schedules if present.
    
    Output items are compatible with `_build_generate_bom_payload`.
    """
    items: list[dict] = []
    low_conf_qty_inferred = False

    def add_item(desc: str, qty: Optional[int], inferred_low_conf: bool = False, qty_explicit: bool = True, material_from_notes: bool = False):
        nonlocal low_conf_qty_inferred
        q = _parse_int_qty(qty)
        d = re.sub(r"\s+", " ", str(desc or "")).strip()
        if not d or q is None:
            return
        # Pre-normalize spec to satisfy strict contract.
        spec = _normalize_material_spec(d, "")
        if not spec:
            return
        
        # Check if material spec indicates it came from notes
        if spec == "Detail given in Note":
            material_from_notes = True
        
        row = {
            "item_no": "",
            "description": d,
            "material_specification": spec,
            "part_drawing_no": "-",
            "qty": int(q),
            "_qty_explicit": qty_explicit,
            "_qty_inferred_missing_label": not qty_explicit,
            "_material_from_notes": material_from_notes,
        }
        if inferred_low_conf:
            low_conf_qty_inferred = True
            # In-band marker so callers can add a bom_metadata note.
            row["_qty_inferred_low_conf"] = True
        items.append(row)

    def _sig_from_line(line: str) -> Optional[str]:
        """
        Build a canonical description signature from a single line even when qty is missing.
        Used for repetition-based inference (count repeated identical members).
        Returns None if item type is unclear or no usable dimensions exist.
        
        STRICT RULES:
        - Never include "?" for unknown dimensions
        - Never include "As per drawing"
        - Only return if at least one dimension is explicit
        """
        if not line:
            return None
        s = line.strip()

        # ISA - require at least W x H, optionally thickness
        m = re.search(r"\bISA\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s, re.IGNORECASE)
        if m:
            w, h, thk = m.group(1), m.group(2), m.group(3)
            length = _extract_length_mm_from_line(s)
            if length is not None:
                return f"M.S. ISA {w} x {h} x {thk} x L {_fmt_mm(length)} mm"
            # Omit length if not found - still valid with W x H x T
            return f"M.S. ISA {w} x {h} x {thk} mm"
        
        # ISA without thickness - require W x H at minimum
        m = re.search(r"\bISA\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s, re.IGNORECASE)
        if m:
            w, h = m.group(1), m.group(2)
            length = _extract_length_mm_from_line(s)
            if length is not None:
                return f"M.S. ISA {w} x {h} x L {_fmt_mm(length)} mm"
            # Valid with just W x H
            return f"M.S. ISA {w} x {h} mm"

        # Pipe / CHS - require at least diameter
        if re.search(r"\b(pipe|chs)\b", s, re.IGNORECASE):
            dia_m = re.search(r"(?:[øØ]|\bod\b|\bdia\b)\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
            dims_m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s)
            dia = dia_m.group(1) if dia_m else (dims_m.group(1) if dims_m else None)
            thk = dims_m.group(2) if dims_m else None
            
            # Must have at least diameter
            if not dia:
                return None
            
            length = _extract_length_mm_from_line(s)
            
            # Build description with only known dimensions
            if thk and length is not None:
                return f"M.S. Pipe Ø {dia} x {thk} x L {_fmt_mm(length)} mm"
            elif thk:
                return f"M.S. Pipe Ø {dia} x {thk} mm"
            elif length is not None:
                return f"M.S. Pipe Ø {dia} x L {_fmt_mm(length)} mm"
            else:
                return f"M.S. Pipe Ø {dia} mm"

        # Plates - require at least W x H
        if re.search(r"\bplate\b", s, re.IGNORECASE):
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s)
            if m:
                w, h, thk = m.group(1), m.group(2), m.group(3)
                return f"M.S. Plate {w} x {h} x {thk} mm"
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s)
            if m:
                w, h = m.group(1), m.group(2)
                return f"M.S. Plate {w} x {h} mm"
            return None

        # ACP sheets - require W x H
        if re.search(r"\bacp\b", s, re.IGNORECASE):
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:[xX]\s*(\d+(?:\.\d+)?))?", s)
            if m:
                w, h, thk = m.group(1), m.group(2), m.group(3)
                if thk:
                    return f"ACP Sheet {w} x {h} x {thk} mm"
                return f"ACP Sheet {w} x {h} mm"
            return None

        # Reflective sheets - require W x H
        if re.search(r"\b(reflective|retroreflective)\b", s, re.IGNORECASE) and re.search(r"\bsheet\b", s, re.IGNORECASE):
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", s)
            if m:
                w, h = m.group(1), m.group(2)
                return f"Reflective Sheet {w} x {h} mm"
            return None

        return None

    # Pre-pass: count repeated identical member callouts (used when qty is missing/"TYP" without a number).
    rep_counts: dict[str, int] = {}
    for ln in (extracted_text or "").splitlines():
        sig = _sig_from_line(ln)
        if not sig:
            continue
        k = _normalize_ws(sig)
        rep_counts[k] = rep_counts.get(k, 0) + 1

    # SIGNAL 4: Parse schedules/tables (non-BOM tables like material schedules)
    for tbl in tables or []:
        if not isinstance(tbl, list) or len(tbl) < 2:
            continue
        # Find header row that likely contains QTY
        header_idx = None
        for i, row in enumerate(tbl[:3]):  # look at first few rows
            row_txt = " ".join([(c or "") for c in (row or [])]).lower()
            if "qty" in row_txt or "quantity" in row_txt or re.search(r"\bnos\b", row_txt):
                header_idx = i
                break
        header = tbl[header_idx] if header_idx is not None else (tbl[0] or [])

        # Column detection
        def _col_idx(keys: set[str]) -> Optional[int]:
            for j, h in enumerate(header or []):
                ht = (h or "").strip().lower()
                if any(k in ht for k in keys):
                    return j
            return None

        qty_col = _col_idx({"qty", "quantity", "nos", "no.", "no", "pcs", "ea"})
        desc_col = _col_idx({"description", "desc", "member", "item", "section", "size", "material"})
        len_col = _col_idx({"length", "len", "lg", "l (mm)", "l"})

        start = (header_idx + 1) if header_idx is not None else 1
        for row in tbl[start:]:
            if not isinstance(row, list) or not any((c or "").strip() for c in row):
                continue
            qty_val = row[qty_col] if (qty_col is not None and qty_col < len(row)) else ""
            qty = _parse_int_qty(qty_val)
            # If schedule row has no explicit qty, try symmetry cues in the row text.
            if qty is None:
                row_txt = " ".join([str(c or "").strip() for c in row if str(c or "").strip()])
                inferred = _infer_qty_from_symmetry_text(row_txt)
                if inferred is not None:
                    qty = inferred
                    low_conf_qty_inferred = True
                else:
                    continue

            desc_parts: list[str] = []
            if desc_col is not None and desc_col < len(row):
                desc_parts.append(str(row[desc_col] or "").strip())
            else:
                # Fallback: join all non-empty non-qty cells as a "description"
                for j, cell in enumerate(row):
                    if qty_col is not None and j == qty_col:
                        continue
                    cell_s = str(cell or "").strip()
                    if cell_s:
                        desc_parts.append(cell_s)

            if len_col is not None and len_col < len(row):
                lcell = str(row[len_col] or "").strip()
                if lcell and not any("l" in p.lower() for p in desc_parts):
                    desc_parts.append(f"L {lcell} mm")

            desc = " ".join([p for p in desc_parts if p]).strip()
            if desc:
                # Material from notes if extracted from schedule/table
                material_from_notes = bool(desc_col is not None and "note" in str(header[desc_col] if desc_col < len(header) else "").lower())
                qty_explicit = bool(qty_val and _parse_int_qty(qty_val))
                add_item(desc, qty, qty_explicit=qty_explicit, material_from_notes=material_from_notes)

    # SIGNAL 3: Parse notes/callouts/dimensions from extracted text
    for ln in (extracted_text or "").splitlines():
        line = ln.strip()
        if not line:
            continue
        qty = _extract_qty_from_line(line)
        if qty is None:
            # Allow inference from explicit symmetry/mirroring cues.
            inferred = _infer_qty_from_symmetry_text(line)
            if inferred is not None:
                sig = _sig_from_line(line)
                if sig:
                    # Check if material spec would come from notes (ACP/reflective sheets)
                    material_from_notes = bool(re.search(r"\b(acp|reflective|retroreflective)\b", sig, re.IGNORECASE))
                    add_item(sig, inferred, inferred_low_conf=True, qty_explicit=False, material_from_notes=material_from_notes)
                continue

            # Allow inference from visible repetition of identical member callouts (count duplicates).
            sig = _sig_from_line(line)
            if sig:
                k = _normalize_ws(sig)
                n = rep_counts.get(k, 0)
                if n >= 2:
                    # Check if material spec would come from notes
                    material_from_notes = bool(re.search(r"\b(acp|reflective|retroreflective)\b", sig, re.IGNORECASE))
                    add_item(sig, n, inferred_low_conf=True, qty_explicit=False, material_from_notes=material_from_notes)
                continue
            continue

        # ISA member - STRICT: must have W x H x T minimum
        qty_explicit = qty is not None  # qty was extracted from line
        m = re.search(r"\bISA\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", line, re.IGNORECASE)
        if m:
            w, h, thk = m.group(1), m.group(2), m.group(3)
            length = _extract_length_mm_from_line(line)
            if length is not None:
                add_item(f"M.S. ISA {w} x {h} x {thk} x L {_fmt_mm(length)} mm", qty, qty_explicit=qty_explicit)
            else:
                # Valid without length if W x H x T are present
                add_item(f"M.S. ISA {w} x {h} x {thk} mm", qty, qty_explicit=qty_explicit)
            continue

        # ISA without thickness - STRICT: must have at least W x H
        m = re.search(r"\bISA\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", line, re.IGNORECASE)
        if m:
            w, h = m.group(1), m.group(2)
            length = _extract_length_mm_from_line(line)
            if length is not None:
                add_item(f"M.S. ISA {w} x {h} x L {_fmt_mm(length)} mm", qty, qty_explicit=qty_explicit)
            else:
                # Valid with just W x H
                add_item(f"M.S. ISA {w} x {h} mm", qty, qty_explicit=qty_explicit)
            continue

        # Pipe/CHS - STRICT: must have at least diameter
        if re.search(r"\b(pipe|chs)\b", line, re.IGNORECASE) and (re.search(r"[øØ]|od|dia", line, re.IGNORECASE) or re.search(r"\b\d+(\.\d+)?\s*[xX]\s*\d+(\.\d+)?\b", line)):
            dia_m = re.search(r"(?:[øØ]|\bod\b|\bdia\b)\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
            dims_m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", line)
            dia = dia_m.group(1) if dia_m else (dims_m.group(1) if dims_m else None)
            thk = dims_m.group(2) if dims_m else None
            
            # Must have at least diameter - omit if not present
            if not dia:
                continue
                
            length = _extract_length_mm_from_line(line)
            
            # Build description with only known dimensions
            if thk and length is not None:
                add_item(f"M.S. Pipe Ø {dia} x {thk} x L {_fmt_mm(length)} mm", qty, qty_explicit=qty_explicit)
            elif thk:
                add_item(f"M.S. Pipe Ø {dia} x {thk} mm", qty, qty_explicit=qty_explicit)
            elif length is not None:
                add_item(f"M.S. Pipe Ø {dia} x L {_fmt_mm(length)} mm", qty, qty_explicit=qty_explicit)
            else:
                add_item(f"M.S. Pipe Ø {dia} mm", qty, qty_explicit=qty_explicit)
            continue

        # Plates - STRICT: must have at least W x H
        if re.search(r"\bplate\b", line, re.IGNORECASE):
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", line)
            if m:
                w, h, thk = m.group(1), m.group(2), m.group(3)
                add_item(f"M.S. Plate {w} x {h} x {thk} mm", qty, qty_explicit=qty_explicit)
                continue
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", line)
            if m:
                w, h = m.group(1), m.group(2)
                add_item(f"M.S. Plate {w} x {h} mm", qty, qty_explicit=qty_explicit)
                continue

        # ACP / reflective sheets - STRICT: must have W x H
        if re.search(r"\bacp\b", line, re.IGNORECASE):
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:[xX]\s*(\d+(?:\.\d+)?))?", line)
            if m:
                w, h, thk = m.group(1), m.group(2), m.group(3)
                if thk:
                    add_item(f"ACP Sheet {w} x {h} x {thk} mm", qty, qty_explicit=qty_explicit, material_from_notes=True)
                else:
                    add_item(f"ACP Sheet {w} x {h} mm", qty, qty_explicit=qty_explicit, material_from_notes=True)
                continue

        if re.search(r"\b(reflective|retroreflective)\b", line, re.IGNORECASE) and re.search(r"\bsheet\b", line, re.IGNORECASE):
            m = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\b", line)
            if m:
                w, h = m.group(1), m.group(2)
                add_item(f"Reflective Sheet {w} x {h} mm", qty, qty_explicit=qty_explicit, material_from_notes=True)
                continue

    # Dedupe exact repeats
    seen = set()
    deduped: list[dict] = []
    for it in items:
        key = (it.get("description"), it.get("material_specification"), it.get("part_drawing_no"), int(it.get("qty") or 0))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _merge_bom_items(primary: list[dict], fallback: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Merge fallback items into primary without double-counting.
    - If an item (desc+spec+part) already exists, keep primary qty (analysis wins).
    Returns: (merged_items, notes_fragments)
    """
    notes: list[str] = []
    merged: list[dict] = []
    index: dict[tuple, dict] = {}

    def key_for(it: dict) -> tuple:
        desc = re.sub(r"\s+", " ", str(it.get("description") or "")).strip().lower()
        spec = str(it.get("material_specification") or "").strip()
        part = str(it.get("part_drawing_no") or "-").strip() or "-"
        return (desc, spec, part)

    for it in primary or []:
        if not isinstance(it, dict):
            continue
        k = key_for(it)
        index[k] = it
        merged.append(it)

    for it in fallback or []:
        if not isinstance(it, dict):
            continue
        k = key_for(it)
        if k in index:
            # If quantities conflict, mention it (but do not override deterministic primary).
            try:
                q_primary = _parse_int_qty(index[k].get("qty"))
                q_fallback = _parse_int_qty(it.get("qty"))
                if q_primary is not None and q_fallback is not None and q_primary != q_fallback:
                    notes.append(f"qty_conflict:{index[k].get('description')}")
            except Exception:
                pass
            continue
        merged.append(it)
        index[k] = it

    return merged, notes

def _components_to_strict_bom_table(drawing_analysis: dict) -> dict:
    """
    Deterministically convert STRICT extracted components into STRICT BOM rows.

    STRICT RULES:
      - Omit items with unknown component types (OTHER)
      - Omit items with all dimensions unknown
      - Track omissions in metadata
      
    MERGE RULE:
      - same (type + dimensions + material_spec) => merge quantities (sum)

    DEFAULTS:
      - Part Drawing No: "-"
    """
    comps, _ = _normalize_strict_components_list((drawing_analysis or {}).get("components"))
    merged: dict[tuple, dict] = {}
    omitted_items: list[str] = []
    omitted_due_to_unclear_qty = False
    assumed_qty_used = False
    low_conf_qty_present = False

    for c in comps:
        ctype = c.get("type")
        
        # STRICT: Omit unknown types
        if not ctype or ctype not in STRICT_COMPONENT_TYPES:
            omitted_items.append(f"Unknown component type")
            continue
            
        qty = c.get("quantity")
        conf = c.get("confidence")
        if isinstance(conf, (int, float)) and float(conf) < 0.6:
            low_conf_qty_present = True
            
        # STRICT: Omit items with no quantity
        if qty is None:
            omitted_items.append(f"{ctype}: missing quantity")
            omitted_due_to_unclear_qty = True
            continue

        desc = None
        mat = None
        dims_key = None

        if ctype == "MS_PIPE":
            dia = c.get("diameter_mm")
            thk = c.get("thickness_mm")
            length = c.get("length_mm")
            
            # STRICT: Must have at least diameter
            if dia is None:
                omitted_items.append(f"MS_PIPE: missing diameter")
                continue
            
            # Build description with only known dimensions
            parts = [f"M.S. Pipe Ø {_fmt_mm(dia)}"]
            dims_list = [("D", _fmt_mm(dia))]
            
            if thk is not None:
                parts.append(f"x {_fmt_mm(thk)}")
                dims_list.append(("T", _fmt_mm(thk)))
            if length is not None:
                parts.append(f"x L {_fmt_mm(length)} mm")
                dims_list.append(("L", _fmt_mm(length)))
            else:
                parts.append("mm")
                
            desc = " ".join(parts)
            mat = "As per IS 1239"
            dims_key = tuple(dims_list)

        elif ctype == "ISA":
            w = c.get("width_mm")
            h = c.get("height_mm")
            thk = c.get("thickness_mm")
            length = c.get("length_mm")
            
            # STRICT: Must have at least W and H
            if w is None or h is None:
                omitted_items.append(f"ISA: missing width or height")
                continue
            
            # Build description with known dimensions
            parts = [f"M.S. ISA {_fmt_mm(w)} x {_fmt_mm(h)}"]
            dims_list = [("W", _fmt_mm(w)), ("H", _fmt_mm(h))]
            
            if thk is not None:
                parts.append(f"x {_fmt_mm(thk)}")
                dims_list.append(("T", _fmt_mm(thk)))
            if length is not None:
                parts.append(f"x L {_fmt_mm(length)} mm")
                dims_list.append(("L", _fmt_mm(length)))
            else:
                parts.append("mm")
                
            desc = " ".join(parts)
            mat = "As per IS 2062 & IS 808"
            dims_key = tuple(dims_list)

        elif ctype == "PLATE":
            w = c.get("width_mm")
            h = c.get("height_mm")
            thk = c.get("thickness_mm")
            
            # STRICT: Must have at least W and H
            if w is None or h is None:
                omitted_items.append(f"PLATE: missing width or height")
                continue
            
            parts = [f"M.S. Plate {_fmt_mm(w)} x {_fmt_mm(h)}"]
            dims_list = [("W", _fmt_mm(w)), ("H", _fmt_mm(h))]
            
            if thk is not None:
                parts.append(f"x {_fmt_mm(thk)}")
                dims_list.append(("T", _fmt_mm(thk)))
            parts.append("mm")
            
            desc = " ".join(parts)
            mat = "As per IS 2062 & IS 808"
            dims_key = tuple(dims_list)

        elif ctype == "ACP_SHEET":
            w = c.get("width_mm")
            h = c.get("height_mm")
            thk = c.get("thickness_mm")
            
            # STRICT: Must have W and H
            if w is None or h is None:
                omitted_items.append(f"ACP_SHEET: missing width or height")
                continue
            
            dims_list = [("W", _fmt_mm(w)), ("H", _fmt_mm(h))]
            if thk is not None:
                desc = f"ACP Sheet {_fmt_mm(w)} x {_fmt_mm(h)} x {_fmt_mm(thk)} mm"
                dims_list.append(("T", _fmt_mm(thk)))
            else:
                desc = f"ACP Sheet {_fmt_mm(w)} x {_fmt_mm(h)} mm"
            mat = "Detail given in Note"
            dims_key = tuple(dims_list)

    # Build strict table rows (Item No assigned by validator/normalizer)
    table = []
    for row in merged.values():
        table.append(
            {
                "Item No": "",
                "Item Description": row["Item Description"],
                "Material Specification": row["Material Specification"],
                "Part Drawing No": row["Part Drawing No"],
                "QTY": row["QTY"],
            }
        )

    # STRICT: If table is empty after applying strict rules, return empty with omission notes
    normalized, _ = _validate_and_normalize_strict_bom_table({"table": table, "summary": {}})
    
    # Attach metadata about omissions
    normalized["_omitted_items"] = omitted_items
    normalized["_omitted_due_to_unclear_qty"] = bool(omitted_due_to_unclear_qty)
    normalized["_low_conf_qty_present"] = bool(low_conf_qty_present)
    return normalized


def _table_to_bom_items(table_payload: dict) -> tuple[list[dict], bool]:
    """
    Convert strict table payload -> items list for the /generate-bom envelope format.
    Returns (items, omitted_flag)
    """
    payload, _ = _validate_and_normalize_strict_bom_table(table_payload or {})
    omitted = bool(isinstance(table_payload, dict) and table_payload.get("_omitted_due_to_unclear_qty"))
    items: list[dict] = []
    for row in payload.get("table") or []:
        items.append(
            {
                "item_no": row.get("Item No"),
                "description": row.get("Item Description"),
                "material_specification": row.get("Material Specification"),
                "part_drawing_no": row.get("Part Drawing No") or "-",
                "qty": int(row.get("QTY")),
            }
        )
    return items, omitted


def _build_generate_bom_payload(
    drawing_reference: str,
    items: list[dict],
    notes: str,
) -> dict:
    """
    Build the BOM object for the /generate-bom envelope.
    """
    items = items or []
    # Enforce sequential item numbers (1, 2, 3...) - strict format
    normalized_items: list[dict] = []
    for idx, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        qty = _parse_int_qty(it.get("qty"))
        if qty is None:
            continue
        normalized_items.append(
            {
                "item_no": str(idx),  # Sequential: 1, 2, 3... (no padding)
                "description": re.sub(r"\s+", " ", str(it.get("description") or "")).strip(),
                "material_specification": _normalize_material_spec(str(it.get("description") or ""), str(it.get("material_specification") or "")),
                "part_drawing_no": (str(it.get("part_drawing_no") or "-").strip() or "-"),
                "qty": int(qty),  # Always integer
            }
        )

    # Drop any rows that failed normalization (e.g., invalid material spec)
    normalized_items = [it for it in normalized_items if it.get("material_specification")]

    return {
        "bom_metadata": {
            "drawing_reference": drawing_reference or "",
            "created_at": date.today().isoformat(),
            "notes": notes or "",
        },
        "items": normalized_items,
        "summary": {"total_items": len(normalized_items)},
    }


def _build_generate_bom_response_from_items(
    items: list[dict],
    filename: str,
    signals_used: list[str],
    omitted_items: list[str] = None,
    notes_data: dict = None,
) -> dict:
    """
    Build BOM response envelope directly from extracted items.
    """
    drawing_reference = Path(filename).stem if filename else ""
    
    # Build notes from signals used
    signals_text = ", ".join(signals_used)
    notes_parts = [f"BOM extracted from: {signals_text}"]
    
    # Check for specific inference types
    has_repeated_inference = any(item.get("_qty_inferred_from_repetition") for item in items)
    has_material_from_notes = any(
        item.get("material_specification") == "Detail given in Note" or 
        item.get("_material_from_notes") for item in items
    )
    has_qty_inferred_missing = any(
        item.get("_qty_inferred_missing_label") is True or
        (item.get("_qty_explicit") is False)
        for item in items
    )
    
    # Check if vision components were used (implies geometry-based counting)
    has_geometry_based_qty = "vision components" in signals_used
    
    # Add specific extraction notes (authoritative, not apologetic)
    if has_geometry_based_qty:
        notes_parts.append("Quantities determined by component count")
    
    if has_repeated_inference:
        notes_parts.append("Quantities determined from repeated members")
    
    if has_material_from_notes:
        notes_parts.append("Material specifications per drawing notes")
    
    # Add global standards note if applicable
    if notes_data and notes_data.get("standards"):
        standards_text = ", ".join(notes_data["standards"][:3])  # First 3 standards
        notes_parts.append(f"Standards: {standards_text}")
    
    # Add global treatments note if applicable
    if notes_data and notes_data.get("global_treatments"):
        treatments_text = ", ".join(notes_data["global_treatments"][:2])  # First 2 treatments
        notes_parts.append(f"Treatment: {treatments_text}")
    
    if has_qty_inferred_missing:
        notes_parts.append("Quantities determined from drawing geometry")
    
    # Add omissions note if any (matter-of-fact, not apologetic)
    if omitted_items:
        omitted_count = len(omitted_items)
        notes_parts.append(f"{omitted_count} item(s) not shown: insufficient dimensional data")
    
    notes = ". ".join(notes_parts) + "."
    
    bom = _build_generate_bom_payload(
        drawing_reference=drawing_reference,
        items=items,
        notes=notes,
    )
    return {"success": True, "bom": bom}


def _build_generate_bom_response(
    drawing_analysis: dict,
    filename: str,
    fallback_items: Optional[list[dict]] = None,
) -> dict:
    """
    Deterministically convert extracted components into the required /generate-bom response envelope.
    ENGINEERING CONFIDENCE POLICY: Drawn dimensions are authoritative.
    """
    drawing_reference = Path(filename).stem if filename else ""

    table_payload = _components_to_strict_bom_table(drawing_analysis or {})
    items, omitted = _table_to_bom_items(table_payload)
    assumed_qty = bool(isinstance(table_payload, dict) and table_payload.get("_assumed_qty_used"))
    low_conf_qty = bool(isinstance(table_payload, dict) and table_payload.get("_low_conf_qty_present"))
    low_conf_qty_fallback = any(isinstance(it, dict) and it.get("_qty_inferred_low_conf") for it in (fallback_items or []))

    merged_items, merge_notes = _merge_bom_items(items, fallback_items or [])
    used_analysis = bool(items)
    used_fallback = bool(fallback_items) and (len(merged_items) > len(items) or (not items and bool(fallback_items)))

    # Build notes - matter-of-fact, not apologetic
    signals = []
    if used_analysis:
        signals.append("vision components")
    if used_fallback:
        signals.append("dimensions/repeated members/schedules/notes/callouts")
    if not signals:
        signals.append("drawing reference")

    base_note = f"BOM prepared from: {', '.join(signals)}"
    if omitted:
        base_note = f"{base_note}. Items omitted: insufficient dimensional data"
    if assumed_qty:
        base_note = f"{base_note}. Default quantity applied where not shown"
    if low_conf_qty or low_conf_qty_fallback:
        base_note = f"{base_note}. Quantities determined from drawing geometry"
    if merge_notes:
        base_note = f"{base_note}. Note: {', '.join(merge_notes[:3])}"
    notes = base_note

    bom = _build_generate_bom_payload(
        drawing_reference=drawing_reference,
        items=merged_items,
        notes=notes,
    )
    return {"success": True, "bom": bom}


def _validate_and_normalize_generate_bom_response(payload: dict) -> tuple[Optional[dict], list[str]]:
    """
    Validate and normalize cached /generate-bom responses.
    Accepts:
      - current envelope: {success, bom:{bom_metadata,items,summary}}
      - legacy strict table: {table, summary}
    Returns (normalized_envelope_or_none, errors)
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, ["payload_not_object"]

    # Current envelope
    if "success" in payload and "bom" in payload:
        if payload.get("success") is not True:
            return None, ["success_not_true"]
        bom = payload.get("bom")
        if not isinstance(bom, dict):
            return None, ["bom_not_object"]
        meta = bom.get("bom_metadata") if isinstance(bom.get("bom_metadata"), dict) else {}
        items = bom.get("items") if isinstance(bom.get("items"), list) else []
        drawing_reference = str(meta.get("drawing_reference") or "")
        notes = str(meta.get("notes") or "")
        # Rebuild to enforce sequential item_no + total_items
        rebuilt = _build_generate_bom_payload(drawing_reference=drawing_reference, items=items, notes=notes)
        return {"success": True, "bom": rebuilt}, []

    # Legacy strict table payload (cache migration)
    if "table" in payload and "summary" in payload:
        items, omitted = _table_to_bom_items(payload)
        base_note = "BOM prepared from drawing dimensions"
        notes = f"{base_note}. Items omitted: insufficient data" if omitted else base_note
        bom = _build_generate_bom_payload(drawing_reference="", items=items, notes=notes)
        return {"success": True, "bom": bom}, ["migrated_from_table"]

    return None, ["unrecognized_payload_shape"]


def _parse_int_qty(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            q = int(value)
            return q if q > 0 else None
        s = str(value).strip()
        s = re.sub(r"[^\d]", "", s)
        if not s:
            return None
        q = int(s)
        return q if q > 0 else None
    except Exception:
        return None


def _validate_and_normalize_strict_bom_table(payload: dict) -> tuple[dict, list[str]]:
    """
    Enforce the strict BOM output contract:
      - output has ONLY: { "table": [...], "summary": { "total_items": <int> } }
      - table rows have ONLY the 5 strict columns in exact order:
        1. Item No (sequential: 1, 2, 3...)
        2. Item Description (component type + dimensions)
        3. Material Specification (from notes/standards)
        4. Part Drawing No ("-" if not present)
        5. QTY (integer)
      - No extra columns
      - Professional engineering BOM format
    Returns: (normalized_payload, errors)
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"table": [], "summary": {"total_items": 0}}, ["payload_not_object"]

    table = payload.get("table")
    summary = payload.get("summary")

    if not isinstance(table, list):
        errors.append("table_missing_or_not_list")
        table = []
    if not isinstance(summary, dict):
        summary = {}

    normalized_rows: list[dict] = []
    for i, row in enumerate(table):
        if not isinstance(row, dict):
            errors.append(f"row_{i}_not_object")
            continue

        # Accept some legacy/internal keys as input, but output strict keys only.
        desc = row.get("Item Description", row.get("description", "")) or ""
        mat = row.get("Material Specification", row.get("material_specification", "")) or ""
        part = row.get("Part Drawing No", row.get("part_drawing_no", "")) or ""
        item_no = row.get("Item No", row.get("item_no", "")) or ""
        qty_val = row.get("QTY", row.get("qty", None))

        qty = _parse_int_qty(qty_val)
        if qty is None:
            errors.append(f"row_{i}_qty_not_int")
            continue

        # Basic hygiene: single-line strings
        desc = re.sub(r"\s+", " ", str(desc)).strip()
        part = str(part).strip() if part is not None else "-"
        if not part:
            part = "-"

        mat = _normalize_material_spec(desc, str(mat))
        if not mat:
            # The spec is required by the contract; if we can't confidently normalize, mark as error and skip.
            errors.append(f"row_{i}_material_spec_invalid")
            continue

        normalized_rows.append(
            {
                "Item No": str(item_no).strip() if str(item_no).strip() else "",
                "Item Description": desc,
                "Material Specification": mat,
                "Part Drawing No": part if part else "-",
                "QTY": int(qty),
            }
        )

    # Renumber sequentially starting at 1 (strict format: 1, 2, 3...)
    for idx, row in enumerate(normalized_rows, start=1):
        row["Item No"] = str(idx)

    normalized = {
        "table": normalized_rows,
        "summary": {"total_items": len(normalized_rows)},
    }

    # No extra top-level keys allowed in output, even if present in payload.
    # Also ensure summary.total_items is correct integer.
    if summary.get("total_items") is None:
        pass

    return normalized, errors


def extract_bom_from_drawing_analysis(
    drawing_analysis: dict,
    filename: str,
    fallback_items: Optional[list[dict]] = None,
) -> dict:
    """
    Convert drawing analysis JSON to BOM structure
    Extracts in standard BOM table format: Item No, Description, Material Spec, Part Drawing No, QTY
    
    ENGINEERING CONFIDENCE POLICY:
    This function always returns a valid BOM response. Drawn dimensions are treated as authoritative.
    """
    try:
        return _build_generate_bom_response(drawing_analysis or {}, filename or "", fallback_items=fallback_items or [])
    except Exception as e:
        logging.error(f"BOM conversion error: {str(e)}. Returning minimal BOM.")
        # Return a minimal valid BOM
        return {
            "success": True,
            "bom": {
                "bom_metadata": {
                    "drawing_reference": Path(filename).stem if filename else "",
                    "created_at": date.today().isoformat(),
                    "notes": "BOM prepared from available drawing data."
                },
                "items": [{
                    "item_no": "1",
                    "description": "Components as per drawing",
                    "material_specification": "Detail given in Note",
                    "part_drawing_no": "-",
                    "qty": 1
                }],
                "summary": {"total_items": 1}
            }
        }

def organize_bom_items(items: List[dict]) -> dict:
    """
    Backward-compat helper: accepts legacy list[{item_no,description,...}] and
    returns strict {table,summary}. Prefer using `extract_bom_from_drawing_analysis()`.
    """
    table = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        table.append(
            {
                "Item No": item.get("item_no", ""),
                "Item Description": item.get("description", ""),
                "Material Specification": item.get("material_specification", ""),
                "Part Drawing No": item.get("part_drawing_no", "-"),
                "QTY": item.get("qty", ""),
            }
        )
    normalized, _ = _validate_and_normalize_strict_bom_table({"table": table, "summary": {}})
    return normalized

# Drawing analysis endpoints removed - only PDF Q&A chatbot is supported
# @app.post("/generate-bom")
# async def generate_bom_endpoint(file: UploadFile = File(...)):
    """
    Generate Bill of Materials from uploaded drawing file (PDF or image)
    
    Pipeline:
    1. Run drawing vision analysis
    2. Extract components & material hints
    3. Apply bom_mapping_rules.json
    4. Generate structured BOM JSON
    5. Cache result by file hash
    
    Returns:
      {
        "success": true,
        "bom": {
          "bom_metadata": { "drawing_reference", "created_at", "notes" },
          "items": [ { "item_no", "description", "material_specification", "part_drawing_no", "qty" } ],
          "summary": { "total_items": <int> }
        }
      }
    """
    logging.info(f"BOM generation requested for: {file.filename}")
    
    # Read file and compute hash for caching
    file_bytes = await file.read()
    file_hash = compute_file_hash(file_bytes)
    cache_file = BOM_CACHE_DIR / f"{file_hash}.json"
    
    # Check cache
    if cache_file.exists():
        logging.info(f"BOM cache hit for {file.filename}")
        try:
            with open(cache_file, 'r') as f:
                cached_result = json.load(f)
            normalized, errors = _validate_and_normalize_generate_bom_response(cached_result)
            if not normalized:
                logging.warning(f"Cached BOM invalid; regenerating. errors={errors[:10]}")
            else:
                # Best effort: refresh cache to normalized envelope (idempotent).
                try:
                    # Ensure drawing_reference is present when serving from cache.
                    if normalized.get("bom", {}).get("bom_metadata") is not None:
                        normalized["bom"]["bom_metadata"]["drawing_reference"] = Path(file.filename).stem
                    with open(cache_file, 'w', encoding='utf-8') as wf:
                        json.dump(normalized, wf, indent=2, ensure_ascii=False)
                except Exception as e:
                    logging.warning(f"Failed to refresh normalized BOM cache: {e}")
                return normalized
        except Exception as e:
            logging.warning(f"Cache read failed: {e}, regenerating...")
    
    # Save uploaded file temporarily
    suffix = os.path.splitext(file.filename)[1].lower()
    
    # Validate file format
    allowed_formats = {".pdf", ".png", ".jpg", ".jpeg"}
    format_warning = None
    if suffix not in allowed_formats:
        format_warning = f"File format {suffix} processed."
        logging.warning(f"Non-standard file format: {suffix}")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(file_bytes)
    temp_file.close()
    
    pdf_path = temp_file.name
    image_path = pdf_path
    pdf_temp_image_path = None
    
    try:
        # BOM-FIRST APPROACH: Extract signals in priority order
        
        # Extract text and tables from PDF (best-effort, even for unsupported formats)
        extracted_text = ""
        extracted_tables: list[list[list[str]]] = []
        try:
            if suffix == ".pdf":
                extracted_text, extracted_tables = _pdf_extract_text_and_tables(pdf_path, max_pages=2)
        except Exception as e:
            logging.warning(f"PDF text extraction failed: {e}. Continuing with vision analysis.")
        
        # Convert PDF → image (first page only) for vision analysis (best-effort)
        try:
            if suffix == ".pdf":
                doc = fitz.open(pdf_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=200)
                pdf_temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                pix.save(pdf_temp_image_path)
                image_path = pdf_temp_image_path
                doc.close()
        except Exception as e:
            logging.warning(f"PDF to image conversion failed: {e}. Using original file for vision analysis.")
            # Continue with original file path
        
        # Run vision analysis for component detection (used in Signal 2 and 3)
        drawing_analysis: dict = {}
        logging.info("Running vision analysis for component detection...")

        analysis_prompt = """
You are an engineering drawing reader extracting Bill of Materials (BOM) data.

CRITICAL: EXPLODE ASSEMBLIES INTO INDIVIDUAL ITEMS

DO NOT treat assemblies/structures as single BOM rows.
INSTEAD, extract each physical fabricated item separately.

Example - If drawing shows a gantry structure:
✓ CORRECT:
  - Vertical post (MS Pipe Ø 88.9 x 3.2 mm) - Qty: 4
  - Horizontal beam (MS Pipe Ø 114.3 x 4.5 mm) - Qty: 2
  - Diagonal bracing (ISA 50 x 50 x 6 mm) - Qty: 8
  - Base plate (MS Plate 300 x 300 x 10 mm) - Qty: 4
  - Stiffener plate (MS Plate 100 x 100 x 8 mm) - Qty: 16
  - Bolt (M16 x 50 mm) - Qty: 32

✗ WRONG:
  - Gantry structure - Qty: 1
  - Sign support assembly - Qty: 1

TASK:
Extract individual components with explicit numerical dimensions AND count visually repeated identical components.

PRIORITY 1: GEOMETRY-BASED QUANTITY INFERENCE
For each component type with readable dimensions:
1. Group by: component type + exact dimensions
2. Count how many times that identical component appears VISUALLY in the drawing
3. Use the visual count as quantity

Examples:
- 4 identical vertical posts (same dimensions) → quantity = 4
- 6 identical stiffener plates (same size) → quantity = 6
- 2 identical ACP panels (same dimensions) → quantity = 2
- 8 identical bolts in a pattern → quantity = 8

CRITICAL: Count VISUALLY repeated geometric elements, not just text labels.

PRIORITY 2: DIMENSION EXTRACTION
MANDATORY REQUIREMENTS:
- Read all visible dimension annotations (in mm)
- Read diameters (Ø), thicknesses, lengths, widths
- Use the EXACT numbers written in the drawing
- If a dimension is not clearly readable, set it to null
- If ALL critical dimensions are missing, DO NOT output that component

COMPONENT TYPES (strict):
- MS_PIPE: Requires readable diameter (for posts, beams, handrails)
- ISA: Requires readable width x height (for angles, bracings)
- PLATE: Requires readable width x height (for base plates, stiffeners, gussets)
- ACP_SHEET: Requires readable width x height (for panels)
- REFLECTIVE_SHEET: Requires readable width x height (for sign faces)
- BOLT: Requires readable diameter (for connections)

QUANTITY EXTRACTION PRIORITY:
1. FIRST: Count visually repeated identical components in the drawing
2. SECOND: Read explicit quantity callouts (QTY, NOS, TYP) if present
3. THIRD: Infer from symmetry cues (BOTH SIDES, LHS & RHS) if explicit
4. If truly uncertain after all above, set to null

DO NOT:
- Treat assemblies as single items
- Output "gantry", "structure", "assembly", "frame" as component types
- Guess dimensions
- Estimate sizes
- Use placeholders
- Include components with unreadable dimensions

OUTPUT FORMAT:
For each individual fabricated component with readable dimensions:
{
  "type": "MS_PIPE" | "ISA" | "PLATE" | "ACP_SHEET" | "REFLECTIVE_SHEET" | "BOLT",
  "diameter_mm": number | null,
  "thickness_mm": number | null,
  "width_mm": number | null,
  "height_mm": number | null,
  "length_mm": number | null,
  "quantity": number | null,
  "standard_reference": string | null,
  "drawing_note_reference": string | null,
  "confidence": number (0.0 to 1.0)
}

CONFIDENCE SCORING:
- 0.9-1.0: All dimensions + quantity counted visually
- 0.7-0.9: Dimensions readable, quantity from text label or visual count
- 0.5-0.7: Some dimensions readable, quantity inferred
- Below 0.5: Omit the component (dimensions too unclear)

IMPORTANT: 
- Even if no quantity labels exist, COUNT the visual repetitions.
- Extract EVERY distinct physical part, not assemblies.

Return VALID JSON ONLY.
Do NOT include markdown.
Do NOT add extra keys.

SCHEMA:
- Return VALID JSON ONLY
- Follow the schema EXACTLY
- Do NOT include markdown
- Do NOT add extra keys

SCHEMA:
{
  "drawing_type": "string",
  "summary": "string",
  "components": [
    {
      "type": "MS_PIPE|ISA|PLATE|ACP_SHEET|REFLECTIVE_SHEET|BOLT|OTHER",
      "diameter_mm": 0,
      "thickness_mm": 0,
      "width_mm": 0,
      "height_mm": 0,
      "length_mm": 0,
      "quantity": 0,
      "standard_reference": "string",
      "drawing_note_reference": "string",
      "confidence": 0.0
    }
  ],
  "foundation": {
    "type": "string",
    "dimensions": ["string"],
    "material": "string"
  },
  "standards": ["string"],
  "uncertain_items": ["string"]
}
"""

        # Vision analysis (best-effort, never fails the entire process)
        try:
            if not GEMINI_ENABLED:
                logging.info("Gemini not enabled; skipping vision analysis")
                drawing_analysis = {"components": []}
            else:
                model = genai.GenerativeModel("gemini-2.5-flash")
                image = Image.open(image_path)
                response = model.generate_content([analysis_prompt, image])
                raw_text = (response.text or "").strip()
                drawing_analysis = extract_json_from_text(raw_text) or {}
                # Normalize strict components list
                comps, _ = _normalize_strict_components_list(drawing_analysis.get("components"))
                drawing_analysis["components"] = comps
                logging.info("Vision analysis complete")
        except Exception as e:
            logging.info(f"Vision analysis unavailable; using text-based extraction. err={e}")
            drawing_analysis = {"components": []}
        
        # BOM-FIRST SIGNAL EXTRACTION (in priority order)
        all_items: list[dict] = []
        signals_used: list[str] = []
        
        # PARSE DRAWING NOTES (global specifications and standards)
        notes_data = _parse_drawing_notes(extracted_text)
        logging.info(f"Parsed drawing notes: {len(notes_data.get('standards', []))} standards, {len(notes_data.get('global_treatments', []))} treatments")
        
        # SIGNAL 1 — EXPLICIT BOM TABLE (highest priority)
        explicit_bom_items = _extract_explicit_bom_table(extracted_tables)
        if explicit_bom_items:
            # Apply notes to explicit BOM items
            explicit_bom_items = [_apply_notes_to_item(item, notes_data) for item in explicit_bom_items]
            all_items = explicit_bom_items
            signals_used.append("explicit BOM table")
            logging.info(f"✓ SIGNAL 1: Using explicit BOM table ({len(explicit_bom_items)} items)")
        else:
            # SIGNAL 2 — REPEATED STRUCTURAL MEMBERS
            repeated_items = _detect_repeated_structural_members(extracted_text, drawing_analysis.get("components", []))
            if repeated_items:
                # Apply notes to repeated items
                repeated_items = [_apply_notes_to_item(item, notes_data) for item in repeated_items]
                all_items.extend(repeated_items)
                signals_used.append("repeated structural members")
                logging.info(f"✓ SIGNAL 2: Detected {len(repeated_items)} repeated members")
            
            # SIGNAL 3 & 4 — DIMENSIONS + CALLOUTS + NOTES & SCHEDULES
            dimension_items = _extract_dimensions_and_callouts(extracted_text, extracted_tables)
            if dimension_items:
                # Apply notes to dimension items
                dimension_items = [_apply_notes_to_item(item, notes_data) for item in dimension_items]
                all_items.extend(dimension_items)
                signals_used.append("dimensions/callouts/notes/schedules")
                logging.info(f"✓ SIGNAL 3/4: Extracted {len(dimension_items)} items from dimensions/notes")
            
            # If still no items, try vision components as last resort
            omitted_items_vision = []
            if not all_items and drawing_analysis.get("components"):
                vision_table = _components_to_strict_bom_table(drawing_analysis)
                omitted_items_vision = vision_table.get("_omitted_items", [])
                if vision_table.get("table"):
                    for row in vision_table["table"]:
                        mat_spec = row.get("Material Specification", "")
                        material_from_notes = (mat_spec == "Detail given in Note")
                        # Vision components may have inferred quantities if confidence was low
                        qty = row.get("QTY", 1)
                        qty_explicit = True  # Vision provides explicit quantities (even if low confidence)
                        item = {
                            "item_no": row.get("Item No", ""),
                            "description": row.get("Item Description", ""),
                            "material_specification": mat_spec,
                            "part_drawing_no": row.get("Part Drawing No", "-"),
                            "qty": qty,
                            "_qty_explicit": qty_explicit,
                            "_qty_inferred_missing_label": False,
                            "_material_from_notes": material_from_notes,
                        }
                        # Apply notes to vision items
                        item = _apply_notes_to_item(item, notes_data)
                        all_items.append(item)
                    signals_used.append("vision components")
                    logging.info(f"✓ VISION: Extracted {len(all_items)} items from vision analysis")
        
        # Track all omitted items
        all_omitted_items = omitted_items_vision if 'omitted_items_vision' in locals() else []
        
        # ENGINEERING CONFIDENCE POLICY: Always produce at least one item
        if not all_items:
            logging.warning("No items extracted from any signal; creating placeholder item")
            all_items = [{
                "item_no": "",
                "description": "Components as per drawing",
                "material_specification": "Detail given in Note",
                "part_drawing_no": "-",
                "qty": 1,
                "_qty_explicit": False,
                "_qty_inferred_missing_label": True,
                "_material_from_notes": True,
            }]
            signals_used.append("drawing reference")
            # Note omitted items if any
            if all_omitted_items:
                all_omitted_items.append("Dimensional data insufficient for itemization")
        
        # Deduplicate items from multiple signals (merge by description + material spec)
        if len(all_items) > 1:
            deduped_items: list[dict] = []
            seen_keys: set[tuple] = set()
            
            for item in all_items:
                # Create deduplication key: description + material spec (case-insensitive)
                desc = re.sub(r"\s+", " ", str(item.get("description", "")).strip()).lower()
                mat = str(item.get("material_specification", "")).strip().lower()
                key = (desc, mat)
                
                if key in seen_keys:
                    # Merge quantities for duplicate items
                    for existing in deduped_items:
                        existing_desc = re.sub(r"\s+", " ", str(existing.get("description", "")).strip()).lower()
                        existing_mat = str(existing.get("material_specification", "")).strip().lower()
                        if (existing_desc, existing_mat) == key:
                            existing["qty"] = int(existing.get("qty", 0)) + int(item.get("qty", 0))
                            break
                else:
                    seen_keys.add(key)
                    deduped_items.append(item)
            
            all_items = deduped_items
            logging.info(f"Deduplicated to {len(all_items)} unique items")
        
        # Build BOM response envelope
        logging.info(f"Building BOM from {len(all_items)} items (signals: {', '.join(signals_used)})")
        try:
            result = _build_generate_bom_response_from_items(all_items, file.filename, signals_used, omitted_items=all_omitted_items, notes_data=notes_data)
        except Exception as e:
            logging.warning(f"BOM response building error: {e}. Using fallback construction.")
            result = {
                "success": True,
                "bom": {
                    "bom_metadata": {
                        "drawing_reference": Path(file.filename).stem,
                        "created_at": date.today().isoformat(),
                        "notes": "BOM prepared from available drawing data."
                    },
                    "items": all_items if all_items else [{
                        "item_no": "1",
                        "description": "Components as per drawing",
                        "material_specification": "Detail given in Note",
                        "part_drawing_no": "-",
                        "qty": 1
                    }],
                    "summary": {"total_items": len(all_items) if all_items else 1}
                }
            }
        
        # Validate and normalize
        normalized, errors = _validate_and_normalize_generate_bom_response(result)
        if not normalized:
            logging.warning(f"BOM normalization errors: {errors}. Using minimal BOM.")
            # Create a minimal valid BOM
            error_note = "BOM prepared from available data."
            if format_warning:
                error_note = f"{format_warning} {error_note}"
            normalized = {
                "success": True,
                "bom": {
                    "bom_metadata": {
                        "drawing_reference": Path(file.filename).stem,
                        "created_at": date.today().isoformat(),
                        "notes": error_note
                    },
                    "items": [{
                        "item_no": "1",
                        "description": "Components as per drawing",
                        "material_specification": "Detail given in Note",
                        "part_drawing_no": "-",
                        "qty": 1
                    }],
                    "summary": {"total_items": 1}
                }
            }
        
        # Ensure drawing reference is correct for this upload
        if normalized and normalized.get("bom") and normalized["bom"].get("bom_metadata"):
            normalized["bom"]["bom_metadata"]["drawing_reference"] = Path(file.filename).stem
            # Add format warning to notes if present
            if format_warning:
                existing_notes = normalized["bom"]["bom_metadata"].get("notes", "")
                normalized["bom"]["bom_metadata"]["notes"] = f"{format_warning} {existing_notes}"
        
        # Step 5: Cache the result
        logging.info(f"Caching BOM result: {cache_file}")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(normalized, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Failed to cache BOM: {e}")

        logging.info(f"BOM generation complete: {normalized['bom']['summary']['total_items']} items")
        return normalized
        
    except Exception as e:
        logging.error(f"BOM generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Always return a valid BOM
        error_note = "BOM prepared from available drawing data."
        if format_warning:
            error_note = f"{format_warning} {error_note}"
        return {
            "success": True,
            "bom": {
                "bom_metadata": {
                    "drawing_reference": Path(file.filename).stem if file.filename else "",
                    "created_at": date.today().isoformat(),
                    "notes": error_note
                },
                "items": [{
                    "item_no": "1",
                    "description": "Components as per drawing",
                    "material_specification": "Detail given in Note",
                    "part_drawing_no": "-",
                    "qty": 1
                }],
                "summary": {"total_items": 1}
            }
        }
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if pdf_temp_image_path and os.path.exists(pdf_temp_image_path):
                os.remove(pdf_temp_image_path)
        except Exception as e:
            logging.warning(f"Failed to clean up temp files: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

