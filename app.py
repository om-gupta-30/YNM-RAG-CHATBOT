"""
FastAPI Backend for RAG PDF System

IMPORTANT: Run uvicorn from the project root directory:
    uvicorn app:app --reload
    OR
    python app.py

Do NOT run from frontend/ directory.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
import json
import os
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Optional
from PIL import Image
from difflib import SequenceMatcher
import re
import logging
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
    logging.warning("GEMINI_API_KEY not set. Gemini-dependent endpoints will be unavailable.")

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
        caption = (response.text or "").strip()
        
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


class GenerateChatTitleRequest(BaseModel):
    question: str


class GenerateChatTitleResponse(BaseModel):
    title: str


@app.get("/health")
async def health():
    gemini_key = os.getenv("GEMINI_API_KEY")
    return {
        "status": "ok",
        "env": {
            "GEMINI_API_KEY": "set" if gemini_key else "missing",
            "GEMINI_ENABLED": bool(gemini_key),
            "PORT": os.getenv("PORT", "not set"),
        },
    }


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


@app.post("/generate-chat-title", response_model=GenerateChatTitleResponse | ErrorResponse)
async def generate_chat_title(request: GenerateChatTitleRequest):
    """
    Generate a short chat title from the user's first question using Gemini.
    Keeps the API key server-side only; never expose it to the frontend.
    """
    if not GEMINI_ENABLED:
        return {"error": "Gemini is not configured (missing GEMINI_API_KEY)."}
    question = (request.question or "").strip()
    if not question:
        return {"error": "question is required."}
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = """Generate a concise chat title (3–6 words) summarizing the topic.
Rules:
- No punctuation
- No markdown
- No filler words like explain, describe
- Output ONLY the title text

User question:
"""
        response = model.generate_content(prompt + question)
        raw = (response.text or "").strip()
        title = re.sub(r"[^\w\s]", "", raw).strip() or "New Chat"
        return {"title": title[:80]}
    except Exception as e:
        logging.exception("generate_chat_title failed: %s", e)
        return {"error": "Failed to generate title."}


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
        _summary_re = re.compile(
            r"\b(summary|summarize|summarise|overview|what is this|about this document|about this pdf|main topics|brief)\b",
            re.IGNORECASE,
        )
        is_summary = bool(_summary_re.search(question))

        if is_summary:
            intro_pages = set(range(1, 10))
            chunks_data = []
            for cid, chunk_data in metadata.items():
                if chunk_data.get("page_number") in intro_pages:
                    chunk_text = chunk_data.get("text", "") or ""
                    if not chunk_text.strip():
                        continue
                    source_type = chunk_data.get("source", "text")
                    page_number = chunk_data.get("page_number")
                    image_path = None
                    if page_number:
                        candidate = f"images/page_{page_number}_img_1.png"
                        if os.path.exists(candidate):
                            image_path = candidate
                    chunks_data.append({
                        "chunk_id": cid,
                        "text": chunk_text,
                        "source": source_type,
                        "page_number": page_number,
                        "image_path": image_path,
                        "distance": 0.1,
                    })
            chunks_data = deduplicate_chunks(chunks_data)
            return chunks_data[:10], None

        try:
            result = genai.embed_content(model="models/gemini-embedding-001", content=question, output_dimensionality=768)
        except Exception as e:
            logging.error(f"Embedding API error: {e}")
            return None, "Embedding service temporarily unavailable. Please try again."
        query_embedding = np.array([result["embedding"]], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        k = 10
        distances, indices = index.search(query_embedding, k)

        chunks_data = []
        for pos, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            chunk_id = str(idx + 1)
            chunk_data = metadata.get(chunk_id, {})
            chunk_text = chunk_data.get("text", "") or ""
            if not chunk_text.strip():
                continue
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
        return chunks_data[:8], None

    # SECTION_QUERY or anything else: semantic search (same as GENERAL)
    try:
        result = genai.embed_content(model="models/gemini-embedding-001", content=question, output_dimensionality=768)
    except Exception as e:
        logging.error(f"Embedding API error: {e}")
        return None, "Embedding service temporarily unavailable. Please try again."
    query_embedding = np.array([result["embedding"]], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    k = 10
    distances, indices = index.search(query_embedding, k)
    chunks_data = []
    for pos, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        chunk_id = str(idx + 1)
        chunk_data = metadata.get(chunk_id, {})
        chunk_text = chunk_data.get("text", "") or ""
        if not chunk_text.strip():
            continue
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
    return chunks_data[:8], None


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
    
    # PDF-only validation: Check if question is relevant to the PDF
    if not chunks_data or len(chunks_data) == 0:
        return {
            "error": "Please ask me only PDF-related questions. I can only answer questions about the content in the uploaded document.",
            "intent": intent_info.get("intent"),
            "target_id": intent_info.get("target_id")
        }
    
    retrieved_chunks = []
    sources_list = []
    
    # IMPORTANT: In strict technical QA mode, do NOT generate or attach vision captions.
    # These are model-generated and are not verbatim document content.
    include_vision_caption = False
    
    # Semantic match score (only meaningful for semantic retrieval modes).
    semantic_match_score = None
    intent_name = intent_info.get("intent", "")
    if intent_name in ("GENERAL_QUERY", "SECTION_QUERY"):
        distances = [c.get("distance") for c in chunks_data if c.get("distance") is not None]
        if distances:
            best_d = min(distances)
            # For normalized vectors: cos_sim = 1 - d²/2
            semantic_match_score = max(0.0, 1.0 - (float(best_d) ** 2) / 2.0)
            
            # Off-topic rejection: with normalized vectors, on-topic d < 0.8, off-topic d > 0.9
            if best_d > 0.85:
                return {
                    "error": "Please ask me only PDF-related questions. I can only answer questions about the content in the uploaded document.",
                    "intent": intent_info.get("intent"),
                    "target_id": intent_info.get("target_id")
                }
    
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
    
    context_parts = []
    for chunk_info in chunks_data:
        page = chunk_info.get("page_number")
        text = chunk_info.get("text", "")
        if page:
            context_parts.append(f"[Page {page}] {text}")
        else:
            context_parts.append(text)
    context = "\n\n".join(context_parts)
    
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
You are a technical document QA assistant. You answer questions based on the provided context extracted from a PDF document.

RULES:
1. Use the provided context to answer the question. The context comes from a PDF document about: road signs (IRC:67-2022).
2. If the question is clearly unrelated to the document's subject matter (e.g., geography, politics, celebrities, cooking, sports, etc.):
   - Respond ONLY with: "Please ask me only PDF-related questions. I can only answer questions about the content in the uploaded document."
3. If the question IS related to the document's subject but the specific detail is not found in the provided context:
   - Answer with whatever relevant information IS available in the context.
   - If truly nothing relevant exists, say: "Not explicitly specified in the document."
4. If the question asks for a figure or table:
   - Focus on the referenced figure/table content only.
5. For summary or overview questions: synthesize the key points from the provided context.

ANSWER STYLE:
- Crisp and concise
- Bullet points when listing multiple items
- No filler or unnecessary repetition
- Stay grounded in the context provided

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
- Base your answer on the context provided.

Query intent:
{json.dumps(intent_info, ensure_ascii=False)}

Context:
{context}

Question:
{question}
"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    raw_text = (response.text or "").strip()
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

# Serve built React SPA (production): must be last so /health, /ask, etc. take precedence
UI_DIST = os.path.abspath("frontend/dist")
if os.path.isdir(UI_DIST):
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        base = UI_DIST
        path = os.path.normpath(os.path.join(base, full_path)) if full_path else base
        try:
            if os.path.commonpath([path, base]) != base:
                return Response(status_code=404)
        except ValueError:
            return Response(status_code=404)
        if os.path.isfile(path):
            return FileResponse(path)
        return FileResponse(os.path.join(base, "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

