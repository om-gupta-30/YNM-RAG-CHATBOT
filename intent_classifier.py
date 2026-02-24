from __future__ import annotations

import re
from typing import Dict, Optional

INTENT_FIGURE = "FIGURE_QUERY"
INTENT_TABLE = "TABLE_QUERY"
INTENT_PAGE = "PAGE_QUERY"
INTENT_SECTION = "SECTION_QUERY"
INTENT_GENERAL = "GENERAL_QUERY"
INTENT_COMPARISON = "COMPARISON_QUERY"


_FIGURE_RE = re.compile(
    r"\b(fig(?:ure)?|illustration)\b\.?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_FIGURE_WORD_RE = re.compile(r"\b(fig(?:ure)?|illustration)\b", re.IGNORECASE)

_TABLE_RE = re.compile(r"\b(table|tab)\b\.?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_TABLE_WORD_RE = re.compile(r"\b(table|tab)\b\.?", re.IGNORECASE)

_PAGE_RE = re.compile(
    r"\b(?:page|pg|p)\b\.?\s*(\d{1,4})\b",
    re.IGNORECASE,
)

_SECTION_WORD_RE = re.compile(
    r"\b(?:section|sec|chapter|ch|heading|clause|appendix|annex|para|paragraph)\b\.?",
    re.IGNORECASE,
)

# A bare number like 3.3 does NOT trigger an intent by itself, but if "fig" is present
# it should still be treated as a figure query (per requirements).
_DOT_NUMBER_RE = re.compile(r"\b\d+\.\d+\b")

_COMPARE_WORD_RE = re.compile(
    r"\b(?:compare|difference\s+between|vs|versus)\b",
    re.IGNORECASE,
)

_PAGE_NUM_RE = re.compile(r"\b(?:page|pg|p)\b\.?\s*(\d{1,4})\b", re.IGNORECASE)


def _extract_all_refs(question: str) -> Dict[str, list[str]]:
    q = (question or "").strip()
    figs = [m.group(2) for m in _FIGURE_RE.finditer(q)]
    # If 'fig' is present but formatting is odd, allow dot numbers as fig candidates.
    if _FIGURE_WORD_RE.search(q):
        figs += [m.group(0) for m in _DOT_NUMBER_RE.finditer(q)]

    tables = [m.group(2) for m in _TABLE_RE.finditer(q)]
    if _TABLE_WORD_RE.search(q):
        tables += [m.group(0) for m in _DOT_NUMBER_RE.finditer(q)]

    pages = [m.group(1) for m in _PAGE_NUM_RE.finditer(q)]
    return {
        "fig": list(dict.fromkeys([f for f in figs if f])),
        "table": list(dict.fromkeys([t for t in tables if t])),
        "page": list(dict.fromkeys([p for p in pages if p])),
    }


def _normalize_target(prefix: str, ref: str) -> str:
    ref = ref.strip()
    prefix = prefix.strip()
    return f"{prefix} {ref}"


def classify_query_intent(question: str) -> Dict[str, Optional[str]]:
    """
    Classify query intent with priority:
      FIGURE > TABLE > PAGE > SECTION > GENERAL

    Output schema:
      { "intent": <one_of>, "target_id": "Fig 3.3" | "Table 6.2" | "Page 27" | null }
    """
    q = (question or "").strip()

    # --- COMPARISON ---
    # Trigger only when there's a comparison keyword AND at least two explicit references.
    if _COMPARE_WORD_RE.search(q):
        refs = _extract_all_refs(q)
        total = len(refs["fig"]) + len(refs["table"]) + len(refs["page"])
        if total >= 2:
            # If multiple intents appear, comparison is more specific than general/section/page,
            # but must not override explicit FIGURE/TABLE priority unless it's a true compare.
            return {"intent": INTENT_COMPARISON, "target_id": None}

    # --- FIGURE ---
    m_fig = _FIGURE_RE.search(q)
    if m_fig:
        return {
            "intent": INTENT_FIGURE,
            "target_id": _normalize_target("Fig", m_fig.group(2)),
        }

    # If "fig/figure/illustration" is present anywhere, treat as FIGURE_QUERY even
    # without a numeric identifier (explicit rule).
    #
    # If there is a dot-number (e.g., 3.3) AND "fig" word exists, still FIGURE_QUERY
    # even if _FIGURE_RE didn't match due to formatting.
    if _FIGURE_WORD_RE.search(q):
        m_dot = _DOT_NUMBER_RE.search(q)
        target = _normalize_target("Fig", m_dot.group(0)) if m_dot else None
        return {"intent": INTENT_FIGURE, "target_id": target}

    # --- TABLE ---
    m_table = _TABLE_RE.search(q)
    if m_table:
        return {
            "intent": INTENT_TABLE,
            "target_id": _normalize_target("Table", m_table.group(2)),
        }

    if _TABLE_WORD_RE.search(q):
        # Explicit "table/tab." mention but no identifiable number.
        return {"intent": INTENT_TABLE, "target_id": None}

    # --- PAGE ---
    m_page = _PAGE_RE.search(q)
    if m_page:
        return {
            "intent": INTENT_PAGE,
            "target_id": _normalize_target("Page", m_page.group(1)),
        }

    # --- SECTION ---
    if _SECTION_WORD_RE.search(q):
        return {"intent": INTENT_SECTION, "target_id": None}

    # --- GENERAL ---
    return {"intent": INTENT_GENERAL, "target_id": None}
