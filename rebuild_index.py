"""
Rebuild metadata.json and faiss.index with improved chunking.

Improvements over the original:
- Sentence-aware splitting (no mid-sentence breaks)
- 100-char overlap between consecutive chunks for context continuity
- Removes junk fragments (< 30 chars)
- Deduplicates text vs image_ocr chunks from the same page
- Preserves all original metadata (page_number, source type)
"""

import json
import re
import faiss
import numpy as np
import time
import os
import google.generativeai as genai

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Export it or add it to .env")
genai.configure(api_key=api_key)

CHUNK_TARGET = 800
CHUNK_MIN = 50
OVERLAP = 100
EMBED_DIM = 768
EMBED_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 80

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*\n")


def sentence_aware_chunks(
    text: str, target: int = CHUNK_TARGET, overlap: int = OVERLAP
) -> list[str]:
    """Split text into chunks at sentence boundaries with overlap."""
    if not text or not text.strip():
        return []

    sentences = SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text.strip()] if len(text.strip()) >= CHUNK_MIN else []

    chunks = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + len(sent) + 1 <= target:
            current = current + " " + sent
        else:
            chunks.append(current)
            # Overlap: carry the tail of the previous chunk
            if overlap > 0 and len(current) > overlap:
                tail = current[-overlap:]
                # Try to start at a word boundary
                space_idx = tail.find(" ")
                if space_idx != -1:
                    tail = tail[space_idx + 1 :]
                current = tail + " " + sent
            else:
                current = sent

    if current and len(current.strip()) >= CHUNK_MIN:
        chunks.append(current.strip())

    return chunks


def text_similarity_quick(a: str, b: str) -> float:
    """Fast Jaccard similarity on word sets."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def main():
    with open("metadata.json", "r", encoding="utf-8") as f:
        old_meta = json.load(f)

    print(f"Original: {len(old_meta)} chunks")

    # Group chunks by page
    pages: dict[int, list[dict]] = {}
    for cid, data in old_meta.items():
        page = data.get("page_number", 0)
        pages.setdefault(page, []).append({**data, "old_id": cid})

    new_meta = {}
    chunk_id = 1

    for page_num in sorted(pages.keys()):
        page_chunks = pages[page_num]

        # Separate text and OCR chunks
        text_chunks = [c for c in page_chunks if c.get("source") == "text"]
        ocr_chunks = [c for c in page_chunks if c.get("source") == "image_ocr"]

        # Combine all text from each source for this page
        text_combined = " ".join(c.get("text", "") for c in text_chunks).strip()
        ocr_combined = " ".join(c.get("text", "") for c in ocr_chunks).strip()

        # Decide which source to use: prefer text, add OCR only if it has unique content
        sources_to_process = []
        if text_combined:
            sources_to_process.append(("text", text_combined))
        if ocr_combined:
            if not text_combined:
                sources_to_process.append(("image_ocr", ocr_combined))
            elif text_similarity_quick(text_combined, ocr_combined) < 0.7:
                # OCR has substantially different content, keep it
                sources_to_process.append(("image_ocr", ocr_combined))

        for source_type, full_text in sources_to_process:
            # Clean up common artifacts
            full_text = re.sub(r"\n{3,}", "\n\n", full_text)
            full_text = re.sub(r" {3,}", " ", full_text)

            sub_chunks = sentence_aware_chunks(full_text)

            for chunk_text in sub_chunks:
                if len(chunk_text.strip()) < CHUNK_MIN:
                    continue

                image_path = None
                if page_num:
                    candidate = f"images/page_{page_num}_img_1.png"
                    if os.path.exists(candidate):
                        image_path = candidate

                new_meta[str(chunk_id)] = {
                    "text": chunk_text,
                    "source": source_type,
                    "page_number": page_num,
                    "image_path": image_path,
                }
                chunk_id += 1

    total = len(new_meta)
    print(f"New: {total} chunks (removed duplicates + junk)")

    # Stats
    lengths = [len(v["text"]) for v in new_meta.values()]
    print(
        f"Chunk sizes: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}"
    )
    print(f"Under 50 chars: {sum(1 for l in lengths if l < 50)}")
    print(f"50-500 chars: {sum(1 for l in lengths if 50 <= l < 500)}")
    print(f"500-1000 chars: {sum(1 for l in lengths if 500 <= l < 1000)}")
    print(f"1000+ chars: {sum(1 for l in lengths if l >= 1000)}")

    # Save new metadata
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2, ensure_ascii=False)
    print("Saved metadata.json")

    # Build FAISS index
    print(f"\nEmbedding {total} chunks with {EMBED_MODEL}...")
    index = faiss.IndexFlatL2(EMBED_DIM)

    chunk_ids = sorted(new_meta.keys(), key=lambda x: int(x))
    texts = [new_meta[cid]["text"] for cid in chunk_ids]

    embedded = 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
                output_dimensionality=EMBED_DIM,
            )
            embeddings = np.array(result["embedding"], dtype=np.float32)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            embedded += len(batch)
            print(f"  {embedded}/{total}")
        except Exception as e:
            print(f"  Batch error at {i}: {e}, retrying individually...")
            for j, t in enumerate(batch):
                try:
                    r = genai.embed_content(
                        model=EMBED_MODEL,
                        content=t if t.strip() else "empty",
                        output_dimensionality=EMBED_DIM,
                    )
                    emb = np.array([r["embedding"]], dtype=np.float32)
                    faiss.normalize_L2(emb)
                    index.add(emb)
                except Exception as e2:
                    print(f"    Chunk {i+j+1} failed: {e2}")
                    index.add(np.zeros((1, EMBED_DIM), dtype=np.float32))
                embedded += 1
        time.sleep(0.1)

    print(f"\nIndex: {index.ntotal} vectors")
    faiss.write_index(index, "faiss.index")
    print("Saved faiss.index")
    print("Done!")


if __name__ == "__main__":
    main()
