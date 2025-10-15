from __future__ import annotations

import json
import re
from pathlib import Path

INPUT_FILE = "ing_out_split_in/docs.jsonl"
OUTPUT_FILE = "split_out_emd_in/docs.jsonl"

# Tunable parameters for chunk sizing
# MAX_TOKENS_PER_CHUNK: approximate cap for each chunk size
# OVERLAP_TOKENS: carry a tail into the next chunk to preserve context
MAX_TOKENS_PER_CHUNK = 700
OVERLAP_TOKENS = 50


def load_docs(jsonl_file: str):
    """Read docs from a JSONL file; each line is a JSON object."""
    docs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def save_chunks(chunks, out_file: str):
    """Write chunk dicts to a JSONL file, one per line."""
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


def estimate_tokens(text: str) -> int:
    """Approximate token count; uses tiktoken if available, else whitespace count."""
    text = (text or "").strip()
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


# --- Sentence splitting / chunking helpers ---
# Split where: punctuation (., ?, !) is followed by whitespace and then an uppercase letter,
# a digit, or an opening quote. This is a lightweight heuristic; it won't be perfect but
# works well for technical docs without abbreviations-heavy prose.
_sentence_split_re = re.compile(r'(?<=[\.?\!])\s+(?=[A-Z0-9“"\'])')


def split_into_sentences(text: str):
    """Split a block of text into sentences.

    Primary: use the punctuation+capital heuristic above after collapsing newlines.
    Fallback: if it looks like one single piece, split on single-or-more newlines.
    Returns a list of non-empty, stripped sentence strings.
    """
    text = text.strip()
    if not text:
        return []
    compact = re.sub(r"\n+", " ", text)
    parts = _sentence_split_re.split(compact)
    if len(parts) == 1:
        parts = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(
    sentences,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    overlap_tokens: int = OVERLAP_TOKENS,
):
    """Greedily pack sentences into chunks up to ~max_tokens with tail overlap.

    - estimate_tokens approximates LLM tokens for size control.
    - When a chunk is flushed, we keep a tail of prior sentences whose token
      count is <= overlap_tokens to preserve continuity across chunk boundaries.
    - A safety flush triggers if we ever exceed 1.5x the target (rare edge cases).
    Returns a list of chunk strings.
    """
    chunks, current, current_tokens = [], [], 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(" ".join(current).strip())
        if overlap_tokens > 0:
            # Keep a backward tail whose total tokens <= overlap_tokens.
            # Insert at the front to preserve original sentence order.
            keep, keep_tokens = [], 0
            for s in reversed(current):
                t = estimate_tokens(s)
                if keep_tokens + t > overlap_tokens:
                    break
                keep.insert(0, s)
                keep_tokens += t
            current, current_tokens = keep, keep_tokens
        else:
            current, current_tokens = [], 0

    for s in sentences:
        s_tokens = estimate_tokens(s)
        if current_tokens + s_tokens > max_tokens and current:
            flush()
        current.append(s)
        current_tokens += s_tokens
        if current_tokens >= max_tokens * 1.5:
            flush()
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c.strip()]


# --- TOC detection (unchanged heuristic) ---
_def_toc_heading_re = re.compile(r"^\d+(?:\.\d+)*(?:\s+.+)?$")


def detect_toc(text: str, max_lines: int = 30):
    """Heuristically detect a Table of Contents.

    Look at the first max_lines non-empty lines and count how many look like
    numbered headings (e.g., 1, 2.1, 3.4.5). If there are at least 3 such lines
    or they form >= one third of the lines, treat this front-matter as a TOC.
    """
    lines = [l.strip() for l in text.splitlines()[:max_lines] if l.strip()]
    heading_like = sum(1 for l in lines if _def_toc_heading_re.match(l))
    return heading_like >= max(3, len(lines) // 3)


# --- Numeric prefix parser ---
_num_prefix_re = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.\)]\s*)?(.*)$")


def parse_leading_number(line: str):
    """
    Returns (num_str, rest) if line starts with numeric prefix like:
      "2. Configuration" -> ("2", "Configuration")
      "2.1. Configure..." -> ("2.1", "Configure...")
      "1. Go to..." -> ("1", "Go to...")
      "4.7. Drilldowns" -> ("4.7", "Drilldowns")
    Returns (None, None) if no leading numeric prefix.
    """
    if not line:
        return None, None
    m = _num_prefix_re.match(line.strip())
    if not m:
        return None, None
    num_str = m.group(1)
    rest = (m.group(2) or "").strip()
    return num_str, rest


# --- Grouping: top-level primary grouping logic ---


def group_by_top_level_sections(text: str):
    """
    Group all content under the current top-level heading (first numeric segment),
    merging nested subheadings and numbered lists into that top-level group until a
    bigger top-level number appears.

    Rules:
      - A numeric line's "top number" is int(num_str.split('.')[0])
      - If current_top is None => the first numeric line starts the first top-level group
      - If we see a numeric line with level == 1 (e.g., "3" as top number) and its top number
        is greater than the current_top_number => start a NEW top-level group
      - Otherwise append the line (including "1.", "2.", subheadings like "2.1", "4.7.1")
        to the current top-level group's buffer
    """
    lines = text.splitlines()
    sections = []
    current_top_number = None
    current_top_heading = None
    buffer = []

    def flush_current():
        nonlocal current_top_number, current_top_heading, buffer
        if current_top_heading is None and not buffer:
            return
        heading_label = current_top_heading or "preamble"
        body = "\n".join(buffer).strip()
        sections.append((heading_label, body))
        current_top_number = None
        current_top_heading = None
        buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()
        num_str, rest = parse_leading_number(line)
        if num_str:
            # compute hierarchical depth (level) and top-level number (e.g., '2' from '2.3.1')
            level = num_str.count(".") + 1
            try:
                top_num = int(num_str.split(".")[0])
            except Exception:
                top_num = None

            if current_top_number is None:
                # Start first top-level group using this line
                current_top_number = top_num
                # If there is a rest (title), use "num rest" as heading label, else keep original line
                current_top_heading = f"{num_str}. {rest}" if rest else f"{num_str}."
                buffer = [line]
                continue

            # If this numeric line is a level-1 and has higher top_num => it's a new top-level start
            if level == 1 and (top_num is not None) and (top_num > current_top_number):
                # flush previous group and start a new one
                flush_current()
                current_top_number = top_num
                current_top_heading = f"{num_str}. {rest}" if rest else f"{num_str}."
                buffer = [line]
                continue
            else:
                # Otherwise it's subordinate (subheading or numbered list) => append
                buffer.append(line)
                continue
        else:
            # Non-numeric line -> append to buffer
            if current_top_number is None and not buffer:
                # no top yet: start a preamble group (un-numbered intro)
                current_top_heading = "preamble"
                buffer = [line]
                continue
            buffer.append(line)

    # flush leftover buffer
    flush_current()
    return sections if sections else [("full", text.strip())]


# --- Main per-document processing ---


def process_single_doc(rec):
    """Convert a normalized ingestion record into one or more chunks.

    - Tables or TOC records are preserved as single chunks (with metadata kept).
    - Otherwise: optional TOC extraction from the front, then group the rest by
      top-level headings and subordinate lines. Groups smaller than the size cap
      are emitted whole; larger ones are sentence-chunked with overlap.
    """
    out = []
    text = rec.get("text", "") or ""
    meta = rec.get("metadata", {}) or {}
    file_name = meta.get("file_name") or rec.get("source") or "unknown"

    # Keep tables and TOC whole - now handles enhanced table metadata
    doc_type = meta.get("type")
    if doc_type in ("table", "table_of_contents"):
        # Preserve all table metadata including table_title, table_data, page, etc.
        chunk_meta = meta.copy()
        out.append({
            "file_name": file_name,
            "chunk_id": 0,
            "strategy": f"{doc_type}_whole",
            "text": text,
            "metadata": chunk_meta,
        })
        return out

    chunk_counter = 0

    # if first part looks like a TOC, extract it as a single chunk
    if detect_toc(text):
        lines = text.splitlines()
        toc_lines, rest = [], []
        toc_done = False
        for line in lines:
            if not toc_done and re.match(r"^\d+(?:\.\d+)*", line.strip()):
                toc_lines.append(line)
            else:
                toc_done = True
                rest.append(line)
        toc_text = "\n".join(toc_lines).strip()
        if toc_text:
            out.append({
                "file_name": file_name,
                "chunk_id": chunk_counter,
                "strategy": "toc_whole",
                "section": "TOC",
                "text": toc_text,
                "metadata": meta,
            })
            chunk_counter += 1
        text = "\n".join(rest).strip()

    # Group by top-level headings but keep subheadings & numbered lists inside the same group
    groups = group_by_top_level_sections(text)
    for heading, body in groups:
        if not body.strip():
            continue
        token_count = estimate_tokens(body)
        if token_count <= MAX_TOKENS_PER_CHUNK:
            out.append({
                "file_name": file_name,
                "chunk_id": chunk_counter,
                "strategy": "merged_headings",
                "section": heading,
                "text": body.strip(),
                "metadata": meta,
            })
            chunk_counter += 1
        else:
            sentences = split_into_sentences(body)
            for ct in chunk_sentences(sentences):
                out.append({
                    "file_name": file_name,
                    "chunk_id": chunk_counter,
                    "strategy": "merged_headings_split",
                    "section": heading,
                    "text": ct,
                    "metadata": meta,
                })
                chunk_counter += 1

    return out


def chunk_all_docs(docs):
    """Process all records and return the concatenated list of chunks."""
    all_chunks = []
    for rec in docs:
        all_chunks.extend(process_single_doc(rec))
    return all_chunks


if __name__ == "__main__":
    print(f"📄 Loading documents from {INPUT_FILE}...")
    docs = load_docs(INPUT_FILE)
    print(f"📚 Loaded {len(docs)} documents.")
    print("🔄 Processing chunks...")
    chunks = chunk_all_docs(docs)
    print(f"✅ Created {len(chunks)} chunks.")
    print(f"💾 Saving chunks to {OUTPUT_FILE}...")
    save_chunks(chunks, OUTPUT_FILE)
    print(f"✅ Saved chunks to {OUTPUT_FILE}")
