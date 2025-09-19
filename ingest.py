
import os
import re
import json
import uuid
from pathlib import Path
from pdfminer.high_level import extract_text as pdf_extract_text
from bs4 import BeautifulSoup
import html2text
from llama_index.core import Document
from docx import Document as DocxReader
import tabula
import pandas as pd


# -------- Load URL mapping --------
URL_MAP_FILE = "url_map.json"
if os.path.exists(URL_MAP_FILE):
    with open(URL_MAP_FILE, "r", encoding="utf-8") as f:
        URL_MAP = json.load(f)
else:
    URL_MAP = {}
    print("⚠️ url_map.json not found, proceeding without external links.")


def clean_text(text: str) -> str:
    """Basic cleaning: remove excessive whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def extract_text_with_pages(filepath: str):
    """Extract text with page-like splits (PDF vs DOCX)."""
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        text = pdf_extract_text(filepath)
        # Store all pages as one string, but keep page mapping for metadata
        pages = text.split("\f") if text else []
        full_text = "\n".join(pages)
        results = []
        if full_text:
            results.append((full_text, {"pages": list(range(1, len(pages)+1))}))
        # Extract tables using tabula-py
        try:
            tables = tabula.read_pdf(filepath, pages="all", multiple_tables=True)
            for idx, table in enumerate(tables):
                if not table.empty:
                    table_text = table.to_string(index=False)
                    results.append((table_text, {
                        "type": "table",
                        "pages": "all",  # tabula does not always give page info
                        "table_index": idx
                    }))
        except Exception as e:
            print(f"Table extraction failed for PDF {filepath}: {e}")
        return results

    elif ext == ".docx":
        doc = DocxReader(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        results = []
        if full_text:
            results.append((full_text, {"paragraphs": len(paragraphs)}))
        # Extract tables from DOCX
        for idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)
            # Convert to string (tabular)
            table_df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
            table_text = table_df.to_string(index=False)
            results.append((table_text, {
                "type": "table",
                "table_index": idx,
                "section": "Unknown"  # Optionally, add logic to detect section
            }))
        return results

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_document(filepath: str):
    """Process PDF or DOCX into LlamaIndex Documents with metadata."""
    records = []
    chunks = extract_text_with_pages(filepath)
    for chunk_text, extra_meta in chunks:
        if not chunk_text.strip():
            continue
        meta = {
            "doc_id": str(uuid.uuid4()),
            "file_name": os.path.basename(filepath),
            "title": Path(filepath).stem,
        }
        meta.update(extra_meta)
        doc = Document(
            text=chunk_text,
            metadata=meta
        )
        records.append(doc)
    return records



def process_html(filepath: str):
    """Extract text from HTML, split by paragraphs (<p>, <div>, <section>)."""
    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Extract all visible text from HTML
    raw_text = soup.get_text(separator=" ")
    cleaned_text = clean_text(raw_text)

    # Save as a single record, similar to DOCX
    records.append({
        "id": str(uuid.uuid4()),
        "source": os.path.basename(filepath),
        "source_url": URL_MAP.get(os.path.basename(filepath), ""),
        "title": soup.title.string if soup.title else Path(filepath).stem,
        "page": None,
        "paragraph_id": 1,
        "text": cleaned_text
    })
    return records

def process_markdown(filepath: str):
    """Extract text from Markdown and split by headings/paragraphs."""
    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        md_content = f.read()

    h = html2text.HTML2Text()
    h.ignore_links = True
    plain_text = h.handle(md_content)
    cleaned_text = clean_text(plain_text)

    # Save as a single record, similar to HTML and DOCX
    records.append({
        "id": str(uuid.uuid4()),
        "source": os.path.basename(filepath),
        "source_url": URL_MAP.get(os.path.basename(filepath), ""),
        "title": Path(filepath).stem,
        "page": None,
        "paragraph_id": 1,
        "text": cleaned_text
    })
    return records


def ingest(data_dir: str, out_file: str):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    for file in Path(data_dir).glob("*"):
        if file.suffix.lower() in [".pdf",".docx"]:
            recs = process_document(str(file))
        elif file.suffix.lower() in [".html", ".htm"]:
            recs = process_html(str(file))
        elif file.suffix.lower() in [".md", ".markdown"]:
            recs = process_markdown(str(file))
        else:
            print(f"Skipping unsupported file type: {file}")
            continue

        print(f"Ingested {len(recs)} records from {file}")
        all_records.extend(recs)

    with open(out_file, "w", encoding="utf-8") as f:
        for rec in all_records:
            if isinstance(rec, Document):
                f.write(json.dumps({"text": rec.text, "metadata": rec.metadata}, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(all_records)} records to {out_file}")


# -------- Fixed paths --------
# INPUT_DIR = "ingestion_input"
# OUTPUT_FILE = "ing_out_split_in/docs.jsonl"

# # Run ingestion directly
# ingest(INPUT_DIR, OUTPUT_FILE)
INPUT_DIR = "input"
OUTPUT_FILE = "ing_out_split_in/docs.jsonl"

# Run ingestion directly
ingest(INPUT_DIR, OUTPUT_FILE)