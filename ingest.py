import os
import re
import json
import uuid
from pathlib import Path
from pdfminer.high_level import extract_text as pdf_extract_text
from bs4 import BeautifulSoup
import html2text


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


def process_pdf(filepath: str):
    """Extract page-wise text from a PDF file."""
    records = []
    text = pdf_extract_text(filepath)
    if not text:
        return records

    pages = text.split("\f")  # page delimiter in pdfminer output
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        page_text = clean_text(page_text)
        records.append({
            "id": str(uuid.uuid4()),
            "source": os.path.basename(filepath),
            "source_url": URL_MAP.get(os.path.basename(filepath), ""),
            "title": Path(filepath).stem,
            "page": page_num,
            "paragraph_id": None,
            "text": page_text
        })
    return records


def process_html(filepath: str):
    """Extract text from HTML, split by paragraphs (<p>, <div>, <section>)."""
    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    paragraphs = soup.find_all(["p", "div", "section", "article"])
    for pid, para in enumerate(paragraphs, start=1):
        text = clean_text(para.get_text(separator=" "))
        if not text:
            continue
        records.append({
            "id": str(uuid.uuid4()),
            "source": os.path.basename(filepath),
            "source_url": URL_MAP.get(os.path.basename(filepath), ""),
            "title": soup.title.string if soup.title else Path(filepath).stem,
            "page": None,
            "paragraph_id": pid,
            "text": text
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

    paragraphs = [clean_text(p) for p in plain_text.split("\n\n") if clean_text(p)]
    for pid, para in enumerate(paragraphs, start=1):
        records.append({
            "id": str(uuid.uuid4()),
            "source": os.path.basename(filepath),
            "source_url": URL_MAP.get(os.path.basename(filepath), ""),
            "title": Path(filepath).stem,
            "page": None,
            "paragraph_id": pid,
            "text": para
        })
    return records


def ingest(data_dir: str, out_file: str):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    for file in Path(data_dir).glob("*"):
        if file.suffix.lower() == ".pdf":
            recs = process_pdf(str(file))
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
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(all_records)} records to {out_file}")


# -------- Fixed paths --------
INPUT_DIR = "ingestion_input"
OUTPUT_FILE = "ing_out_split_in/docs.jsonl"

# Run ingestion directly
ingest(INPUT_DIR, OUTPUT_FILE)
