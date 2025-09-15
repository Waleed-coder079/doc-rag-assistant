import json
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document


def load_docs(jsonl_file):
    """Load documents from JSONL file."""
    docs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def save_chunks(chunks, out_file):
    """Save processed chunks to JSONL file."""
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


def chunk_pdf_docs(docs, chunk_size=800, overlap=200):
    """Use SentenceSplitter for PDF docs."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for rec in docs:
        parts = splitter.split_text(rec["text"])
        for idx, part in enumerate(parts):
            chunks.append({
                "parent_id": rec["id"],
                "source": rec["source"],            # local filename
                "source_url": rec.get("source_url"), # ✅ keep actual URL
                "title": rec["title"],
                "page": rec.get("page"),
                "paragraph_id": rec.get("paragraph_id"),
                "chunk_id": idx,
                "text": part,
                "strategy": "sentence"
            })
    return chunks


def chunk_other_docs(docs, chunk_size=800, embed_model=None):
    """Use SemanticSplitterNodeParser for non-PDF docs."""
    embed_model = embed_model or HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    splitter = SemanticSplitterNodeParser(embed_model=embed_model, chunk_size=chunk_size)

    chunks = []
    for rec in docs:
        # Create Document object from the text
        doc = Document(text=rec["text"], id_=rec["id"])
        nodes = splitter.get_nodes_from_documents([doc])
        for idx, node in enumerate(nodes):
            chunks.append({
                "parent_id": rec["id"],
                "source": rec["source"],            # local filename
                "source_url": rec.get("source_url"), # ✅ keep actual URL
                "title": rec["title"],
                "page": rec.get("page"),
                "paragraph_id": rec.get("paragraph_id"),
                "chunk_id": idx,
                "text": node.get_content(),
                "strategy": "semantic"
            })
    return chunks

# Fixed configuration
INPUT_FILE = "ing_out_split_in/docs.jsonl"
OUTPUT_FILE = "split_out_emd_in/docs.jsonl"
PDF_CHUNK_SIZE = 800
PDF_OVERLAP = 200
OTHER_CHUNK_SIZE = 800

# Load and process documents
docs = load_docs(INPUT_FILE)

pdf_docs = [d for d in docs if d["source"].lower().endswith(".pdf")]
other_docs = [d for d in docs if not d["source"].lower().endswith(".pdf")]

chunks = []
if pdf_docs:
    chunks.extend(chunk_pdf_docs(pdf_docs, PDF_CHUNK_SIZE, PDF_OVERLAP))
if other_docs:
    chunks.extend(chunk_other_docs(other_docs, OTHER_CHUNK_SIZE))

print(f"✅ Created {len(chunks)} chunks "
      f"(from {len(pdf_docs)} PDF docs & {len(other_docs)} other docs).")

save_chunks(chunks, OUTPUT_FILE)
