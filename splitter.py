import json
from pathlib import Path
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser
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



def chunk_all_docs(docs, chunk_size=800):
    """Use SemanticSplitterNodeParser for all text chunks, keep table chunks whole."""
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = SemanticSplitterNodeParser(embed_model=embed_model, chunk_size=chunk_size)
    chunks = []
    for rec in docs:
        meta = rec.get("metadata", {})
        # Table chunk: keep whole
        if meta.get("type") == "table":
            node_meta = meta.copy()
            node_meta.update({
                "parent_id": rec.get("id"),
                "source": rec.get("source"),
                "source_url": rec.get("source_url"),
                "title": rec.get("title"),
                "chunk_id": 0,
                "strategy": "table_whole"
            })
            chunks.append({
                **node_meta,
                "text": rec["text"]
            })
        else:
            doc = Document(text=rec["text"], metadata=meta)
            nodes = splitter.get_nodes_from_documents([doc])
            for idx, node in enumerate(nodes):
                node_meta = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                node_meta.update({
                    "parent_id": rec.get("id"),
                    "source": rec.get("source"),
                    "source_url": rec.get("source_url"),
                    "title": rec.get("title"),
                    "page": rec.get("page"),
                    "paragraph_id": rec.get("paragraph_id"),
                    "chunk_id": idx,
                    "strategy": "semantic"
                })
                chunks.append({
                    **node_meta,
                    "text": node.get_content()
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

def get_source(doc):
    if "source" in doc:
        return doc["source"]
    return doc.get("metadata", {}).get("source", "")


# Single logic for all docs
chunks = chunk_all_docs(docs, PDF_CHUNK_SIZE)

print(f"✅ Created {len(chunks)} chunks.")

save_chunks(chunks, OUTPUT_FILE)
