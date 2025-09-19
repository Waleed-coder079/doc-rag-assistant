import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_chunks(jsonl_file):
    """Load chunked docs from splitter.py output."""
    chunks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_faiss_index(chunks, embed_model_name, out_dir):
    """Create FAISS index + metadata.pkl (keeps source_url)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Init embedder
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Compute embeddings
    texts = [c["text"] for c in chunks]
    embeddings = [embed_model.get_text_embedding(t) for t in texts]
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index built with {index.ntotal} vectors, dim={dim}")

    # Save FAISS index
    faiss.write_index(index, f"{out_dir}/faiss.index")

    # Save metadata (idx → chunk info, with correct source_url)
    metadata = {}
    for i, ch in enumerate(chunks):
        entry = {
            "text": ch["text"],
            "source": ch.get("source"),           # local path or filename
            "source_url": ch.get("source_url"),   # ✅ actual link
            "title": ch.get("title"),
            "page": ch.get("page"),
            "paragraph_id": ch.get("paragraph_id"),
            "chunk_id": ch.get("chunk_id"),
            "strategy": ch.get("strategy"),
            "parent_id": ch.get("parent_id"),
        }
        # Add table metadata if present
        if ch.get("strategy") == "table_whole":
            entry["type"] = ch.get("type")
            entry["table_index"] = ch.get("table_index")
            entry["section"] = ch.get("section")
            entry["pages"] = ch.get("pages")
        metadata[i] = entry

    with open(f"{out_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved FAISS index + metadata to '{out_dir}'")


# Configuration + direct execution
INPUT_FILE = "split_out_emd_in/docs.jsonl"
OUTPUT_DIR = "emd_out_retr_in"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

chunks = load_chunks(INPUT_FILE)
build_faiss_index(chunks, embed_model_name=EMBED_MODEL, out_dir=OUTPUT_DIR)