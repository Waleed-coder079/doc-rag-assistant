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


def separate_text_and_tables(chunks):
    """Separate chunks into text chunks and table chunks."""
    text_chunks = []
    table_chunks = []
    
    for chunk in chunks:
        chunk_type = chunk.get("metadata", {}).get("type")
        
        if chunk_type == "table":
            table_chunks.append(chunk)
        else:
            # Everything else (text, TOC, etc.)
            text_chunks.append(chunk)
    
    return text_chunks, table_chunks


def build_faiss_index(chunks, embed_model_name, out_dir, index_name="faiss"):
    """Create FAISS index + metadata.pkl for a set of chunks."""
    if not chunks:
        print(f"⚠️ No chunks to index for {index_name}")
        return
    
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
    print(f"✅ {index_name} index built with {index.ntotal} vectors, dim={dim}")

    # Save FAISS index
    index_path = f"{out_dir}/{index_name}.index"
    faiss.write_index(index, index_path)

    # Save metadata with enhanced table information
    metadata = {}
    for i, ch in enumerate(chunks):
        meta = ch.get("metadata", {})
        entry = {
            "text": ch["text"],
            "file_name": ch.get("file_name", "unknown"),
            "chunk_id": ch.get("chunk_id", 0),
            "strategy": ch.get("strategy", "unknown"),
            "type": meta.get("type"),
        }
        
        # If it's a table, preserve table-specific metadata
        if meta.get("type") == "table":
            entry["table_title"] = meta.get("table_title")
            entry["table_data"] = meta.get("table_data")
            entry["page"] = meta.get("page")
            entry["title_confidence"] = meta.get("title_confidence")
            entry["source_file"] = meta.get("source_file")
        
        # Preserve section info for text chunks
        if "section" in ch:
            entry["section"] = ch["section"]
        
        metadata[i] = entry

    metadata_path = f"{out_dir}/{index_name}_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

        print(f"✅ Saved {index_name} index + metadata to '{out_dir}'")


# Configuration
INPUT_FILE = "split_out_emd_in/docs.jsonl"
OUTPUT_DIR = "emd_out_retr_in"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load all chunks
print(f"📄 Loading chunks from {INPUT_FILE}...")
chunks = load_chunks(INPUT_FILE)
print(f"📚 Loaded {len(chunks)} total chunks")

# Separate text and table chunks
text_chunks, table_chunks = separate_text_and_tables(chunks)
print(f"📝 Text chunks: {len(text_chunks)}")
print(f"📊 Table chunks: {len(table_chunks)}")

# Build TEXT index
print("\n🔨 Building TEXT index...")
build_faiss_index(
    text_chunks, 
    embed_model_name=EMBED_MODEL, 
    out_dir=OUTPUT_DIR,
    index_name="text_index"
)

# Build TABLE index
print("\n🔨 Building TABLE index...")
build_faiss_index(
    table_chunks, 
    embed_model_name=EMBED_MODEL, 
    out_dir=OUTPUT_DIR,
    index_name="table_index"
)

print(f"\n✅ Created 2 separate indices:")
print(f"   1. Text Index: {len(text_chunks)} chunks → {OUTPUT_DIR}/text_index.index")
print(f"   2. Table Index: {len(table_chunks)} chunks → {OUTPUT_DIR}/table_index.index")