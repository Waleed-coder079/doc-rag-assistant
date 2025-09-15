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
    """Create FAISS index + metadata."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Init embedder
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    texts = [c["text"] for c in chunks]
    embeddings = [embed_model.get_text_embedding(t) for t in texts]
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatL2(dim)  # L2 distance (you can swap to cosine if needed)

    index.add(embeddings)
    print(f"✅ FAISS index built with {index.ntotal} vectors, dim={dim}")

    # Save FAISS index
    faiss.write_index(index, f"{out_dir}/faiss.index")

    # Save metadata (mapping idx → chunk info)
    metadata = {i: chunks[i] for i in range(len(chunks))}
    with open(f"{out_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved FAISS index + metadata to '{out_dir}'")


# Fixed configuration
INPUT_FILE = "split_out_emd_in/docs.jsonl"
OUTPUT_DIR = "emd_out_retr_in"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load and process chunks
chunks = load_chunks(INPUT_FILE)
build_faiss_index(chunks, embed_model_name=EMBED_MODEL, out_dir=OUTPUT_DIR)