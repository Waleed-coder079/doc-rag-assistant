import pickle
import faiss
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Fixed configuration
INDEX_DIR = "RAG_DATA/emd_out_retr_in"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    """Load FAISS index and metadata from the fixed index directory."""
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(query, index, metadata, embed_model, k=5, use_cosine=True):
    """Search top-k results from FAISS index."""
    query_emb = np.array([embed_model.get_text_embedding(query)]).astype("float32")

    if use_cosine:
        # Normalize both query & index for cosine sim
        faiss.normalize_L2(query_emb)

    D, I = index.search(query_emb, k)  # D=distances, I=indices

    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        result = {
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "text": meta["text"],
            "source": meta["source"],
            "title": meta.get("title"),
            "page": meta.get("page"),
            "paragraph_id": meta.get("paragraph_id"),
            "strategy": meta.get("strategy")
        }
        # Add table metadata if present
        if meta.get("strategy") == "table_whole":
            result["type"] = meta.get("type")
            result["table_index"] = meta.get("table_index")
            result["section"] = meta.get("section")
            result["pages"] = meta.get("pages")
        results.append(result)
    return results


# Initialize the embedding model (can be reused across queries)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)