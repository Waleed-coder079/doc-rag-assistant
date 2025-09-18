import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai

# Fixed configuration
INDEX_DIR = "RAG_DATA/emd_out_retr_in"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gemini-1.5-flash-latest"

# Initialize Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)


# ---------- Load Index ----------
def load_index():
    """Load FAISS index and metadata from the fixed index directory."""
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ---------- Retriever ----------
def search(query, index, metadata, embed_model, k=5, use_cosine=True):
    """Search top-k results from FAISS index."""
    query_emb = np.array([embed_model.get_text_embedding(query)]).astype("float32")

    if use_cosine:
        faiss.normalize_L2(query_emb)

    D, I = index.search(query_emb, k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        meta = metadata[idx]

        # ✅ Prefer URL if available, fallback to local source
        source_link = meta.get("source_url") or meta.get("source")

        result = {
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "text": meta["text"],
            "source": source_link,   # unified source field
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

# ---------- Generator ----------
def generate_answer(query, retrieved_chunks, model_name=DEFAULT_MODEL):
    """Call Gemini API with retrieved context and return answer + citations."""
    # Build context with citation anchors
    context = ""
    citation_map = {}
    for i, ch in enumerate(retrieved_chunks, start=1):
        anchor = f"[{i}]"
        snippet = ch["text"].replace("\n", " ")
        context += f"{anchor} {snippet}\n"
        citation_map[i] = ch   # keeps full chunk info (with URL/source)

    prompt = f"""
You are a helpful AI assistant. Use only the provided context to answer the query. 
If multiple parts of the context are relevant, combine them into a detailed explanation in clear, natural language. 
Cite sources with inline numbers [1], [2] only when referencing specific facts. 
Do not start answers with phrases like "Based on the provided text." 
Always provide as much useful detail as the context allows.

Query: {query}

Context:
{context}

Answer:
"""

    response = genai.GenerativeModel(model_name).generate_content(prompt)
    return response.text, citation_map


# Load index and metadata at module level
index, metadata = load_index()
