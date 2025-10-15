import pickle
import faiss
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Fixed configuration (paths and model are used across functions)
INDEX_DIR = "emd_out_retr_in"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LanGraph State Definition
class RAGState(TypedDict):
        """State for the (simplified) agentic RAG workflow.

        Fields:
            query: original user query
            query_type: classification label (text | table | hybrid)
            retrieved_chunks: final list of retrieved chunk dicts (possibly reranked)
            debug_info: metrics + routing metadata
            agent_steps: ordered human‑readable trace of what the agent did
            k: target top-k (after optional reranking)
            use_reranker: flag passed through from caller
        """
        query: str
        query_type: str
        retrieved_chunks: List[Dict[str, Any]]
        debug_info: Dict[str, Any]
        agent_steps: Annotated[List[str], add_messages]
        k: int
        use_reranker: bool


def load_index(index_name="text_index"):
    """Load FAISS index and metadata for a specific index (text_index or table_index)."""
    index_path = f"{INDEX_DIR}/{index_name}.index"
    metadata_path = f"{INDEX_DIR}/{index_name}_metadata.pkl"
    
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except FileNotFoundError:
        print(f"⚠️ Index not found: {index_name}")
        return None, None


def load_all_indices():
    """Load both text and table indices."""
    text_index, text_metadata = load_index("text_index")
    table_index, table_metadata = load_index("table_index")
    
    return {
        "text": {"index": text_index, "metadata": text_metadata},
        "table": {"index": table_index, "metadata": table_metadata}
    }


def classify_query(query):
    """Classify query as 'text', 'table', or 'hybrid' using rule-based heuristics.

    Simple keyword matching to infer intent:
      - table: structured requests (columns/rows/comparisons/countries/etc.)
      - text: procedural/how-to/explanations
      - hybrid: signals for both, or ambiguous → search both
    Returns one of: 'text' | 'table' | 'hybrid'
    """
    query_lower = query.lower()
    
    # Table-related keywords and patterns
    table_keywords = [
        r'\btable\b', r'\bcolumn\b', r'\brow\b', r'\bdata\b',
        r'\blist\s+of\b', r'\blist\s+all\b', r'\bshow\s+all\b',
        r'\bcompare\b', r'\bcomparison\b',
        r'\bcountries\b', r'\bcountry\b',
        r'\brequirement\b', r'\brequirements\b',
        r'\bobligation\b', r'\bobligations\b',
        r'\bsupport\b', r'\bsupported\b',
        r'\bfiling\s+method\b', r'\bfiling\s+methods\b',
        r'\bvat\s+rate\b', r'\bvat\s+rates\b',
        r'\bdeadline\b', r'\bdeadlines\b',
        r'\bfrequency\b', r'\bfrequencies\b',
        r'\bformat\b', r'\bformats\b',
    ]
    
    # Text/procedural keywords
    text_keywords = [
        r'\bhow\s+to\b', r'\bstep\b', r'\bsteps\b',
        r'\bprocess\b', r'\bprocedure\b',
        r'\bconfigure\b', r'\bconfiguration\b',
        r'\bsetup\b', r'\bset\s+up\b',
        r'\bexplain\b', r'\bdescribe\b',
        r'\bwhat\s+is\b', r'\bwhy\b',
        r'\bguide\b', r'\btutorial\b',
        r'\bcreate\b', r'\bupload\b', r'\bdownload\b',
    ]
    
    # Hybrid indicators (both structured and procedural)
    hybrid_keywords = [
        r'\bwhich\s+countries\b',
        r'\blist.*how\b',
        r'\bshow.*steps\b',
        r'\bcompare.*process\b',
    ]
    
    # Count matches for each class
    table_score = sum(1 for pattern in table_keywords if re.search(pattern, query_lower))
    text_score = sum(1 for pattern in text_keywords if re.search(pattern, query_lower))
    hybrid_score = sum(1 for pattern in hybrid_keywords if re.search(pattern, query_lower))
    
    # Decision logic
    if hybrid_score > 0:
        return "hybrid"
    
    # If both scores are significant, it's hybrid
    if table_score >= 2 and text_score >= 1:
        return "hybrid"
    
    # If strong table signal
    if table_score > text_score:
        return "table"
    
    # If strong text signal
    if text_score > table_score:
        return "text"
    
    # Default: hybrid (search both if unclear)
    return "hybrid"

"""We intentionally keep only ONE retrieval layer used by the agentic flow.

What remains:
  - _single_index_search: minimal helper for a FAISS index
  - _route_and_search: classification + routing + merge (private)
  - search_agentic: public adaptive retrieval API consumed by generator/app
"""

def _single_index_search(query: str, index, metadata, k: int, embed_model) -> List[Dict[str, Any]]:
    """Lightweight semantic search inside one FAISS index (already built with L2-normed vectors)."""
    if index is None or metadata is None:
        return []
    q_emb = np.array([embed_model.get_text_embedding(query)]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    out = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        r: Dict[str, Any] = {
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "text": meta.get("text", ""),
            "file_name": meta.get("file_name", "unknown"),
            "chunk_id": meta.get("chunk_id", 0),
            "strategy": meta.get("strategy", "unknown"),
            "type": meta.get("type"),
        }
        if meta.get("type") == "table":
            r["table_title"] = meta.get("table_title")
            r["table_data"] = meta.get("table_data")
            r["page"] = meta.get("page")
        if "section" in meta:
            r["section"] = meta.get("section")
        out.append(r)
    return out


def _route_and_search(query: str, k: int, embed_model, indices) -> tuple[List[Dict[str, Any]], str, List[str]]:
    """Classify the query then search appropriate index(es). Returns (results, query_type, searched_indices).

    For hybrid queries we retrieve k from each index, merge, then keep top-k overall
    via raw distance (lower=better). This is intentionally simple; reranker (if on)
    operates afterwards.
    """
    q_type = classify_query(query)
    searched: List[str] = []
    if q_type == "text":
        res = _single_index_search(query, indices["text"]["index"], indices["text"]["metadata"], k, embed_model)
        for r in res:
            r["source_index"] = "text"
        searched.append("text")
        return res, q_type, searched
    if q_type == "table":
        res = _single_index_search(query, indices["table"]["index"], indices["table"]["metadata"], k, embed_model)
        for r in res:
            r["source_index"] = "table"
        searched.append("table")
        return res, q_type, searched
    # hybrid
    both: List[Dict[str, Any]] = []
    if indices["text"]["index"] is not None:
        tr = _single_index_search(query, indices["text"]["index"], indices["text"]["metadata"], k, embed_model)
        for r in tr:
            r["source_index"] = "text"
        both.extend(tr)
        searched.append("text")
    if indices["table"]["index"] is not None:
        tb = _single_index_search(query, indices["table"]["index"], indices["table"]["metadata"], k, embed_model)
        for r in tb:
            r["source_index"] = "table"
        both.extend(tb)
        searched.append("table")
    both.sort(key=lambda x: x.get("score", 1e9))
    both = both[:k]
    for i, r in enumerate(both):
        r["rank"] = i + 1
    return both, q_type, searched


# ========== LanGraph Agentic Workflow ==========

def classify_node(state: RAGState) -> RAGState:
    """Agent Node 1: Classify query type.

    (Refinement removed) We keep only classification to drive routing.
    """
    q = state["query"]
    q_type = classify_query(q)
    state["query_type"] = q_type
    state["agent_steps"].append(f"✓ Classified query as: {q_type.upper()}")
    state["debug_info"] = {"query_type": q_type}
    return state


_reranker_model = None  # cached cross-encoder

def _load_reranker():
    """Lazy-load and cache the cross-encoder reranker model.

    Safe to call multiple times; returns None on failure and we gracefully
    skip reranking in that case.
    """
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"⚠️ Could not load reranker: {e}")
            _reranker_model = None
    return _reranker_model

def _apply_rerank(query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Apply cross-encoder reranking and return top-k.

    Truncates text to 512 chars per pair to keep scoring fast.
    Falls back silently if model/predict fails.
    """
    model = _load_reranker()
    if model is None or not results:
        return results[:k]
    pairs = []
    for r in results:
        pairs.append([query, r.get("text", "")[:512]])
    try:
        scores = model.predict(pairs)
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        for i, r in enumerate(reranked[:k]):
            r["rank"] = i + 1
        return reranked[:k]
    except Exception as e:
        print(f"⚠️ Rerank failed: {e}")
        return results[:k]

def retrieve_node(state: RAGState) -> RAGState:
    """Agent Node 2 (final): Route + retrieve + optional rerank."""
    query = state["query"]
    k = state.get("k", 5)
    use_reranker = state.get("use_reranker", False)
    effective_k = k * 4 if use_reranker else k

    results, q_type, searched_indices = _route_and_search(
        query=query,
        k=effective_k,
        embed_model=embed_model,
        indices=all_indices
    )

    debug_info = {
        "query_type": q_type,
        "searched_indices": searched_indices,
        "total_results": len(results)
    }

    if use_reranker:
        before = len(results)
        results = _apply_rerank(query, results, k)
        state["agent_steps"].append(f"✓ Reranked {before}→{len(results)} with cross-encoder")
        debug_info["reranker"] = True
    else:
        for i, r in enumerate(results[:k]):
            r["rank"] = i + 1
        results = results[:k]
        state["agent_steps"].append(f"✓ Retrieved {len(results)} chunks from {searched_indices}")

    state["retrieved_chunks"] = results
    state["query_type"] = q_type  # ensure propagated
    # merge debug info
    if "debug_info" not in state or not state["debug_info"]:
        state["debug_info"] = {}
    state["debug_info"].update(debug_info)
    return state


def build_agentic_retriever() -> StateGraph:
    """Build (minimal) LangGraph: classify → retrieve.

    We removed the refine stage because it only echoed the query. This keeps
    the graph lean and the agent_steps trace focused.
    """
    workflow = StateGraph(RAGState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", END)
    return workflow.compile()



def _heuristic_initial_k(query: str) -> int:
    """Conservative initial k with a bump for enumerative queries.

    For list-style questions (e.g., countries, steps), start a bit higher
    to avoid missing tail items in the first pass.
    """
    return 5


def _extract_aspects_basic(query: str, max_aspects: int = 10) -> list:
    """Coarse aspect extraction from the query for coverage estimation.

    Splits on commas and simple conjunctions, removes stopwords, and truncates
    to short phrases. Not meant to be perfect—just enough to gauge coverage.
    """
    import re as _re
    cleaned = _re.sub(r"[?;:.]", "", query)
    parts = _re.split(r",| and | & |/|\\n", cleaned)
    aspects = []
    stop = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "with", "on"}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        tokens = [t for t in p.split() if t.lower() not in stop]
        if not tokens:
            continue
        phrase = " ".join(tokens[:6])
        if 2 <= len(phrase) <= 80 and phrase.lower() not in [a.lower() for a in aspects]:
            aspects.append(phrase)
        if len(aspects) >= max_aspects:
            break
    return aspects


def _enumerative_flag(query: str) -> bool:
    """Heuristic: does the query look enumerative (lists, comparisons, steps)?"""
    q = query.lower()
    enumerative_terms = ["list", "all", "countries", "steps", "obligations", "compare", "supported", "requirements", "deadlines"]
    return any(t in q for t in enumerative_terms) or q.count(',') >= 3


def _embedding_sim(vec_a, vec_b):
    """Cosine similarity between two vectors (lists/ndarrays)."""
    import numpy as _np
    va = _np.array(vec_a, dtype="float32"); vb = _np.array(vec_b, dtype="float32")
    # cosine similarity
    denom = ( (va**2).sum() ** 0.5 ) * ( (vb**2).sum() ** 0.5 )
    if denom == 0:
        return 0.0
    return float( (va * vb).sum() / denom )


def _compute_coverage(aspects: list, chunks: list, threshold: float = 0.55) -> float:
    """Embedding-based coverage of aspects by retrieved chunk texts.

    Each aspect is "covered" if ANY chunk reaches similarity >= threshold.
    If there are no aspects, return full coverage (1.0).
    """
    if not aspects:
        return 1.0
    # Precompute chunk embeddings once
    chunk_embs = []
    for c in chunks:
        text = c.get("text", "")[:400]
        try:
            emb = embed_model.get_text_embedding(text)
        except Exception:
            emb = [0.0]
        chunk_embs.append(emb)

    covered = 0
    for a in aspects:
        try:
            a_emb = embed_model.get_text_embedding(a)
        except Exception:
            a_emb = [0.0]
        max_sim = 0.0
        for ce in chunk_embs:
            sim = _embedding_sim(a_emb, ce)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= threshold:
                break
        if max_sim >= threshold:
            covered += 1
    return covered / len(aspects)


def search_agentic(
    query: str,
    k: int = 5,
    use_reranker: bool = False,
    adaptive: bool = True,
    coverage_target: float = 0.8,
    k_cap: int = 25,
    max_expansions: int = 3
) -> tuple:
    """Agentic retrieval with conservative progressive expansion.

    Pipeline:
      1) Build a LangGraph (classify → refine → retrieve) and run with k=5
      2) Compute embedding-based coverage of coarse aspects
      3) If coverage < target, expand k by a small delta (2 or 3 if enumerative)
      4) Repeat until coverage target reached OR expansions/cap exhausted
    Returns: (retrieved_chunks, query_type, debug_info, agent_steps)
    """
    if not adaptive:
        k = max(1, k)

    base_k = _heuristic_initial_k(query) if adaptive else k
    internal_k = base_k

    def _run(current_k: int):
        app = build_agentic_retriever()
        state = {
            "query": query,
            "query_type": "",
            "retrieved_chunks": [],
            "debug_info": {},
            "agent_steps": [],
            "k": current_k,
            "use_reranker": use_reranker
        }
        return app.invoke(state)

    expansions = 0
    aspects = _extract_aspects_basic(query)
    state = _run(internal_k)
    coverage = _compute_coverage(aspects, state["retrieved_chunks"])
    enumerative = _enumerative_flag(query)

    state["debug_info"].update({
        "initial_k": internal_k,
        "final_k": internal_k,
        "adaptive": adaptive,
        "coverage": round(coverage, 3),
        "coverage_target": coverage_target,
        "expansions": expansions,
        "aspects_count": len(aspects),
        "coverage_method": "embedding",
        "enumerative": enumerative
    })
    state["agent_steps"].append(f"✓ Coverage {coverage:.2f} k={internal_k}")

    while adaptive and coverage < coverage_target and expansions < max_expansions and internal_k < k_cap:
        # Expand more aggressively for enumerative queries to capture long lists
        delta = 4 if enumerative else 2
        new_k = min(internal_k + delta, k_cap)
        if new_k == internal_k:
            break
        expansions += 1
        expanded = _run(new_k)
        new_coverage = _compute_coverage(aspects, expanded["retrieved_chunks"])
        expanded["debug_info"].update({
            "initial_k": base_k,
            "final_k": new_k,
            "adaptive": adaptive,
            "coverage_before": round(coverage, 3),
            "coverage": round(new_coverage, 3),
            "coverage_target": coverage_target,
            "expansions": expansions,
            "aspects_count": len(aspects),
            "coverage_method": "embedding",
            "enumerative": enumerative
        })
        expanded["agent_steps"].extend(state["agent_steps"])  # retain history
        expanded["agent_steps"].append(f"↻ Expand {internal_k}->{new_k} cov {coverage:.2f}->{new_coverage:.2f}")
        state = expanded
        internal_k = new_k
        coverage = new_coverage
        if coverage >= coverage_target:
            state["agent_steps"].append("✓ Coverage target reached")
            break

    # Ensure reported final_k
    state["debug_info"]["final_k"] = internal_k

    return (
        state["retrieved_chunks"],
        state["query_type"],
        state["debug_info"],
        state["agent_steps"]
    )


# Initialize the embedding model (can be reused across queries)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# Load all indices at module level for reuse
all_indices = load_all_indices()
