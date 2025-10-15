import numpy as np
from sentence_transformers import CrossEncoder
import pickle
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class RerankerRetriever:
    """Enhanced retriever with cross-encoder reranking for better relevance."""
    
    def __init__(self, index_path="emd_out_retr_in"):
        self.index_path = index_path
        self.index, self.metadata = self._load_index()
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize cross-encoder for reranking
        print("🔄 Loading cross-encoder reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Reranker loaded!")
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        index = faiss.read_index(f"{self.index_path}/faiss.index")
        with open(f"{self.index_path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    
    def retrieve_with_reranking(self, query, k=5, initial_k=20):
        """
        Two-stage retrieval: FAISS + Cross-encoder reranking
        
        Args:
            query: Search query
            k: Final number of results to return
            initial_k: Initial candidates from FAISS (should be > k)
        """
        # Stage 1: Get more candidates from FAISS
        initial_results = self._faiss_search(query, k=initial_k)
        
        if len(initial_results) <= k:
            return initial_results
        
        # Stage 2: Rerank with cross-encoder
        reranked_results = self._rerank_results(query, initial_results, k=k)
        
        return reranked_results
    
    def _faiss_search(self, query, k=20):
        """Stage 1: FAISS semantic search."""
        query_emb = np.array([self.embed_model.get_text_embedding(query)]).astype("float32")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_emb)
        
        D, I = self.index.search(query_emb, k)
        
        results = []
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            
            meta = self.metadata[idx]
            result = {
                "rank": rank + 1,
                "faiss_score": float(D[0][rank]),
                "text": meta["text"],
                "file_name": meta.get("file_name", "unknown"),
                "chunk_id": meta.get("chunk_id", 0),
                "strategy": meta.get("strategy", "unknown")
            }
            results.append(result)
        
        return results
    
    def _rerank_results(self, query, candidates, k=5):
        """Stage 2: Cross-encoder reranking."""
        # Prepare query-document pairs for reranking
        query_doc_pairs = []
        for candidate in candidates:
            # Truncate text to avoid token limits
            text_preview = candidate['text'][:512]  # First 512 chars
            query_doc_pairs.append([query, text_preview])
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Add rerank scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(rerank_scores[i])
            candidate['original_rank'] = candidate['rank']
        
        # Sort by rerank score (higher = more relevant)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks and return top-k
        for i, result in enumerate(reranked[:k]):
            result['rank'] = i + 1
            result['score'] = result['rerank_score']  # Use rerank score as final score
        
        return reranked[:k]
    
    def retrieve_hybrid(self, query, k=5, alpha=0.7):
        """
        Hybrid approach: Combine FAISS and rerank scores
        
        Args:
            alpha: Weight for rerank score (0.7 = 70% rerank, 30% FAISS)
        """
        initial_results = self._faiss_search(query, k=k*3)  # Get 3x candidates
        reranked = self._rerank_results(query, initial_results, k=len(initial_results))
        
        # Hybrid scoring: combine FAISS and rerank scores
        for result in reranked:
            # Normalize scores to 0-1 range
            faiss_norm = 1.0 / (1.0 + abs(result['faiss_score']))  # Convert distance to similarity
            rerank_norm = (result['rerank_score'] + 1) / 2  # Convert to 0-1 range
            
            # Weighted combination
            result['hybrid_score'] = alpha * rerank_norm + (1 - alpha) * faiss_norm
            result['score'] = result['hybrid_score']
        
        # Sort by hybrid score and return top-k
        final_results = sorted(reranked, key=lambda x: x['hybrid_score'], reverse=True)
        
        for i, result in enumerate(final_results[:k]):
            result['rank'] = i + 1
        
        return final_results[:k]


# Wrapper functions for backward compatibility
def load_index():
    """Load FAISS index and metadata from the fixed index directory."""
    index = faiss.read_index("emd_out_retr_in/faiss.index")
    with open("emd_out_retr_in/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search_with_reranking(query, k=5, method="rerank"):
    """
    Enhanced search with reranking options.
    
    Methods:
    - "rerank": Two-stage FAISS + Cross-encoder
    - "hybrid": Weighted combination of scores
    - "faiss": Original FAISS only
    """
    retriever = RerankerRetriever()
    
    if method == "rerank":
        return retriever.retrieve_with_reranking(query, k=k, initial_k=k*4)
    elif method == "hybrid":
        return retriever.retrieve_hybrid(query, k=k)
    else:  # faiss
        return retriever._faiss_search(query, k=k)


# For testing
if __name__ == "__main__":
    print("🧪 Testing reranker retrieval...")
    
    # Test queries
    test_queries = [
        "VAT Mapping update Aug 2022 2023 changes Import Services Triangulation boxes",
        "How to upload files in Comply system",
        "Estonia VAT return processing steps"
    ]
    
    retriever = RerankerRetriever()
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("=" * 60)
        
        # Test different methods
        methods = ["faiss", "rerank", "hybrid"]
        
        for method in methods:
            print(f"\n📊 Method: {method.upper()}")
            if method == "faiss":
                results = retriever._faiss_search(query, k=5)
            elif method == "rerank":
                results = retriever.retrieve_with_reranking(query, k=5)
            else:
                results = retriever.retrieve_hybrid(query, k=5)
            
            for i, result in enumerate(results[:3], 1):
                score_type = "faiss_score" if method == "faiss" else "score"
                score = result.get(score_type, result.get('rerank_score', 0))
                
                print(f"  {i}. Score: {score:.3f} | {result['file_name']}")
                print(f"     Text: {result['text'][:100]}...")