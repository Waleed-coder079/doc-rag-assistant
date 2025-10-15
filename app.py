import os
import time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

from generator import generate_answer
from retriver import load_index, load_all_indices


# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.title("📚 RAG docs chat")

    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in .env file.")
        return
    genai.configure(api_key=api_key)

    # Create Gemini model (generation config is set in generator.py)
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    # Load index + embeddings once
    if "indices" not in st.session_state:
        with st.spinner("Loading indices..."):
            # Load both text and table indices
            st.session_state.indices = load_all_indices()
            
            # For backward compatibility, also load default text index
            st.session_state.index, st.session_state.metadata = load_index()

    # User input
    query = st.text_input("🔍 Enter your query:")

    # Generate button in main area
    generate = st.button("Generate Answer")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
        # k removed from UI (adaptive internal). Base starting point for adaptive logic:
    base_k = 5
    use_reranker = st.sidebar.checkbox("Use Cross-Encoder Reranker", value=False, help="Improves relevance (slower)")
    
    st.sidebar.divider()
    st.sidebar.info("🤖 **Agentic Mode Active**\\n\\nUsing LanGraph workflow with dual-index smart routing")
    st.sidebar.caption("� Automatically routes queries to Text/Table indices")

    if generate and query:
        start_time = time.time()

        # Generate answer using agentic workflow (only mode)
        answer, citation_map, detected_query_type, agent_steps, debug_info = generate_answer(
            query,
            model=model,
                k=base_k,
            use_reranker=use_reranker,
            adaptive=True
        )
        latency = time.time() - start_time

        # Display main answer
        st.subheader("📝 Answer")
        
        # Show query classification
        classification_emoji = {
            "table": "📊",
            "text": "📝", 
            "hybrid": "🔀"
        }
        emoji = classification_emoji.get(detected_query_type, "❓")
        
        # Get searched indices from first chunk if available
        searched_indices = []
        if citation_map and 1 in citation_map:
            searched_indices = citation_map[1].get("_searched_indices", [])

        rerank_badge = " | 🔄 Reranked" if use_reranker else ""
        st.info(f"{emoji} **Query Type:** {detected_query_type.upper()} | **Searched:** {', '.join(searched_indices) if searched_indices else 'N/A'}{rerank_badge}")
        
        # Show agent workflow steps (always available in agentic mode)
        with st.expander("🤖 Agent Workflow Steps", expanded=False):
            for step in agent_steps:
                st.text(step)
        
        st.write(answer)

        # Sources in sidebar
        st.sidebar.header("📚 Sources")
        for i, meta in citation_map.items():
            with st.sidebar.expander(f"Source [{i}] - {meta.get('file_name', 'Unknown')}"):
                snippet = meta["text"][:400].replace("\n", " ")

                # Build metadata display
                info_parts = []
                info_parts.append(f"file: {meta.get('file_name', 'unknown')}")
                info_parts.append(f"chunk: {meta.get('chunk_id', 0)}")
                info_parts.append(f"strategy: {meta.get('strategy', 'unknown')}")
                
                info_text = ", ".join(info_parts)

                st.markdown(
                    f"*{info_text}*  \n\n"
                    f"> {snippet}..."
                )

        # Observability
        with st.expander("🔎 Debug Info"):
            # Get searched indices from first chunk
            searched_indices = []
            if citation_map and 1 in citation_map:
                searched_indices = citation_map[1].get("_searched_indices", [])
            
            # Pull adaptive metrics if available
            internal_initial_k = debug_info.get("initial_k")
            internal_final_k = debug_info.get("final_k", internal_initial_k)
            coverage = debug_info.get("coverage")
            expansions = debug_info.get("expansions")

            debug_data = {
                "query": query,
                "query_type": detected_query_type,
                "searched_indices": searched_indices,
                "workflow": "LanGraph Agentic",
                "reranker": use_reranker,
                    "base_k_start": base_k,
                "initial_k_internal": internal_initial_k,
                "final_k_internal": internal_final_k,
                "retrieved_count": len(citation_map),
                "coverage": coverage,
                "expansions": expansions,
                "agent_steps": agent_steps,
                "retrieved_chunks": [
                    {
                        "text": ch["text"][:100],
                        "file_name": ch.get("file_name"),
                        "chunk_id": ch.get("chunk_id"),
                        "strategy": ch.get("strategy"),
                        "type": ch.get("type"),
                        "source_index": ch.get("source_index", "N/A")
                    }
                    for ch in citation_map.values()
                ],
            }
            
            st.json(debug_data)

if __name__ == "__main__":
    main()
