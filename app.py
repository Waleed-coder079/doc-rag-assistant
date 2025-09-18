import os
import time
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai

from generator import load_index, search, generate_answer


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

    # Hidden system configs
    model_name = "gemini-1.5-flash"  # fixed model

    # Load index + embeddings once
    if "index" not in st.session_state:
        with st.spinner("Loading index..."):
            st.session_state.index, st.session_state.metadata = load_index()
            st.session_state.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    # User input
    query = st.text_input("🔍 Enter your query:")

    # Generate button in main area
    generate = st.button("Generate Answer")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    k = st.sidebar.slider("Top-k retrieved chunks", 1, 10, 5)

    if generate and query:
        start_time = time.time()

        # Retrieve
        retrieved = search(
            query,
            st.session_state.index,
            st.session_state.metadata,
            st.session_state.embed_model,
            k=k,
        )

        # Generate
        answer, citation_map = generate_answer(query, retrieved, model_name=model_name)
        latency = time.time() - start_time

        # Display main answer
        st.subheader("📝 Answer")
        st.write(answer)

        # Sources in sidebar
        st.sidebar.header("📚 Sources")
        for i, meta in citation_map.items():
            with st.sidebar.expander(f"Source [{i}] - {meta.get('title', 'Untitled')}"):
                snippet = meta["text"][:400].replace("\n", " ")

                # ✅ Prefer original URL if available, else fallback to local file
                source_url = meta.get("source_url") or meta.get("source")

                anchor = []
                if meta.get("page"):
                    anchor.append(f"page={meta['page']}")
                if meta.get("paragraph_id"):
                    anchor.append(f"para={meta['paragraph_id']}")
                # Table metadata
                if meta.get("strategy") == "table_whole":
                    if meta.get("type"): anchor.append(f"type={meta['type']}")
                    if meta.get("table_index") is not None: anchor.append(f"table_index={meta['table_index']}")
                    if meta.get("section"): anchor.append(f"section={meta['section']}")
                    if meta.get("pages"): anchor.append(f"pages={meta['pages']}")
                anchor_text = ", ".join(anchor) if anchor else "no position info"

                st.markdown(
                    f"[Open Document]({source_url})  \n"
                    f"*{anchor_text}*  \n\n"
                    f"> {snippet}..."
                )

        # Observability
        with st.expander("🔎 Debug Info"):
            st.json(
                {
                    "query": query,
                    "top_k": k,
                    "retrieved_count": len(retrieved),
                    "latency_sec": round(latency, 2),
                    "retrieved_chunks": [
                        {
                            "text": r["text"][:100],
                            "source": r.get("source"),
                            "title": r.get("title"),
                            "strategy": r.get("strategy"),
                            "type": r.get("type"),
                            "table_index": r.get("table_index"),
                            "section": r.get("section"),
                            "pages": r.get("pages"),
                        }
                        for r in retrieved
                    ],
                }
            )


if __name__ == "__main__":
    main()
