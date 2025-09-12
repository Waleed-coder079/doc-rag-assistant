import os
import time
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai


from generator import load_index
from generator import search
from generator import generate_answer


# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.title("📚 Retrieval-Augmented Generation (RAG) Demo")

    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in .env file.")
        return
    genai.configure(api_key=api_key)

    # Hidden system configs
    index_dir = "emd_out_retr_in"   # fixed directory
    model_name = "gemini-1.5-flash" # fixed model

    # Load index + embeddings once
    if "index" not in st.session_state:
        with st.spinner("Loading index..."):
            st.session_state.index, st.session_state.metadata = load_index()
            st.session_state.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    query = st.text_input("🔍 Enter your query:")

    # Generate button in main area
    generate = st.button("Generate Answer")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    k = st.sidebar.slider("Top-k retrieved chunks", 1, 10, 5)

    if generate and query:
        start_time = time.time()

        # Retrieve
        retrieved = search(query, st.session_state.index, st.session_state.metadata,
                           st.session_state.embed_model, k=k)

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
                snippet = meta['text'][:400].replace("\n", " ")

                # Build clickable local file link
                source_path = meta['source']
                file_path = os.path.join("ingestion_input", os.path.basename(source_path))
                file_url = f"file://{os.path.abspath(file_path)}"

                anchor = f"page={meta.get('page')}, para={meta.get('paragraph_id')}"
                st.markdown(f"[Open Document]({file_url})  \n"
                            f"*{anchor}*  \n\n"
                            f"> {snippet}...")

        # Observability
        with st.expander("🔎 Debug Info"):
            st.json({
                "query": query,
                "top_k": k,
                "retrieved_count": len(retrieved),
                "latency_sec": round(latency, 2),
                "retrieved_chunks": [r["text"][:100] for r in retrieved]
            })


if __name__ == "__main__":
    main()
