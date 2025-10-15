from retriver import search_agentic


# ---------- Generator ----------
def generate_answer(query, model, k=5, use_reranker=False, adaptive=True):
    """Generate answer using LanGraph agentic workflow with dual-index routing.

    Args:
        query: user question
        model: LLM client
        k: top-k final chunks after (optional) reranking
        use_reranker: whether to apply cross-encoder reranking
    """

    retrieved_chunks, query_type, debug_info, agent_steps = search_agentic(
        query,
        k=k,
        use_reranker=use_reranker,
        adaptive=adaptive
    )

    if retrieved_chunks:
        retrieved_chunks[0]["_query_classification"] = query_type
        retrieved_chunks[0]["_searched_indices"] = debug_info.get("searched_indices", [])
        if use_reranker:
            retrieved_chunks[0]["_reranker"] = True

    context = ""
    citation_map = {}
    for i, ch in enumerate(retrieved_chunks, start=1):
        anchor = f"[{i}]"
        if ch.get("type") == "table" and ch.get("table_title"):
            snippet = f"TABLE: {ch['table_title']}\n{ch['text']}"
        else:
            snippet = ch.get("text", "").replace("\n", " ")
        context += f"{anchor} {snippet}\n"
        citation_map[i] = ch

    prompt = f"""Answer the query ONLY using the context. Include ALL items if it's a list.

Query: {query}

Context:
{context}

Rules:
- If the context lists countries/steps/items you MUST enumerate every single one without omission.
- Use inline citations [1], [2], etc. immediately after each factual span or group.
- Do not invent items not present in context.
- If insufficient info, explicitly say so and cite what you do have.

Answer:"""

    # Keep a high output budget so long enumerations (e.g., country lists) don't truncate
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "stop_sequences": []
        }
    )

    return response.text, citation_map, query_type, agent_steps, debug_info
