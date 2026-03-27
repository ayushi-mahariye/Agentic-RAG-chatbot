"""
query_graph.py — LangGraph pipeline for RAG query + answer generation.

Flow:
  START → embed_query → retrieve → grade_chunks → generate_answer → validate → END
                                        ↓ (no relevant chunks)
                                     no_answer → END

  If validate decides answer is not grounded AND retry_count < 2:
    → retry_expand → retrieve  (top_k grows by +3 on each retry)

  Otherwise:
    → END
"""
from __future__ import annotations

import time
from typing import TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

logger = structlog.get_logger()


# ── State ─────────────────────────────────────────────────────────────────────


class QueryState(TypedDict, total=False):
    # ── Inputs ──
    app_id: str        # from JWT — for logging
    tenant_id: str     # from JWT — WHERE tenant_id = :tid inside each table
    schema_name: str   # PostgreSQL schema = app isolation  e.g. "crm_app"
    # table_name is NOT required — retrieve searches all tables in the schema
    query: str
    retrieval_query: str      # English translation of query (for non-ASCII queries)
    top_k: int
    embedding_model: str      # selected by caller based on doc_type_hint or default

    # ── User-doc retrieval (set when called from messages endpoint) ──
    user_id: str | None           # if set → also search user_docs schema
    conversation_id: str | None   # scope user doc search to this conversation

    # ── History & language ──
    conversation_history: list[dict]   # [{"role": "user"/"assistant", "content": "..."}]
    language: str                      # "auto" | "en" | "hinglish" | "punjabi" | "marathi"

    # ── Retrieval ──
    query_vector: list[float]
    retrieved: list[dict]     # [{text, metadata, score}]
    graded: list[dict]        # subset after relevance grading
    grade_passed: bool

    # ── Generation ──
    answer: str
    citations: list[dict]
    is_grounded: bool

    # ── Control ──
    retry_count: int
    error: str | None

    # ── Timings (ms) ──
    embed_ms: int
    retrieval_ms: int
    grading_ms: int
    generation_ms: int


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def embed_query(state: QueryState) -> dict:
    """
    Embed the user query. Uses Gemini text-embedding-004 when user_id is set
    (user-doc retrieval path), sentence-transformers otherwise (app RAG path).
    Non-ASCII queries are translated to English first.
    """
    from rag.llm_clients import translate_query_for_retrieval

    t0 = time.monotonic()
    try:
        retrieval_query = await translate_query_for_retrieval(state["query"])

        if state.get("user_id"):
            from rag.embedder import embed_query_gemini
            vec = await embed_query_gemini(retrieval_query)
        else:
            from rag.embedder import embed_query as _embed
            model_name = state.get("embedding_model", "all-mpnet-base-v2")
            vec = await _embed(retrieval_query, model_name)

        ms = int((time.monotonic() - t0) * 1000)
        logger.info("query_embed_done", embed_ms=ms)
        return {
            "query_vector": vec.tolist(),
            "retrieval_query": retrieval_query,
            "embed_ms": ms,
            "error": None,
        }
    except Exception as exc:
        logger.error("query_embed_failed", error=str(exc))
        return {"error": f"Query embedding failed: {exc}", "embed_ms": 0}


async def retrieve(state: QueryState) -> dict:
    """Retrieve top-k candidate chunks from pgvector."""
    if state.get("error"):
        return {}

    from rag.vector_store import PgVectorStore

    t0 = time.monotonic()
    try:
        store = PgVectorStore()
        top_k = state.get("top_k", 5)
        combined: list[dict] = []

        # App-level RAG retrieval (sentence-transformers)
        app_results = await store.search_schema(
            schema=state["schema_name"],
            tenant_id=state["tenant_id"],
            query_vector=state["query_vector"],
            top_k=top_k,
            embedding_model=state.get("embedding_model", "all-mpnet-base-v2"),
        )
        for r in app_results:
            r["source"] = "app_rag"
        combined.extend(app_results)

        # User-doc retrieval (Gemini embeddings) — only when called from messages endpoint
        if state.get("user_id") and state.get("conversation_id"):
            user_results = await store.search_user_docs(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                tenant_id=state["tenant_id"],
                query_vector=state["query_vector"],
                top_k=top_k,
            )
            combined.extend(user_results)

        # Re-rank merged results by score, keep global top_k
        combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        results = combined[:top_k]

        ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "query_retrieved",
            schema=state["schema_name"], tenant_id=state["tenant_id"],
            found=len(results), retrieval_ms=ms,
        )
        return {"retrieved": results, "retrieval_ms": ms}
    except Exception as exc:
        ms = int((time.monotonic() - t0) * 1000)
        logger.error("query_retrieve_failed", error=str(exc), retrieval_ms=ms)
        return {"error": f"Retrieval failed: {exc}", "retrieved": [], "retrieval_ms": ms}


def _cosine_threshold() -> float:
    from config import settings
    return settings.rag_cosine_threshold


async def grade_chunks(state: QueryState) -> dict:
    """
    Filter retrieved chunks by cosine similarity score — no LLM call needed.
    Pgvector already computes score = 1 - cosine_distance during retrieval.
    Chunks above the threshold are already semantically relevant.
    """
    if state.get("error"):
        return {}

    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"graded": [], "grade_passed": False, "grading_ms": 0}

    t0 = time.monotonic()
    graded = [
        {**chunk, "relevance_score": chunk.get("score", 0.0)}
        for chunk in retrieved
        if chunk.get("score", 0.0) >= _cosine_threshold()
    ]
    graded.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    ms = int((time.monotonic() - t0) * 1000)
    logger.info("query_graded", total=len(retrieved), kept=len(graded), grading_ms=ms)
    return {"graded": graded, "grade_passed": len(graded) > 0, "grading_ms": ms}


async def generate_answer(state: QueryState) -> dict:
    """
    Generate a cited answer from the graded context chunks using GPT-4o.
    """
    if state.get("error"):
        return {}

    from rag.llm_clients import generate_with_context

    graded = state.get("graded", [])
    if not graded:
        return {
            "answer": "I could not find relevant information to answer your question.",
            "citations": [],
            "is_grounded": False,
        }

    t0 = time.monotonic()
    try:
        original_query = state["query"]
        gen_query = state.get("retrieval_query") or original_query
        answer = await generate_with_context(
            query=gen_query,
            context_chunks=graded,
            history=state.get("conversation_history", []),
            language=state.get("language", "auto"),
            original_query=original_query if gen_query != original_query else None,
        )
        ms = int((time.monotonic() - t0) * 1000)
        citations = [
            {
                "index": i + 1,
                "text": c["text"][:300],
                "metadata": c.get("metadata", {}),
                "score": c.get("relevance_score", 0),
            }
            for i, c in enumerate(graded)
        ]
        logger.info("query_generated", generation_ms=ms)
        return {"answer": answer, "citations": citations, "is_grounded": True, "generation_ms": ms}
    except Exception as exc:
        ms = int((time.monotonic() - t0) * 1000)
        logger.error("query_generate_failed", error=str(exc), generation_ms=ms)
        return {
            "answer": f"Generation failed: {exc}",
            "citations": [],
            "is_grounded": False,
            "generation_ms": ms,
        }


async def validate(state: QueryState) -> dict:
    """
    Lightweight grounding check: ask GPT-4o-mini if the answer is fully
    supported by the retrieved context.  Sets is_grounded accordingly.
    """
    graded = state.get("graded", [])
    answer = state.get("answer", "")

    if not graded or not answer:
        return {"is_grounded": False}

    from rag.llm_clients import check_grounding

    try:
        is_grounded = await check_grounding(
            context_chunks=graded,
            answer=answer,
        )
        logger.info("query_validated", is_grounded=is_grounded)
        return {"is_grounded": is_grounded}
    except Exception:
        # Assume grounded if validation itself errors out
        return {"is_grounded": True}


async def no_answer(state: QueryState) -> dict:
    """Terminal node when no relevant chunks were found."""
    logger.warning("query_no_relevant_chunks", schema=state.get("schema_name"))
    return {
        "answer": (
            "I couldn't find relevant information in the knowledge base for your query. "
            "Try rephrasing or ingesting more documents into this namespace."
        ),
        "citations": [],
        "is_grounded": False,
    }


async def retry_expand(state: QueryState) -> dict:
    """Expand the search before retrying retrieval."""
    new_top_k = state.get("top_k", 5) + 3
    new_retry = state.get("retry_count", 0) + 1
    logger.info("query_retrying", retry=new_retry, new_top_k=new_top_k)
    return {"top_k": new_top_k, "retry_count": new_retry}


# ── Conditional routing ───────────────────────────────────────────────────────


def _after_grade(state: QueryState) -> str:
    if state.get("error"):
        return END
    return "generate_answer" if state.get("grade_passed") else "no_answer"


def _after_validate(state: QueryState) -> str:
    if state.get("is_grounded", True):
        return END
    if state.get("retry_count", 0) < 2:
        return "retry_expand"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_query_graph():
    graph: StateGraph = StateGraph(QueryState)

    graph.add_node("embed_query", embed_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_chunks", grade_chunks)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate", validate)
    graph.add_node("no_answer", no_answer)
    graph.add_node("retry_expand", retry_expand)

    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges("grade_chunks", _after_grade)
    graph.add_edge("generate_answer", "validate")
    graph.add_conditional_edges("validate", _after_validate)
    graph.add_edge("no_answer", END)
    graph.add_edge("retry_expand", "retrieve")

    return graph.compile()


# Singleton — compiled once at import time
query_pipeline = build_query_graph()
