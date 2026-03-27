"""
ingestion_graph.py — LangGraph pipeline for document ingestion.

Flow:
  START → detect_type → chunk_document → embed_chunks → store_vectors → END
               ↓ (error)       ↓ (error)       ↓ (error)       ↓ (error)
             on_error        on_error         on_error        on_error

State keys mutated at each node are returned as partial dicts — LangGraph
merges them into the running state automatically.
"""
from __future__ import annotations

from typing import TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

logger = structlog.get_logger()


# ── State ─────────────────────────────────────────────────────────────────────


class IngestionState(TypedDict, total=False):
    # ── Inputs (caller sets these) ──
    app_id: str           # from JWT — determines table name prefix
    tenant_id: str        # from JWT — stored as row column for row-level isolation
    document_name: str    # original filename, used to derive the table name
    document_bytes: bytes # raw file content

    # ── User-upload scope (set when file comes from messages endpoint) ──
    user_id: str | None           # if set → store in user_docs schema
    conversation_id: str | None   # scope user docs to this conversation

    # ── Resolved by store_vectors node ──
    schema_name: str      # e.g. "crm_app"       ← PostgreSQL schema = app isolation
    table_name: str       # e.g. "policy_vector"  ← table = document

    # ── Set by detect_type node ──
    doc_type: str              # "pdf" | "txt" | "csv" | "json" | "db_rows"
    embedding_model: str       # "all-mpnet-base-v2" | "all-MiniLM-L6-v2"
    chunk_strategy: str        # "semantic" | "sliding_window" | "row" | "record"

    # ── Set by chunk_document node ──
    chunks: list[str]
    chunk_metadatas: list[dict]

    # ── Set by embed_chunks node ──
    vectors: list[list[float]]

    # ── Set by store_vectors node ──
    total_chunks: int

    # ── Error flag (any node can set this to short-circuit to on_error) ──
    error: str | None


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def detect_type(state: IngestionState) -> dict:
    """Classify doc type and select embedding model + chunking strategy."""
    from rag.document_type_detector import detect_doc_type

    try:
        result = detect_doc_type(
            filename=state["document_name"],
            content=state["document_bytes"],
        )
        logger.info(
            "ingestion_detected",
            name=state["document_name"],
            doc_type=result.doc_type,
            model=result.embedding_model,
            strategy=result.chunk_strategy,
        )
        return {
            "doc_type": result.doc_type,
            "embedding_model": result.embedding_model,
            "chunk_strategy": result.chunk_strategy,
            "error": None,
        }
    except Exception as exc:
        logger.error("ingestion_detect_failed", error=str(exc))
        return {"error": f"Type detection failed: {exc}"}


async def chunk_document(state: IngestionState) -> dict:
    """Split document bytes into text chunks using the selected strategy."""
    if state.get("error"):
        return {}

    from rag.chunkers import Chunk, get_chunker

    doc_type = state.get("doc_type", "txt")
    chunker = get_chunker(doc_type)

    try:
        raw_chunks: list[Chunk] = chunker.chunk(
            state["document_bytes"], source=state["document_name"]
        )
    except Exception as exc:
        logger.error("ingestion_chunk_failed", doc_type=doc_type, error=str(exc))
        return {"error": f"Chunking failed: {exc}", "chunks": [], "chunk_metadatas": []}

    if not raw_chunks:
        return {
            "error": "No content could be extracted from the document.",
            "chunks": [],
            "chunk_metadatas": [],
        }

    return {
        "chunks": [c.text for c in raw_chunks],
        "chunk_metadatas": [c.metadata for c in raw_chunks],
    }


async def embed_chunks(state: IngestionState) -> dict:
    """
    Embed all chunks. Uses Gemini text-embedding-004 for user-uploaded docs
    (when user_id is set), sentence-transformers for app-level RAG docs.
    """
    if state.get("error"):
        return {}

    chunks = state.get("chunks", [])
    if not chunks:
        return {"error": "No chunks to embed.", "vectors": []}

    # User-uploaded docs → Gemini embeddings (single unified model)
    if state.get("user_id"):
        from rag.embedder import GEMINI_EMBED_MODEL, embed_texts_gemini

        try:
            vectors_np = await embed_texts_gemini(chunks)
            logger.info("ingestion_embedded_gemini", count=len(chunks))
            return {"vectors": vectors_np.tolist(), "embedding_model": GEMINI_EMBED_MODEL}
        except Exception as exc:
            logger.error("ingestion_embed_gemini_failed", error=str(exc))
            return {"error": f"Gemini embedding failed: {exc}", "vectors": []}

    # App-level RAG docs → sentence-transformers (existing behaviour)
    from rag.embedder import embed_texts

    model_name = state.get("embedding_model", "all-mpnet-base-v2")
    try:
        vectors_np = await embed_texts(chunks, model_name)
        logger.info("ingestion_embedded", model=model_name, count=len(chunks))
        return {"vectors": vectors_np.tolist()}
    except Exception as exc:
        logger.error("ingestion_embed_failed", model=model_name, error=str(exc))
        return {"error": f"Embedding failed: {exc}", "vectors": []}


async def store_vectors(state: IngestionState) -> dict:
    """Persist embedded chunks into pgvector (rag_embeddings table)."""
    if state.get("error"):
        return {}

    from rag.vector_store import PgVectorStore

    chunks = state.get("chunks", [])
    vectors = state.get("vectors", [])
    if not chunks or not vectors:
        return {"error": "Nothing to store.", "total_chunks": 0}

    try:
        store = PgVectorStore()

        # User-uploaded docs → store in user_docs schema scoped to conversation
        if state.get("user_id") and state.get("conversation_id"):
            schema, tbl, total = await store.add_user_doc(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                tenant_id=state["tenant_id"],
                filename=state["document_name"],
                texts=chunks,
                vectors=vectors,
                metadatas=state.get("chunk_metadatas"),
            )
            logger.info(
                "ingestion_stored_user_doc",
                schema=schema, table=tbl,
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                total=total,
            )
            return {"schema_name": schema, "table_name": tbl, "total_chunks": total}

        # App-level RAG → store in app schema (existing behaviour)
        schema, tbl, total = await store.add(
            app_id=state["app_id"],
            tenant_id=state["tenant_id"],
            filename=state["document_name"],
            texts=chunks,
            vectors=vectors,
            metadatas=state.get("chunk_metadatas"),
            embedding_model=state.get("embedding_model", "all-mpnet-base-v2"),
        )
        logger.info(
            "ingestion_stored",
            schema=schema, table=tbl, tenant_id=state["tenant_id"], total=total,
        )
        return {"schema_name": schema, "table_name": tbl, "total_chunks": total}
    except Exception as exc:
        logger.error("ingestion_store_failed", error=str(exc))
        return {"error": f"Vector store failed: {exc}", "total_chunks": 0}


async def on_error(state: IngestionState) -> dict:
    """Terminal error handler — logs and preserves error message."""
    logger.error(
        "ingestion_pipeline_error",
        error=state.get("error"),
        namespace=state.get("namespace"),
    )
    return {}


# ── Conditional routing ───────────────────────────────────────────────────────


def _route_or_error(next_node: str):
    """Return a conditional-edge function that goes to next_node or on_error."""

    def _decide(state: IngestionState) -> str:
        return "on_error" if state.get("error") else next_node

    return _decide


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_ingestion_graph():
    graph: StateGraph = StateGraph(IngestionState)

    graph.add_node("detect_type", detect_type)
    graph.add_node("chunk_document", chunk_document)
    graph.add_node("embed_chunks", embed_chunks)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("on_error", on_error)

    graph.add_edge(START, "detect_type")
    graph.add_conditional_edges("detect_type", _route_or_error("chunk_document"))
    graph.add_conditional_edges("chunk_document", _route_or_error("embed_chunks"))
    graph.add_conditional_edges("embed_chunks", _route_or_error("store_vectors"))
    graph.add_conditional_edges("store_vectors", _route_or_error(END))
    graph.add_edge("on_error", END)

    return graph.compile()


# Singleton — compiled once at import time
ingestion_pipeline = build_ingestion_graph()
