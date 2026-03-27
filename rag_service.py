"""
rag_service.py — top-level orchestrator for RAG operations.

Mirrors SqlPipeline in structure: no FastAPI coupling, accepts plain
Python types, returns typed result dataclasses.  Langfuse spans wrap
each pipeline call following the langfuse_config.py pattern.

Public API:
  service = RagService()
  result  = await service.ingest(namespace, file_bytes, filename, mime_type)
  result  = await service.query(namespace, query, top_k, doc_type_hint)
  deleted = await service.delete_namespace(namespace)
  count   = await service.doc_count(namespace)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass(slots=True)
class IngestionResult:
    schema_name: str      # PostgreSQL schema = app  e.g. "crm_app"
    table_name: str       # document table            e.g. "policy_vector"
    document_name: str
    doc_type: str
    embedding_model: str
    chunk_strategy: str
    chunks_stored: int
    status: str              # "ok" | "error"
    error: str | None = None


@dataclass(slots=True)
class QueryResult:
    schema_name: str
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    chunks_retrieved: int = 0
    chunks_used: int = 0
    is_grounded: bool = False
    embedding_model: str = ""
    language: str = "auto"
    conversation_id: str | None = None
    # Timings (milliseconds)
    embed_ms: int = 0
    retrieval_ms: int = 0
    grading_ms: int = 0
    generation_ms: int = 0
    status: str = "ok"
    error: str | None = None


# ── Service ───────────────────────────────────────────────────────────────────


class RagService:
    """
    Orchestrates the ingestion and query LangGraph pipelines.
    Decodes raw bytes, delegates to the compiled graphs, and wraps results.
    """

    async def ingest(
        self,
        app_id: str,
        tenant_id: str,
        file_bytes: bytes,
        filename: str,
        mime_type: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> IngestionResult:
        from rag.ingestion_graph import IngestionState, ingestion_pipeline

        logger.info(
            "rag_ingest_start",
            app_id=app_id, tenant_id=tenant_id,
            filename=filename, size=len(file_bytes),
        )

        # Open Langfuse span if available
        span = self._start_span(
            "rag_ingest",
            {"app_id": app_id, "tenant_id": tenant_id, "filename": filename},
        )

        initial: IngestionState = {
            "app_id": app_id,
            "tenant_id": tenant_id,
            "document_name": filename,
            "document_bytes": file_bytes,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "doc_type": "",
            "embedding_model": "",
            "chunk_strategy": "",
            "chunks": [],
            "chunk_metadatas": [],
            "vectors": [],
            "total_chunks": 0,
            "error": None,
        }

        try:
            result = await ingestion_pipeline.ainvoke(initial)
        except Exception as exc:
            logger.error("rag_ingest_pipeline_failed", error=str(exc))
            self._end_span(span, error=str(exc))
            return IngestionResult(
                schema_name="",
                table_name="",
                document_name=filename,
                doc_type="",
                embedding_model="",
                chunk_strategy="",
                chunks_stored=0,
                status="error",
                error=str(exc),
            )

        self._end_span(span)
        error = result.get("error")
        return IngestionResult(
            schema_name=result.get("schema_name", ""),
            table_name=result.get("table_name", ""),
            document_name=filename,
            doc_type=result.get("doc_type", ""),
            embedding_model=result.get("embedding_model", ""),
            chunk_strategy=result.get("chunk_strategy", ""),
            chunks_stored=result.get("total_chunks", 0),
            status="error" if error else "ok",
            error=error,
        )

    async def query(
        self,
        app_id: str,
        tenant_id: str,
        schema_name: str,
        query_text: str,
        top_k: int = 5,
        doc_type_hint: str | None = None,
        conversation_id: str | None = None,
        language: str = "auto",
        user_id: str | None = None,
        db=None,
    ) -> QueryResult:
        from rag.embedder import model_for_doc_type
        from rag.query_graph import QueryState, query_pipeline

        embedding_model = model_for_doc_type(doc_type_hint or "txt")
        logger.info(
            "rag_query_start",
            app_id=app_id, tenant_id=tenant_id,
            schema=schema_name, model=embedding_model,
            conversation_id=conversation_id, language=language,
        )

        span = self._start_span("rag_query", {
            "app_id": app_id, "tenant_id": tenant_id,
            "schema": schema_name,
        })

        # ── Load conversation history ──────────────────────────────────────────
        history: list[dict] = []
        if db and conversation_id:
            try:
                from services.conversation_service import ConversationService
                svc = ConversationService(db)
                msgs = await svc.get_recent_messages(conversation_id, n=10)
                history = [{"role": m.role, "content": m.content} for m in msgs]
            except Exception as exc:
                logger.warning("rag_history_load_failed", error=str(exc))

        initial: QueryState = {
            "app_id": app_id,
            "tenant_id": tenant_id,
            "schema_name": schema_name,
            "query": query_text,
            "top_k": top_k,
            "embedding_model": embedding_model,
            "conversation_history": history,
            "language": language,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_vector": [],
            "retrieved": [],
            "graded": [],
            "grade_passed": False,
            "answer": "",
            "citations": [],
            "is_grounded": False,
            "retry_count": 0,
            "error": None,
        }

        try:
            result = await query_pipeline.ainvoke(initial)
        except Exception as exc:
            logger.error("rag_query_pipeline_failed", error=str(exc))
            self._end_span(span, error=str(exc))
            return QueryResult(
                schema_name=schema_name,
                query=query_text,
                answer="Query pipeline failed.",
                status="error",
                error=str(exc),
                embedding_model=embedding_model,
                language=language,
                conversation_id=conversation_id,
            )

        self._end_span(span)
        error = result.get("error")
        answer = result.get("answer", "")

        # ── Persist conversation messages ──────────────────────────────────────
        if db:
            try:
                import uuid as _uuid

                from db.models import Conversation
                from services.conversation_service import ConversationService
                svc = ConversationService(db)
                if not conversation_id:
                    conv = Conversation(
                        id=str(_uuid.uuid4()),
                        employee_code=user_id or f"{app_id}:{tenant_id}",
                        app_id=app_id,
                        title=query_text[:60],
                        metadata_={"tenant_id": tenant_id},
                    )
                    db.add(conv)
                    await db.commit()
                    await db.refresh(conv)
                    conversation_id = conv.id
                await svc.save_message(conversation_id, "user", query_text)
                if not error:
                    await svc.save_message(conversation_id, "assistant", answer)
            except Exception as exc:
                logger.warning("rag_history_save_failed", error=str(exc))

        return QueryResult(
            schema_name=schema_name,
            query=query_text,
            answer=answer,
            citations=result.get("citations", []),
            chunks_retrieved=len(result.get("retrieved", [])),
            chunks_used=len(result.get("graded", [])),
            is_grounded=result.get("is_grounded", False),
            embedding_model=embedding_model,
            language=language,
            conversation_id=conversation_id,
            embed_ms=result.get("embed_ms", 0),
            retrieval_ms=result.get("retrieval_ms", 0),
            grading_ms=result.get("grading_ms", 0),
            generation_ms=result.get("generation_ms", 0),
            status="error" if error else "ok",
            error=error,
        )

    async def delete_table_data(self, schema_name: str, table_name: str, tenant_id: str) -> bool:
        from rag.vector_store import PgVectorStore

        try:
            store = PgVectorStore()
            await store.delete_tenant_data(schema_name, table_name, tenant_id)
            return True
        except Exception as exc:
            logger.error(
                "rag_delete_failed",
                schema=schema_name, table=table_name, tenant_id=tenant_id, error=str(exc),
            )
            return False

    async def doc_count(self, schema_name: str, table_name: str, tenant_id: str) -> int:
        from rag.vector_store import PgVectorStore

        try:
            store = PgVectorStore()
            return await store.doc_count(schema_name, table_name, tenant_id)
        except Exception:
            return 0

    # ── Langfuse helpers (graceful no-op if Langfuse not configured) ──────────

    @staticmethod
    def _start_span(name: str, metadata: dict | None = None):
        try:
            from langfuse_config import langfuse_client

            if langfuse_client:
                return langfuse_client.trace(name=name, metadata=metadata or {})
        except Exception:
            pass
        return None

    @staticmethod
    def _end_span(span, error: str | None = None) -> None:
        try:
            if span:
                span.update(level="ERROR" if error else "DEFAULT", status_message=error or "")
        except Exception:
            pass
