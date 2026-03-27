"""
vector_store.py — pgvector-backed semantic vector store with schema-based isolation.

PostgreSQL schema per app, table per document:
    Schema  : sanitize(app_id)       e.g.  crm_app
    Table   : sanitize(filename)_vector  e.g.  policy_vector
    Full ref: crm_app.policy_vector

Isolation hierarchy:
    schema (app)  → PostgreSQL-native isolation, schema per app_id
    table  (doc)  → one table per uploaded document within the schema
    tenant_id col → row-level isolation for multi-tenant within one app

Schema introspection (like SQL chatbot's SchemaIntrospector):
    information_schema.tables  → list all vector tables for an app
    information_schema.schemata → check if app schema exists

Each table schema:
    id              TEXT PRIMARY KEY
    tenant_id       TEXT NOT NULL
    text            TEXT NOT NULL
    embedding_model TEXT NOT NULL
    embedding_mpnet vector(768)   -- all-mpnet-base-v2  (PDF, TXT)
    embedding_minilm vector(384)  -- all-MiniLM-L6-v2   (CSV, JSON)
    metadata        JSON DEFAULT '{}'
    created_at      TIMESTAMPTZ DEFAULT now()

Uses raw SQL with CAST(:vec AS vector) to avoid asyncpg codec issues.
Score = 1 - cosine_distance  (higher = more similar).
"""
from __future__ import annotations

import json
import re
import uuid

import db.session as _db_session
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

_MODEL_COLUMN = {
    "all-mpnet-base-v2": "embedding_mpnet",
    "all-MiniLM-L6-v2":  "embedding_minilm",
    "text-embedding-004": "embedding_gemini",
}

USER_DOCS_SCHEMA = "user_docs"  # fixed schema for all user-uploaded conversation docs


# ── Naming helpers ────────────────────────────────────────────────────────────

def _sanitize(s: str) -> str:
    """Lowercase, replace non-alphanumeric runs with underscore, strip edges."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def make_schema_name(app_id: str) -> str:
    """
    PostgreSQL schema name for an app.
    e.g. "crm-app" → "crm_app",  "MyApp 2" → "myapp_2"
    Max 63 chars (PostgreSQL identifier limit).
    """
    return _sanitize(app_id)[:63]


def make_user_table_name(conversation_id: str, filename: str) -> str:
    """
    Table name for a user-uploaded document within the user_docs schema.
    Scoped to conversation so each conversation's files are isolated.
    e.g. conversation_id="abc123...", filename="report.pdf"
      → "conv_abc123_report_vector"
    """
    conv_prefix = _sanitize(conversation_id[:8])
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    name = f"conv_{conv_prefix}_{_sanitize(base)}_vector"
    return name[:63]


def make_table_name(filename: str) -> str:
    """
    Table name (within the app's schema) for a document.
    e.g. "Policy Report 2024.pdf" → "policy_report_2024_vector"
    Max 63 chars.
    """
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    name = _sanitize(base) + "_vector"
    if len(name) > 63:
        name = _sanitize(base)[:59] + "_vec"
    return name


def _safe_id(s: str) -> str:
    """Validate a PostgreSQL identifier (schema or table name)."""
    if not re.match(r"^[a-z][a-z0-9_]{0,62}$", s):
        raise ValueError(f"Unsafe PostgreSQL identifier: {s!r}")
    return s


# ── PgVectorStore ─────────────────────────────────────────────────────────────

class PgVectorStore:
    """
    Async pgvector store using PostgreSQL schemas for app isolation.

    Schema  = app_id   (one schema per app, created on first ingest)
    Table   = filename (one table per document, within the app's schema)
    Row col = tenant_id (filters rows within a table for multi-tenant apps)
    """

    # ── Schema introspection (mirrors SQL SchemaIntrospector) ─────────────────

    async def schema_exists(self, schema: str) -> bool:
        """Check if an app schema exists in this database."""
        async with self._session() as session:
            count = await session.scalar(
                text(
                    "SELECT COUNT(*) FROM information_schema.schemata "
                    "WHERE schema_name = :schema"
                ),
                {"schema": schema},
            )
        return (count or 0) > 0

    async def list_tables(self, schema: str) -> list[str]:
        """
        List all vector tables in an app's schema.
        Equivalent to SchemaIntrospector.get_or_refresh for RAG.
        """
        async with self._session() as session:
            rows = (await session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :schema "
                    "  AND table_name LIKE '%_vector' "
                    "ORDER BY table_name"
                ),
                {"schema": schema},
            )).all()
        return [r.table_name for r in rows]

    async def table_exists(self, schema: str, table: str) -> bool:
        """Check if a document vector table exists within the app's schema."""
        async with self._session() as session:
            count = await session.scalar(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = :schema AND table_name = :table"
                ),
                {"schema": schema, "table": table},
            )
        return (count or 0) > 0

    async def tenant_has_data(self, schema: str, table: str, tenant_id: str) -> bool:
        """Return True if this tenant has at least one chunk in schema.table."""
        if not await self.table_exists(schema, table):
            return False
        s, t = _safe_id(schema), _safe_id(table)
        async with self._session() as session:
            count = await session.scalar(
                text(f"SELECT COUNT(*) FROM {s}.{t} WHERE tenant_id = :tid LIMIT 1"),
                {"tid": tenant_id},
            )
        return (count or 0) > 0

    async def doc_count(self, schema: str, table: str, tenant_id: str) -> int:
        if not await self.table_exists(schema, table):
            return 0
        s, t = _safe_id(schema), _safe_id(table)
        async with self._session() as session:
            count = await session.scalar(
                text(f"SELECT COUNT(*) FROM {s}.{t} WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )
        return count or 0

    # ── DDL ──────────────────────────────────────────────────────────────────

    async def ensure_schema(self, schema: str) -> None:
        """Create the app's PostgreSQL schema if it does not exist."""
        s = _safe_id(schema)
        async with self._session() as session:
            await session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {s}"))
            await session.commit()
        logger.info("pgvector_schema_ensured", schema=s)

    async def ensure_table(self, schema: str, table: str) -> None:
        """
        Create the document vector table and HNSW indexes inside the app schema.
        Idempotent — safe to call on every ingest request.
        Each statement is executed separately (asyncpg rejects multi-statement strings).
        HNSW chosen over IVFFlat: works with zero existing rows at creation time.
        """
        s, t = _safe_id(schema), _safe_id(table)
        statements = [
            f"""CREATE TABLE IF NOT EXISTS {s}.{t} (
                id               TEXT PRIMARY KEY,
                tenant_id        TEXT NOT NULL,
                text             TEXT NOT NULL,
                embedding_model  TEXT NOT NULL,
                embedding_mpnet  vector(768),
                embedding_minilm vector(384),
                embedding_gemini vector(768),
                metadata         JSON DEFAULT '{{}}',
                created_at       TIMESTAMPTZ DEFAULT now()
            )""",
            f"CREATE INDEX IF NOT EXISTS {t}_tenant_idx ON {s}.{t} (tenant_id)",
            (
                f"CREATE INDEX IF NOT EXISTS {t}_mpnet_idx ON {s}.{t} "
                f"USING hnsw (embedding_mpnet vector_cosine_ops)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS {t}_minilm_idx ON {s}.{t} "
                f"USING hnsw (embedding_minilm vector_cosine_ops)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS {t}_gemini_idx ON {s}.{t} "
                f"USING hnsw (embedding_gemini vector_cosine_ops)"
            ),
        ]
        async with self._session() as session:
            for stmt in statements:
                await session.execute(text(stmt))
            await session.commit()
        logger.info("pgvector_table_ensured", schema=s, table=t)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def add(
        self,
        app_id: str,
        tenant_id: str,
        filename: str,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
        embedding_model: str = "all-mpnet-base-v2",
    ) -> tuple[str, str, int]:
        """
        Insert chunks into schema.table for (app_id, filename).
        Creates the schema and table automatically if they don't exist.

        Returns (schema_name, table_name, total_rows_for_tenant).
        """
        if not texts or not vectors:
            schema = make_schema_name(app_id)
            table  = make_table_name(filename)
            return schema, table, 0

        schema = make_schema_name(app_id)
        table  = make_table_name(filename)
        col    = _MODEL_COLUMN.get(embedding_model, "embedding_mpnet")
        metas  = metadatas or [{} for _ in texts]

        await self.ensure_schema(schema)
        await self.ensure_table(schema, table)

        s, t = _safe_id(schema), _safe_id(table)

        insert_sql = text(f"""
            INSERT INTO {s}.{t}
                (id, tenant_id, text, embedding_model, {col}, metadata)
            VALUES
                (:id, :tid, :txt, :model, CAST(:vec AS vector), CAST(:meta AS json))
        """)

        async with self._session() as session:
            for txt, vec, meta in zip(texts, vectors, metas):
                await session.execute(insert_sql, {
                    "id":    str(uuid.uuid4()),
                    "tid":   tenant_id,
                    "txt":   txt,
                    "model": embedding_model,
                    "vec":   _vec_str(vec),
                    "meta":  json.dumps(meta),
                })
            await session.commit()

            total = await session.scalar(
                text(f"SELECT COUNT(*) FROM {s}.{t} WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )

        logger.info(
            "pgvector_add",
            schema=schema, table=table, tenant_id=tenant_id,
            added=len(texts), total=total,
        )
        return schema, table, total or 0

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        schema: str,
        table: str,
        tenant_id: str,
        query_vector: list[float],
        top_k: int = 5,
        embedding_model: str = "all-mpnet-base-v2",
    ) -> list[dict]:
        col   = _MODEL_COLUMN.get(embedding_model, "embedding_mpnet")
        vec_s = _vec_str(query_vector)
        s, t  = _safe_id(schema), _safe_id(table)

        sql = text(f"""
            SELECT text, metadata,
                   1 - ({col} <=> CAST(:vec AS vector)) AS score
            FROM {s}.{t}
            WHERE tenant_id = :tid
              AND {col} IS NOT NULL
            ORDER BY {col} <=> CAST(:vec AS vector)
            LIMIT :k
        """)

        async with self._session() as session:
            rows = (await session.execute(
                sql, {"vec": vec_s, "tid": tenant_id, "k": top_k}
            )).all()

        results = [
            {"text": r.text, "metadata": r.metadata or {}, "score": float(r.score)}
            for r in rows
        ]
        logger.info(
            "pgvector_search",
            schema=schema, table=table, tenant_id=tenant_id, found=len(results),
        )
        return results

    async def search_schema(
        self,
        schema: str,
        tenant_id: str,
        query_vector: list[float],
        top_k: int = 5,
        embedding_model: str = "all-mpnet-base-v2",
    ) -> list[dict]:
        """
        Search across ALL vector tables in the app's schema for this tenant.
        Results from every table are merged and re-ranked by cosine score.
        Each result includes a `source_table` field so the caller knows which
        document the chunk came from.
        """
        tables = await self.list_tables(schema)
        if not tables:
            logger.warning("pgvector_schema_empty", schema=schema, tenant_id=tenant_id)
            return []

        combined: list[dict] = []
        for table in tables:
            if not await self.tenant_has_data(schema, table, tenant_id):
                continue
            rows = await self.search(schema, table, tenant_id, query_vector, top_k, embedding_model)
            for r in rows:
                r["source_table"] = table
            combined.extend(rows)

        # Re-rank across all tables by score, keep global top_k
        combined.sort(key=lambda x: x["score"], reverse=True)
        results = combined[:top_k]

        logger.info(
            "pgvector_schema_search",
            schema=schema, tenant_id=tenant_id,
            tables_searched=len(tables), found=len(results),
        )
        return results

    # ── User-doc DDL ─────────────────────────────────────────────────────────

    async def ensure_user_table(self, table: str) -> None:
        """
        Create a user-doc table inside the user_docs schema.
        Columns include user_id and conversation_id for scoped retrieval.
        """
        s = _safe_id(USER_DOCS_SCHEMA)
        t = _safe_id(table)
        statements = [
            f"CREATE SCHEMA IF NOT EXISTS {s}",
            f"""CREATE TABLE IF NOT EXISTS {s}.{t} (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                tenant_id       TEXT NOT NULL,
                text            TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding       vector(768),
                metadata        JSON DEFAULT '{{}}',
                created_at      TIMESTAMPTZ DEFAULT now()
            )""",
            (
                f"CREATE INDEX IF NOT EXISTS {t}_conv_idx "
                f"ON {s}.{t} (user_id, conversation_id)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS {t}_emb_idx ON {s}.{t} "
                f"USING hnsw (embedding vector_cosine_ops)"
            ),
        ]
        async with self._session() as session:
            for stmt in statements:
                await session.execute(text(stmt))
            await session.commit()
        logger.info("user_table_ensured", schema=s, table=t)

    async def add_user_doc(
        self,
        user_id: str,
        conversation_id: str,
        tenant_id: str,
        filename: str,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> tuple[str, str, int]:
        """
        Insert user-uploaded document chunks into user_docs schema.
        Table is scoped to (conversation_id, filename).
        Returns (schema_name, table_name, total_chunks).
        """
        if not texts or not vectors:
            table = make_user_table_name(conversation_id, filename)
            return USER_DOCS_SCHEMA, table, 0

        table = make_user_table_name(conversation_id, filename)
        metas = metadatas or [{} for _ in texts]

        await self.ensure_user_table(table)

        s = _safe_id(USER_DOCS_SCHEMA)
        t = _safe_id(table)
        insert_sql = text(f"""
            INSERT INTO {s}.{t}
                (id, user_id, conversation_id, tenant_id, text, embedding_model, embedding, metadata)
            VALUES
                (:id, :uid, :cid, :tid, :txt, :model, CAST(:vec AS vector), CAST(:meta AS json))
        """)

        from rag.embedder import GEMINI_EMBED_MODEL

        async with self._session() as session:
            for txt, vec, meta in zip(texts, vectors, metas):
                await session.execute(insert_sql, {
                    "id":    str(uuid.uuid4()),
                    "uid":   user_id,
                    "cid":   conversation_id,
                    "tid":   tenant_id,
                    "txt":   txt,
                    "model": GEMINI_EMBED_MODEL,
                    "vec":   _vec_str(vec),
                    "meta":  json.dumps(meta),
                })
            await session.commit()
            total = await session.scalar(
                text(
                    f"SELECT COUNT(*) FROM {s}.{t} "
                    "WHERE user_id = :uid AND conversation_id = :cid"
                ),
                {"uid": user_id, "cid": conversation_id},
            )

        logger.info(
            "user_doc_added",
            table=table, user_id=user_id, conversation_id=conversation_id,
            added=len(texts), total=total,
        )
        return USER_DOCS_SCHEMA, table, total or 0

    async def search_user_docs(
        self,
        user_id: str,
        conversation_id: str,
        tenant_id: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search all user-uploaded document tables for this conversation.
        Only returns chunks belonging to the given user_id + conversation_id.
        """
        s = _safe_id(USER_DOCS_SCHEMA)
        if not await self.schema_exists(USER_DOCS_SCHEMA):
            return []

        tables = await self.list_tables(USER_DOCS_SCHEMA)
        conv_prefix = _sanitize(conversation_id[:8])
        conv_tables = [t for t in tables if t.startswith(f"conv_{conv_prefix}_")]
        if not conv_tables:
            return []

        vec_s = _vec_str(query_vector)
        combined: list[dict] = []
        for table in conv_tables:
            t = _safe_id(table)
            sql = text(f"""
                SELECT text, metadata,
                       1 - (embedding <=> CAST(:vec AS vector)) AS score
                FROM {s}.{t}
                WHERE user_id = :uid
                  AND conversation_id = :cid
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """)
            async with self._session() as session:
                rows = (await session.execute(
                    sql, {"vec": vec_s, "uid": user_id, "cid": conversation_id, "k": top_k}
                )).all()
            for r in rows:
                combined.append({
                    "text": r.text,
                    "metadata": r.metadata or {},
                    "score": float(r.score),
                    "source_table": table,
                    "source": "user_upload",
                })

        combined.sort(key=lambda x: x["score"], reverse=True)
        results = combined[:top_k]
        logger.info(
            "user_docs_searched",
            conversation_id=conversation_id, user_id=user_id,
            tables_searched=len(conv_tables), found=len(results),
        )
        return results

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_tenant_data(self, schema: str, table: str, tenant_id: str) -> None:
        """Delete all rows for this tenant from schema.table (table is kept)."""
        if not await self.table_exists(schema, table):
            return
        s, t = _safe_id(schema), _safe_id(table)
        async with self._session() as session:
            await session.execute(
                text(f"DELETE FROM {s}.{t} WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )
            await session.commit()
        logger.info("pgvector_tenant_deleted", schema=schema, table=table, tenant_id=tenant_id)

    # ── Session ───────────────────────────────────────────────────────────────

    @staticmethod
    def _session() -> AsyncSession:
        if _db_session._rag_session_factory is None:
            raise RuntimeError("Database not initialised. Ensure init_db() ran at startup.")
        return _db_session._rag_session_factory()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vec_str(vec: list[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"
