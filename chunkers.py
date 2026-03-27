"""
chunkers.py — document chunking strategie per content type.

Factory: get_chunker(doc_type) → BaseChunkers

Chunker          | Input             | Strategy
-----------------|-------------------|----------------------------------
PdfChunker       | bytes (PDF)       | PyMuPDF page-level semantic split
TxtChunker       | str / bytes       | Sentence-boundary sliding window
CsvChunker       | str / bytes       | One chunk per CSV row
JsonChunker      | str / bytes       | One chunk per top-level JSON record
DbRowsChunker    | list[dict]        | Serialise each dict row to text
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import structlog

logger = structlog.get_logger()

# ── Data model ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ── Protocol / Base ──────────────────────────────────────────────────────────


@runtime_checkable
class BaseChunker(Protocol):
    def chunk(self, content: bytes | str | list[dict], source: str = "") -> list[Chunk]: ...


# ── PDF Chunker ───────────────────────────────────────────────────────────────


class PdfChunker:
    """
    Semantic page-level PDF chunking using PyMuPDF (fitz).

    Strategy:
      1. Extract text page-by-page (preserves natural section boundaries).
      2. Split each page by double-newline (paragraph blocks).
      3. Merge tiny fragments (<min_chars) into the next block.
      4. Cap each chunk at max_chars to avoid oversized embeddings.
    """

    def __init__(self, min_chars: int = 80, max_chars: int = 1500) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, content: bytes | str | list[dict], source: str = "") -> list[Chunk]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF") from exc

        raw = content if isinstance(content, bytes) else str(content).encode()
        chunks: list[Chunk] = []

        try:
            doc = fitz.open(stream=raw, filetype="pdf")
        except Exception as exc:
            logger.error("pdf_open_failed", source=source, error=str(exc))
            return chunks

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if not text.strip():
                continue

            blocks = text.split("\n\n")
            buffer = ""
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                candidate = (buffer + " " + block).strip() if buffer else block
                if len(candidate) > self.max_chars:
                    if buffer:
                        chunks.append(Chunk(
                            text=buffer.strip(),
                            metadata={"source": source, "page": page_num + 1},
                        ))
                    buffer = block
                elif len(candidate) >= self.min_chars:
                    buffer = candidate
                else:
                    buffer = candidate  # still accumulating

            if buffer.strip():
                chunks.append(Chunk(
                    text=buffer.strip(),
                    metadata={"source": source, "page": page_num + 1},
                ))

        doc.close()
        logger.info("pdf_chunked", source=source, total_chunks=len(chunks))
        return chunks


# ── Text Chunker ──────────────────────────────────────────────────────────────


class TxtChunker:
    """
    Sliding-window sentence-boundary chunker for plain text.

    Groups sentences into chunks of at most `chunk_size` characters,
    then carries over the last `overlap` characters into the next window
    so context doesn't disappear at boundaries.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, content: bytes | str | list[dict], source: str = "") -> list[Chunk]:
        text = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        buffer = ""
        i = 0

        while i < len(sentences):
            s = sentences[i]
            candidate = (buffer + " " + s).strip() if buffer else s

            if len(candidate) <= self.chunk_size:
                buffer = candidate
                i += 1
            else:
                if buffer:
                    chunks.append(Chunk(
                        text=buffer.strip(),
                        metadata={"source": source, "chunk_index": len(chunks)},
                    ))
                    # Carry overlap forward
                    buffer = buffer[-self.overlap:].strip() if self.overlap > 0 else ""
                else:
                    # Single sentence is larger than chunk_size — emit as-is
                    chunks.append(Chunk(
                        text=s.strip(),
                        metadata={"source": source, "chunk_index": len(chunks)},
                    ))
                    buffer = ""
                    i += 1

        if buffer.strip():
            chunks.append(Chunk(
                text=buffer.strip(),
                metadata={"source": source, "chunk_index": len(chunks)},
            ))

        logger.info("txt_chunked", source=source, total_chunks=len(chunks))
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Lightweight sentence tokeniser — no external NLP library required."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


# ── CSV Chunker ───────────────────────────────────────────────────────────────


class CsvChunker:
    """
    Row-level CSV chunker.

    Each row becomes: "col1: val1 | col2: val2 | ..."
    Column names provide context so the embedding captures schema meaning.
    """

    def chunk(self, content: bytes | str | list[dict], source: str = "") -> list[Chunk]:
        if isinstance(content, list):
            return _dict_rows_to_chunks(content, source)

        raw = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        chunks: list[Chunk] = []
        try:
            reader = csv.DictReader(io.StringIO(raw))
            for row_idx, row in enumerate(reader):
                text = " | ".join(
                    f"{k}: {v}" for k, v in row.items() if v is not None and str(v).strip()
                )
                if text:
                    chunks.append(Chunk(
                        text=text,
                        metadata={"source": source, "row": row_idx, "columns": list(row.keys())},
                    ))
        except csv.Error as exc:
            logger.error("csv_chunk_failed", source=source, error=str(exc))
        logger.info("csv_chunked", source=source, total_chunks=len(chunks))
        return chunks


# ── JSON Chunker ──────────────────────────────────────────────────────────────


class JsonChunker:
    """
    Record-level JSON chunker.

    Handles both JSON arrays (each element → chunk) and single objects
    (each top-level key-value pair → chunk).
    """

    def chunk(self, content: bytes | str | list[dict], source: str = "") -> list[Chunk]:
        if isinstance(content, list):
            return _dict_rows_to_chunks(content, source)

        raw = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        chunks: list[Chunk] = []
        try:
            data = json.loads(raw)
            records = data if isinstance(data, list) else [data]
            for idx, record in enumerate(records):
                if isinstance(record, dict):
                    text = " | ".join(
                        f"{k}: {v}" for k, v in record.items() if v is not None
                    )
                else:
                    text = json.dumps(record, ensure_ascii=False)
                if text:
                    chunks.append(Chunk(
                        text=text,
                        metadata={"source": source, "record_index": idx},
                    ))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("json_chunk_failed", source=source, error=str(exc))
        logger.info("json_chunked", source=source, total_chunks=len(chunks))
        return chunks


# ── DB Rows Chunker ───────────────────────────────────────────────────────────


class DbRowsChunker:
    """
    Accepts a list[dict] (database query result rows) and serialises each
    row to a readable text chunk.
    """

    def chunk(self, content: bytes | str | list[dict], source: str = "") -> list[Chunk]:
        rows = content if isinstance(content, list) else []
        return _dict_rows_to_chunks(rows, source)


# ── Factory ───────────────────────────────────────────────────────────────────


def get_chunker(doc_type: str) -> BaseChunker:
    """Return the appropriate chunker for the given doc_type string."""
    _map: dict[str, BaseChunker] = {
        "pdf":     PdfChunker(),
        "txt":     TxtChunker(chunk_size=512, overlap=64),
        "csv":     CsvChunker(),
        "json":    JsonChunker(),
        "db_rows": DbRowsChunker(),
    }
    return _map.get(doc_type, TxtChunker())


# ── Shared helper ─────────────────────────────────────────────────────────────


def _dict_rows_to_chunks(rows: list[dict], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, row in enumerate(rows):
        text = " | ".join(
            f"{k}: {v}" for k, v in row.items() if v is not None and str(v).strip()
        )
        if text:
            chunks.append(Chunk(
                text=text,
                metadata={"source": source, "row": idx},
            ))
    return chunks
