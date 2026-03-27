"""
document_type_detector.py — infers document type and selects the optimal
embedding model + chunking strategy.

Decision table:
  doc_type   | embedding_model          | chunk_strategy
  -----------|--------------------------|----------------
  pdf        | all-mpnet-base-v2        | semantic
  txt        | all-mpnet-base-v2        | sliding_window
  csv        | all-MiniLM-L6-v2         | row
  json       | all-MiniLM-L6-v2         | record
  db_rows    | all-MiniLM-L6-v2         | row

Unstructured text (pdf, txt) gets the heavier MPNet model for richer
semantic embeddings.  Structured/tabular data (csv, json, db) gets the
lighter MiniLM model — shorter texts, speed matters more than depth.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Literal

DocType = Literal["pdf", "txt", "csv", "json", "db_rows"]
EmbeddingModelName = Literal["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
ChunkStrategy = Literal["semantic", "sliding_window", "row", "record"]

# Open-source sentence-transformer model IDs
MPNET_MODEL = "all-mpnet-base-v2"      # unstructured / long texts
MINILM_MODEL = "all-MiniLM-L6-v2"     # structured / short texts

_PDF_MAGIC = b"%PDF"
_JSON_START = (b"{", b"[")

_EXT_MAP: dict[str, DocType] = {
    "pdf":  "pdf",
    "txt":  "txt",
    "md":   "txt",
    "rst":  "txt",
    "log":  "txt",
    "csv":  "csv",
    "tsv":  "csv",
    "json": "json",
    "jsonl": "json",
}

_MIME_MAP: dict[str, DocType] = {
    "application/pdf":   "pdf",
    "text/plain":        "txt",
    "text/markdown":     "txt",
    "text/csv":          "csv",
    "application/csv":   "csv",
    "application/json":  "json",
    "text/json":         "json",
}

_CONFIG: dict[DocType, tuple[EmbeddingModelName, ChunkStrategy]] = {
    "pdf":     (MPNET_MODEL,  "semantic"),  # type: ignore[dict-item]
    "txt":     (MPNET_MODEL,  "sliding_window"),  # type: ignore[dict-item]
    "csv":     (MINILM_MODEL, "row"),  # type: ignore[dict-item]
    "json":    (MINILM_MODEL, "record"),  # type: ignore[dict-item]
    "db_rows": (MINILM_MODEL, "row"),  # type: ignore[dict-item]
}


@dataclass(slots=True)
class DetectionResult:
    doc_type: DocType
    embedding_model: EmbeddingModelName
    chunk_strategy: ChunkStrategy


def detect_doc_type(
    filename: str,
    content: bytes,
    mime_type: str | None = None,
) -> DetectionResult:
    """
    Detect document type from filename extension, MIME type, and magic bytes.
    Returns a DetectionResult with the resolved doc_type, embedding model,
    and chunking strategy.
    """
    doc_type = _resolve_type(filename, content, mime_type)
    embedding_model, chunk_strategy = _CONFIG.get(doc_type, (MPNET_MODEL, "sliding_window"))
    return DetectionResult(
        doc_type=doc_type,
        embedding_model=embedding_model,  # type: ignore[arg-type]
        chunk_strategy=chunk_strategy,  # type: ignore[arg-type]
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_type(filename: str, content: bytes, mime_type: str | None) -> DocType:
    # 1. Extension (fastest, most reliable when present)
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in _EXT_MAP:
            return _EXT_MAP[ext]

    # 2. MIME type hint from the upload
    if mime_type:
        base_mime = mime_type.split(";")[0].strip().lower()
        if base_mime in _MIME_MAP:
            return _MIME_MAP[base_mime]

    # 3. Magic bytes / content sniffing
    if content[:4] == _PDF_MAGIC:
        return "pdf"
    if content[:1] in _JSON_START:
        return _probe_json(content)
    if _looks_like_csv(content):
        return "csv"

    # 4. Fallback — treat as plain text
    return "txt"


def _probe_json(content: bytes) -> DocType:
    try:
        data = json.loads(content[:8192])
        if isinstance(data, (list, dict)):
            return "json"
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return "txt"


def _looks_like_csv(content: bytes) -> bool:
    """Sniff first 2 KB for CSV-like structure (delimiter + consistent columns)."""
    try:
        sample = content[:2048].decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        # Confirm at least 2 columns
        reader = csv.reader(io.StringIO(sample), dialect)
        first_row = next(reader, [])
        return len(first_row) >= 2
    except (csv.Error, StopIteration):
        return False
