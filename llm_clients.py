"""
llm_clients.py — thin async wrappers around Vertex AI (Gemini) for RAG-specific calls.

Three operations:
  grade_relevance(query, chunk_text) → float [0.0 – 1.0]
  generate_with_context(query, context_chunks) → str
  check_grounding(context_chunks, answer) → bool

All use a lazy-initialised google-genai Client (Vertex AI backend).
Model names, token limits, and thresholds come from settings (config/*.py).
"""
from __future__ import annotations

import json
import os
import pathlib

import structlog
from google import genai
from google.genai import types

logger = structlog.get_logger()

# ── Lazy Vertex AI client ─────────────────────────────────────────────────────

_client: genai.Client | None = None

# App root = CAP-F1-orchestration-router/ (3 levels above this file in src/rag/)
_APP_ROOT = pathlib.Path(__file__).parent.parent.parent


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        from config import settings

        # Resolve credentials — handle relative paths from .env.local
        creds = settings.google_application_credentials
        if creds:
            creds_path = pathlib.Path(creds)
            if not creds_path.is_absolute():
                # Try the filename in the app root directory first
                candidate = _APP_ROOT / creds_path.name
                if candidate.exists():
                    creds = str(candidate)
                elif (_APP_ROOT / creds_path).exists():
                    creds = str(_APP_ROOT / creds_path)
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds)

        _client = genai.Client(
            vertexai=True,
            project=settings.vertex_ai_project_id,
            location=settings.vertex_ai_location,
        )
    return _client


def _cfg():
    from config import settings
    return settings


def _to_contents(history: list[dict], user_message: str) -> list[types.Content]:
    """Build a Vertex AI contents list from OpenAI-style history + new user message."""
    contents: list[types.Content] = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    return contents


# ── translate_query_for_retrieval ─────────────────────────────────────────────


async def translate_query_for_retrieval(query: str) -> str:
    """
    If the query contains non-ASCII characters (Devanagari, Gurmukhi, etc.),
    translate it to English so the English-only embedding model can match it
    against English document chunks.  Falls back to original query on any error.
    """
    if query.isascii():
        return query  # already English / Roman script — no translation needed

    prompt = (
        "Translate this search query to English for semantic document retrieval.\n"
        "IMPORTANT: Keep proper nouns, brand names, and product names in English form"
        " (e.g. 'ओनिफाइड', 'ਓਨਿਫਾਈਡ', 'ऑनिफाइड' should remain 'Onified').\n"
        f"Query: {query}\n\n"
        "Respond ONLY with the English translation, nothing else."
    )
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=_cfg().rag_fast_model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=200, temperature=0),
        )
        translated = (response.text or "").strip()
        logger.info("query_translated", original=query[:80], translated=translated[:80])
        return translated or query
    except Exception as exc:
        logger.warning("query_translation_failed", error=str(exc))
        return query


# ── grade_relevance ───────────────────────────────────────────────────────────


async def grade_relevance(query: str, chunk_text: str) -> float:
    """
    Ask the fast model to rate how relevant a chunk is to the query.
    Returns a float in [0.0, 1.0].  Falls back to 0.6 on any error.
    """
    prompt = (
        f"Query: {query}\n\n"
        f"Chunk: {chunk_text[:800]}\n\n"
        "Rate how relevant this chunk is to the query on a scale of 0.0 to 1.0. "
        'Respond ONLY with valid JSON: {"score": <number>, "reason": "<short reason>"}'
    )
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=_cfg().rag_fast_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=80,
                temperature=0,
            ),
        )
        data = json.loads(response.text or "{}")
        return float(data.get("score", 0.6))
    except Exception as exc:
        logger.warning("grade_relevance_failed", error=str(exc))
        return 0.6  # Permissive default


# ── Language instructions ─────────────────────────────────────────────────────

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": (
        "Answer in clear, professional English."
    ),
    "hinglish": (
        "Answer in Hinglish — a natural mix of Hindi and English written entirely in Roman script "
        "(e.g., 'Is document mein likha hai ki...'). Keep it conversational and friendly."
    ),
    "punjabi": (
        "Answer entirely in Punjabi using the native Gurmukhi script (ਪੰਜਾਬੀ). "
        "Do not use Roman script for Punjabi words. Be clear and natural."
    ),
    "marathi": (
        "Answer entirely in Marathi using the native Devanagari script (मराठी). "
        "Do not use Roman script for Marathi words. Be clear and natural."
    ),
    "auto": (
        "Detect the language of the user's question and respond in the SAME language. "
        "If the question is in English respond in English; if in Hindi/Hinglish respond in Hinglish"
        " (Roman script); if in Punjabi respond in Punjabi (Gurmukhi script); "
        "if in Marathi respond in Marathi (Devanagari script)."
    ),
}


# ── generate_with_context ─────────────────────────────────────────────────────


async def generate_with_context(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    language: str = "auto",
    original_query: str | None = None,
) -> str:
    """
    Generate a cited answer using the configured Vertex AI Gemini model.

    - history: prior conversation turns as [{"role": "user"/"assistant", "content": "..."}]
    - language: one of 'auto', 'en', 'hinglish', 'punjabi', 'marathi'
    - original_query: the user's raw query (non-ASCII), when query is its English translation
    Context is formatted as numbered references so the model can cite
    them as [1], [2], etc. in the answer body.
    """
    cfg = _cfg()
    numbered = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["auto"])

    system = (
        "You are a concise, helpful assistant. Answer in 2-3 sentences maximum using ONLY "
        "the provided context. Cite references as [1], [2], etc. if helpful. "
        "If the context does not contain sufficient information, say so briefly. "
        f"Do not pad or repeat. LANGUAGE INSTRUCTION: {lang_instruction}"
    )

    if original_query and original_query != query:
        user_content = (
            f"Context references:\n{numbered}\n\n"
            f"User's question: {original_query}\n"
            f"(English equivalent for context matching: {query})\n\n"
            f"Answer in the SAME language as the user's question, with inline citations:"
        )
    else:
        user_content = (
            f"Context references:\n{numbered}\n\nQuestion: {query}\n\n"
            "Answer (with inline citations):"
        )

    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=cfg.rag_max_tokens,
        temperature=0.2,
    )

    client = _get_client()
    trimmed_history = (history or [])[-cfg.rag_history_turns:]
    contents = _to_contents(trimmed_history, user_content) if trimmed_history else user_content

    response = await client.aio.models.generate_content(
        model=cfg.rag_llm_model,
        contents=contents,
        config=gen_config,
    )
    return response.text or ""


# ── generate_with_context_stream ──────────────────────────────────────────────


async def generate_with_context_stream(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    language: str = "auto",
    original_query: str | None = None,
):
    """
    Async generator that yields answer tokens one by one using streaming.
    Same prompt/language logic as generate_with_context.
    """
    cfg = _cfg()
    numbered = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks))
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["auto"])

    system = (
        "You are a concise, helpful assistant. Answer in 2-3 sentences maximum using ONLY "
        "the provided context. Cite references as [1], [2], etc. if helpful. "
        "If the context does not contain sufficient information, say so briefly. "
        f"Do not pad or repeat. LANGUAGE INSTRUCTION: {lang_instruction}"
    )

    if original_query and original_query != query:
        user_content = (
            f"Context references:\n{numbered}\n\n"
            f"User's question: {original_query}\n"
            f"(English equivalent for context matching: {query})\n\n"
            f"Answer in the SAME language as the user's question, with inline citations:"
        )
    else:
        user_content = (
            f"Context references:\n{numbered}\n\nQuestion: {query}\n\n"
            "Answer (with inline citations):"
        )

    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=cfg.rag_max_tokens,
        temperature=0.2,
    )

    client = _get_client()
    trimmed_history = (history or [])[-cfg.rag_history_turns:]
    contents = _to_contents(trimmed_history, user_content) if trimmed_history else user_content

    async for chunk in await client.aio.models.generate_content_stream(
        model=cfg.rag_llm_model,
        contents=contents,
        config=gen_config,
    ):
        if chunk.text:
            yield chunk.text


# ── check_grounding ───────────────────────────────────────────────────────────


async def check_grounding(context_chunks: list[dict], answer: str) -> bool:
    """
    Ask the fast model whether the answer is fully supported by the context.
    Returns True (grounded) or False (potential hallucination).
    Falls back to True if the check itself fails.
    """
    cfg = _cfg()
    context_text = " ".join(c["text"] for c in context_chunks)[:3000]
    prompt = (
        f"Context: {context_text}\n\n"
        f"Answer: {answer[:cfg.rag_max_tokens]}\n\n"
        "Is the answer fully supported by the context above, with no information fabricated "
        "or assumed beyond what is present? "
        'Respond ONLY with valid JSON: {"grounded": true|false}'
    )
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=cfg.rag_fast_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=20,
                temperature=0,
            ),
        )
        data = json.loads(response.text or "{}")
        return bool(data.get("grounded", True))
    except Exception as exc:
        logger.warning("check_grounding_failed", error=str(exc))
        return True  # Assume grounded on check failure
