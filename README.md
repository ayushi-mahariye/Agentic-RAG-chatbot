# 🧠 Agentic RAG Chatbot

> A production-grade Agentic Retrieval-Augmented Generation (RAG) chatbot with voice support — featuring intelligent document ingestion, vector search, multi-LLM support, and a graph-based agentic query pipeline.

---

## 📌 Overview

The **Agentic RAG Chatbot** combines the power of Retrieval-Augmented Generation with an agentic execution graph to deliver accurate, context-aware responses grounded in your documents. It supports voice input/output, automatically detects document types, chunks and embeds content into a vector store, and uses a structured query graph to reason over retrieved data before responding.
<img width="1918" height="914" alt="Screenshot 2026-03-27 162346" src="https://github.com/user-attachments/assets/e5e3bfde-052c-4a65-93e7-53d98c87ee27" />

---

## 🗂️ Project Structure

```
Agentic-RAG-chatbot/
├── ingestion_graph.py         # Agentic graph for document ingestion pipeline
├── query_graph.py             # Agentic graph for query processing & reasoning
├── rag_service.py             # Core RAG orchestration — retrieval + generation
├── chunkers.py                # Document chunking strategies (fixed, semantic, etc.)
├── embedder.py                # Text embedding generation & management
├── vector_store.py            # Vector database interface — store & similarity search
├── document_type_detector.py  # Auto-detect document formats (PDF, DOCX, TXT, etc.)
└── llm_clients.py             # Unified LLM client abstraction (multi-provider support)
```

---

## ✨ Features

- 🕸️ **Agentic Graph Pipeline** — Graph-based ingestion and query execution for multi-step reasoning
- 📄 **Auto Document Type Detection** — Automatically identifies and parses PDF, DOCX, TXT, HTML, and more
- ✂️ **Smart Chunking** — Multiple chunking strategies (fixed-size, semantic, recursive) via `chunkers.py`
- 🔢 **Text Embedding** — Converts document chunks into dense vector embeddings for semantic search
- 🗃️ **Vector Store** — Stores and retrieves embeddings via similarity search for relevant context
- 🤖 **Multi-LLM Support** — Swap between OpenAI, Anthropic, or other providers via `llm_clients.py`
- 🎙️ **Voice Support** — Voice input and output for hands-free interaction
- 🔄 **End-to-End RAG Pipeline** — From raw document to grounded AI response in one flow

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────┐
                        │        User Input            │
                        │   (Voice 🎙️ / Text ⌨️)       │
                        └────────────┬────────────────┘
                                     │
              ┌──────────────────────▼─────────────────────────┐
              │              INGESTION PIPELINE                 │
              │                                                 │
              │  Document ──► Type Detector ──► Chunker        │
              │                                    │            │
              │                               Embedder          │
              │                                    │            │
              │                            Vector Store 🗃️      │
              └─────────────────────────────────────────────────┘
                                     │
              ┌──────────────────────▼─────────────────────────┐
              │               QUERY PIPELINE                    │
              │                                                 │
              │  Query ──► Embedder ──► Vector Search           │
              │                              │                  │
              │                      Retrieved Context          │
              │                              │                  │
              │                        LLM Client 🤖            │
              │                              │                  │
              │                     Grounded Response           │
              └─────────────────────────────────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │      Output              │
                        │  (Voice 🎙️ / Text 💬)    │
                        └──────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- A vector database (Chroma, Pinecone, FAISS, or Weaviate)
- An LLM API key (Anthropic Claude, OpenAI GPT, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushi-mahariye/Agentic-RAG-chatbot.git
   cd Agentic-RAG-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file:
   ```env
   LLM_PROVIDER=anthropic               # or openai, cohere, etc.
   LLM_API_KEY=your_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   VECTOR_STORE_TYPE=chroma             # or pinecone, faiss, weaviate
   VECTOR_STORE_PATH=./vector_db
   CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   VOICE_ENABLED=true
   ```

4. **Run the chatbot**
   ```bash
   python rag_service.py
   ```

---

## 🔄 Ingestion Pipeline

The `ingestion_graph.py` orchestrates document ingestion as an agentic graph:

```python
from ingestion_graph import run_ingestion

# Ingest a document
run_ingestion("path/to/your/document.pdf")
```

**Steps executed:**
1. **Detect** document type via `document_type_detector.py`
2. **Parse** raw content from the document
3. **Chunk** content into segments via `chunkers.py`
4. **Embed** chunks into vectors via `embedder.py`
5. **Store** vectors in the vector database via `vector_store.py`

---

## 🔍 Query Pipeline

The `query_graph.py` handles user queries as an agentic reasoning graph:

```python
from query_graph import run_query

response = run_query("What are the key findings in the uploaded report?")
print(response)
```

**Steps executed:**
1. **Embed** the user query
2. **Search** the vector store for top-k relevant chunks
3. **Rank & filter** retrieved context
4. **Reason** over context using the LLM
5. **Return** a grounded, cited response

---

## ✂️ Chunking Strategies

`chunkers.py` supports multiple strategies depending on document type:

| Strategy | Best For | Description |
|---|---|---|
| Fixed-size | Uniform docs | Split by token/character count |
| Semantic | Articles, reports | Split at sentence/paragraph boundaries |
| Recursive | Mixed content | Tries multiple separators hierarchically |
| Sliding window | Dense content | Overlapping chunks for better context |

---

## 📄 Supported Document Types

`document_type_detector.py` auto-detects and handles:

| Format | Extension | Status |
|---|---|---|
| PDF | `.pdf` | ✅ Supported |
| Word Document | `.docx` | ✅ Supported |
| Plain Text | `.txt` | ✅ Supported |
| HTML | `.html` | ✅ Supported |
| Markdown | `.md` | ✅ Supported |
| CSV | `.csv` | 🔜 Planned |
| JSON | `.json` | 🔜 Planned |

---

## 🤖 LLM Clients

`llm_clients.py` provides a unified interface across providers:

```python
from llm_clients import get_llm_response

response = get_llm_response(
    provider="anthropic",
    messages=messages,
    context=retrieved_chunks
)
```

| Provider | Status |
|---|---|
| Anthropic Claude | ✅ Supported |
| OpenAI GPT | ✅ Supported |
| Cohere | 🔜 Planned |
| Ollama (local) | 🔜 Planned |

---

## 🎙️ Voice Support

The chatbot supports voice interaction:

- **Voice Input** — Speak your query; it is transcribed and passed to the RAG pipeline
- **Voice Output** — AI responses are converted to speech using TTS
- Toggle voice mode via the `VOICE_ENABLED` environment variable or at runtime

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Ayushi Mahariye** — [@ayushi-mahariye](https://github.com/ayushi-mahariye)

---

> ⭐ Star this repo if you're building with RAG!
