# semantiCache

**A FEDERATED METADATA-GUIDED FRAMEWORK FOR RELIABLE SEMANTIC CACHING IN LLM SYSTEMS**

Current prototype: **FastAPI + SQLite + Ollama/Phi-3 exact-match cache**.

## Setup

1. Clone the repository:

   ```bash
   git clone git@github.com:slept3arly/semantiCache.git
   cd semantiCache
   ```

2. Create and activate the Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Ollama is installed and `phi3` is available:

   ```bash
   ollama list
   ```

5. Start the API:

   ```bash
   uvicorn app.main:app --reload
   ```

6. Open:

   ```text
   http://127.0.0.1:8000/docs
   ```

## Current Functionality

```text
Request
  ↓
FastAPI
  ↓
SQLite exact-cache lookup
  ├── HIT  → cached response
  └── MISS → Ollama/Phi-3 → store response
```

Semantic retrieval with **SentenceTransformers + ChromaDB** is the next implementation stage.

## Development

Do not commit `.venv/`, SQLite databases, ChromaDB runtime data, or environment files. Use Git to synchronize source code and `requirements.txt`.
