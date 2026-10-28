# semantiCache

semantiCache is a research prototype for studying cache reuse in multi-user
LLM applications. Its research question is whether lexical and semantic
retrieval can reduce repeated generation while preserving user, session,
conversation, category, and freshness constraints. The prototype implements
retrieval policies; it does not establish answer quality or research novelty.

## Architecture and request flow

FastAPI accepts a query and its scope metadata. SQLAlchemy stores responses and
metadata in SQLite; ChromaDB stores embeddings and semantic metadata;
SentenceTransformers generates `all-MiniLM-L6-v2` embeddings; Ollama generates
responses on misses.

The default `proposed` policy follows:

```text
Validate request → exact SQLite → lexical SQLite → scoped semantic ChromaDB
                 → Ollama fallback → policy-specific persistence
```

Generated entries have a one-hour TTL. SQLite exact and lexical candidates
must match `user_id`, `session_id`, `conversation_id`, and `category`, with
`expires_at` strictly later than the current UTC time. Chroma metadata stores
the same scope and expiry as a UTC-derived numeric timestamp; semantic search
filters on that metadata and validates it again before returning a hit.

Exact lookup requires the query text to match. Lexical lookup lowercases the
query, extracts Unicode alphanumeric tokens, and computes set Jaccard similarity
`|A ∩ B| / |A ∪ B|`; empty token sets do not match. It accepts scores of **0.8
or higher**, an initial conservative cutoff that has not been empirically
tuned. Equal-scoring candidates resolve to the lowest SQLite row ID. Lexical
overlap is only a retrieval signal. Semantic lookup uses the existing Chroma
distance rule, **distance `< 0.6`**, and the same metadata eligibility checks.
Neither kind of hit establishes that an answer is relevant or factually
correct.

## Cache policies

`CacheQueryRequest` requires `user_id`, `session_id`, `conversation_id`,
`query`, and `category`. It accepts a validated `cache_policy` and an optional
`cache_namespace` (1–64 characters). The policy defaults to `proposed`, so
existing clients that omit these fields keep the existing retrieval behavior
and cache keys. The response shape remains `cache_hit`, `answer`, and `source`.

| Policy | Reads | Writes after a miss |
| --- | --- | --- |
| `no_cache` | None | None; calls Ollama every time |
| `exact_only` | Scoped exact SQLite lookup | SQLite only |
| `pure_semantic` | Scoped metadata-validated Chroma lookup; no exact or lexical stage | SQLite and ChromaDB |
| `proposed` | Exact → lexical → metadata-aware semantic | SQLite and ChromaDB |

`pure_semantic` is a metadata-constrained semantic-only retrieval baseline; it
does not disable isolation. When `cache_namespace` is supplied, the application
derives an internal user key from the namespace and user ID. This partitions
experiment entries across namespaces without changing the database schema.
Without a namespace, the original user ID is used.

## Setup

The project targets Python 3.12. Install the pinned dependencies from the
repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama and ensure the configured model is available. The
client defaults to model `phi3` and host `http://172.31.224.1:11434`; set
`OLLAMA_MODEL` or `OLLAMA_HOST` if your Ollama service uses different values.
For a local default Ollama endpoint, for example:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
ollama pull phi3
```

Start the API:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API documentation is available at <http://127.0.0.1:8000/docs>. SQLite is
stored at `data/cache.db`; persistent ChromaDB data is stored at `data/chroma/`.
Use a disposable deployment for experiments that write cache entries.

## Benchmark and tests

The benchmark sends the same query workload to each selected policy, uses a
distinct cache namespace per policy, and writes per-request CSV plus per-policy
JSON summaries. See [benchmarks/README.md](benchmarks/README.md) for workload,
metrics, and methodology. A small comparison command is:

```bash
python benchmarks/benchmark.py --base-url http://127.0.0.1:8000 --repetitions 2 --policies no_cache exact_only pure_semantic proposed
```

This sends live requests and can call Ollama; it is not part of the unit test
suite. Run deterministic tests with:

```bash
PYTHONPATH=. pytest -q
```

The benchmark measures cache decisions and request latency, not answer
correctness. Rigorous evaluation still requires repeated controlled runs,
answer-quality assessment independent of cache hits, and analysis of cache
correctness, freshness, cross-user isolation, hit rate, and latency
distributions. No performance improvement is claimed here.

The exploratory, fixed-pair semantic distance evaluation is documented in
[evaluation/README.md](evaluation/README.md). It runs against an ephemeral
Chroma collection and does not change production thresholds.

The API has no authentication layer: it scopes entries by the `user_id` and
other identifiers supplied in each request. This prototype therefore does not
verify that a caller is entitled to claim a particular user ID.

## Development

Do not commit virtual environments, SQLite databases, ChromaDB runtime data,
or environment files.
