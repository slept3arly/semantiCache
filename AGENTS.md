# AGENTS.md — semantiCache

## 1. Project Purpose

This repository contains a university research prototype for semantic caching in multi-user LLM systems. The objective is to investigate whether semantically similar requests can reuse cached responses while maintaining cache correctness, freshness, and user isolation.

Do not claim research novelty or performance improvements without supporting literature review and experimental results.

## 2. Technology Stack

- Python 3.12
- FastAPI for the API
- SQLite with SQLAlchemy for relational metadata and cached responses
- ChromaDB for vector storage and similarity retrieval
- Sentence Transformers using `all-MiniLM-L6-v2` for embeddings
- Ollama for LLM-generated responses
- pytest for automated testing

Do not introduce new dependencies or infrastructure without explicit approval.

## 3. Known Project Structure

The project has included the following files:

- `app/main.py` — FastAPI endpoints and cache request handling
- `app/semantic_cache.py` — embedding generation and ChromaDB operations
- `app/database.py` — SQLAlchemy database configuration
- `app/models.py` — relational database models
- `app/schemas.py` — request and response schemas
- `app/ollama_client.py` — Ollama response generation
- `tests/test_semantic_cache.py` — semantic-cache tests
- `data/chroma/` — persistent ChromaDB data

This structure is based on the previously observed project. Inspect the actual repository before relying on it; files and implementations may have changed.

## 4. Cache Policies and Request Pipeline

`CacheQueryRequest.cache_policy` is validated and defaults to `proposed`.
Preserve these implemented policy definitions:

- `no_cache`: bypass cache reads and writes; call Ollama.
- `exact_only`: scoped exact SQLite lookup, then Ollama; write misses to SQLite.
- `pure_semantic`: metadata-scoped Chroma retrieval, then Ollama; skip exact and
  lexical retrieval while preserving metadata validation and the semantic
  threshold.
- `proposed`: scoped exact lookup, lexical lookup, metadata-aware semantic
  lookup, then Ollama.

Validate request and policy fields first. When `cache_namespace` is present,
derive the internal cache user key from that namespace and the request user ID.
Persist a generated response only according to the selected policy. Do not
reorder stages or claim a policy exists without checking the implementation.

## 5. Metadata and Cache Correctness

Cache eligibility must be established independently of semantic similarity.

- Enforce user isolation.
- Enforce session isolation.
- Enforce category compatibility.
- Reject expired entries.
- Preserve any additional isolation constraints already required by the existing implementation, including conversation scope where applicable.
- Never return a cached response if its required metadata is missing, malformed, or cannot be validated.
- A rejected semantic candidate must not prevent other eligible candidates from being considered.
- If no eligible candidate meets the configured similarity threshold, treat the request as a cache miss.
- Never weaken isolation or expiry checks to increase the hit rate.

Use UTC consistently for expiration handling. Choose a metadata representation compatible with ChromaDB's supported scalar types and filtering operations.

## 6. Coding Standards

- Use clear, typed Python functions.
- Keep changes minimal and consistent with existing patterns.
- Reuse existing database models and schemas where appropriate.
- Avoid duplicated embedding, retrieval, or validation logic.
- Handle empty search results, missing metadata, malformed records, and duplicate IDs safely.
- Do not silently swallow exceptions or convert unrelated errors into successful cache hits.
- Preserve existing API contracts unless a change is necessary and explicitly justified.
- Do not store sensitive request information in logs unnecessarily.

## 7. Testing Requirements

Add or update tests for:

- Exact-match cache behavior.
- Eligible semantic hits.
- Different-user isolation.
- Different-session isolation.
- Category mismatch.
- Expired entries.
- Missing or malformed metadata.
- Similarity-threshold rejection.
- Empty vector-search results.
- Correct fallback to the LLM.
- Persistence and compatibility where relevant.

Use deterministic test fixtures and isolated test data. Avoid depending on network access, downloading models, or calling a live Ollama instance in unit tests unless explicitly required.

Do not report a test as passing unless it has actually passed.

## 8. Research Integrity

The API implements four cache modes for comparison:

A. No cache  
B. Exact-match cache  
C. Pure semantic cache  
D. Proposed metadata-aware semantic cache

Future evaluation should measure cache hit rate, median and P95 latency,
decision correctness, answer quality independently of cache hits, freshness,
and cross-user isolation. Automated unit-test success is not evidence of
performance improvement or semantic answer quality.

Keep implemented functionality separate from proposed functionality. Do not fabricate benchmark results, statistical significance, guarantees, or literature findings.

## 9. Required Workflow for Codex

Before making changes:

1. Inspect the relevant source files, models, schemas, and tests.
2. Explain the current behavior and identify the precise implementation gap.
3. Identify any schema or persistent-data compatibility risks.
4. Implement only the requested feature.

After making changes:

1. Run relevant tests and, where practical, the full test suite.
2. Report the exact commands executed and their results.
3. Summarize files changed and explain important implementation decisions.
4. Identify any remaining limitations or untested behavior.
5. Never claim that the application works end-to-end based only on static inspection.

Prefer a small, correct, reproducible implementation over a large architectural rewrite.
