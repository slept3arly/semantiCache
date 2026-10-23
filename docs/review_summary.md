# semantiCache — Review Summary

## Objective and research problem

This university prototype studies whether cached LLM responses can be reused
for repeated or similar requests while respecting user, session, conversation,
category, and freshness constraints. The research question is whether
metadata-constrained lexical and semantic retrieval can improve cache reuse
without returning entries outside the request's permitted scope. Implemented
functionality is described here; research novelty, performance benefit, and
answer correctness have not been established.

## Approach and architecture

FastAPI serves requests. SQLAlchemy and SQLite store cache responses and
relational metadata; ChromaDB stores embeddings and semantic metadata;
SentenceTransformers (`all-MiniLM-L6-v2`) produces embeddings; Ollama generates
responses on misses. A newly generated entry receives a one-hour TTL. Expiry is
checked against UTC time; Chroma stores expiry as a numeric UTC timestamp.

The default `proposed` pipeline is:

```text
validated request → exact SQLite → lexical SQLite → scoped semantic ChromaDB
                  → Ollama fallback → policy-specific writes
```

Exact and lexical candidates are scoped by user, session, conversation,
category, and unexpired TTL. Lexical retrieval uses lowercase Unicode
alphanumeric token sets and Jaccard similarity, accepting scores `>= 0.8`.
Ties choose the lowest SQLite row ID. Semantic search filters and revalidates
the same metadata and expiry, with the existing Chroma distance threshold
`< 0.6`. Lexical overlap and vector distance are retrieval signals, not answer
quality judgments.

## Implemented policies

- `no_cache`: no cache reads or writes; generate every response.
- `exact_only`: exact SQLite reads; misses are generated and written to SQLite.
- `pure_semantic`: skip exact and lexical reads; use metadata-scoped semantic
  retrieval and the existing validation; write misses to SQLite and ChromaDB.
- `proposed`: exact → lexical → metadata-aware semantic; write misses to SQLite
  and ChromaDB. This is the default when `cache_policy` is omitted.

`cache_policy` is validated by the request schema. Optional `cache_namespace`
derives a hashed internal user key from namespace and user ID, partitioning
experiment entries without a schema migration. Omitting it preserves the
existing cache key behavior. This namespace isolates benchmark policies; it is
not authentication. The API has no authentication layer and trusts the
request-supplied user identifier for scoping.

## Validation performed

The automated suite was run with:

```text
PYTHONPATH=. pytest -q
40 passed in 4.82s
```

The tests cover policy validation and routing, in-memory SQLite isolation,
metadata and expiry checks, lexical behavior, semantic fallback, benchmark
workload separation, and metric aggregation. Unit tests use mocks and isolated
storage; they do not establish live service behavior or answer quality.

`git diff --check` passed for this documentation update.

## Existing live benchmark artifact

An existing local, untracked benchmark artifact was created at
`2026-10-23T09:09:07Z`, labeled
`current_exact_then_metadata_aware_semantic`. It records six HTTP 200 responses,
one `exact_cache` hit, five `ollama` responses, and zero errors. Its recorded
median latency is **3,989.47 ms**, nearest-rank P95 is **11,526.49 ms**, and hit
rate is **1/6 (16.7%)**. This is a small historical single-configuration run;
it predates the current four-policy comparison and is not evidence of a
comparative performance improvement. It did not measure answer quality.
The recorded files are [CSV](../benchmark-results/393012d3888742f5b43bd2b213a83af2.csv)
and [summary JSON](../benchmark-results/393012d3888742f5b43bd2b213a83af2.summary.json).

No new live benchmark was run for this review.

## Limitations and evaluation next steps

The lexical cutoff is an initial heuristic, not an empirically calibrated
boundary. Semantic hits are not judged for factual correctness. The benchmark
is sequential, calls a live Ollama service on misses, and is sensitive to host
load, model warm-up, and service state. Reusing a run ID also reuses its
namespaces. User IDs are caller-supplied and are not authenticated. Persistent
SQLite and ChromaDB stores require a disposable deployment for experiments.

A rigorous study should use repeated runs with controlled hardware and model
settings, document warm-up and concurrency, preserve isolated policy
namespaces, and report latency distributions, hit rate, freshness, and
cross-user decision correctness. Evaluate answer quality separately using a
defined dataset and independent criteria; compare policies without inferring
answer correctness from cache hits.
