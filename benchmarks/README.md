# semantiCache benchmark

`benchmark.py` sends the same fixed sequence of requests to a running
semantiCache API under each selected cache policy. It writes all per-request
measurements to CSV and per-policy summaries to adjacent JSON. Latency is
end-to-end HTTP time, including API lookup, embedding work, and any Ollama
generation.

## Run

Start the application and its configured Ollama service, then run from the
repository root:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
python benchmarks/benchmark.py --base-url http://127.0.0.1:8000 --repetitions 2
```

The default run compares all four policies. A small explicit comparison is:

```bash
python benchmarks/benchmark.py --base-url http://127.0.0.1:8000 --repetitions 2 --policies no_cache exact_only pure_semantic proposed
```

The default run writes files under `benchmark-results/`. Use `--output
benchmark-results/run.csv` to choose a CSV path; the summary is written next to
it with `.summary.json` appended in place of `.csv`. Each run gets a generated
run ID and distinct namespace per policy. Raw user/session/conversation IDs and
queries are held constant across policies; the namespace is an additional
internal partition. Use a fresh run ID for a cold comparison. Reusing it can
make a later run warm.

The workload contains three experiments:

- `warm_exact`: first request is expected to miss; identical repeats are
  expected to hit under the selected policy (the `source` identifies whether
  the hit was exact or semantic).
- `cold_unique`: each request has a separate user/session/conversation scope and
  is expected to miss.
- `semantic_probe`: a seed question followed by a paraphrase. The paraphrase's
  cache-hit expectation is undefined for semantic-enabled policies because
  model similarity is measured, not assumed. It is expected to miss for
  `no_cache` and `exact_only`.

The first request in each repeated-query or semantic-probe sequence is the
measured seed; there is no unmeasured warm-up phase. `no_cache` expects misses
throughout. For caching policies, the first repeated-query request is expected
to miss and subsequent identical requests to hit. Under `pure_semantic`, those
repeats are semantic hits rather than exact hits. Cold-unique requests use a
different scope per item and are expected misses.

`decision_correct` compares actual `cache_hit` with the expected decision only
when that expectation is defined. It is a cache-decision check, not a judgment
that a semantically returned answer is relevant or factually correct. No answer
quality evaluation is performed. Hit rate uses successful responses that
returned a boolean `cache_hit`; errors are counted separately. P95 uses the
nearest-rank definition (`ceil(0.95 * n)`). Empty latency or hit-rate samples
are reported as JSON `null`.

## Available configurations and limits

The API accepts a validated `cache_policy` request field and optional
`cache_namespace`. `proposed` is the default when omitted. Policies are:

- `no_cache`: bypass all cache reads and writes; call Ollama per request.
- `exact_only`: exact SQLite lookup, then Ollama; generated responses are
  written to SQLite only.
- `pure_semantic`: skip exact and lexical lookup; use the existing
  metadata-scoped Chroma search and `distance < 0.6`, then Ollama. It writes to
  SQLite and Chroma. This is a semantic-only retrieval baseline with user,
  session, conversation, category, and expiry enforcement, not an unscoped or
  unsafe semantic search.
- `proposed`: exact → lexical (`Jaccard >= 0.8`) → metadata-aware semantic →
  Ollama. It writes to SQLite and Chroma.

Per-policy summaries include request count, median and nearest-rank P95
latency, hit rate, and error count. The CSV records policy, experiment,
expected and actual cache decisions, source, status, latency, and errors.
Decision correctness is not answer-quality evaluation; no response correctness
or semantic equivalence is inferred from a cache hit.

This is a request-level prototype benchmark, not a controlled performance or
answer-quality study. Cold requests call the configured LLM and can fail or vary
with service availability, model state, and generation behavior. Latency also
depends on the host, persistent database contents, model warm-up, and concurrent
load. The namespace partitions cache entries across policies using a stable
hashed internal user key; omitting the namespace preserves ordinary application
behavior. It does not isolate one benchmark run from a prior run that reuses
the same run ID. Run against a disposable/test deployment when possible:
caching policies populate its SQLite and ChromaDB stores.
