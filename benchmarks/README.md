# semantiCache benchmark

`benchmark.py` sends a deterministic request sequence to a running API under
each selected policy. It writes per-request CSV observations and per-policy
JSON summaries. Latency covers the HTTP request through response parsing,
including lookup, embedding work, and Ollama generation. The benchmark does not
evaluate answer quality.

## Run safely

Run against a disposable deployment with its own SQLite and ChromaDB stores.
Cache policies write misses to those stores, and this harness does not reset or
delete data. Start the API and configured Ollama service in that environment,
then from the repository root run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
python benchmarks/benchmark.py --base-url http://127.0.0.1:8000 --repetitions 2 \
  --policies no_cache exact_only pure_semantic proposed
```

The default run ID is a fresh random UUID. A fixed `--run-id` reproduces the
same workload and measured namespaces. Reusing that ID deliberately reuses its
persistent entries, so it is a warm-state continuation, not a cold repetition.
Use a **new run ID for a cold run**. Output defaults to
`benchmark-results/<run-id>.csv` and an adjacent `.summary.json`. The output
path is overwritten if it already exists.

Each run ID and policy pair maps to a distinct `cache_namespace`. Requests
within a case share their intended user/session/conversation scope, while cases use separate
scope IDs. This prevents cases from seeding one another. Namespaces are derived
from the run ID and policy, and the API hashes namespace plus user ID for the
internal cache key. Stores are not cleared: reusing a run ID can retrieve old
entries, and the truncated namespace hash has a theoretical collision risk.
The JSON summary records that namespaces are isolated but a fresh run ID is
required; the CSV records the namespace on each request.

Before measured requests, the harness sends one unmeasured `proposed` request by
default. Set `--warmup-requests 0` to disable it or choose another nonnegative
count. The warm-up namespace includes a fresh per-invocation UUID and is
separate from measured namespaces; warm-up data is left in the persistent stores
and is not deleted. A successful proposed miss loads the lazy embedding model,
calls Ollama, and stores only under the warm-up namespace. Warm-up errors are
recorded in the summary and do not become measured rows. This reduces first-use
effects but does not eliminate backend, hardware, or service variability.

## Workload and labels

With `--repetitions N`, each policy receives `2 × N + 11` requests. The fixed
queries and case IDs are constructed in `build_workload`:

| Case | Requests and purpose | Expected decision |
| --- | --- | --- |
| `exact_repeat` | Seed with “What is the capital of France?” and issue N identical repeats. | Seed misses. Repeats hit for `exact_only`, `pure_semantic`, and `proposed`; `no_cache` always misses. |
| `paraphrase` | Seed “What is photosynthesis?” then ask “How do plants convert sunlight into chemical energy?” | Manually labeled reusable. Semantic-enabled policies are expected to hit; `no_cache` and `exact_only` are expected to miss. A semantic miss is recorded as incorrect relative to this label, not rewritten. |
| `related_distinct` | Seed “What is photosynthesis?” then ask “What is the chemical equation for photosynthesis?” | Manually labeled non-reusable; probe expected to miss under every policy. |
| `unrelated` | Seed “What is photosynthesis?” then ask “How do I reset my email password?” | Manually labeled non-reusable; probe expected to miss under every policy. |
| `cold_unique` | Submit N unique query and scope combinations. | Every request is expected to miss. |
| `metadata_isolation` | Seed a boiling-point question, then repeat the identical question once as another user and once in another session. | Answer content is labeled reusable for all rows, but every request is expected to miss: probes test user and session authorization boundaries. |

`expected_reusable` labels whether the cached answer content is appropriate for
the target query, independent of cache scope or authorization. `expected_cache_hit`
is the expected behavior of the complete policy, including metadata constraints;
`actual_cache_hit` is the API's observed response. Thus the metadata-isolation
probes have reusable content and an expected miss. The paraphrase label is an
explicit hypothesis, not a guarantee that the current embedding model and
`distance < 0.6` rule will accept it. Related and unrelated probes are labeled
non-reusable, so a hit is incorrect against the content label. These manual
labels do not establish that an answer is factually correct or useful.

The sequence is deterministic for a fixed run ID, repetition count, and
policies. Policy workloads use the same request IDs, case IDs, queries, and raw
metadata identifiers. For each case role and repeat iteration, the harness
issues that phase for all policies, rotating the selected policy order by one
position from the previous phase. Seed and probe ordering within each policy is
preserved. This reduces fixed-order bias; it does not eliminate backend,
hardware, or service variability. The workload is not publication-ready by
itself.

## Metrics and output

The CSV records case/request IDs, case type and role, policy, namespace, query
and metadata identifiers, expected reuse and cache decision, actual hit,
source, latency, HTTP status, decision correctness, and error. `decision_correct`
is blank when the expectation is undefined, the response lacks a valid hit
boolean, or the request failed.

Per-policy summaries report request count, median latency, nearest-rank P95
(`ceil(0.95 × n)`), hit rate, decision accuracy and its evaluation count, and
error count. Hit rate uses successful responses with a boolean `cache_hit` as
its denominator. Decision accuracy uses only successful requests with a defined
expected decision and a boolean actual result. Failed requests are not counted
as cache misses in either denominator. Latency summaries include recorded
failed-request durations. Empty metric samples are JSON `null` where applicable.

Decision accuracy compares `actual_cache_hit` with `expected_cache_hit`; it is a
whole-policy cache-decision metric, not a semantic false-accept/false-reject
metric. A metadata-ineligible probe that misses is correctly scored as a policy
decision, not counted as a semantic false reject. The benchmark does not
currently calculate semantic false accepts or false rejects. A future semantic
error analysis should use only content-labeled probes that are eligible under
the requested metadata scope, and report metadata-denied cases separately.

The four policies retain their API definitions: `no_cache` bypasses reads and
writes; `exact_only` uses scoped exact SQLite lookup; `pure_semantic` uses
metadata-scoped Chroma retrieval with `distance < 0.6`; and `proposed` follows
exact, lexical (`Jaccard >= 0.8`), metadata-aware semantic retrieval, then
Ollama. Cache hits and decision accuracy are retrieval measures, not answer
quality measures.

The JSON summary records repetitions, workload version, request timeout, the
actual policy order for each phase, warm-up count/namespace/errors, cache
namespace strategy, thresholds, one-hour TTL, benchmark-process Python,
ChromaDB, and Sentence Transformers versions, Ollama model identifier and
endpoint, and embedding model name. The Ollama values come from the benchmark
process environment and local source defaults; the API process does not expose
its runtime settings, so they may differ. Likewise, runtime package versions
are those of the benchmark process. The embedding revision is reported as
unknown: production constructs SentenceTransformer from a model name without
pinning or exposing a resolved revision. Project constants are read from the
local source checkout without loading the embedding model or opening the
persistent Chroma collection. For a remote API or a server running a different
checkout, those local constants may not describe the serving process.
