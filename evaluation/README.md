# Exploratory semantic-threshold evaluation

This evaluation measures ChromaDB distances for a fixed set of nine source and
target query pairs using `all-MiniLM-L6-v2`. It includes three manually labeled
paraphrase pairs, three related-but-distinct pairs, and three unrelated pairs.
The label and rationale for each pair are in [semantic_pairs.json](semantic_pairs.json).
`expected_reuse` is defined independently of distance: whether one cached
answer to the source query would reasonably answer the target query.

## Run

The model must already be available locally; evaluation is local-only and does
not download models or call Ollama:

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 PYTHONPATH=. python evaluation/semantic_threshold_eval.py
```

The script uses an ephemeral Chroma collection and the production
`add_to_cache`, `search_cache`, and candidate metadata validator. Each pair has
an isolated user/session/conversation/category scope and a future expiry, so
the production metadata filter and expiry revalidation apply. The ephemeral
collection uses Chroma's default `l2` space, as does the current production
collection. No production cache records are inserted. The output records model
revision, ChromaDB version, measured pair distances, scope/expiry eligibility,
current-threshold acceptance, and accepted/rejected pair IDs for each candidate
threshold. Default thresholds are `0.4`, `0.6`, `0.8`, `1.0`, and `1.2`; all use
the strict rule `distance < threshold`.

JSON and CSV results are written to `evaluation/results/`. The recorded
[result JSON](results/semantic_threshold_evaluation.json) and
[per-pair CSV](results/semantic_threshold_evaluation.csv) were measured on
2026-10-27 using model revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and ChromaDB 1.5.9.

At the existing production threshold `0.6`, the nine labeled pairs produced
three accepted pairs, one false accept, and one false reject. The false reject
was the photosynthesis paraphrase (distance `0.7318053`); the false accept was
the related-but-distinct photosynthesis question (distance `0.5643823`). At
`0.4`, there were no false accepts and one false reject; at `0.8`, there were
three false accepts and no false rejects. These are exploratory counts on this
small, manually labeled dataset, not calibrated estimates or evidence for a
production threshold change.

Embedding distance alone does not establish that cached answers are correct,
equivalent, or useful. The evaluation measures retrieval decisions against
small manually defined pair labels; it does not evaluate generated answer
quality, user experience, latency, or performance under load. Threshold choice
requires a larger, independently reviewed dataset and answer-quality evaluation
separate from cache-hit classification.
