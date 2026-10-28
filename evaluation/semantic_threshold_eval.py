#!/usr/bin/env python3
"""Measure Chroma semantic distances for a small, labeled query-pair set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_THRESHOLDS = (0.4, 0.6, 0.8, 1.0, 1.2)
EVALUATION_NAMESPACE = "semantic-threshold-evaluation"
EXPIRY_TTL = timedelta(hours=1)


def evaluate_thresholds(
    records: list[dict[str, Any]], thresholds: tuple[float, ...] | list[float]
) -> list[dict[str, Any]]:
    """Classify metadata-eligible pairs using Chroma's strict distance cutoff."""
    reports = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("thresholds must be finite positive numbers")

        accepted = [
            record["pair_id"]
            for record in records
            if record["metadata_eligible"] and record["distance"] < threshold
        ]
        rejected = [
            record["pair_id"]
            for record in records
            if not record["metadata_eligible"] or record["distance"] >= threshold
        ]
        false_accepts = [
            record["pair_id"]
            for record in records
            if record["metadata_eligible"]
            and record["distance"] < threshold
            and not record["expected_reuse"]
        ]
        false_rejects = [
            record["pair_id"]
            for record in records
            if record["metadata_eligible"]
            and record["distance"] >= threshold
            and record["expected_reuse"]
        ]
        reports.append(
            {
                "threshold": threshold,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "accepted_pair_ids": accepted,
                "rejected_pair_ids": rejected,
                "false_accept_count": len(false_accepts),
                "false_accept_pair_ids": false_accepts,
                "false_reject_count": len(false_rejects),
                "false_reject_pair_ids": false_rejects,
                "labeled_pair_count": sum(bool(record["metadata_eligible"]) for record in records),
                "ineligible_pair_ids": [
                    record["pair_id"] for record in records if not record["metadata_eligible"]
                ],
            }
        )
    return reports


def _model_revision(model: Any) -> str | None:
    for module in model.modules():
        config = getattr(getattr(module, "auto_model", None), "config", None)
        if config is not None:
            return getattr(config, "_commit_hash", None)
    return None


def measure_pairs(
    dataset: dict[str, Any],
    *,
    model_name_or_path: str,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Measure with the production embedder/metadata helpers and an ephemeral collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    from app import semantic_cache
    from app.cache_policy import scoped_cache_user_id

    # local_files_only avoids network access and keeps this script from silently
    # benchmarking a newly downloaded model revision.
    model = SentenceTransformer(model_name_or_path, local_files_only=True)
    observed_at = datetime.now(timezone.utc)
    expires_at = observed_at + EXPIRY_TTL
    client = chromadb.EphemeralClient()
    ephemeral_collection = client.create_collection("semantic_threshold_eval")

    original_collection = semantic_cache.collection
    original_get_embedder = semantic_cache._get_embedder
    semantic_cache.collection = ephemeral_collection
    semantic_cache._get_embedder = lambda: model

    records = []
    try:
        for pair in dataset["pairs"]:
            pair_id = pair["pair_id"]
            user_id = scoped_cache_user_id(f"eval-{pair_id}", EVALUATION_NAMESPACE)
            scope = {
                "user_id": user_id,
                "session_id": "threshold-eval-session",
                "conversation_id": pair_id,
                "category": "threshold-evaluation",
            }
            semantic_cache.add_to_cache(
                cache_id=pair_id,
                prompt=pair["source_query"],
                response="evaluation-only response placeholder",
                **scope,
                expires_at=expires_at,
            )
            results = semantic_cache.search_cache(
                pair["target_query"],
                **scope,
                now=observed_at,
                n_results=1,
            )
            docs = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            candidate_count = min(len(docs), len(metadatas), len(distances))
            if candidate_count != 1:
                raise RuntimeError(
                    f"Expected one filtered candidate for {pair_id}, found {candidate_count}"
                )

            metadata = metadatas[0]
            distance = float(distances[0])
            # A neutral zero distance delegates scope, response, and expiry
            # revalidation to the same production validator while threshold
            # classification remains an independent offline calculation.
            metadata_eligible = semantic_cache.valid_candidate(
                metadata,
                0.0,
                **scope,
                now=observed_at,
            )
            records.append(
                {
                    "pair_id": pair_id,
                    "group": pair["group"],
                    "source_query": pair["source_query"],
                    "target_query": pair["target_query"],
                    "expected_reuse": pair["expected_reuse"],
                    "label_rationale": pair["label_rationale"],
                    "candidate_count": candidate_count,
                    "distance": distance,
                    "metadata_eligible": metadata_eligible,
                    "candidate_scope_matches": all(
                        metadata.get(field) == value for field, value in scope.items()
                    ),
                    "candidate_expires_at": metadata.get("expires_at"),
                    "passes_current_threshold": (
                        metadata_eligible and distance < semantic_cache.SIMILARITY_THRESHOLD
                    ),
                }
            )
    finally:
        semantic_cache.collection = original_collection
        semantic_cache._get_embedder = original_get_embedder

    return {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["dataset_version"],
        "label_definition": dataset["label_definition"],
        "measured_at_utc": observed_at.isoformat(),
        "model_name_or_path": model_name_or_path,
        "model_revision": _model_revision(model),
        "chromadb_version": version("chromadb"),
        "collection_space": ephemeral_collection.configuration["hnsw"]["space"],
        "current_production_threshold": semantic_cache.SIMILARITY_THRESHOLD,
        "candidate_thresholds_use_rule": "distance < threshold",
        "expiry_ttl_hours": EXPIRY_TTL.total_seconds() / 3600,
        "collection_type": "ephemeral",
        "records": records,
        "threshold_reports": evaluate_thresholds(records, thresholds),
    }


def write_results(output_path: Path, results: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    fields = [
        "pair_id",
        "group",
        "source_query",
        "target_query",
        "distance",
        "expected_reuse",
        "passes_current_threshold",
        "metadata_eligible",
        "candidate_count",
        "candidate_scope_matches",
        "candidate_expires_at",
        "label_rationale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results["records"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("semantic_pairs.json"),
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "semantic_threshold_evaluation.json",
    )
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = measure_pairs(
        dataset,
        model_name_or_path=args.model,
        thresholds=tuple(args.thresholds),
    )
    write_results(args.output, results)
    print(json.dumps({
        "output_json": str(args.output),
        "output_csv": str(args.output.with_suffix('.csv')),
        "model_revision": results["model_revision"],
        "current_threshold": results["current_production_threshold"],
        "threshold_reports": results["threshold_reports"],
    }, indent=2))


if __name__ == "__main__":
    main()
