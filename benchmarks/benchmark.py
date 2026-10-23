#!/usr/bin/env python3
"""Exercise the running semantiCache API and write per-request measurements."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache_policy import CachePolicy


CSV_FIELDS = [
    "experiment",
    "cache_policy",
    "cache_namespace",
    "request_id",
    "query",
    "user_id",
    "session_id",
    "conversation_id",
    "category",
    "expected_cache_hit",
    "actual_cache_hit",
    "source",
    "latency_ms",
    "http_status",
    "decision_correct",
    "error",
]


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return nearest-rank percentile (rank = ceil(p * n)), or None if empty."""
    if not values:
        return None
    if not 0 < percentile_value <= 1:
        raise ValueError("percentile_value must be in (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(percentile_value * len(ordered))
    return ordered[rank - 1]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in records]
    good_records = [
        record
        for record in records
        if isinstance(record.get("actual_cache_hit"), bool)
        and record.get("error") in (None, "")
    ]
    errors = sum(bool(record.get("error")) for record in records)
    hits = sum(record["actual_cache_hit"] for record in good_records)
    return {
        "request_count": len(records),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "hit_rate": hits / len(good_records) if good_records else None,
        "hit_rate_request_count": len(good_records),
        "error_count": errors,
    }


def build_workload(
    run_id: str,
    repetitions: int,
    policy: CachePolicy | str = CachePolicy.PROPOSED,
) -> list[dict[str, Any]]:
    """Build identical query and metadata sequences under an isolated namespace."""
    policy = CachePolicy(policy)
    workload: list[dict[str, Any]] = []
    prefix = f"bench_{run_id[:24]}"
    run_namespace = sha256(run_id.encode("utf-8")).hexdigest()[:16]
    cache_namespace = f"{run_namespace}_{policy.value}"

    def add(experiment: str, query: str, scope: str, expected: bool | None) -> None:
        number = len(workload) + 1
        workload.append(
            {
                "experiment": experiment,
                "cache_policy": policy.value,
                "cache_namespace": cache_namespace,
                "request_id": f"{run_id}-{number:04d}",
                "query": query,
                "user_id": f"{prefix}_{scope}_user",
                "session_id": f"{prefix}_{scope}_session",
                "conversation_id": f"{prefix}_{scope}_conversation",
                "category": "benchmark",
                "expected_cache_hit": expected,
            }
        )

    for index in range(repetitions):
        add(
            "warm_exact",
            "What is the capital of France?",
            "warm_exact",
            expected=(
                False
                if policy == CachePolicy.NO_CACHE and index == 0
                else policy != CachePolicy.NO_CACHE and index > 0
            ),
        )

    for index in range(repetitions):
        add(
            "cold_unique",
            f"Define benchmark cold request number {index + 1}.",
            f"cold_{index:04d}",
            expected=False,
        )

    add(
        "semantic_probe",
        "What is photosynthesis?",
        "semantic_probe",
        expected=False,
    )
    add(
        "semantic_probe",
        "Can you explain how plants turn sunlight into food?",
        "semantic_probe",
        expected=(
            False
            if policy in (CachePolicy.NO_CACHE, CachePolicy.EXACT_ONLY)
            else None
        ),
    )
    return workload


def _request_one(base_url: str, item: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    status: int | None = None
    actual_hit: bool | None = None
    source: str | None = None
    error: str | None = None
    payload = {
        key: item[key]
        for key in (
            "user_id",
            "session_id",
            "conversation_id",
            "category",
            "query",
            "cache_policy",
            "cache_namespace",
        )
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/cache/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body.get("cache_hit"), bool) or not isinstance(body.get("source"), str):
                raise ValueError("response is missing a boolean cache_hit or string source")
            actual_hit = body["cache_hit"]
            source = body["source"]
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}: {exc.reason}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000
    expected = item["expected_cache_hit"]
    return {
        **item,
        "actual_cache_hit": actual_hit,
        "source": source,
        "latency_ms": latency_ms,
        "http_status": status,
        "decision_correct": actual_hit == expected if expected is not None and actual_hit is not None else None,
        "error": error,
    }


def run_benchmark(
    base_url: str,
    run_id: str,
    repetitions: int,
    timeout: float,
    policies: list[CachePolicy] | None = None,
) -> list[dict[str, Any]]:
    selected_policies = policies or list(CachePolicy)
    records = []
    for policy in selected_policies:
        for item in build_workload(run_id, repetitions, policy):
            records.append(_request_one(base_url, item, timeout))
    return records


def summarize_by_policy(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["cache_policy"]].append(record)
    return {policy: summarize(items) for policy, items in sorted(grouped.items())}


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "expected_cache_hit": _csv_bool(record["expected_cache_hit"]),
                    "actual_cache_hit": _csv_bool(record["actual_cache_hit"]),
                    "decision_correct": _csv_bool(record["decision_correct"]),
                }
            )


def _csv_bool(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=125.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=[policy.value for policy in CachePolicy],
        default=[policy.value for policy in CachePolicy],
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    policies = [CachePolicy(value) for value in args.policies]
    records = run_benchmark(args.base_url, args.run_id, args.repetitions, args.timeout, policies)
    output = args.output or Path("benchmark-results") / f"{args.run_id}.csv"
    write_csv(output, records)
    summary = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "configurations": [policy.value for policy in policies],
        "summaries_by_configuration": summarize_by_policy(records),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"CSV: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
