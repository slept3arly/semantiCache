#!/usr/bin/env python3
"""Exercise the running semantiCache API and write per-request measurements."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
import json
import math
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache_policy import CachePolicy


CSV_FIELDS = [
    "experiment",
    "case_id",
    "case_type",
    "case_role",
    "cache_policy",
    "cache_namespace",
    "request_id",
    "query",
    "user_id",
    "session_id",
    "conversation_id",
    "category",
    "expected_reusable",
    "expected_cache_hit",
    "actual_cache_hit",
    "source",
    "latency_ms",
    "http_status",
    "decision_correct",
    "error",
]
WORKLOAD_VERSION = "policy-comparable-v2"


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
    decision_records = [
        record
        for record in records
        if record.get("expected_cache_hit") is not None
        and isinstance(record.get("actual_cache_hit"), bool)
        and record.get("error") in (None, "")
    ]
    correct_decisions = sum(
        record["actual_cache_hit"] == record["expected_cache_hit"]
        for record in decision_records
    )
    return {
        "request_count": len(records),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "hit_rate": hits / len(good_records) if good_records else None,
        "hit_rate_request_count": len(good_records),
        "decision_correct_count": correct_decisions,
        "decision_evaluation_count": len(decision_records),
        "decision_accuracy": (
            correct_decisions / len(decision_records) if decision_records else None
        ),
        "error_count": errors,
    }


def build_workload(
    run_id: str,
    repetitions: int,
    policy: CachePolicy | str = CachePolicy.PROPOSED,
) -> list[dict[str, Any]]:
    """Build labeled, policy-comparable cases within a run/policy namespace."""
    policy = CachePolicy(policy)
    workload: list[dict[str, Any]] = []
    prefix = f"bench_{run_id[:24]}"
    run_namespace = sha256(run_id.encode("utf-8")).hexdigest()[:16]
    cache_namespace = f"{run_namespace}_{policy.value}"

    def add(
        case_id: str,
        case_type: str,
        role: str,
        query: str,
        scope: str,
        *,
        expected: bool | None,
        reusable: bool | None,
        user_suffix: str = "",
        session_suffix: str = "",
    ) -> None:
        number = len(workload) + 1
        workload.append(
            {
                "experiment": case_type,
                "case_id": case_id,
                "case_type": case_type,
                "case_role": role,
                "cache_policy": policy.value,
                "cache_namespace": cache_namespace,
                "request_id": f"{run_id}-{number:04d}",
                "query": query,
                "user_id": f"{prefix}_{scope}{user_suffix}_user",
                "session_id": f"{prefix}_{scope}{session_suffix}_session",
                "conversation_id": f"{prefix}_{scope}_conversation",
                "category": "benchmark",
                "expected_reusable": reusable,
                "expected_cache_hit": expected,
            }
        )

    def labeled_pair(
        case_type: str,
        seed_query: str,
        probe_query: str,
        *,
        reusable: bool,
    ) -> None:
        scope = case_type
        semantic_policy = policy in (CachePolicy.PURE_SEMANTIC, CachePolicy.PROPOSED)
        probe_expected = reusable if semantic_policy else False
        add(
            case_type,
            case_type,
            "seed",
            seed_query,
            scope,
            expected=False,
            reusable=reusable,
        )
        add(
            case_type,
            case_type,
            "probe",
            probe_query,
            scope,
            expected=probe_expected,
            reusable=reusable,
        )

    for index in range(repetitions + 1):
        add(
            "exact_repeat-capital-france",
            "exact_repeat",
            "seed" if index == 0 else "repeat",
            "What is the capital of France?",
            "warm_exact",
            expected=False if index == 0 or policy == CachePolicy.NO_CACHE else True,
            reusable=True,
        )

    labeled_pair(
        "paraphrase",
        "What is photosynthesis?",
        "How do plants convert sunlight into chemical energy?",
        reusable=True,
    )
    labeled_pair(
        "related_distinct",
        "What is photosynthesis?",
        "What is the chemical equation for photosynthesis?",
        reusable=False,
    )
    labeled_pair(
        "unrelated",
        "What is photosynthesis?",
        "How do I reset my email password?",
        reusable=False,
    )

    for index in range(repetitions):
        add(
            f"cold_unique-{index + 1:04d}",
            "cold_unique",
            "unique",
            f"Define benchmark cold request number {index + 1}.",
            f"cold_{index:04d}",
            expected=False,
            reusable=None,
        )

    isolation_query = "What is the boiling point of water?"
    for variant, suffix_kind in (("user", "user"), ("session", "session")):
        scope = f"metadata_isolation_{variant}"
        add(
            f"metadata_isolation-{variant}",
            "metadata_isolation",
            "seed",
            isolation_query,
            scope,
            expected=False,
            reusable=True,
        )
        add(
            f"metadata_isolation-{variant}",
            "metadata_isolation",
            f"probe_different_{variant}",
            isolation_query,
            scope,
            expected=False,
            reusable=True,
            user_suffix="_other" if suffix_kind == "user" else "",
            session_suffix="_other" if suffix_kind == "session" else "",
        )
    return workload


def build_warmup_workload(execution_id: str, request_count: int) -> list[dict[str, Any]]:
    """Build unmeasured proposed requests in a namespace isolated from measured data."""
    namespace = f"warmup_{execution_id}"
    items = []
    for index in range(request_count):
        scope = f"warmup_{execution_id}_{index:04d}"
        items.append(
            {
                "experiment": "warmup",
                "case_id": f"warmup-{index + 1:04d}",
                "case_type": "warmup",
                "case_role": "initialization",
                "cache_policy": CachePolicy.PROPOSED.value,
                "cache_namespace": namespace,
                "request_id": f"warmup-{execution_id}-{index + 1:04d}",
                "query": f"Benchmark component warm-up request {index + 1}.",
                "user_id": f"{scope}_user",
                "session_id": f"{scope}_session",
                "conversation_id": f"{scope}_conversation",
                "category": "benchmark_warmup",
                "expected_reusable": None,
                "expected_cache_hit": False,
            }
        )
    return items


def build_execution_plan(
    run_id: str,
    repetitions: int,
    policies: list[CachePolicy],
) -> list[dict[str, Any]]:
    """Rotate policy order by request phase, keeping each seed/probe sequence intact."""
    if not policies:
        return []
    if len(set(policies)) != len(policies):
        raise ValueError("policies must not contain duplicates")
    workloads = {
        policy.value: build_workload(run_id, repetitions, policy)
        for policy in policies
    }
    first_workload = workloads[policies[0].value]
    case_ids = list(dict.fromkeys(item["case_id"] for item in first_workload))
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for policy in policies:
        cases: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for item in workloads[policy.value]:
            cases[item["case_id"]][item["case_role"]].append(item)
        grouped[policy.value] = cases

    role_order = {
        case_id: list(dict.fromkeys(item["case_role"] for item in first_workload if item["case_id"] == case_id))
        for case_id in case_ids
    }
    plan: list[dict[str, Any]] = []
    rotation = 0
    for case_id in case_ids:
        for role in role_order[case_id]:
            phase_count = len(grouped[policies[0].value][case_id][role])
            for index in range(phase_count):
                offset = rotation % len(policies)
                policy_order = policies[offset:] + policies[:offset]
                plan.append(
                    {
                        "case_id": case_id,
                        "case_role": role,
                        "iteration": index + 1,
                        "policy_order": [policy.value for policy in policy_order],
                        "requests": {
                            policy.value: grouped[policy.value][case_id][role][index]
                            for policy in policies
                        },
                    }
                )
                rotation += 1
    return plan


def run_warmup(
    base_url: str,
    execution_id: str,
    request_count: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """Prime the lazy embedder and Ollama using only the isolated warm-up namespace."""
    return [
        _request_one(base_url, item, timeout)
        for item in build_warmup_workload(execution_id, request_count)
    ]


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
        "decision_correct": (
            actual_hit == expected
            if expected is not None and actual_hit is not None and error is None
            else None
        ),
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
    for phase in build_execution_plan(run_id, repetitions, selected_policies):
        for policy in phase["policy_order"]:
            records.append(
                _request_one(base_url, phase["requests"][policy], timeout)
            )
    return records


def _assigned_literal(relative_path: str, variable: str) -> Any:
    """Read a literal source setting without importing modules with side effects."""
    source_path = Path(__file__).resolve().parents[1] / relative_path
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        value: ast.expr | None = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in targets
        ):
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError):
                return None
    return None


def _environment_default(relative_path: str, variable: str) -> str | None:
    """Resolve a simple os.getenv(name, default) assignment from app source."""
    source_path = Path(__file__).resolve().parents[1] / relative_path
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        ):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or len(call.args) < 2:
            return None
        try:
            env_name = ast.literal_eval(call.args[0])
            default = ast.literal_eval(call.args[1])
        except (ValueError, TypeError):
            return None
        return os.environ.get(env_name, default)
    return None


def _safe_endpoint(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return None
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return None


def _call_keyword_literal(relative_path: str, function_name: str, keyword_name: str) -> Any:
    source_path = Path(__file__).resolve().parents[1] / relative_path
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if name != function_name:
            continue
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                try:
                    return ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    return None
    return None


def _cache_ttl_hours() -> int | float | None:
    source_path = Path(__file__).resolve().parents[1] / "app/main.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "expires_at"
            for target in node.targets
        ):
            continue
        value = node.value
        if not isinstance(value, ast.BinOp) or not isinstance(value.right, ast.Call):
            continue
        if not isinstance(value.right.func, ast.Name) or value.right.func.id != "timedelta":
            continue
        for keyword in value.right.keywords:
            if keyword.arg == "hours":
                try:
                    result = ast.literal_eval(keyword.value)
                    return result if isinstance(result, (int, float)) else None
                except (ValueError, TypeError):
                    return None
    return None


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def experiment_settings() -> dict[str, Any]:
    """Collect result-affecting project settings without initializing Chroma/models."""
    return {
        "semantic_threshold": _assigned_literal(
            "app/semantic_cache.py", "SIMILARITY_THRESHOLD"
        ),
        "lexical_threshold": _assigned_literal(
            "app/lexical_cache.py", "LEXICAL_SIMILARITY_THRESHOLD"
        ),
        "cache_ttl_hours": _cache_ttl_hours(),
        "embedding_model": _assigned_literal("app/semantic_cache.py", "MODEL_NAME"),
        "embedding_model_revision": None,
        "embedding_model_revision_note": "The application does not pin or expose a resolved model revision.",
        "ollama_model": _environment_default("app/ollama_client.py", "OLLAMA_MODEL"),
        "ollama_model_source_note": "Resolved from the benchmark process environment and local source default; the API process setting is not exposed.",
        "ollama_endpoint": _safe_endpoint(
            _environment_default("app/ollama_client.py", "OLLAMA_HOST")
        ),
        "ollama_endpoint_source_note": "Resolved from the benchmark process environment and local source default; the API process setting is not exposed.",
        "ollama_request_timeout_seconds": _call_keyword_literal(
            "app/ollama_client.py", "AsyncClient", "timeout"
        ),
        "chroma_collection": _call_keyword_literal(
            "app/semantic_cache.py", "get_or_create_collection", "name"
        ),
        "chroma_distance_space": "collection default; production code does not set hnsw:space",
        "benchmark_python_version": platform.python_version(),
        "benchmark_chromadb_version": _package_version("chromadb"),
        "benchmark_sentence_transformers_version": _package_version("sentence-transformers"),
        "configuration_source_note": "Project values come from the local checkout; remote API code and runtime configuration are not exposed to the benchmark client.",
    }


def build_run_summary(
    *,
    run_id: str,
    base_url: str,
    repetitions: int,
    timeout: float,
    policies: list[CachePolicy],
    execution_plan: list[dict[str, Any]],
    execution_id: str,
    warmup_requests: int,
    warmup_records: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    warmup_count = warmup_requests
    return {
        "run_id": run_id,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "repetitions": repetitions,
        "workload_version": WORKLOAD_VERSION,
        "request_timeout_seconds": timeout,
        "configurations": [policy.value for policy in policies],
        "policy_execution_order": {
            "strategy": "rotate selected policy order left once per case role and repetition",
            "phases": [
                {
                    "case_id": phase["case_id"],
                    "case_role": phase["case_role"],
                    "iteration": phase["iteration"],
                    "policies": phase["policy_order"],
                }
                for phase in execution_plan
            ],
        },
        "warmup": {
            "configured_requests": warmup_count,
            "policy": CachePolicy.PROPOSED.value,
            "namespace": f"warmup_{execution_id}" if warmup_count else None,
            "error_count": sum(bool(record.get("error")) for record in warmup_records),
            "measured": False,
            "isolation": "unique per invocation and separate from measured policy namespaces",
        },
        "cache_isolation": {
            "strategy": "first 64 bits of SHA-256(run_id) plus policy; API derives its internal user key as SHA-256(namespace + NUL + user_id)",
            "run_namespace_bits": 64,
            "collision_note": "A namespace collision is theoretically possible, though unlikely for ordinary experiment counts.",
            "persistent_stores_reset": False,
            "reusing_run_id": "reuses the same measured namespaces and may reuse prior cache entries",
            "cold_run_requirement": "use a fresh run_id; the default is a random UUID",
        },
        "experiment_settings": experiment_settings(),
        "summaries_by_configuration": summaries,
    }


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
                    "expected_reusable": _csv_bool(record["expected_reusable"]),
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
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="unmeasured proposed-policy requests in an isolated namespace (0 disables)",
    )
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
    if args.warmup_requests < 0:
        parser.error("--warmup-requests cannot be negative")

    policies = [CachePolicy(value) for value in args.policies]
    if len(set(policies)) != len(policies):
        parser.error("--policies must not contain duplicates")
    execution_id = uuid.uuid4().hex
    warmup_records = run_warmup(
        args.base_url,
        execution_id,
        args.warmup_requests,
        args.timeout,
    )
    records = run_benchmark(args.base_url, args.run_id, args.repetitions, args.timeout, policies)
    execution_plan = build_execution_plan(args.run_id, args.repetitions, policies)
    output = args.output or Path("benchmark-results") / f"{args.run_id}.csv"
    write_csv(output, records)
    summary = build_run_summary(
        run_id=args.run_id,
        base_url=args.base_url,
        repetitions=args.repetitions,
        timeout=args.timeout,
        policies=policies,
        execution_plan=execution_plan,
        execution_id=execution_id,
        warmup_requests=args.warmup_requests,
        warmup_records=warmup_records,
        summaries=summarize_by_policy(records),
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"CSV: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
