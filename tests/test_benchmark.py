import pytest

from benchmarks import benchmark
from benchmarks.benchmark import build_workload, percentile, summarize, summarize_by_policy


def test_summary_calculates_median_p95_hit_rate_and_errors():
    records = [
        {"latency_ms": 10.0, "actual_cache_hit": False, "error": None},
        {"latency_ms": 20.0, "actual_cache_hit": True, "error": None},
        {"latency_ms": 30.0, "actual_cache_hit": None, "error": "HTTP 500"},
    ]

    summary = summarize(records)

    assert summary == {
        "request_count": 3,
        "median_latency_ms": 20.0,
        "p95_latency_ms": 30.0,
        "hit_rate": 0.5,
        "hit_rate_request_count": 2,
        "error_count": 1,
    }


def test_empty_summary_has_null_latency_and_hit_rate():
    assert summarize([]) == {
        "request_count": 0,
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "hit_rate": None,
        "hit_rate_request_count": 0,
        "error_count": 0,
    }


def test_nearest_rank_percentile_and_empty_input():
    assert percentile([9, 1, 5, 3], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert percentile([], 0.95) is None
    with pytest.raises(ValueError):
        percentile([1], 0)


def test_workload_repeatability_and_expected_decisions():
    first = build_workload("stable-run", 2)
    second = build_workload("stable-run", 2)

    assert first == second
    assert [item["expected_cache_hit"] for item in first[:2]] == [False, True]
    assert [item["expected_cache_hit"] for item in first[2:4]] == [False, False]
    assert first[-1]["experiment"] == "semantic_probe"
    assert first[-1]["expected_cache_hit"] is None
    cold_scopes = {(item["user_id"], item["session_id"], item["conversation_id"]) for item in first[2:4]}
    assert len(cold_scopes) == 2


def test_workload_is_same_across_policies_but_namespaces_are_isolated():
    policies = ["no_cache", "exact_only", "pure_semantic", "proposed"]
    workloads = [build_workload("shared-run", 2, policy) for policy in policies]
    comparable_fields = (
        "experiment",
        "request_id",
        "query",
        "user_id",
        "session_id",
        "conversation_id",
        "category",
    )

    for index in range(1, len(workloads)):
        assert [tuple(row[field] for field in comparable_fields) for row in workloads[index]] == [
            tuple(row[field] for field in comparable_fields) for row in workloads[0]
        ]
    namespaces = {workload[0]["cache_namespace"] for workload in workloads}
    assert len(namespaces) == len(policies)
    assert [row["expected_cache_hit"] for row in workloads[0][:2]] == [False, False]
    assert [row["expected_cache_hit"] for row in workloads[1][:2]] == [False, True]


def test_summary_aggregation_keeps_configurations_separate():
    records = [
        {"cache_policy": "no_cache", "latency_ms": 10, "actual_cache_hit": False, "error": None},
        {"cache_policy": "exact_only", "latency_ms": 20, "actual_cache_hit": True, "error": None},
    ]

    summaries = summarize_by_policy(records)

    assert summaries["no_cache"]["hit_rate"] == 0
    assert summaries["exact_only"]["hit_rate"] == 1


def test_benchmark_preserves_lexical_source(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"cache_hit": true, "source": "lexical_cache"}'

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda *_a, **_k: Response())
    item = {
        "experiment": "lexical",
        "cache_policy": "proposed",
        "cache_namespace": "bench-namespace",
        "request_id": "r1",
        "query": "similar query",
        "user_id": "u",
        "session_id": "s",
        "conversation_id": "c",
        "category": "general",
        "expected_cache_hit": True,
    }

    record = benchmark._request_one("http://example.test", item, 1.0)

    assert record["actual_cache_hit"] is True
    assert record["source"] == "lexical_cache"
    assert record["decision_correct"] is True
