import pytest

from app.cache_policy import CachePolicy, scoped_cache_user_id
from benchmarks import benchmark
from benchmarks.benchmark import (
    build_execution_plan,
    build_run_summary,
    build_warmup_workload,
    build_workload,
    experiment_settings,
    percentile,
    summarize,
    summarize_by_policy,
)


def test_summary_calculates_median_p95_hit_rate_and_errors():
    records = [
        {"latency_ms": 10.0, "actual_cache_hit": False, "expected_cache_hit": False, "error": None},
        {"latency_ms": 20.0, "actual_cache_hit": True, "expected_cache_hit": True, "error": None},
        {"latency_ms": 30.0, "actual_cache_hit": None, "expected_cache_hit": False, "error": "HTTP 500"},
    ]

    summary = summarize(records)

    assert summary == {
        "request_count": 3,
        "median_latency_ms": 20.0,
        "p95_latency_ms": 30.0,
        "hit_rate": 0.5,
        "hit_rate_request_count": 2,
        "decision_correct_count": 2,
        "decision_evaluation_count": 2,
        "decision_accuracy": 1.0,
        "error_count": 1,
    }


def test_empty_summary_has_null_latency_and_hit_rate():
    assert summarize([]) == {
        "request_count": 0,
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "hit_rate": None,
        "hit_rate_request_count": 0,
        "decision_correct_count": 0,
        "decision_evaluation_count": 0,
        "decision_accuracy": None,
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
    assert [item["expected_cache_hit"] for item in first[:3]] == [False, True, True]
    assert first[0]["case_id"] == first[1]["case_id"] == first[2]["case_id"]
    assert [item["case_role"] for item in first[:3]] == ["seed", "repeat", "repeat"]
    cold_rows = [item for item in first if item["case_type"] == "cold_unique"]
    cold_scopes = {(item["user_id"], item["session_id"], item["conversation_id"]) for item in cold_rows}
    assert len(cold_scopes) == 2


def test_workload_is_same_across_policies_but_namespaces_are_isolated():
    policies = ["no_cache", "exact_only", "pure_semantic", "proposed"]
    workloads = [build_workload("shared-run", 2, policy) for policy in policies]
    comparable_fields = (
        "experiment",
        "case_id",
        "case_type",
        "case_role",
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
    assert [row["expected_cache_hit"] for row in workloads[0][:3]] == [False, False, False]
    assert [row["expected_cache_hit"] for row in workloads[1][:3]] == [False, True, True]


def test_workload_case_labels_and_expected_policy_decisions():
    expected = {
        "no_cache": {"paraphrase": False, "related_distinct": False, "unrelated": False},
        "exact_only": {"paraphrase": False, "related_distinct": False, "unrelated": False},
        "pure_semantic": {"paraphrase": True, "related_distinct": False, "unrelated": False},
        "proposed": {"paraphrase": True, "related_distinct": False, "unrelated": False},
    }
    for policy, cases in expected.items():
        rows = build_workload("labels", 2, policy)
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row)
        for case_type, target_decision in cases.items():
            pair = next(
                (items for items in by_case.values() if items[0]["case_type"] == case_type),
                None,
            )
            assert pair is not None
            assert [item["case_role"] for item in pair] == ["seed", "probe"]
            assert pair[0]["expected_cache_hit"] is False
            assert pair[1]["expected_reusable"] is (case_type == "paraphrase")
            assert pair[1]["expected_cache_hit"] is target_decision

        isolation = [row for row in rows if row["case_type"] == "metadata_isolation"]
        assert len(isolation) == 4
        assert all(row["expected_cache_hit"] is False for row in isolation)
        assert all(row["expected_reusable"] is True for row in isolation)
        user_seed, user_probe, session_seed, session_probe = isolation
        assert user_seed["user_id"] != user_probe["user_id"]
        assert user_seed["session_id"] == user_probe["session_id"]
        assert user_seed["conversation_id"] == user_probe["conversation_id"]
        assert user_seed["category"] == user_probe["category"]
        assert user_seed["cache_namespace"] == user_probe["cache_namespace"]
        assert session_seed["user_id"] == session_probe["user_id"]
        assert session_seed["session_id"] != session_probe["session_id"]
        assert session_seed["conversation_id"] == session_probe["conversation_id"]
        assert session_seed["category"] == session_probe["category"]
        assert session_seed["cache_namespace"] == session_probe["cache_namespace"]


def test_seed_and_probe_share_scope_and_namespace_but_runs_and_policies_do_not():
    seed_rows = build_workload("run-a", 1, "proposed")
    probe_rows = build_workload("run-a", 1, "proposed")
    paraphrase = [row for row in seed_rows if row["case_type"] == "paraphrase"]
    repeat = [row for row in seed_rows if row["case_type"] == "exact_repeat"]
    assert paraphrase[0]["cache_namespace"] == paraphrase[1]["cache_namespace"]
    for field in ("user_id", "session_id", "conversation_id", "category"):
        assert paraphrase[0][field] == paraphrase[1][field]
    assert [row["case_id"] for row in seed_rows] == [row["case_id"] for row in probe_rows]
    assert repeat[0]["case_id"] == repeat[1]["case_id"]

    other_run = build_workload("run-b", 1, "proposed")
    other_policy = build_workload("run-a", 1, "pure_semantic")
    assert seed_rows[0]["cache_namespace"] != other_run[0]["cache_namespace"]
    assert seed_rows[0]["cache_namespace"] != other_policy[0]["cache_namespace"]
    assert scoped_cache_user_id("same-user", seed_rows[0]["cache_namespace"]) != scoped_cache_user_id(
        "same-user", other_run[0]["cache_namespace"]
    )


def test_execution_plan_rotates_policy_order_deterministically(monkeypatch):
    policies = list(CachePolicy)
    first = build_execution_plan("fixed-order", 2, policies)
    second = build_execution_plan("fixed-order", 2, policies)

    assert first == second
    assert first[0]["case_id"] == "exact_repeat-capital-france"
    assert first[0]["case_role"] == "seed"
    assert first[0]["policy_order"] == [policy.value for policy in policies]
    assert first[1]["case_role"] == "repeat"
    assert first[1]["iteration"] == 1
    assert first[1]["policy_order"] == [
        policy.value for policy in policies[1:] + policies[:1]
    ]
    assert first[2]["iteration"] == 2

    calls = []

    def fake_request(_base_url, item, _timeout):
        calls.append((item["case_id"], item["case_role"], item["cache_policy"]))
        return {**item, "actual_cache_hit": False, "error": None}

    monkeypatch.setattr(benchmark, "_request_one", fake_request)
    benchmark.run_benchmark("http://example.test", "fixed-order", 2, 1.0, policies)

    expected_calls = [
        (phase["case_id"], phase["case_role"], policy)
        for phase in first
        for policy in phase["policy_order"]
    ]
    assert calls == expected_calls


def test_execution_plan_rejects_duplicate_policy_names():
    with pytest.raises(ValueError, match="duplicates"):
        build_execution_plan("duplicate-policy", 1, [CachePolicy.PROPOSED, CachePolicy.PROPOSED])


def test_warmup_requests_use_a_separate_namespace_from_measured_workloads(monkeypatch):
    warmup = build_warmup_workload("invocation-a", 2)
    measured = build_workload("run-a", 2, "proposed")

    assert len(warmup) == 2
    assert all(item["cache_policy"] == "proposed" for item in warmup)
    assert len({item["cache_namespace"] for item in warmup}) == 1
    assert warmup[0]["cache_namespace"] == "warmup_invocation-a"
    assert warmup[0]["cache_namespace"] != measured[0]["cache_namespace"]
    assert len({item["user_id"] for item in warmup}) == 2
    assert build_warmup_workload("invocation-b", 1)[0]["cache_namespace"] != warmup[0]["cache_namespace"]

    calls = []

    def fake_request(_base_url, item, timeout):
        calls.append((item, timeout))
        return {**item, "error": None}

    monkeypatch.setattr(benchmark, "_request_one", fake_request)
    benchmark.run_warmup("http://example.test", "invocation-c", 1, 15.0)
    assert calls[0][0]["cache_namespace"] == "warmup_invocation-c"
    assert calls[0][0]["cache_namespace"] != measured[0]["cache_namespace"]
    assert calls[0][1] == 15.0


def test_run_summary_records_execution_warmup_and_model_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "benchmark-test-model")
    monkeypatch.setenv("OLLAMA_HOST", "https://user:secret@localhost:11434/ollama?token=hidden")
    policies = list(CachePolicy)
    plan = build_execution_plan("summary-run", 1, policies)
    summary = build_run_summary(
        run_id="summary-run",
        base_url="http://127.0.0.1:8000",
        repetitions=1,
        timeout=30.0,
        policies=policies,
        execution_plan=plan,
        execution_id="invocation-x",
        warmup_requests=1,
        warmup_records=[{"error": None}],
        summaries={"proposed": {"request_count": 1}},
        created_at_utc="2026-10-28T00:00:00+00:00",
    )
    settings = experiment_settings()

    assert summary["repetitions"] == 1
    assert summary["workload_version"] == "policy-comparable-v2"
    assert summary["request_timeout_seconds"] == 30.0
    assert summary["policy_execution_order"]["phases"] == [
        {
            "case_id": phase["case_id"],
            "case_role": phase["case_role"],
            "iteration": phase["iteration"],
            "policies": phase["policy_order"],
        }
        for phase in plan
    ]
    assert summary["warmup"]["configured_requests"] == 1
    assert summary["warmup"]["namespace"] == "warmup_invocation-x"
    assert summary["warmup"]["error_count"] == 0
    assert summary["cache_isolation"]["persistent_stores_reset"] is False
    assert summary["cache_isolation"]["run_namespace_bits"] == 64
    assert summary["experiment_settings"] == settings
    assert settings["semantic_threshold"] == 0.6
    assert settings["lexical_threshold"] == 0.8
    assert settings["cache_ttl_hours"] == 1
    assert settings["embedding_model"] == "all-MiniLM-L6-v2"
    assert settings["embedding_model_revision"] is None
    assert settings["ollama_model"] == "benchmark-test-model"
    assert settings["ollama_endpoint"] == "https://localhost:11434"
    assert settings["benchmark_python_version"]
    assert "secret" not in str(settings)


def test_summary_does_not_score_undefined_or_failed_decisions():
    summary = summarize(
        [
            {"latency_ms": 1, "actual_cache_hit": True, "expected_cache_hit": None, "error": None},
            {"latency_ms": 2, "actual_cache_hit": None, "expected_cache_hit": True, "error": "timeout"},
            {"latency_ms": 3, "actual_cache_hit": False, "expected_cache_hit": False, "error": None},
        ]
    )
    assert summary["decision_evaluation_count"] == 1
    assert summary["decision_correct_count"] == 1
    assert summary["decision_accuracy"] == 1.0
    assert summary["hit_rate_request_count"] == 2
    assert summary["error_count"] == 1


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
        "case_id": "lexical-case",
        "case_type": "paraphrase",
        "case_role": "probe",
        "cache_policy": "proposed",
        "cache_namespace": "bench-namespace",
        "request_id": "r1",
        "query": "similar query",
        "user_id": "u",
        "session_id": "s",
        "conversation_id": "c",
        "category": "general",
        "expected_reusable": True,
        "expected_cache_hit": True,
    }

    record = benchmark._request_one("http://example.test", item, 1.0)

    assert record["actual_cache_hit"] is True
    assert record["source"] == "lexical_cache"
    assert record["decision_correct"] is True
