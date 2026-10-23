import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.cache_policy import CachePolicy, scoped_cache_user_id
from app.database import Base
from app.models import CacheEntry
from app.schemas import CacheQueryRequest


NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
BASE_REQUEST = {
    "user_id": "policy-user",
    "session_id": "policy-session",
    "conversation_id": "policy-conversation",
    "category": "research",
    "query": "a deterministic query",
}


class FakeQuery:
    def __init__(self, result=None):
        self.result = result

    def filter(self, *_args):
        return self

    def first(self):
        return self.result


class FakeDB:
    def __init__(self, exact=None):
        self.exact = exact
        self.added = []
        self.commits = 0

    def query(self, _model):
        return FakeQuery(self.exact)

    def add(self, entry):
        self.added.append(entry)

    def commit(self):
        self.commits += 1


@pytest.fixture(autouse=True)
def fixed_time_and_disabled_external_writes(monkeypatch):
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))
    monkeypatch.setattr(main, "add_to_cache", lambda **_kwargs: None)


def make_request(**overrides):
    return CacheQueryRequest(**(BASE_REQUEST | overrides))


def valid_semantic_results(user_id="policy-user"):
    return {
        "documents": [["stored query"]],
        "metadatas": [
            [
                {
                    "response": "semantic answer",
                    "user_id": user_id,
                    "session_id": BASE_REQUEST["session_id"],
                    "conversation_id": BASE_REQUEST["conversation_id"],
                    "category": BASE_REQUEST["category"],
                    "expires_at": (NOW + timedelta(minutes=5)).timestamp(),
                }
            ]
        ],
        "distances": [[0.2]],
    }


def test_request_policy_is_validated_and_defaults_to_proposed():
    assert make_request().cache_policy == CachePolicy.PROPOSED
    with pytest.raises(ValidationError):
        make_request(cache_policy="unsafe_unscoped")


def test_no_cache_bypasses_reads_and_writes(monkeypatch):
    async def generate(_query):
        return "generated each time"

    monkeypatch.setattr(main, "generate_response", generate)
    monkeypatch.setattr(main, "find_lexical_candidate", lambda *_a, **_k: pytest.fail("lexical read"))
    monkeypatch.setattr(main, "search_cache", lambda *_a, **_k: pytest.fail("vector read"))
    db = FakeDB()
    db.query = lambda *_a: pytest.fail("SQLite read")

    response = asyncio.run(main.query_cache(make_request(cache_policy="no_cache"), db))

    assert response.cache_hit is False
    assert response.source == "ollama"
    assert db.added == [] and db.commits == 0


def test_exact_only_skips_lexical_and_vector_and_writes_sqlite(monkeypatch):
    async def generate(_query):
        return "exact-only generation"

    monkeypatch.setattr(main, "generate_response", generate)
    monkeypatch.setattr(main, "find_lexical_candidate", lambda *_a, **_k: pytest.fail("lexical read"))
    monkeypatch.setattr(main, "search_cache", lambda *_a, **_k: pytest.fail("vector read"))
    monkeypatch.setattr(main, "add_to_cache", lambda **_kwargs: pytest.fail("vector write"))
    db = FakeDB()

    response = asyncio.run(main.query_cache(make_request(cache_policy="exact_only"), db))

    assert response.cache_hit is False and response.source == "ollama"
    assert len(db.added) == 1 and db.commits == 1


def test_pure_semantic_skips_exact_and_lexical_but_keeps_metadata_validation(monkeypatch):
    monkeypatch.setattr(main, "find_lexical_candidate", lambda *_a, **_k: pytest.fail("lexical read"))
    captured = {}

    def search(_query, **kwargs):
        captured.update(kwargs)
        return valid_semantic_results()

    monkeypatch.setattr(main, "search_cache", search)
    monkeypatch.setattr(main, "generate_response", lambda *_a: pytest.fail("LLM after semantic hit"))
    db = FakeDB()
    db.query = lambda *_a: pytest.fail("exact SQLite read")

    response = asyncio.run(main.query_cache(make_request(cache_policy="pure_semantic"), db))

    assert response.cache_hit is True
    assert response.source == "semantic_cache"
    assert response.answer == "semantic answer"
    assert captured["user_id"] == BASE_REQUEST["user_id"]
    assert captured["session_id"] == BASE_REQUEST["session_id"]
    assert captured["conversation_id"] == BASE_REQUEST["conversation_id"]
    assert captured["category"] == BASE_REQUEST["category"]


def test_experiment_namespace_separates_same_user_without_changing_default_key():
    assert scoped_cache_user_id("user-a", None) == "user-a"
    assert scoped_cache_user_id("user-a", "run-policy-a") != scoped_cache_user_id(
        "user-a", "run-policy-b"
    )
    assert scoped_cache_user_id("user-a", "run-policy-a") == scoped_cache_user_id(
        "user-a", "run-policy-a"
    )


def test_exact_policy_namespace_prevents_cross_experiment_hits(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    generated = []

    async def generate(_query):
        generated.append(True)
        return "isolated answer"

    monkeypatch.setattr(main, "generate_response", generate)
    monkeypatch.setattr(main, "find_lexical_candidate", lambda *_a, **_k: pytest.fail("lexical read"))
    monkeypatch.setattr(main, "search_cache", lambda *_a, **_k: pytest.fail("vector read"))

    first = make_request(cache_policy="exact_only", cache_namespace="experiment-a")
    other = make_request(cache_policy="exact_only", cache_namespace="experiment-b")
    first_result = asyncio.run(main.query_cache(first, db))
    other_result = asyncio.run(main.query_cache(other, db))
    warm_result = asyncio.run(main.query_cache(first, db))

    assert first_result.cache_hit is False
    assert other_result.cache_hit is False
    assert warm_result.cache_hit is True and warm_result.source == "exact_cache"
    assert len(generated) == 2
    db.close()
    engine.dispose()
