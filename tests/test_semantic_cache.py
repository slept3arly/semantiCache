import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main, semantic_cache
from app.database import Base
from app.models import CacheEntry
from app.schemas import CacheQueryRequest


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCOPE = {
    "user_id": "user-a",
    "session_id": "session-a",
    "conversation_id": "conversation-a",
    "category": "science",
}


@pytest.fixture(autouse=True)
def avoid_real_vector_store(monkeypatch):
    monkeypatch.setattr(main, "add_to_cache", lambda **_kwargs: None)


class FakeQuery:
    def __init__(self, result):
        self.result = result

    def filter(self, *args):
        return self

    def order_by(self, *_args):
        return self

    def all(self):
        return []

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


def request(query="A new question", **overrides):
    return CacheQueryRequest(query=query, **(SCOPE | overrides))


def metadata(**overrides):
    return {
        "response": "cached answer",
        **SCOPE,
        "expires_at": (NOW + timedelta(minutes=5)).timestamp(),
        **overrides,
    }


def results(records):
    return {
        "documents": [["prompt"] * len(records)],
        "metadatas": [[record[0] for record in records]],
        "distances": [[record[1] for record in records]],
    }


def test_exact_match_returns_existing_cached_response(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    db.add(
        CacheEntry(
            **SCOPE,
            prompt="exact prompt",
            response="exact answer",
            created_at=NOW,
            expires_at=NOW + timedelta(minutes=5),
        )
    )
    db.commit()
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))
    monkeypatch.setattr(main, "search_cache", lambda *a, **k: pytest.fail("semantic query after exact hit"))

    response = asyncio.run(main.query_cache(request(query="exact prompt"), db))

    assert response.model_dump() == {"cache_hit": True, "answer": "exact answer", "source": "exact_cache"}
    db.close()
    engine.dispose()


def test_valid_semantic_hit_and_scope_filters(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))

    def search(_query, **kwargs):
        captured.update(kwargs)
        return results([(metadata(), 0.2)])

    monkeypatch.setattr(main, "search_cache", search)
    monkeypatch.setattr(main, "generate_response", lambda *_: pytest.fail("generation after semantic hit"))
    response = asyncio.run(main.query_cache(request(), FakeDB()))

    assert response.cache_hit and response.answer == "cached answer"
    assert response.source == "semantic_cache"
    assert {key: captured[key] for key in SCOPE} == SCOPE


@pytest.mark.parametrize(
    "overrides,distance",
    [
        ({"user_id": "user-b"}, 0.2),
        ({"session_id": "session-b"}, 0.2),
        ({"conversation_id": "conversation-b"}, 0.2),
        ({"category": "other"}, 0.2),
        ({"expires_at": (NOW - timedelta(seconds=1)).timestamp()}, 0.2),
        ({"expires_at": "not-a-timestamp"}, 0.2),
        ({"session_id": None}, 0.2),
        ({}, 0.6),
    ],
)
def test_rejected_candidate_falls_back_without_false_hit(monkeypatch, overrides, distance):
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))
    monkeypatch.setattr(main, "search_cache", lambda *a, **k: results([(metadata(**overrides), distance)]))
    async def generate(_query):
        return "fresh answer"
    monkeypatch.setattr(main, "generate_response", generate)
    db = FakeDB()

    response = asyncio.run(main.query_cache(request(), db))

    assert not response.cache_hit
    assert response.answer == "fresh answer"
    assert response.source == "ollama"


def test_missing_metadata_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))
    monkeypatch.setattr(main, "search_cache", lambda *a, **k: results([({"response": "unsafe"}, 0.1)]))
    async def generate(_query):
        return "fallback"
    monkeypatch.setattr(main, "generate_response", generate)
    response = asyncio.run(main.query_cache(request(), FakeDB()))
    assert not response.cache_hit and response.answer == "fallback"


def test_rejected_nearest_candidate_does_not_hide_valid_candidate(monkeypatch):
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))
    monkeypatch.setattr(
        main,
        "search_cache",
        lambda *a, **k: results([({"response": "bad"}, 0.1), (metadata(), 0.2)]),
    )
    response = asyncio.run(main.query_cache(request(), FakeDB()))
    assert response.cache_hit and response.answer == "cached answer"


def test_empty_vector_results_trigger_llm_and_store_consistent_metadata(monkeypatch):
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))
    monkeypatch.setattr(main, "search_cache", lambda *a, **k: results([]))
    async def generate(_query):
        return "generated"
    monkeypatch.setattr(main, "generate_response", generate)
    added = {}
    monkeypatch.setattr(main, "add_to_cache", lambda **kwargs: added.update(kwargs))
    db = FakeDB()

    response = asyncio.run(main.query_cache(request(), db))

    assert response.model_dump() == {"cache_hit": False, "answer": "generated", "source": "ollama"}
    assert len(db.added) == 1 and db.commits == 1
    assert added["expires_at"] == db.added[0].expires_at
    assert {key: added[key] for key in SCOPE} == SCOPE
    assert added["response"] == "generated"


def test_chroma_search_uses_metadata_filters_and_numeric_expiry(monkeypatch):
    class Embedder:
        def encode(self, _query):
            return SimpleNamespace(tolist=lambda: [0.1, 0.2])

    class Collection:
        def query(self, **kwargs):
            self.kwargs = kwargs
            return results([])

    collection = Collection()
    monkeypatch.setattr(semantic_cache, "_get_embedder", lambda: Embedder())
    monkeypatch.setattr(semantic_cache, "collection", collection)

    semantic_cache.search_cache("query", **SCOPE, now=NOW)

    assert collection.kwargs["where"] == {
        "$and": [
            {"user_id": SCOPE["user_id"]},
            {"session_id": SCOPE["session_id"]},
            {"conversation_id": SCOPE["conversation_id"]},
            {"category": SCOPE["category"]},
            {"expires_at": {"$gt": NOW.timestamp()}},
        ]
    }


def test_insert_uses_upsert_and_stores_numeric_utc_expiry(monkeypatch):
    class Embedder:
        def encode(self, _prompt):
            return SimpleNamespace(tolist=lambda: [0.3])

    class Collection:
        def upsert(self, **kwargs):
            self.kwargs = kwargs

    collection = Collection()
    monkeypatch.setattr(semantic_cache, "_get_embedder", lambda: Embedder())
    monkeypatch.setattr(semantic_cache, "collection", collection)

    semantic_cache.add_to_cache("id-1", "prompt", "response", **SCOPE, expires_at=NOW)

    assert collection.kwargs["ids"] == ["id-1"]
    stored = collection.kwargs["metadatas"][0]
    assert {key: stored[key] for key in SCOPE} == SCOPE
    assert stored["expires_at"] == NOW.timestamp()


def test_naive_expiry_is_interpreted_as_utc():
    assert semantic_cache._utc_timestamp(datetime(2026, 1, 1)) == NOW.timestamp()
