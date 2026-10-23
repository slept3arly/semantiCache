import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.database import Base
from app.lexical_cache import (
    LEXICAL_SIMILARITY_THRESHOLD,
    find_lexical_candidate,
    lexical_similarity,
)
from app.models import CacheEntry
from app.schemas import CacheQueryRequest


NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)
SCOPE = {
    "user_id": "lex-user-a",
    "session_id": "lex-session-a",
    "conversation_id": "lex-conversation-a",
    "category": "general",
}


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def avoid_external_services(monkeypatch):
    monkeypatch.setattr(main, "add_to_cache", lambda **_kwargs: None)


def add_entry(db, prompt, response, *, expires_at=None, **scope_overrides):
    entry = CacheEntry(
        **(SCOPE | scope_overrides),
        prompt=prompt,
        response=response,
        created_at=NOW - timedelta(minutes=1),
        expires_at=expires_at or NOW + timedelta(hours=1),
    )
    db.add(entry)
    db.commit()
    return entry


def request(query, **overrides):
    return CacheQueryRequest(query=query, **(SCOPE | overrides))


def empty_semantic_results():
    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def patch_time(monkeypatch):
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))


def test_lexical_positive_returns_cached_answer_before_semantic_or_llm(db_session, monkeypatch):
    add_entry(
        db_session,
        "ALPHA, beta gamma delta!",
        "cached lexical answer",
    )
    patch_time(monkeypatch)
    monkeypatch.setattr(main, "search_cache", lambda *_a, **_k: pytest.fail("semantic stage should not run"))
    monkeypatch.setattr(main, "generate_response", lambda *_a: pytest.fail("LLM should not run"))

    response = asyncio.run(
        main.query_cache(request("alpha beta gamma delta epsilon"), db_session)
    )

    assert LEXICAL_SIMILARITY_THRESHOLD == 0.8
    assert response.cache_hit is True
    assert response.answer == "cached lexical answer"
    assert response.source == "lexical_cache"


def test_lexical_miss_continues_to_semantic_stage(db_session, monkeypatch):
    add_entry(db_session, "quantum entanglement particle pairs", "unrelated answer")
    patch_time(monkeypatch)
    called = []
    monkeypatch.setattr(main, "search_cache", lambda *_a, **_k: called.append(True) or empty_semantic_results())

    async def generate(_query):
        return "fresh answer"

    monkeypatch.setattr(main, "generate_response", generate)
    response = asyncio.run(main.query_cache(request("capital of France"), db_session))

    assert called == [True]
    assert response.cache_hit is False
    assert response.answer == "fresh answer"
    assert response.source == "ollama"


def test_lexical_miss_can_still_return_valid_semantic_hit(db_session, monkeypatch):
    add_entry(db_session, "quantum entanglement particle pairs", "unrelated answer")
    patch_time(monkeypatch)
    semantic_candidate = {
        "response": "semantic answer",
        **SCOPE,
        "expires_at": (NOW + timedelta(minutes=5)).timestamp(),
    }
    monkeypatch.setattr(
        main,
        "search_cache",
        lambda *_a, **_k: {
            "documents": [["other prompt"]],
            "metadatas": [[semantic_candidate]],
            "distances": [[0.2]],
        },
    )
    monkeypatch.setattr(main, "generate_response", lambda *_a: pytest.fail("LLM after semantic hit"))

    response = asyncio.run(
        main.query_cache(request("capital of France"), db_session)
    )

    assert response.cache_hit is True
    assert response.answer == "semantic answer"
    assert response.source == "semantic_cache"


@pytest.mark.parametrize(
    "scope_override",
    [
        {"user_id": "lex-user-b"},
        {"session_id": "lex-session-b"},
        {"conversation_id": "lex-conversation-b"},
        {"category": "different"},
    ],
)
def test_lexical_candidate_respects_all_metadata_scope_fields(
    db_session, monkeypatch, scope_override
):
    add_entry(db_session, "alpha beta gamma delta", "scoped answer")
    patch_time(monkeypatch)
    monkeypatch.setattr(main, "search_cache", lambda *_a, **_k: empty_semantic_results())

    async def generate(_query):
        return "fallback"

    monkeypatch.setattr(main, "generate_response", generate)
    response = asyncio.run(
        main.query_cache(
            request("alpha beta gamma delta epsilon", **scope_override),
            db_session,
        )
    )

    assert response.cache_hit is False
    assert response.answer == "fallback"
    assert response.source == "ollama"


def test_expired_candidate_is_ignored_and_unexpired_candidate_is_selected(db_session):
    add_entry(
        db_session,
        "alpha beta gamma delta",
        "expired answer",
        expires_at=NOW,
    )
    active = add_entry(db_session, "alpha beta gamma delta", "active answer")

    candidate = find_lexical_candidate(
        db_session,
        "alpha beta gamma delta epsilon",
        **SCOPE,
        now=NOW,
    )

    assert candidate is not None
    assert candidate.id == active.id
    assert candidate.response == "active answer"


def test_empty_and_punctuation_only_queries_do_not_match(db_session):
    add_entry(db_session, "!!!", "punctuation answer")
    add_entry(db_session, " ", "empty answer")

    assert lexical_similarity("", "!!!") == 0.0
    assert lexical_similarity("!!!", "???") == 0.0
    assert find_lexical_candidate(
        db_session,
        "!!!",
        **SCOPE,
        now=NOW,
    ) is None


def test_equal_scores_use_lowest_id_reproducibly(db_session):
    first = add_entry(
        db_session,
        "beta gamma delta epsilon zeta eta theta iota kappa lambda",
        "lowest id answer",
    )
    add_entry(
        db_session,
        "alpha gamma delta epsilon zeta eta theta iota kappa lambda",
        "higher id answer",
    )
    query = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    selected = [
        find_lexical_candidate(db_session, query, **SCOPE, now=NOW)
        for _ in range(3)
    ]

    assert all(candidate.id == first.id for candidate in selected)


def test_exact_match_still_precedes_lexical_lookup(db_session, monkeypatch):
    add_entry(db_session, "exact query", "exact answer")
    patch_time(monkeypatch)
    monkeypatch.setattr(
        main,
        "find_lexical_candidate",
        lambda *_a, **_k: pytest.fail("lexical stage ran before exact cache"),
    )

    response = asyncio.run(main.query_cache(request("exact query"), db_session))

    assert response.cache_hit is True
    assert response.answer == "exact answer"
    assert response.source == "exact_cache"
