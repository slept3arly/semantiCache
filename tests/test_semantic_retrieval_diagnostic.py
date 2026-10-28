import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import chromadb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main, semantic_cache
from app.cache_policy import scoped_cache_user_id
from app.database import Base
from app.lexical_cache import lexical_similarity
from app.models import CacheEntry
from app.schemas import CacheQueryRequest


NOW = datetime(2026, 10, 23, 9, 48, 0, tzinfo=timezone.utc)
REQUEST_SCOPE = {
    "user_id": "semantic_review_user",
    "session_id": "semantic_review_session",
    "conversation_id": "semantic_review_run_01",
    "category": "science",
    "cache_namespace": "semantic_review",
    "cache_policy": "proposed",
}


def test_proposed_miss_persists_scoped_vector_and_next_request_reaches_semantic_hit(monkeypatch):
    client = chromadb.EphemeralClient()
    backing_collection = client.create_collection("semantic_retrieval_diagnostic")
    first_prompt = "What is photosynthesis?"
    second_prompt = "How do plants convert sunlight into energy?"
    vectors = {
        first_prompt: [0.0, 0.0],
        second_prompt: [0.2, 0.2],
    }

    class Embedder:
        def encode(self, text):
            return SimpleNamespace(tolist=lambda: vectors[text])

    class DiagnosticCollection:
        def __init__(self):
            self.write_metadata = []
            self.queries = []

        def upsert(self, **kwargs):
            self.write_metadata.extend(kwargs["metadatas"])
            backing_collection.upsert(**kwargs)

        def query(self, **kwargs):
            result = backing_collection.query(**kwargs)
            self.queries.append(
                {
                    "where": kwargs["where"],
                    "candidate_count": len(result["ids"][0]),
                    "distances": list(result["distances"][0]),
                }
            )
            return result

        def get(self, **kwargs):
            return backing_collection.get(**kwargs)

    diagnostic_collection = DiagnosticCollection()
    monkeypatch.setattr(semantic_cache, "_get_embedder", lambda: Embedder())
    monkeypatch.setattr(semantic_cache, "collection", diagnostic_collection)
    monkeypatch.setattr(main, "datetime", SimpleNamespace(now=lambda *_: NOW))

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
        return "generated photosynthesis answer"

    monkeypatch.setattr(main, "generate_response", generate)

    def make_request(query):
        return CacheQueryRequest(query=query, **REQUEST_SCOPE)

    try:
        first_response = asyncio.run(main.query_cache(make_request(first_prompt), db))
        expected_user_key = scoped_cache_user_id(
            REQUEST_SCOPE["user_id"], REQUEST_SCOPE["cache_namespace"]
        )
        stored_sqlite = (
            db.query(CacheEntry)
            .filter(
                CacheEntry.user_id == expected_user_key,
                CacheEntry.session_id == REQUEST_SCOPE["session_id"],
                CacheEntry.conversation_id == REQUEST_SCOPE["conversation_id"],
                CacheEntry.category == REQUEST_SCOPE["category"],
            )
            .one()
        )
        stored_vector = diagnostic_collection.get(
            where={"user_id": expected_user_key},
            include=["documents", "metadatas"],
        )

        assert lexical_similarity(first_prompt, second_prompt) == 0.0
        second_response = asyncio.run(main.query_cache(make_request(second_prompt), db))

        lookup = diagnostic_collection.queries[-1]
        stored_metadata = diagnostic_collection.write_metadata[0]
        vector_metadata = stored_vector["metadatas"][0]
        distance = lookup["distances"][0]
        diagnostics = {
            "sqlite_miss_write": bool(stored_sqlite.response),
            "sqlite_expiry": stored_sqlite.expires_at.isoformat(),
            "chroma_write_count": len(diagnostic_collection.write_metadata),
            "stored_metadata_user_key": stored_metadata["user_id"],
            "lookup_metadata_user_key": lookup["where"]["$and"][0]["user_id"],
            "persisted_vector_count": len(stored_vector["ids"]),
            "candidate_count_on_second_query": lookup["candidate_count"],
            "candidate_distance": distance,
            "candidate_expiry": vector_metadata["expires_at"],
        }
        assert first_response.source == "ollama", diagnostics
        assert stored_metadata["user_id"] == expected_user_key, diagnostics
        assert lookup["where"]["$and"][0]["user_id"] == expected_user_key, diagnostics
        assert stored_metadata["session_id"] == REQUEST_SCOPE["session_id"], diagnostics
        assert stored_metadata["conversation_id"] == REQUEST_SCOPE["conversation_id"], diagnostics
        assert stored_metadata["category"] == REQUEST_SCOPE["category"], diagnostics
        assert stored_metadata["expires_at"] > NOW.timestamp(), diagnostics
        assert len(stored_vector["ids"]) == 1, diagnostics
        assert lookup["candidate_count"] == 1, diagnostics
        assert distance < semantic_cache.SIMILARITY_THRESHOLD, diagnostics
        assert semantic_cache.valid_candidate(
            vector_metadata,
            distance,
            user_id=expected_user_key,
            session_id=REQUEST_SCOPE["session_id"],
            conversation_id=REQUEST_SCOPE["conversation_id"],
            category=REQUEST_SCOPE["category"],
            now=NOW + timedelta(seconds=1),
        ), diagnostics
        assert second_response.cache_hit is True
        assert second_response.source == "semantic_cache"
        assert second_response.answer == "generated photosynthesis answer"
        assert len(generated) == 1
    finally:
        db.close()
        engine.dispose()


def test_observed_live_distance_is_rejected_by_the_unchanged_threshold():
    first_expiry = datetime(2026, 10, 23, 10, 47, 36, tzinfo=timezone.utc).timestamp()
    metadata = {
        "response": "stored answer",
        "user_id": scoped_cache_user_id("semantic_review_user", "semantic_review"),
        "session_id": REQUEST_SCOPE["session_id"],
        "conversation_id": REQUEST_SCOPE["conversation_id"],
        "category": REQUEST_SCOPE["category"],
        "expires_at": first_expiry,
    }
    distance = 0.7318053841590881

    diagnostic = {
        "scope_metadata_matches": True,
        "unexpired_at_second_request": first_expiry > NOW.timestamp(),
        "distance": distance,
        "threshold": semantic_cache.SIMILARITY_THRESHOLD,
        "rejection": "distance_threshold" if distance >= semantic_cache.SIMILARITY_THRESHOLD else None,
    }
    assert semantic_cache.SIMILARITY_THRESHOLD == 0.6, diagnostic
    assert first_expiry > NOW.timestamp(), diagnostic
    assert not semantic_cache.valid_candidate(
        metadata,
        distance,
        user_id=metadata["user_id"],
        session_id=REQUEST_SCOPE["session_id"],
        conversation_id=REQUEST_SCOPE["conversation_id"],
        category=REQUEST_SCOPE["category"],
        now=NOW,
    ), diagnostic
