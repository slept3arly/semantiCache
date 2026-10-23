from datetime import datetime, timezone
from functools import lru_cache
import math
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(name="semantic_cache")


def _utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).timestamp()


def add_to_cache(
    cache_id: str,
    prompt: str,
    response: str,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    category: str,
    expires_at: datetime,
) -> None:
    embedding = _get_embedder().encode(prompt).tolist()
    collection.upsert(
        ids=[cache_id],
        embeddings=[embedding],
        documents=[prompt],
        metadatas=[
            {
                "response": response,
                "user_id": user_id,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "category": category,
                "expires_at": _utc_timestamp(expires_at),
            }
        ],
    )


def search_cache(
    query: str,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    category: str,
    now: datetime,
    n_results: int = 10,
) -> dict[str, Any]:
    """Return only close, in-scope candidates; callers must validate metadata again."""
    if not all((user_id, session_id, conversation_id, category)):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    embedding = _get_embedder().encode(query).tolist()
    current_timestamp = _utc_timestamp(now)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where={
            "$and": [
                {"user_id": user_id},
                {"session_id": session_id},
                {"conversation_id": conversation_id},
                {"category": category},
                {"expires_at": {"$gt": current_timestamp}},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )
    return results


def valid_candidate(
    metadata: Any,
    distance: Any,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    category: str,
    now: datetime,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    try:
        expiry = metadata["expires_at"]
    except KeyError:
        return False
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)):
        return False
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return False
    score = float(distance)
    expiry_timestamp = float(expiry)

    if not math.isfinite(expiry_timestamp) or not (expiry_timestamp > _utc_timestamp(now)):
        return False
    if not math.isfinite(score) or not (score < SIMILARITY_THRESHOLD):
        return False
    return (
        metadata.get("user_id") == user_id
        and metadata.get("session_id") == session_id
        and metadata.get("conversation_id") == conversation_id
        and metadata.get("category") == category
        and isinstance(metadata.get("response"), str)
        and bool(metadata["response"])
    )
