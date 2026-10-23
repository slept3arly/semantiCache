"""Deterministic token-overlap retrieval over eligible SQLite cache entries."""

import re
from datetime import datetime

from sqlalchemy.orm import Session

from .models import CacheEntry


LEXICAL_SIMILARITY_THRESHOLD = 0.8
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _normalized_tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def lexical_similarity(first: str, second: str) -> float:
    """Compute Jaccard similarity over lowercase alphanumeric token sets."""
    first_tokens = _normalized_tokens(first)
    second_tokens = _normalized_tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def find_lexical_candidate(
    db: Session,
    query: str,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    category: str,
    now: datetime,
) -> CacheEntry | None:
    """Find the best eligible lexical candidate; equal scores favor lowest ID."""
    query_tokens = _normalized_tokens(query)
    if not query_tokens:
        return None

    candidates = (
        db.query(CacheEntry)
        .filter(
            CacheEntry.user_id == user_id,
            CacheEntry.session_id == session_id,
            CacheEntry.conversation_id == conversation_id,
            CacheEntry.category == category,
            CacheEntry.expires_at > now,
        )
        .order_by(CacheEntry.id.asc())
        .all()
    )

    best: CacheEntry | None = None
    best_score = -1.0
    for candidate in candidates:
        if not isinstance(candidate.prompt, str) or not isinstance(candidate.response, str):
            continue
        candidate_tokens = _normalized_tokens(candidate.prompt)
        if not candidate_tokens:
            continue
        score = len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
        if score >= LEXICAL_SIMILARITY_THRESHOLD and score > best_score:
            best = candidate
            best_score = score
    return best
