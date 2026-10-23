"""Cache policy selection and optional experiment namespace helpers."""

from enum import Enum
from hashlib import sha256


class CachePolicy(str, Enum):
    NO_CACHE = "no_cache"
    EXACT_ONLY = "exact_only"
    PURE_SEMANTIC = "pure_semantic"
    PROPOSED = "proposed"


def scoped_cache_user_id(user_id: str, namespace: str | None) -> str:
    """Keep namespaced experiment entries disjoint without changing DB schema."""
    if namespace is None:
        return user_id
    return sha256(f"{namespace}\0{user_id}".encode("utf-8")).hexdigest()
