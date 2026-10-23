from pydantic import BaseModel, Field

from .cache_policy import CachePolicy


class CacheQueryRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    cache_policy: CachePolicy = CachePolicy.PROPOSED
    cache_namespace: str | None = Field(default=None, min_length=1, max_length=64)


class CacheQueryResponse(BaseModel):
    cache_hit: bool
    answer: str
    source: str
