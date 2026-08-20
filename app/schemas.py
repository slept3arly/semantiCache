from pydantic import BaseModel, Field


class CacheQueryRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)


class CacheQueryResponse(BaseModel):
    cache_hit: bool
    answer: str
    source: str