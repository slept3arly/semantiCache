from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from .ollama_client import generate_response
from .database import Base, SessionLocal, engine
from .models import CacheEntry
from .schemas import CacheQueryRequest, CacheQueryResponse
from .semantic_cache import add_to_cache, search_cache


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="semantiCache",
    version="0.1.0",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/cache/query", response_model=CacheQueryResponse)
async def query_cache(
    request: CacheQueryRequest,
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)

    cached_entry = (
        db.query(CacheEntry)
        .filter(
            CacheEntry.user_id == request.user_id,
            CacheEntry.session_id == request.session_id,
            CacheEntry.conversation_id == request.conversation_id,
            CacheEntry.prompt == request.query,
            CacheEntry.category == request.category,
            CacheEntry.expires_at > now,
        )
        .first()
    )

    if cached_entry:
        return CacheQueryResponse(
            cache_hit=True,
            answer=cached_entry.response,
            source="exact_cache",
        )

    # Semantic cache lookup
    semantic_results = search_cache(request.query, n_results=1)

    if semantic_results["documents"][0]:
        distance = semantic_results["distances"][0][0]

        # Lower distance = more semantically similar
        if distance < 0.6:
            cached_response = semantic_results["metadatas"][0][0]["response"]

            return CacheQueryResponse(
                cache_hit=True,
                answer=cached_response,
                source="semantic_cache",
            )

    response = await generate_response(request.query)

    entry = CacheEntry(
        user_id=request.user_id,
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        prompt=request.query,
        response=response,
        category=request.category,
        created_at=now,
        expires_at=now + timedelta(hours=1),
        embedding_id=None,
    )

    db.add(entry)
    db.commit()

    add_to_cache(
        cache_id=str(uuid4()),
        prompt=request.query,
        response=response,
    )
    
    return CacheQueryResponse(
        cache_hit=False,
        answer=response,
        source="ollama",
    )