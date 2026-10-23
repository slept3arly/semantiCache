from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from .cache_policy import CachePolicy, scoped_cache_user_id
from .ollama_client import generate_response
from .database import Base, SessionLocal, engine
from .models import CacheEntry
from .schemas import CacheQueryRequest, CacheQueryResponse
from .lexical_cache import find_lexical_candidate
from .semantic_cache import add_to_cache, search_cache, valid_candidate


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
    policy = request.cache_policy
    cache_user_id = scoped_cache_user_id(request.user_id, request.cache_namespace)

    if policy == CachePolicy.NO_CACHE:
        response = await generate_response(request.query)
        return CacheQueryResponse(cache_hit=False, answer=response, source="ollama")

    if policy in (CachePolicy.EXACT_ONLY, CachePolicy.PROPOSED):
        cached_entry = (
            db.query(CacheEntry)
            .filter(
                CacheEntry.user_id == cache_user_id,
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

    if policy == CachePolicy.PROPOSED:
        lexical_entry = find_lexical_candidate(
            db,
            request.query,
            user_id=cache_user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            category=request.category,
            now=now,
        )
        if lexical_entry:
            return CacheQueryResponse(
                cache_hit=True,
                answer=lexical_entry.response,
                source="lexical_cache",
            )

    if policy in (CachePolicy.PURE_SEMANTIC, CachePolicy.PROPOSED):
        semantic_results = search_cache(
            request.query,
            user_id=cache_user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            category=request.category,
            now=now,
        )

        metadatas = semantic_results.get("metadatas") or [[]]
        distances = semantic_results.get("distances") or [[]]
        candidate_metadatas = metadatas[0] if metadatas and metadatas[0] else []
        candidate_distances = distances[0] if distances and distances[0] else []
        for metadata, distance in zip(candidate_metadatas, candidate_distances):
            if valid_candidate(
                metadata,
                distance,
                user_id=cache_user_id,
                session_id=request.session_id,
                conversation_id=request.conversation_id,
                category=request.category,
                now=now,
            ):
                return CacheQueryResponse(
                    cache_hit=True,
                    answer=metadata["response"],
                    source="semantic_cache",
                )

    response = await generate_response(request.query)

    expires_at = now + timedelta(hours=1)
    if policy != CachePolicy.NO_CACHE:
        entry = CacheEntry(
            user_id=cache_user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            prompt=request.query,
            response=response,
            category=request.category,
            created_at=now,
            expires_at=expires_at,
            embedding_id=None,
        )

        db.add(entry)
        db.commit()

        if policy in (CachePolicy.PURE_SEMANTIC, CachePolicy.PROPOSED):
            add_to_cache(
                cache_id=str(uuid4()),
                prompt=request.query,
                response=response,
                user_id=cache_user_id,
                session_id=request.session_id,
                conversation_id=request.conversation_id,
                category=request.category,
                expires_at=expires_at,
            )
    
    return CacheQueryResponse(
        cache_hit=False,
        answer=response,
        source="ollama",
    )
