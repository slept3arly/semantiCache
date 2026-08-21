import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="data/chroma")

collection = client.get_or_create_collection(
    name="semantic_cache"
)


def add_to_cache(
    cache_id: str,
    prompt: str,
    response: str,
) -> None:
    embedding = embedder.encode(prompt).tolist()

    collection.add(
        ids=[cache_id],
        embeddings=[embedding],
        documents=[prompt],
        metadatas=[{"response": response}],
    )


def search_cache(
    query: str,
    n_results: int = 1,
):
    embedding = embedder.encode(query).tolist()

    return collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )