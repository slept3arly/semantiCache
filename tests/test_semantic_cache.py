from app.semantic_cache import add_to_cache, search_cache


def test_semantic_search():
    add_to_cache(
        cache_id="test-1",
        prompt="What is machine learning?",
        response="Machine learning is a method where systems learn patterns from data.",
    )

    results = search_cache(
        "Can you explain what machine learning is?"
    )

    print(results)

    assert len(results["documents"][0]) == 1