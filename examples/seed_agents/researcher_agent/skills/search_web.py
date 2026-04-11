"""Information retrieval skill — search and extract relevant information."""

from typing import Any


def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search for information matching a query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts with 'title', 'snippet', and 'relevance'.
    """
    # Tokenize and normalize the query
    tokens = _tokenize(query)

    # Simulated search results (in a real agent, this would call an API)
    results = _generate_results(tokens, max_results)

    return results


def extract_keywords(text: str) -> list[str]:
    """Extract key terms from text for search refinement.

    Args:
        text: Input text to extract keywords from.

    Returns:
        List of keyword strings, ordered by relevance.
    """
    words = _tokenize(text)
    # Filter stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    return text.lower().split()


def _generate_results(tokens: list[str], max_results: int) -> list[dict[str, str]]:
    """Generate search results based on query tokens."""
    results = []
    for i in range(min(max_results, len(tokens) + 1)):
        results.append({
            "title": f"Result {i + 1}: {' '.join(tokens[:3])}",
            "snippet": f"Information about {' '.join(tokens)}...",
            "relevance": str(round(1.0 - i * 0.15, 2)),
        })
    return results
