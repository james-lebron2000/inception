"""Text summarization skill — condense text while preserving key information."""

from typing import Any


def summarize(text: str, max_length: int = 200) -> str:
    """Summarize text to a target length.

    Uses extractive summarization: selects the most important sentences.

    Args:
        text: The text to summarize.
        max_length: Maximum character length of the summary.

    Returns:
        A condensed summary string.
    """
    if len(text) <= max_length:
        return text

    sentences = _split_sentences(text)
    if not sentences:
        return text[:max_length]

    # Score sentences by importance
    scored = [(s, _sentence_score(s, sentences)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build summary from top sentences, maintaining original order
    selected = []
    current_length = 0
    top_sentences = {s for s, _ in scored[:len(scored) // 2 + 1]}

    for sentence in sentences:
        if sentence in top_sentences and current_length + len(sentence) <= max_length:
            selected.append(sentence)
            current_length += len(sentence) + 1

    return " ".join(selected) if selected else text[:max_length]


def extract_key_points(text: str, num_points: int = 5) -> list[str]:
    """Extract the key points from a text.

    Args:
        text: Input text.
        num_points: Number of key points to extract.

    Returns:
        List of key point strings.
    """
    sentences = _split_sentences(text)
    scored = [(s, _sentence_score(s, sentences)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:num_points]]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _sentence_score(sentence: str, all_sentences: list[str]) -> float:
    """Score a sentence's importance based on word frequency."""
    # Build word frequency from all sentences
    word_freq: dict[str, int] = {}
    for s in all_sentences:
        for word in s.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    # Score = sum of word frequencies / sentence length
    words = sentence.lower().split()
    if not words:
        return 0.0

    score = sum(word_freq.get(w, 0) for w in words) / len(words)

    # Bonus for sentence position (first sentences are often important)
    if sentence == all_sentences[0]:
        score *= 1.5

    return score
