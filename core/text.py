"""
core/text.py — tiny text helpers shared by the chunkers, the mock LLM and the judge.
"""

import re

# Split after . ! ? when followed by whitespace and a capital letter / digit / quote.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
    "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "their", "can", "could", "do",
    "does", "did", "what", "which", "who", "whom", "how", "when", "where", "why",
    "will", "would", "should", "shall", "may", "might", "must", "have", "has",
    "had", "not", "no", "so", "than", "then", "there", "about", "into", "any",
    "all", "each", "up", "out", "only", "also", "much", "many", "get", "gets",
    "there", "here", "per", "after", "before", "within", "same", "such", "s",
}


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (good enough for our clean KB text)."""
    text = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _stem(word: str) -> str:
    """Very light stemming so 'consultations' matches 'consultation'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> set[str]:
    """Lower-cased, stemmed, stop-word-free tokens (numbers are kept)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS}
