"""
core/chunkers.py — the two chunking strategies compared in Part 1.

  1. fixed_size_chunks : sliding window of N words with an overlap of M words.
                         Simple, but can cut a sentence (and a fact) in half.
  2. sentence_chunks   : groups of K whole sentences, never cutting a sentence.

Each KB file is parsed into {doc_id, title, body}. Chunks keep the doc_id so
retrieval can be scored at document level (precision / recall).
"""

from pathlib import Path

from core.text import split_sentences


def load_kb(kb_dir: Path) -> list[dict]:
    """Read every kb/*.md file. The first '# ' line is the title; the rest is the body."""
    docs = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        docs.append({"doc_id": path.name, "title": title, "body": body})
    return docs


def fixed_size_chunks(text: str, size: int = 40, overlap: int = 10) -> list[str]:
    """Sliding window over words: chunk i starts at i*(size-overlap)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        chunks.append(" ".join(chunk))
        if start + size >= len(words):   # last window already reached the end
            break
    return chunks


def sentence_chunks(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Group consecutive whole sentences, K at a time."""
    sentences = split_sentences(text)
    return [
        " ".join(sentences[i:i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]


def chunk_document(doc: dict, strategy: str, cfg) -> list[dict]:
    """Return chunk records ready to upsert: id, text, metadata."""
    if strategy == "fixed":
        pieces = fixed_size_chunks(doc["body"], cfg.FIXED_CHUNK_WORDS, cfg.FIXED_OVERLAP_WORDS)
    elif strategy == "sentence":
        pieces = sentence_chunks(doc["body"], cfg.SENTENCES_PER_CHUNK)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return [
        {
            "id": f"{doc['doc_id']}::{strategy}::{i}",   # stable id -> upsert is idempotent
            "text": piece,
            "metadata": {"doc_id": doc["doc_id"], "title": doc["title"], "chunk_index": i},
        }
        for i, piece in enumerate(pieces)
    ]
