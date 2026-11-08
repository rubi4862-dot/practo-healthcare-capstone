"""
rag/store.py — KBStore: local embeddings + two separate ChromaDB collections.

  practo_kb_fixed     <- fixed-size + overlap chunks
  practo_kb_sentence  <- sentence-based chunks

Both use cosine space, so similarity = 1 - distance returned by Chroma.
Embeddings are L2-normalised SentenceTransformer vectors computed locally.
"""

import config  # noqa: F401  (telemetry off before chromadb import)

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from core.chunkers import chunk_document, load_kb

STRATEGIES = ("fixed", "sentence")


class KBStore:
    def __init__(self, kb_dir=config.KB_DIR):
        self.model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self.collections = {
            s: self.client.get_or_create_collection(
                name=f"practo_kb_{s}", metadata={"hnsw:space": "cosine"})
            for s in STRATEGIES
        }
        self.docs = load_kb(kb_dir)
        for doc in self.docs:
            self.add_document(doc)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def add_document(self, doc: dict) -> dict[str, int]:
        """Chunk one document both ways and upsert() into each collection."""
        counts = {}
        for strategy, coll in self.collections.items():
            chunks = chunk_document(doc, strategy, config)
            # The title is prepended for EMBEDDING only, so a chunk like
            # "Cancellations made less than 4 hours..." still knows its topic.
            coll.upsert(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=self.embed([f"{doc['title']}. {c['text']}" for c in chunks]),
                metadatas=[c["metadata"] for c in chunks],
            )
            counts[strategy] = len(chunks)
        return counts

    def query(self, strategy: str, text: str, k: int = config.TOP_K) -> list[dict]:
        res = self.collections[strategy].query(
            query_embeddings=self.embed([text]), n_results=k,
            include=["documents", "metadatas", "distances"])
        return [
            {"text": doc, "doc_id": meta["doc_id"], "title": meta["title"],
             "score": round(1 - dist, 4)}                 # cosine similarity
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def chunk_counts(self) -> dict[str, int]:
        return {s: c.count() for s, c in self.collections.items()}
