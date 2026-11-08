"""
prefetch_model.py — ONE-TIME setup step, the only step that uses the network.

Downloads the local embedding model (~90 MB) into the Hugging Face cache.
After this, run_all.sh sets HF_HUB_OFFLINE=1, so every graded run is offline.

Run:  python prefetch_model.py
"""

import config
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
print(f"Cached {config.EMBED_MODEL}; embedding dimension = {model.get_sentence_embedding_dimension()}")
