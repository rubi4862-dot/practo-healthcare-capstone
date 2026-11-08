"""
config.py — shared settings. Import this FIRST in every script.

It switches off all telemetry (CrewAI, OpenTelemetry, ChromaDB, Hugging Face)
so the project runs with zero outbound network calls, and it enforces MOCK_LLM.
"""

import os
from pathlib import Path

# --- Telemetry off (must be set before crewai / chromadb are imported) -------
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")      # ChromaDB
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Mock LLM -----------------------------------------------------------------
# Every LLM call in this project goes through the deterministic mock in
# core/mock_llm.py. There is no real-LLM code path, so no API key is ever needed.
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"
if not MOCK_LLM:
    raise RuntimeError(
        "This capstone only implements the deterministic MOCK_LLM. Run with MOCK_LLM=1."
    )

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "kb"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
README_PATH = ROOT / "README.md"
CALIBRATION_FILE = RESULTS_DIR / "calibration.json"

RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# --- RAG settings -------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # runs locally on CPU
TOP_K = 3
FIXED_CHUNK_WORDS = 40       # fixed-size chunker: words per chunk
FIXED_OVERLAP_WORDS = 10     # fixed-size chunker: words shared with previous chunk
SENTENCES_PER_CHUNK = 2      # sentence chunker: sentences per chunk
