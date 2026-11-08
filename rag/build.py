"""rag/build.py — one call to get the store + calibrated pipeline (after part1_rag.py ran)."""

import json

import config
from core.pipeline import SupportPipeline, load_threshold
from rag.store import KBStore


def build_pipeline(store: KBStore | None = None, **kwargs) -> SupportPipeline:
    store = store or KBStore()
    threshold = load_threshold(config.CALIBRATION_FILE)
    strategy = json.loads(config.CALIBRATION_FILE.read_text())["strategy"]
    return SupportPipeline(store, threshold, strategy=strategy, top_k=config.TOP_K, **kwargs)
