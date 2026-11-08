"""
core/request_log.py — JSON-Lines request logging (Part 3).

One JSON object per line: timestamp, trace_id, event, latency_ms, ...
Every string value passes through mask_pii() first, so a phone number typed by
a patient never reaches the log file.
"""

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.guardrails import mask_pii


def _mask(value):
    if isinstance(value, str):
        return mask_pii(value)[0]
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    return value


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields) -> dict:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
        trace_id = fields.pop("trace_id", None)   # server-generated hex: never mask it (its digits
        if trace_id:                              # can look like a phone number)
            record["trace_id"] = trace_id
        record.update(_mask(fields))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    @contextmanager
    def trace(self, event: str, **fields):
        """with logger.trace("ask", question=q) as t: ... t["refused"] = True
        Writes ONE line at the end with the trace_id and latency_ms."""
        info = {"trace_id": uuid.uuid4().hex[:16], **fields}
        start = time.perf_counter()
        try:
            yield info
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            info["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.log(event=event, **info)
