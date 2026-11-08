"""
core/pipeline.py — the end-to-end support answer flow used by the API, the
LangChain memory demo, the evaluation and the cache demo.

    question
      -> budget cap (Part 4)                    -> BudgetExceeded
      -> input guardrail: prompt injection       -> refuse
      -> input guardrail: mask phone / email
      -> status tool if an APT-XXXX id is present
      -> retrieve top-k chunks (ChromaDB)
      -> "I don't know" if top-1 cosine < calibrated threshold
      -> MOCK_LLM grounded composition
      -> output guardrail: groundedness check    -> refuse
      -> SupportResponse (validated by Pydantic)

`retriever` is any object with .query(strategy, text, k) -> list[chunk dicts]
(rag/store.py: KBStore). Keeping it injectable makes this file testable.
"""

import json
import re

from core import mock_llm
from core.governance import enforce_budget
from core.guardrails import (INJECTION_REFUSAL, UNGROUNDED_REFUSAL, check_groundedness,
                             detect_injection, mask_pii)
from core.schemas import AppointmentStatus, SupportResponse
from core.status_tool import check_appointment_status, describe_status

RECORD_ID_RE = re.compile(r"\bAPT-?\d{4}\b", re.IGNORECASE)


def policy_query(text: str) -> str:
    """Text to embed for KB search: the question minus any APT-XXXX id.
    The id is for the status tool; to the embedding model it is noise that drags the
    top-1 cosine down (measured: 0.31 -> 0.47 for the APT-1024 follow-up question)."""
    return " ".join(RECORD_ID_RE.sub(" ", text).split())


def load_threshold(calibration_file) -> float:
    try:
        return json.loads(calibration_file.read_text())["threshold"]
    except FileNotFoundError:
        raise RuntimeError(
            "No calibrated threshold yet - run `python part1_rag.py` first."
        ) from None


class SupportPipeline:
    def __init__(self, retriever, threshold: float, strategy: str = "sentence", top_k: int = 3,
                 generator=None):
        self.retriever = retriever
        # generator(question, chunks) -> (answer, sources). Default: the MOCK_LLM.
        # part2_guardrails.py swaps in a deliberately hallucinating one to show
        # the output guardrail refusing.
        self.generator = generator or mock_llm.compose_grounded_answer
        self.threshold = threshold
        self.strategy = strategy
        self.top_k = top_k
        self.llm_calls = 0          # counts MOCK_LLM generations (cache evidence, Part 4)
        self.last_context = ""      # exposed for the evaluation judge

    def answer(self, question: str) -> SupportResponse:
        enforce_budget(question)                      # raises BudgetExceeded

        if detect_injection(question):
            self.last_context = ""
            return SupportResponse(answer=INJECTION_REFUSAL, grounded=True,
                                   refused=True, refusal_reason="injection")

        masked, _ = mask_pii(question)                # nothing downstream sees the raw number

        appointment = None
        status_text = ""
        m = RECORD_ID_RE.search(masked)
        if m:
            rid = "APT-" + m.group(0)[-4:]            # accepts apt-1024 / APT1024
            status = check_appointment_status(rid)
            appointment = AppointmentStatus.model_validate(status)
            status_text = describe_status(status)

        chunks = self.retriever.query(self.strategy, policy_query(masked), self.top_k)
        top1 = chunks[0]["score"] if chunks else 0.0
        in_scope = top1 >= self.threshold

        if not in_scope and not appointment:
            self.last_context = ""
            return SupportResponse(answer=mock_llm.IDK_ANSWER, grounded=True, refused=True,
                                   refusal_reason="out_of_scope", top1_score=round(top1, 4))

        self.llm_calls += 1
        policy_answer, sources = ("", [])
        if in_scope:
            policy_answer, sources = self.generator(masked, chunks)
        answer = " ".join(p for p in (status_text, policy_answer) if p)

        context = " ".join(c["text"] for c in chunks) + " " + status_text
        self.last_context = context
        ok, unsupported = check_groundedness(answer, context)
        if not ok:
            return SupportResponse(answer=UNGROUNDED_REFUSAL, grounded=False, refused=True,
                                   refusal_reason="ungrounded", top1_score=round(top1, 4))

        return SupportResponse(answer=answer, sources=sources, grounded=True,
                               top1_score=round(top1, 4), appointment=appointment)
