"""
core/governance.py — Part 4 controls: least autonomy, budget cap, response cache.
"""

import math
import re
import string

# ---------------------------------------------------------------------------
# 1. Least autonomy: which agent may call which tool. The CrewAI crew is BUILT
#    from this table (part2_crew.py), so it is the single source of truth.
#    AutoGen reviewers get no tools at all.
# ---------------------------------------------------------------------------
AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "Retrieval Agent": {"rag_search"},
    "Lookup Agent": {"check_appointment_status"},   # the ONLY agent that sees patient records
    "Composer Agent": set(),
    "Policy_Compliance_Reviewer": set(),
    "Final_Editor": set(),
}


def authorize(agent: str, tool: str) -> None:
    """Raise PermissionError unless `agent` is allowed to call `tool`."""
    if tool not in AGENT_TOOL_ALLOWLIST.get(agent, set()):
        raise PermissionError(f"{agent!r} is not allowed to call {tool!r}")


# ---------------------------------------------------------------------------
# 2. Per-request token / cost budget.
#    Token estimate: ceil(characters / 4)  (the usual ~4 chars-per-token rule).
#    Prices are illustrative (the mock costs nothing) but the cap is enforced
#    BEFORE any model call, exactly as it would be with a paid model.
# ---------------------------------------------------------------------------
MAX_INPUT_TOKENS = 500             # a support question never needs more than this
CONTEXT_TOKENS_ESTIMATE = 3 * 120  # top-3 chunks of ~120 tokens each
OUTPUT_TOKENS_ESTIMATE = 200
PRICE_PER_1K_INPUT_USD = 0.003
PRICE_PER_1K_OUTPUT_USD = 0.015
MAX_COST_PER_REQUEST_USD = 0.005


class BudgetExceeded(Exception):
    pass


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def estimate_request(question: str) -> dict:
    q_tokens = estimate_tokens(question)
    input_tokens = q_tokens + CONTEXT_TOKENS_ESTIMATE
    cost = (input_tokens / 1000) * PRICE_PER_1K_INPUT_USD + (
        OUTPUT_TOKENS_ESTIMATE / 1000) * PRICE_PER_1K_OUTPUT_USD
    return {"question_tokens": q_tokens, "input_tokens": input_tokens,
            "output_tokens": OUTPUT_TOKENS_ESTIMATE, "est_cost_usd": round(cost, 6)}


def enforce_budget(question: str) -> dict:
    """Return the estimate, or raise BudgetExceeded if either cap is broken."""
    est = estimate_request(question)
    if est["question_tokens"] > MAX_INPUT_TOKENS:
        raise BudgetExceeded(
            f"question is ~{est['question_tokens']} tokens; the cap is {MAX_INPUT_TOKENS}")
    if est["est_cost_usd"] > MAX_COST_PER_REQUEST_USD:
        raise BudgetExceeded(
            f"estimated cost ${est['est_cost_usd']} exceeds cap ${MAX_COST_PER_REQUEST_USD}")
    return est


# ---------------------------------------------------------------------------
# 3. In-memory response cache keyed on the NORMALISED query.
# ---------------------------------------------------------------------------
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_query(q: str) -> str:
    """'  What is the FEE for Cardiology?? ' -> 'what is the fee for cardiology'"""
    return " ".join(_PUNCT.sub(" ", q.lower()).split())


class ResponseCache:
    def __init__(self):
        self._store: dict[str, object] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, query: str, compute):
        key = normalize_query(query)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = compute(query)
        self._store[key] = value
        return value
