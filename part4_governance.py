"""
part4_governance.py — Part 4: least autonomy, budget cap, response cache.

Run:  MOCK_LLM=1 python part4_governance.py
"""

import config  # noqa: F401
from core.governance import (AGENT_TOOL_ALLOWLIST, MAX_COST_PER_REQUEST_USD, MAX_INPUT_TOKENS,
                             BudgetExceeded, ResponseCache, authorize, enforce_budget,
                             normalize_query)
from rag.build import build_pipeline


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


hr("1. LEAST AUTONOMY - tool allowlist (the crew in part2_crew.py is built from this)")
for agent, tools in AGENT_TOOL_ALLOWLIST.items():
    print(f"  {agent:<28} {sorted(tools) or '(no tools)'}")
for agent, tool in [("Lookup Agent", "check_appointment_status"),
                    ("Retrieval Agent", "check_appointment_status"),
                    ("Composer Agent", "rag_search"),
                    ("Final_Editor", "check_appointment_status")]:
    try:
        authorize(agent, tool)
        print(f"  ALLOWED : {agent} -> {tool}")
    except PermissionError as exc:
        print(f"  DENIED  : {exc}")

hr("2. PER-REQUEST BUDGET CAP")
print(f"caps: question <= {MAX_INPUT_TOKENS} tokens, estimated cost <= ${MAX_COST_PER_REQUEST_USD}")
for label, q in [("normal", "What is the refund if I cancel late?"),
                 ("oversized", "Tell me everything about refunds and cancellations. " * 40)]:
    try:
        print(f"  {label:<9} OK      {enforce_budget(q)}")
    except BudgetExceeded as exc:
        print(f"  {label:<9} REJECTED  {exc}")

hr("3. IN-MEMORY RESPONSE CACHE (key = normalised query)")
pipe = build_pipeline()
cache = ResponseCache()
queries = [
    "How much does a cardiology consultation cost?",
    "how much does a CARDIOLOGY consultation cost",
    "  How much does a cardiology consultation cost?? ",
    "How long do thyroid reports take?",
]
for q in queries:
    before = pipe.llm_calls
    resp = cache.get_or_compute(q, pipe.answer)
    print(f"  key={normalize_query(q)!r}\n    llm_calls {before} -> {pipe.llm_calls}   "
          f"answer: {resp.answer[:70]}...")
print(f"\nCache hits={cache.hits} misses={cache.misses}; MOCK_LLM generations={pipe.llm_calls} "
      f"for {len(queries)} requests.")
