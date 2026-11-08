# Capstone handoff — Practo (Healthcare) Domain Support Agent

**Course:** Masai — Certification Program in Agentic Systems and Design, "Final Capstone" (10% weightage)
**Deadline:** 8 Nov 2026, 11:59 PM IST = 10:29 AM PST
**Full brief:** https://abvenu.github.io/capstone-projects/index.html#/iitr-as-260313/03healthcare
**Submission (assess.masaischool.com):** Q1 = exactly `Healthcare` · Q2 = one public GitHub repo link. Mismatch = 0 marks.

## Done so far (in this zip)
- `dataset.py`: seeded generator, SEED=30, 40 records, all checks pass (follow-up 20.0%, fees INR 300–1450, per-specialty fee bands)
- `kb/`: 12 knowledge-base docs, one per required topic, 4 sentences each, fabricated content; fee bands match dataset.py

## Still to build (marks)
**Part 1 (30):** two chunkers (fixed-size+overlap, sentence-based) → SentenceTransformers (local) → two separate ChromaDB collections via `upsert()`; grounded generation; calibrate the "I don't know" threshold from measured top-1 cosine on ≥3 in-scope + ≥2 out-of-scope queries (no preset 0.5/0.6/0.7), state values in README; demo ≥5 in-scope + 1 out-of-scope; doc-level precision/recall for both collections with per-query arithmetic + 2–3 sentence recommendation.

**Part 2 (30):** `check_appointment_status(record_id) -> dict` with status, fee, `escalation_score` in [0,1] combining follow_up_required + normalized recency from days_since_created (state formula + percentile-justified threshold); CrewAI crew with ≥3 agents (Retrieval w/ RAG tool, Lookup w/ status tool, Composer), run via `.kickoff()`, show both tools invoked; LangChain session memory (InMemoryChatMessageHistory + RunnableWithMessageHistory; deprecation warning is fine) with multi-turn transcript + fresh-session transcript; Pydantic response schema validated in code; guardrails: input PII masking (contact phone number) + prompt-injection detection, output groundedness refusal — each shown firing.

**Part 3 (20):** FastAPI ≥2 POST endpoints (e.g. /ask, /add-document) with Pydantic models + a WebSocket chat endpoint handling `WebSocketDisconnect`; JSON-Lines logging per request with trace ID + timing, PII masked in logs; LLM-as-judge (MOCK_LLM) on 15 queries (≥1 per KB topic, ≥2 out-of-scope/edge) scoring Accuracy, Grounding, Completeness, Safety + averages.

**Part 4 (20):** Autogen `RoundRobinGroupChat` (Policy-Compliance-Reviewer + Final-Editor), `max_turns=2`; Final-Editor `output_content_type=Verdict` (approved: bool, final_answer: str, reason: str) and team `custom_message_types=[StructuredMessage[Verdict]]`; show one approve-unchanged and one revise (injected ungrounded claim); least autonomy (only Lookup Agent has the status tool) demonstrated + explained; risk classification (High — medical data) with paragraph; per-request token/cost budget cap rejecting an oversized request; in-memory response cache keyed on normalized query, with call-counter evidence.

## Hard rules
- Everything must run under deterministic **MOCK_LLM** with zero API keys and zero network; graded transcripts use it.
- MOCK_LLM for CrewAI: subclass `crewai.llms.base_llm.BaseLLM`.
  - Pitfall 1: don't search the conversation for "Observation:" — CrewAI's ReAct system prompt contains "Observation: the result of the action". Parse only model-generated text.
  - Pitfall 2: dispatch tool args from the tool's declared args schema, not by substring-matching the tool name (`rag_lookup` contains "lookup").
- Set `CREWAI_DISABLE_TELEMETRY=true` (and `OTEL_SDK_DISABLED=true`) and say so in README.
- README top: "Completed the Practo (Healthcare) track" + dataset choices (seed, weights, fee range).
- No images/PDFs/slides/video; everything is code or text in the repo. Transcripts saved as text files.
- Student must be able to explain the code and data flow — keep code readable and commented.
