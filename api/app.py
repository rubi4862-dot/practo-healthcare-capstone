"""
api/app.py — Part 3: FastAPI service.

  POST /ask           AskRequest -> AskResponse        (pipeline answer)
  POST /ask-crew      AskRequest -> AskResponse        (Part 2 CrewAI crew, run via .kickoff())
  POST /add-document  AddDocumentRequest -> AddDocumentResponse (upsert into both collections)
  WS   /ws/chat       {"question": ...} -> SupportResponse JSON, until the client disconnects

Every request writes ONE JSON line to logs/requests.jsonl with a trace_id and
latency_ms; questions are PII-masked before they are logged.

Run the server:  MOCK_LLM=1 uvicorn api.app:app --port 8000
"""

import config  # noqa: F401

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from core.governance import BudgetExceeded, enforce_budget
from core.guardrails import INJECTION_REFUSAL, detect_injection, mask_pii
from core.request_log import JsonlLogger
from core.schemas import (AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse,
                          SupportResponse)
from rag.build import build_pipeline

app = FastAPI(title="Practo Support Agent (MOCK_LLM)")
LOG = JsonlLogger(config.LOGS_DIR / "requests.jsonl")
_PIPELINE = None
_CREW = None


def pipeline():
    """Build once, on first use (loads the embedding model + KB)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_pipeline()
    return _PIPELINE


def crew_answer(question: str) -> SupportResponse:
    """Run the Part 2 CrewAI crew (built once) and validate its output schema."""
    global _CREW
    import part2_crew                      # imported lazily: only /ask-crew needs CrewAI
    if _CREW is None:
        part2_crew.PIPELINE = pipeline()   # the crew's RAG tool uses the same store + threshold
        _CREW = part2_crew.build_crew(part2_crew.MockCrewLLM())
        _CREW.verbose = False
    result = _CREW.kickoff(inputs={"question": question})
    return SupportResponse.model_validate_json(result.raw)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    with LOG.trace("POST /ask", question=req.question, session_id=req.session_id) as t:
        try:
            resp = pipeline().answer(req.question)
        except BudgetExceeded as exc:
            t["status_code"] = 413
            raise HTTPException(status_code=413, detail=f"Request over budget: {exc}") from exc
        t.update(status_code=200, refused=resp.refused, refusal_reason=resp.refusal_reason,
                 top1_score=resp.top1_score, sources=resp.sources)
        return AskResponse(trace_id=t["trace_id"], response=resp)


@app.post("/ask-crew", response_model=AskResponse)
def ask_crew(req: AskRequest) -> AskResponse:
    """Same input guardrails as /ask (budget, injection, PII masking), then the CrewAI crew."""
    with LOG.trace("POST /ask-crew", question=req.question, session_id=req.session_id) as t:
        try:
            enforce_budget(req.question)
        except BudgetExceeded as exc:
            t["status_code"] = 413
            raise HTTPException(status_code=413, detail=f"Request over budget: {exc}") from exc
        if detect_injection(req.question):
            resp = SupportResponse(answer=INJECTION_REFUSAL, grounded=True,
                                   refused=True, refusal_reason="injection")
        else:
            masked, _ = mask_pii(req.question)            # the crew never sees the raw number
            resp = crew_answer(masked)
        t.update(status_code=200, refused=resp.refused, refusal_reason=resp.refusal_reason,
                 sources=resp.sources)
        return AskResponse(trace_id=t["trace_id"], response=resp)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    with LOG.trace("POST /add-document", doc_id=req.doc_id) as t:
        counts = pipeline().retriever.add_document(
            {"doc_id": req.doc_id, "title": req.title, "body": " ".join(req.content.split())})
        t.update(status_code=200, chunks_upserted=counts)
        return AddDocumentResponse(trace_id=t["trace_id"], doc_id=req.doc_id, chunks_upserted=counts)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    LOG.log(event="WS connect")
    try:
        while True:
            data = await ws.receive_json()
            with LOG.trace("WS message", question=data.get("question", "")) as t:
                try:
                    resp = pipeline().answer(AskRequest(**data).question)
                    payload = {"trace_id": t["trace_id"], "response": resp.model_dump()}
                    t.update(refused=resp.refused, refusal_reason=resp.refusal_reason)
                except BudgetExceeded as exc:
                    payload = {"trace_id": t["trace_id"], "error": f"Request over budget: {exc}"}
                    t["error"] = "BudgetExceeded"
            if "response" in payload:
                SupportResponse.model_validate(payload["response"])     # schema check before sending
            await ws.send_json(payload)
    except WebSocketDisconnect as exc:
        LOG.log(event="WS disconnect", code=exc.code)     # clean shutdown, no traceback
