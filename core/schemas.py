"""
core/schemas.py — Pydantic models. Every agent / API response is validated
against these in code (model_validate / model_validate_json).
"""

from pydantic import BaseModel, Field


class AppointmentStatus(BaseModel):
    found: bool
    record_id: str
    category: str | None = None
    status: str | None = None
    consultation_fee_inr: int | None = Field(default=None, ge=0)
    follow_up_required: bool | None = None
    days_since_created: int | None = Field(default=None, ge=0, le=30)
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    escalation_threshold: float | None = None
    escalate: bool | None = None
    error: str | None = None


class SupportResponse(BaseModel):
    """The single response schema used by the pipeline, the crew and the API."""
    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    grounded: bool
    refused: bool = False
    refusal_reason: str | None = None          # "out_of_scope" | "injection" | "ungrounded" | None
    top1_score: float | None = None
    appointment: AppointmentStatus | None = None


class Verdict(BaseModel):
    """Part 4: Final-Editor's structured output."""
    approved: bool
    final_answer: str
    reason: str


# ---- API request / response models (Part 3) ----------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None


class AskResponse(BaseModel):
    trace_id: str
    response: SupportResponse


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[\w-]+\.md$", examples=["13_parking_policy.md"])
    title: str = Field(min_length=3)
    content: str = Field(min_length=20)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_upserted: dict[str, int]
