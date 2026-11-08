"""
core/status_tool.py — check_appointment_status(record_id) -> dict  (Part 2)

Escalation score (always in [0, 1]):

    recency          = 1 - days_since_created / 30          (30 = max of the dataset range)
    escalation_score = 0.6 * follow_up_required + 0.4 * recency

  * follow_up_required (0 or 1) carries the larger weight (0.6) because a
    doctor-flagged follow-up is the strongest signal that the patient needs action.
  * recency (0..1) carries 0.4: the follow-up discount policy (kb/10) makes a
    follow-up free within 7 days and 50% off within 14 days, so a RECENT record
    with a pending follow-up is the one where acting now saves the patient money.
  * Max = 0.6 + 0.4 = 1.0 and min = 0, so the score is already normalised.

Threshold: the 90th percentile of the score over all 40 records (computed below,
not hard-coded). Escalating the top ~10% (4 of 40) matches a small support team's
proactive-outreach capacity. With SEED=30 the P90 cut-off is 0.725, which in
practice means "follow-up required AND record created at most 16 days ago".
"""

import numpy as np

from dataset import APPOINTMENTS, APPOINTMENTS_BY_ID

W_FOLLOW_UP = 0.6
W_RECENCY = 0.4
MAX_DAYS = 30
ESCALATION_PERCENTILE = 90


def recency(days_since_created: int) -> float:
    """1.0 for a record created today, 0.0 for one created 30 days ago."""
    return 1 - min(days_since_created, MAX_DAYS) / MAX_DAYS


def escalation_score(record: dict) -> float:
    score = W_FOLLOW_UP * int(record["follow_up_required"]) + W_RECENCY * recency(
        record["days_since_created"]
    )
    return round(score, 3)


def compute_threshold(records: list[dict], percentile: int = ESCALATION_PERCENTILE) -> float:
    scores = [escalation_score(r) for r in records]
    return round(float(np.percentile(scores, percentile)), 3)


ESCALATION_THRESHOLD = compute_threshold(APPOINTMENTS)


def check_appointment_status(record_id: str) -> dict:
    """Look up one appointment. Never raises: unknown IDs return found=False."""
    rid = record_id.strip().upper()
    record = APPOINTMENTS_BY_ID.get(rid)
    if record is None:
        return {"found": False, "record_id": rid, "error": f"No appointment with ID {rid}."}
    score = escalation_score(record)
    return {
        "found": True,
        "record_id": rid,
        "category": record["category"],
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "follow_up_required": record["follow_up_required"],
        "days_since_created": record["days_since_created"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalate": score >= ESCALATION_THRESHOLD,
    }


def describe_status(s: dict) -> str:
    """One plain sentence built ONLY from the tool output (so it is grounded)."""
    if not s.get("found"):
        return s["error"]
    text = (
        f"Appointment {s['record_id']} ({s['category']}) is {s['status']} with a "
        f"consultation fee of INR {s['consultation_fee_inr']}."
    )
    if s["follow_up_required"]:
        text += " The doctor marked a follow-up as required."
    if s["escalate"]:
        text += (
            f" Escalation score {s['escalation_score']} is at or above the threshold "
            f"{s['escalation_threshold']}, so it is escalated to the support team."
        )
    return text
