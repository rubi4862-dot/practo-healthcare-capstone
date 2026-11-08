"""
dataset.py — Seeded, deterministic generator for the Practo (Healthcare) appointment dataset.

Design choices (also stated in README.md):
  - SEED = 30, N_RECORDS = 40. The first seed tried (42) passed but sat at the
    edge of the band (10.0% follow-ups); seed 30 lands mid-band at 20.0%.
  - Category weights: General Medicine 0.30, Cardiology 0.15, Dermatology 0.20,
    Pediatrics 0.20, Orthopedics 0.15
  - Status weights: Scheduled 0.30, Completed 0.35, Cancelled 0.12,
    No-Show 0.08, Rescheduled 0.15
  - follow_up_required probability: 0.20 (target band 10%-30%)
  - consultation_fee_inr: INR 300-1500 overall, drawn from a per-specialty band
    and rounded to the nearest 50. Reasoning: typical Indian private outpatient
    consultations run from about INR 300 for a GP to about INR 1500 for a
    specialist such as a cardiologist, so specialty-specific bands keep fees realistic.
  - days_since_created: uniform integer 0-30

Records are never hand-edited. If validation fails, change SEED or weights and regenerate.
Run:  python dataset.py
"""

import random
from collections import Counter

SEED = 30
N_RECORDS = 40

CATEGORY_WEIGHTS = {
    "General Medicine": 0.30,
    "Cardiology": 0.15,
    "Dermatology": 0.20,
    "Pediatrics": 0.20,
    "Orthopedics": 0.15,
}

STATUS_WEIGHTS = {
    "Scheduled": 0.30,
    "Completed": 0.35,
    "Cancelled": 0.12,
    "No-Show": 0.08,
    "Rescheduled": 0.15,
}

# Per-specialty fee bands in INR (min, max). Overall range: 300-1500.
FEE_BANDS = {
    "General Medicine": (300, 600),
    "Pediatrics": (400, 800),
    "Dermatology": (500, 1000),
    "Orthopedics": (600, 1200),
    "Cardiology": (800, 1500),
}

FOLLOW_UP_PROB = 0.20


def generate_appointments(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    categories = list(CATEGORY_WEIGHTS)
    statuses = list(STATUS_WEIGHTS)
    records = []
    for i in range(n):
        category = rng.choices(categories, weights=list(CATEGORY_WEIGHTS.values()))[0]
        status = rng.choices(statuses, weights=list(STATUS_WEIGHTS.values()))[0]
        lo, hi = FEE_BANDS[category]
        fee = round(rng.randint(lo, hi) / 50) * 50
        records.append(
            {
                "record_id": f"APT-{1001 + i}",
                "category": category,
                "status": status,
                "consultation_fee_inr": fee,
                "days_since_created": rng.randint(0, 30),
                "follow_up_required": rng.random() < FOLLOW_UP_PROB,
            }
        )
    return records


APPOINTMENTS = generate_appointments()
APPOINTMENTS_BY_ID = {r["record_id"]: r for r in APPOINTMENTS}


def validate(records: list[dict]) -> dict:
    """Check every structural threshold from the brief. Raises AssertionError on failure."""
    cat_counts = Counter(r["category"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    follow_pct = 100 * sum(r["follow_up_required"] for r in records) / len(records)

    assert len(records) >= 40, f"need >=40 records, got {len(records)}"
    for c in CATEGORY_WEIGHTS:
        assert cat_counts[c] >= 3, f"category {c} has {cat_counts[c]} (<3)"
    for s in STATUS_WEIGHTS:
        assert status_counts[s] >= 1, f"status {s} missing"
    assert 10 <= follow_pct <= 30, f"follow_up_required {follow_pct:.1f}% outside 10-30%"
    assert all(0 <= r["days_since_created"] <= 30 for r in records)
    return {"categories": cat_counts, "statuses": status_counts, "follow_up_pct": follow_pct}


if __name__ == "__main__":
    report = validate(APPOINTMENTS)
    print(f"Generated {len(APPOINTMENTS)} appointments (SEED={SEED})\n")
    print("Count per category:")
    for c in CATEGORY_WEIGHTS:
        print(f"  {c:<18} {report['categories'][c]}")
    print("\nCount per status:")
    for s in STATUS_WEIGHTS:
        print(f"  {s:<18} {report['statuses'][s]}")
    print(f"\nfollow_up_required=True: {report['follow_up_pct']:.1f}%  (target 10-30%)")
    fees = [r["consultation_fee_inr"] for r in APPOINTMENTS]
    print(f"consultation_fee_inr range: {min(fees)}-{max(fees)} INR")
    print("\nAll validation checks passed.")
