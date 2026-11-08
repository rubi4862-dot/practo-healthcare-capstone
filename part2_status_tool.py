"""
part2_status_tool.py — Part 2: check_appointment_status() + escalation score.

Shows the formula, every record's score, the percentile table used to pick the
threshold, and three lookups validated against the AppointmentStatus schema.
Run:  MOCK_LLM=1 python part2_status_tool.py
"""

import numpy as np

import config  # noqa: F401
from core.schemas import AppointmentStatus
from core.status_tool import (ESCALATION_PERCENTILE, ESCALATION_THRESHOLD, W_FOLLOW_UP,
                              W_RECENCY, check_appointment_status, escalation_score)
from dataset import APPOINTMENTS

print("escalation_score = "
      f"{W_FOLLOW_UP} * follow_up_required + {W_RECENCY} * (1 - days_since_created / 30)\n")

scores = sorted(((escalation_score(r), r) for r in APPOINTMENTS), key=lambda x: -x[0])
print(f"{'record':<10}{'follow_up':<11}{'days':<6}score")
for s, r in scores:
    print(f"{r['record_id']:<10}{str(r['follow_up_required']):<11}{r['days_since_created']:<6}{s:.3f}")

values = [s for s, _ in scores]
print("\nPercentiles of the score over all 40 records:")
for p in (50, 75, 80, 85, 90, 95):
    print(f"  P{p:<3} = {np.percentile(values, p):.4f}")
n_esc = sum(v >= ESCALATION_THRESHOLD for v in values)
print(f"\nThreshold = P{ESCALATION_PERCENTILE} = {ESCALATION_THRESHOLD} "
      f"-> {n_esc}/40 records ({100 * n_esc / 40:.0f}%) escalate.")
print("Why P90: the top ~10% is what a small support team can contact proactively, and here it")
print("selects exactly the follow-up-required records created <=16 days ago, i.e. still close")
print("to the 7-day free / 14-day half-price follow-up window in kb/10.\n")

for rid in ["APT-1024", "APT-1025", "APT-1022", "APT-9999"]:
    result = check_appointment_status(rid)
    AppointmentStatus.model_validate(result)          # schema check in code
    print(f"check_appointment_status({rid!r}) ->\n  {result}")
