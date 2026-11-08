"""
part1_rag.py — Part 1: chunking, two ChromaDB collections, threshold calibration,
grounded generation demo, and document-level precision / recall.

Run:  MOCK_LLM=1 python part1_rag.py
Writes results/calibration.json and fills the CALIBRATION + RETRIEVAL blocks in README.md.
"""

import json

import config
from core.pipeline import SupportPipeline
from core.readme_blocks import write_block
from rag.store import STRATEGIES, KBStore

# ---- Queries used to MEASURE top-1 cosine (no preset threshold) --------------
CALIBRATION_IN_SCOPE = [
    "How much does a dermatology consultation cost?",
    "Can I reschedule my appointment and how many times?",
    "When will my blood test report be ready?",
    "Is video consultation available for pediatrics?",
    "Does Practo share my medical records with my employer?",
]
CALIBRATION_OUT_OF_SCOPE = [
    "What is the weather in Mumbai today?",
    "Who won the IPL final this year?",
    "Recommend a good laptop for programming.",
    "Which medicine should I take for a migraine?",
]

# ---- Demo: >=5 in-scope + 1 out-of-scope -------------------------------------
DEMO_QUERIES = [
    "What is the cancellation refund if I cancel 2 hours before my slot?",
    "How much does an orthopedics consultation cost?",
    "Can I get a prescription refill for a controlled drug?",
    "How long do culture test reports take?",
    "Is there an extra charge for a home visit?",
    "Can you suggest a good pizza place in Bangalore?",          # out of scope
]

# ---- Retrieval evaluation set: ground-truth RELEVANT documents per query ------
PR_QUERIES = [
    ("How much does a cardiology consultation cost?", {"03_consultation_fee_structure.md"}),
    ("Can I get a refund if I cancel 2 hours before my appointment?", {"02_cancellation_rescheduling_window.md"}),
    ("How long until my thyroid test results are ready?", {"06_lab_test_turnaround_times.md"}),
    ("Is a follow-up visit free and can I do it by video call?",
     {"10_follow_up_visit_discount_policy.md", "07_telemedicine_eligibility.md"}),
    ("Can I claim insurance for my consultation and will Practo share my records with the insurer?",
     {"04_insurance_claim_process.md", "09_patient_data_privacy_policy.md"}),
    ("My father is 70 and cannot travel, can a doctor come home?", {"12_home_visit_eligibility.md"}),
    ("I have chest pain right now, what should I do?", {"08_emergency_visit_protocol.md"}),
    ("Can I refill my prescription without seeing the doctor again?", {"05_prescription_refill_policy.md"}),
    ("How do I get a second opinion from another doctor?", {"11_second_opinion_process.md"}),
    ("How far in advance can I book and how many appointments can I hold?", {"01_appointment_booking_policy.md"}),
]


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def precision_recall(store, strategy):
    rows = []
    for q, relevant in PR_QUERIES:
        retrieved = list(dict.fromkeys(c["doc_id"] for c in store.query(strategy, q, config.TOP_K)))
        hits = len(relevant & set(retrieved))
        rows.append({"query": q, "retrieved": retrieved, "relevant": sorted(relevant), "hits": hits,
                     "precision": hits / len(retrieved), "recall": hits / len(relevant)})
    p = sum(r["precision"] for r in rows) / len(rows)
    r = sum(r["recall"] for r in rows) / len(rows)
    return rows, p, r


def main():
    hr("1. Build two ChromaDB collections (upsert)")
    store = KBStore()
    print(f"Embedding model : {config.EMBED_MODEL} (local, CPU)")
    print(f"Fixed-size      : {config.FIXED_CHUNK_WORDS} words, overlap {config.FIXED_OVERLAP_WORDS} words")
    print(f"Sentence-based  : {config.SENTENCES_PER_CHUNK} sentences per chunk")
    print(f"Chunks per collection: {store.chunk_counts()}")
    sample = store.docs[1]
    from core.chunkers import chunk_document
    for s in STRATEGIES:
        print(f"\n  {s} chunks of {sample['doc_id']}:")
        for c in chunk_document(sample, s, config):
            print(f"    - {c['text']}")

    hr("2. Document-level precision / recall @ top-3 chunks")
    pr = {}
    for s in STRATEGIES:
        rows, p, r = precision_recall(store, s)
        pr[s] = {"precision": round(p, 3), "recall": round(r, 3), "rows": rows}
        print(f"\n--- collection: practo_kb_{s} ---")
        for i, row in enumerate(rows, 1):
            print(f"Q{i:02d} {row['query']}")
            print(f"     retrieved docs: {row['retrieved']}")
            print(f"     relevant docs : {row['relevant']}")
            print(f"     P = {row['hits']}/{len(row['retrieved'])} = {row['precision']:.3f}   "
                  f"R = {row['hits']}/{len(row['relevant'])} = {row['recall']:.3f}")
        print(f"MEAN  precision = {p:.3f}   recall = {r:.3f}")

    # Choose the collection: higher precision+recall; ties -> sentence (whole facts).
    score = {s: pr[s]["precision"] + pr[s]["recall"] for s in STRATEGIES}
    chosen = "sentence" if score["sentence"] >= score["fixed"] else "fixed"
    other = "fixed" if chosen == "sentence" else "sentence"

    hr("3. Calibrate the 'I don't know' threshold from measured top-1 cosine")
    calib = {}
    for s in STRATEGIES:
        ins = [(q, store.query(s, q, 1)[0]["score"]) for q in CALIBRATION_IN_SCOPE]
        oos = [(q, store.query(s, q, 1)[0]["score"]) for q in CALIBRATION_OUT_OF_SCOPE]
        min_in, max_out = min(v for _, v in ins), max(v for _, v in oos)
        threshold = round((min_in + max_out) / 2, 3)     # midpoint of the gap
        calib[s] = {"in_scope": ins, "out_of_scope": oos, "min_in_scope": min_in,
                    "max_out_of_scope": max_out, "threshold": threshold,
                    "separated": min_in > max_out}
        print(f"\n--- collection: practo_kb_{s} ---")
        for q, v in ins:
            print(f"  IN   top1={v:.4f}  {q}")
        for q, v in oos:
            print(f"  OUT  top1={v:.4f}  {q}")
        print(f"  min(in-scope)={min_in:.4f}  max(out-of-scope)={max_out:.4f}  "
              f"-> threshold = ({min_in:.4f} + {max_out:.4f}) / 2 = {threshold}")
        if not calib[s]["separated"]:
            print("  WARNING: in-scope and out-of-scope scores overlap; some queries will be misrouted.")

    threshold = calib[chosen]["threshold"]
    config.CALIBRATION_FILE.write_text(json.dumps(
        {"strategy": chosen, "threshold": threshold,
         "per_strategy": {s: {k: v for k, v in c.items()} for s, c in calib.items()}}, indent=2))
    print(f"\nChosen collection: practo_kb_{chosen}, threshold {threshold} "
          f"-> saved to {config.CALIBRATION_FILE.relative_to(config.ROOT)}")

    hr("4. Grounded generation demo (MOCK_LLM, collection practo_kb_%s)" % chosen)
    pipe = SupportPipeline(store, threshold, strategy=chosen)
    for q in DEMO_QUERIES:
        r = pipe.answer(q)
        print(f"\nQ: {q}\n   top1={r.top1_score}  refused={r.refused} ({r.refusal_reason})")
        print(f"A: {r.answer}\n   sources: {r.sources}")

    # ---- write measured numbers into README ----------------------------------
    lines = ["| Collection | in-scope top-1 cosine (min) | out-of-scope top-1 cosine (max) | threshold |",
             "|---|---|---|---|"]
    for s in STRATEGIES:
        c = calib[s]
        lines.append(f"| practo_kb_{s} | {c['min_in_scope']:.4f} | {c['max_out_of_scope']:.4f} | {c['threshold']} |")
    lines.append("")
    for s in STRATEGIES:
        c = calib[s]
        lines.append(f"`{s}` in-scope: " + ", ".join(f"{v:.3f}" for _, v in c["in_scope"])
                     + f" · out-of-scope: " + ", ".join(f"{v:.3f}" for _, v in c["out_of_scope"]))
    lines.append(f"\n**Used at runtime:** `practo_kb_{chosen}`, threshold **{threshold}** "
                 "(midpoint between the lowest in-scope and highest out-of-scope top-1 cosine).")
    write_block("CALIBRATION", "\n".join(lines))

    pr_lines = ["| Collection | mean precision | mean recall |", "|---|---|---|"]
    for s in STRATEGIES:
        pr_lines.append(f"| practo_kb_{s} | {pr[s]['precision']:.3f} | {pr[s]['recall']:.3f} |")
    pr_lines.append("")
    why = {
        "sentence": "Sentence chunks keep each policy fact whole (a fee band or a refund rule is never "
                    "split across two chunks), so the best-matching chunk carries the complete fact.",
        "fixed": "On these short 4-sentence documents the overlapping windows repeat boundary facts in "
                 "two chunks, which gave the retriever a second chance to surface the right document.",
    }[chosen]
    pr_lines.append(
        f"**Recommendation:** use `practo_kb_{chosen}` (precision {pr[chosen]['precision']:.3f}, "
        f"recall {pr[chosen]['recall']:.3f} vs {pr[other]['precision']:.3f} / {pr[other]['recall']:.3f} "
        f"for `{other}`). {why} Per-query arithmetic is in `transcripts/01_part1_rag.txt`.")
    write_block("RETRIEVAL", "\n".join(pr_lines))
    print("\nREADME.md CALIBRATION and RETRIEVAL blocks updated with the numbers above.")


if __name__ == "__main__":
    main()
