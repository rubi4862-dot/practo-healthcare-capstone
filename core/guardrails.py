"""
core/guardrails.py — input and output guardrails (Part 2).

  INPUT  1. mask_pii()          : replaces contact phone numbers (and emails) with tags
         2. detect_injection()  : flags prompt-injection attempts -> request refused
  OUTPUT 3. check_groundedness(): every answer sentence must be supported by the
                                  retrieved context, otherwise the answer is refused
"""

import re

from core.text import content_tokens, split_sentences

# Indian mobile numbers: optional +91 / 0 prefix, 10 digits starting 6-9,
# optionally split 5+5 by a space or dash. Plus a generic 10-12 digit fallback.
PHONE_RE = re.compile(
    r"(?:\+?91[\s-]?|\b0)?[6-9]\d{4}[\s-]?\d{5}\b|\b\d{3}[\s-]\d{3}[\s-]\d{4}\b"
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|your) (instructions|rules)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in) ",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"act as (if you are |an? )?(unrestricted|unfiltered|admin|root)",
    r"(list|dump|print|show) (all |every )?(patients?'?|users?'?) (phone|contact|data|records)",
]
_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

INJECTION_REFUSAL = (
    "I can't help with that request. I can only answer questions about Practo "
    "appointments and policies."
)
UNGROUNDED_REFUSAL = (
    "I'm not able to confirm that from the Practo knowledge base, so I won't guess. "
    "Please contact Practo support for help."
)
GROUNDING_MIN_OVERLAP = 0.7   # share of a sentence's content words that must appear in context


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, list_of_what_was_masked)."""
    found = []
    if PHONE_RE.search(text):
        found.append("phone")
        text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    if EMAIL_RE.search(text):
        found.append("email")
        text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    return text, found


def detect_injection(text: str) -> str | None:
    """Return the matched pattern if the text looks like a prompt injection, else None."""
    for rx in _INJECTION_RES:
        if rx.search(text):
            return rx.pattern
    return None


def sentence_supported(sentence: str, context: str) -> float:
    """Fraction of the sentence's content words that occur in the context."""
    words = content_tokens(sentence)
    if not words:
        return 1.0
    return len(words & content_tokens(context)) / len(words)


def check_groundedness(answer: str, context: str) -> tuple[bool, list[str]]:
    """(is_grounded, unsupported_sentences)."""
    unsupported = [
        s for s in split_sentences(answer)
        if sentence_supported(s, context) < GROUNDING_MIN_OVERLAP
    ]
    return (not unsupported), unsupported
