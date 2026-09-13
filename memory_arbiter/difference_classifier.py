"""Difference-based clearance for check-route pairs (0.16.2 plan §1.4).

Owner principle: a conflict must carry a DIFFERENCE — pairs with an
extractable value difference stay for agent judgment, pairs without one are
duplicates/evolution noise and are machine-cleared. Similarity alone never
decides (the v5 "high-similarity rescue" route was falsified on real data:
every rescued pair was a duplicate or an evolution note).

Two check-route shapes and their keep rules, calibrated on the production
library (27,832 pending pairs, 2026-09-13):
- numeric_value_candidate: character-bigram set Ochiai cosine >= 0.50 keeps
  a "same-sentence shape" pair — the only shape that can carry "same claim,
  two values" (MySQL 5.7 vs 8.0 in one sentence). Below the line the numbers
  coincide across unrelated sentences → clear.
- semantic_similarity_only: a small token symmetric difference (common >= 2,
  unique per side <= 2) keeps the literal value-difference shape the rule
  engine cannot parse (MySQL/PostgreSQL class). Anything else has no
  extractable difference → clear.

notify-route pairs (polarity_changed / todo_resolved) NEVER pass through
here — real-conflict recall has no threshold; the routing layer protects
them, not similarity.

Shared by the live pipeline and the boot stock-clearance migration so the
two can never drift (plan review note).
"""
from __future__ import annotations

import re

# numeric keep line (plan §1.4.4: knee of the sensitivity curve; pluggable
# constant — retune from miss feedback, the sensitivity table is the guide).
DIFFERENCE_COSINE_KEEP = 0.50
# similarity-route small-symmetric-difference shape (plan §1.4.1 layer 2,
# document-literal (2, 2); the planner session's looser estimate is recorded
# in plan §1.4.3 and NOT implemented).
SYMDIFF_MIN_COMMON = 2
SYMDIFF_MAX_UNIQUE = 2
# garbage label (counting only — never flips a keep verdict): separator
# lines, bare dates, or <8 content characters after stripping.
GARBAGE_MIN_CONTENT_CHARS = 8

_DATE_RE = re.compile(
    r"\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}日?"
    r"|\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?"
    r"|\d{2}:\d{2}(:\d{2})?"
)
_SEPARATOR_RE = re.compile(r"^-{3,}$|^={3,}$|^[~_*=]{3,}$")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def is_garbage(quote: "str | None") -> bool:
    """Counting label for cleared rows (plan §1.4.1): separator line, bare
    date, or <8 content chars after date/punctuation stripping."""
    if not quote:
        return True
    s = _SEPARATOR_RE.sub("", quote.strip())
    if not s:
        return True
    if _DATE_RE.fullmatch(s):
        return True
    stripped = _PUNCT_RE.sub("", _DATE_RE.sub("", s))
    return len(stripped) < GARBAGE_MIN_CONTENT_CHARS


def _bigrams(text: str) -> set[str]:
    s = re.sub(r"\s+", "", text)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def name_cosine(a: str, b: str) -> float:
    """Character-bigram set Ochiai cosine (0..1)."""
    set_a, set_b = _bigrams(a), _bigrams(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / ((len(set_a) * len(set_b)) ** 0.5)


def _tokens(text: str) -> set[str]:
    """Latin/digit runs kept whole + CJK chars individually, casefolded."""
    return set(_TOKEN_RE.findall(text.casefold()))


def has_small_symmetric_difference(a: str, b: str) -> bool:
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    common = tokens_a & tokens_b
    return (
        len(common) >= SYMDIFF_MIN_COMMON
        and len(tokens_a - tokens_b) <= SYMDIFF_MAX_UNIQUE
        and len(tokens_b - tokens_a) <= SYMDIFF_MAX_UNIQUE
    )


def classify_pair(
    quote_a: "str | None", quote_b: "str | None", *, route: str,
    entity_a: "str | None" = None, entity_b: "str | None" = None,
) -> str:
    """Return ``"keep"`` (enqueue for agent judgment) or ``"clear"``
    (machine-cleared, count only, never lands in conflicts).

    ``route`` is the check-route reason (``numeric_value_candidate`` or any
    other check reason, treated as the similarity route). The entity layer
    (owner ⑪) clears pairs whose BOTH-side metadata.entity values exist and
    differ — true different-subject pairs; zero hits on the current library,
    free to run, useful as the library grows.
    """
    if not quote_a or not quote_b:
        return "clear"
    if entity_a and entity_b and entity_a.strip() != entity_b.strip():
        return "clear"
    if "numeric_value_candidate" in route:
        return "keep" if name_cosine(quote_a, quote_b) >= DIFFERENCE_COSINE_KEEP else "clear"
    return "keep" if has_small_symmetric_difference(quote_a, quote_b) else "clear"
