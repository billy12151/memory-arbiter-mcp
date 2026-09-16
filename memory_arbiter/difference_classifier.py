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

2026-09-16 (eval cf-oppo): the (2, 2) budget only fits near-identical
sentences — a real same-skeleton value opposition in CJK prose blows past it
on ordinary wording (MySQL vs PostgreSQL died at 8/8 unique tokens). A third
keep shape therefore asks the owner question directly: does each side carry
a VALUE-shaped token the other side lacks, under a shared-token anchor
(has_value_opposition). Dotted versions (V1.1/V1.2) and bare years are
deliberately NOT values — version/date fragments are evolution noise the
calibration cleared on purpose; Chinese numerals canonicalise to Arabic so
"12 个" vs "十二个" still reads as a duplicate, not an opposition.

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
# value-opposition keep (2026-09-16, eval cf-oppo): minimum shared-token
# anchor for the two-sided shape (each side has a value the other lacks) and
# the stricter anchor for the one-sided shape (only one side carries a value
# — "必须持有 ISO9001" vs "无需任何认证").
VALUE_OPPOSITION_MIN_COMMON = 4
VALUE_OPPOSITION_ONE_SIDED_MIN_COMMON = 8

_DATE_RE = re.compile(
    r"\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}日?"
    r"|\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?"
    r"|\d{2}:\d{2}(:\d{2})?"
)
_SEPARATOR_RE = re.compile(r"^-{3,}$|^={3,}$|^[~_*=]{3,}$")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


# ── internal (same-memory) noise shapes (0.16.3, live-library calibrated) ──
# Two structural shapes that rule-fire as contradictions but never are:
#
# 1. Markdown TABLE units: a status matrix re-sliced into several units makes
#    its ENUM values (待受理/已受理/已完成) collide with the polarity/todo
#    word lists — 26/280 live rows, sampled all false. Row-level table
#    semantics are beyond the rules; a genuine in-table contradiction is
#    agent work via other channels, not a unit-pair rule.
# 2. Quote META lines ("> 更新：… | 来源：… | 对应文档：…"): evolution
#    stamps of the note itself, not claims.
#
# Deliberately NOT added: a cosine gate for internal numeric pairs. The
# cross-memorory calibration (low cos = coincidence) does NOT transfer —
# inside ONE document, two sentences stating DIFFERENT values of the same
# metric are exactly the low-cos shape (live row #34943: "0.29 人力" vs
# "0.25 人力", cos 0.44, a TRUE conflict). The 0.16.0 genuine-shape gate
# stays.
_TABLE_SEPARATOR_RE = re.compile(r"\|-{2,}")
_META_LINE_RE = re.compile(r"更新|来源|对应文档|原文|链接")
_META_LINE_MAX_CHARS = 200


def _is_table_unit(quote: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.search(quote)) or quote.count("|") >= 5


def _is_meta_line(quote: str) -> bool:
    t = quote.strip()
    return (
        t.startswith(">")
        and len(t) <= _META_LINE_MAX_CHARS
        and bool(_META_LINE_RE.search(t[:40]))
    )


def internal_noise_pair(quote_a: "str | None", quote_b: "str | None") -> bool:
    """True when a same-memory unit pair is structural noise, never a
    contradiction (table slices, note-meta lines). Applied by BOTH the
    write-time internal examination and the scan-side _examine_internal."""
    a = quote_a or ""
    b = quote_b or ""
    if _is_table_unit(a) or _is_table_unit(b):
        return True
    return _is_meta_line(a) or _is_meta_line(b)


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
    return float(len(set_a & set_b)) / float((len(set_a) * len(set_b)) ** 0.5)


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


# ── value-opposition keep (2026-09-16, eval cf-oppo-01/06/07/11) ──
# "Value-shaped" tokens, two grades: STRONG = digit-bearing tokens (p6,
# iso9001, 500ms, 24) and Chinese numerals before a measure/unit char
# (两个 → 2, 四倍 → 4) canonicalised to Arabic; LATIN = bare identifiers
# (mysql, postgresql, lpr). Dotted versions and dates are scrubbed BEFORE
# tokenising — "0.16.0" shatters into digit tokens (0, 16, 0) that are
# evolution noise, never values (the V1.1/V1.2 calibration pair must stay
# cleared); bare years are excluded at the token level.
_DOTTED_VERSION_SPAN_RE = re.compile(r"v?\d+(?:\.\d+)+")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_LATIN_VALUE_RE = re.compile(r"[a-z][a-z0-9]+")
_LATIN_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "http", "https",
    "www", "com", "org", "net", "io",
})
_CN_NUM_VALUE_RE = re.compile(r"[零一二三四五六七八九十百千万两]+(?=[个倍天次条人台轮])")
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_to_int(text: str) -> "int | None":
    """Minimal Chinese-numeral parser (digits + 十百千万); None on non-numeral."""
    total = section = number = 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if unit == 10000:
                total += (section + number) * unit
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number


def _value_tokens(text: str) -> "tuple[set[str], set[str]]":
    """Return (strong, latin) value-token sets; see the block comment above."""
    scrubbed = _DOTTED_VERSION_SPAN_RE.sub(" ", _DATE_RE.sub(" ", text.casefold()))
    strong: set[str] = set()
    latin: set[str] = set()
    for token in _tokens(scrubbed):
        if _YEAR_RE.fullmatch(token):
            continue
        if any(ch.isdigit() for ch in token):
            strong.add(token)
        elif _LATIN_VALUE_RE.fullmatch(token) and token not in _LATIN_STOPWORDS:
            latin.add(token)
    for match in _CN_NUM_VALUE_RE.finditer(scrubbed):
        number = _cn_to_int(match.group(0))
        if number is not None:
            strong.add(str(number))
    return strong, latin


def has_value_opposition(a: str, b: str) -> bool:
    """Each side carries a value the other lacks, under a shared-token anchor.

    Two-sided (both have exclusive values → MySQL vs PostgreSQL) needs the
    loose anchor and admits latin identifiers. One-sided (only one side
    carries a value → 必须持有 ISO9001 vs 无需任何认证) counts STRONG values
    only and needs the strict anchor — a bare one-sided latin word ("main")
    is topic noise, and "same topic, one side mentions a number" is common
    enough to demand a strong shared skeleton. Chinese numerals canonicalise
    through the value-set subtraction: "12 个" vs "十二个" cancels and stays
    a duplicate.
    """
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    common = len(tokens_a & tokens_b)
    strong_a, latin_a = _value_tokens(a)
    strong_b, latin_b = _value_tokens(b)
    values_a, values_b = strong_a | latin_a, strong_b | latin_b
    only_a = values_a - tokens_b - values_b
    only_b = values_b - tokens_a - values_a
    if only_a and only_b:
        return common >= VALUE_OPPOSITION_MIN_COMMON
    if (only_a & strong_a) or (only_b & strong_b):
        return common >= VALUE_OPPOSITION_ONE_SIDED_MIN_COMMON
    return False



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
    return "keep" if (
        has_small_symmetric_difference(quote_a, quote_b)
        or has_value_opposition(quote_a, quote_b)
    ) else "clear"
