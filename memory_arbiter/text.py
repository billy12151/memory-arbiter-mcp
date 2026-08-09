"""Shared text/CJK/tag utilities — single implementation, many import sites.

Phase 1 (v0.12.4) consolidation. This module owns the ONE implementation of
text helpers that were previously duplicated across ``db.py``, ``search.py``,
``anchors.py``, and ``claims.py``. Original locations keep re-export aliases so
existing imports (and tests that import private names) keep working.

CJK: TWO constants are kept deliberately (R11 / test_cjk_characterization.py):
  * ``CJK_RE_SEARCH`` — the FTS-trigram/token path regex (matches U+3400-4DBF,
    4E00-9FFF, F900-FAFF, 3040-309F, 30A0-30FF, AC00-D7AF).
  * ``CJK_RE_SUBJECT`` — the subject-tokenisation regex; a strict SUPERSET that
    additionally matches U+4DC0-4DFF (Yijing hexagram symbols).
  The match sets differ, so they MUST NOT be merged.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# CJK regexes (TWO, not one — see module docstring)
# ---------------------------------------------------------------------------

# FTS / token path. Shared by search.py and anchors.py.
CJK_RE_SEARCH = re.compile('[㐀-䶿一-鿿豈-﫿぀-ゟ゠-ヿ가-힯]')

# Subject-tokenisation path (db.py). Superset: 㐀-鿿 is one CONTIGUOUS
# range, so it also matches U+4DC0-4DFF (Yijing symbols) that CJK_RE_SEARCH skips.
CJK_RE_SUBJECT = re.compile('[㐀-鿿豈-﫿぀-ヿ가-힯]')


# ---------------------------------------------------------------------------
# CJK token helpers
# ---------------------------------------------------------------------------

def is_cjk_token(token: str) -> bool:
    """True if the token contains ANY CJK char (FTS trigram path)."""
    return bool(CJK_RE_SEARCH.search(token))


def split_cjk_token(token: str) -> list[str]:
    """Split a CJK run into overlapping 3-character trigrams (unquoted).

    The FTS5 table uses ``tokenize='trigram'``: it only matches queries that
    produce ≥3-char tokens, and a strict CJK phrase silently misses when the
    query is even slightly overspecified. Splitting into OR-joined trigrams
    restores recall for Chinese queries. A <3-char CJK token yields nothing
    (caller falls back to LIKE).
    """
    cleaned = "".join(c for c in token if CJK_RE_SEARCH.search(c) or c.isalnum())
    if len(cleaned) < 3:
        return []
    return [cleaned[i : i + 3] for i in range(len(cleaned) - 2)]


def is_pure_cjk_token(token: str) -> bool:
    """True if the token contains NO ASCII alphanumeric chars.

    OPPOSITE of ``is_cjk_token`` (which is True on ANY CJK char). Used by the
    tag-scoring path to pick prefix/suffix vs equality matching. Do NOT merge
    these two helpers.
    """
    return not any(c.isascii() and c.isalnum() for c in token)


def normalize_token_for_tag_match(token: str) -> str:
    """Normalize a token for tag-level matching (applied to query AND tags).

    Strips a leading ``v`` only when it prefixes a version-like token
    (``v0.7.2`` → ``0.7.2``); words like ``vue`` are left alone.
    """
    s = (token or "").lower().strip()
    if len(s) > 1 and s[0] == "v" and s[1].isdigit():
        s = s[1:]
    return s


def cjk_substring_match(tag_norm: str, query_token_norm: str) -> bool:
    """CJK substring match — prefix/suffix only, never middle.

    The ``len >= 2`` gate excludes single-char tags (they would over-match).
    """
    if tag_norm == query_token_norm:
        return True
    if len(tag_norm) >= 2 and len(query_token_norm) >= 2:
        return query_token_norm.startswith(tag_norm) or query_token_norm.endswith(tag_norm)
    return False


def is_cjk_char(ch: str) -> bool:
    """True if a single char is in the CJK range (anchors path)."""
    return bool(CJK_RE_SEARCH.match(ch))


def cjk_bigrams(cjk_run: str) -> list[str]:
    """Split a CJK run into overlapping 2-char bigrams (anchors path)."""
    return [cjk_run[i : i + 2] for i in range(len(cjk_run) - 1)]


def subject_tokens(subject: str) -> list[str]:
    """Split a subject into tokens for LIKE-based candidate recall.

    CJK: 2-char sliding windows. ASCII: split on whitespace, keep tokens ≥ 2.
    """
    if not subject:
        return []
    tokens: list[str] = []
    for word in subject.split():
        if not word:
            continue
        if CJK_RE_SUBJECT.search(word):
            chars = "".join(c for c in word if CJK_RE_SUBJECT.match(c) or c.isalnum())
            for i in range(len(chars) - 1):
                tokens.append(chars[i : i + 2])
        elif len(word) >= 2:
            tokens.append(word.lower())
    return tokens


# ---------------------------------------------------------------------------
# Canonicalisation (entity/scope) — mirrors claims.py semantics
# ---------------------------------------------------------------------------

def canon_token(value: Any) -> str:
    """Canonicalise entity/attribute/scope without semantic aliasing."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    return text.strip("。，,;；:：.、· \t-—")


def canon_scope(value: Any) -> str:
    """Same lexical normalisation as entity; kept separate for API clarity."""
    return canon_token(value)


def canon_entity(value: Any) -> str:
    """Entity canonicalisation (identical to scope; separate name for clarity)."""
    return canon_token(value)


# ---------------------------------------------------------------------------
# Tags normalisation (ONE implementation; db/search both re-export it)
# ---------------------------------------------------------------------------

def coerce_tags(raw: Any) -> list[str]:
    """Normalise a ``tags`` value into a deduped ``list[str]``.

    Accepts list / JSON string / malformed / scalar / None; never raises —
    bad shapes yield []. Dedupes preserving first-seen order.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        tags = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            tags = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError, TypeError):
            return []
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if isinstance(t, str) and t not in seen:
            seen.add(t)
            out.append(t)
    return out
