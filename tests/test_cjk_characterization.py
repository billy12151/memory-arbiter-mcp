"""Characterization tests: lock the CURRENT (pre-refactor) CJK regex behavior.

These tests prove "what the two CJK regexes match today" so Phase 1 cannot
silently change recall/subject behavior by merging them (R11).

Authoritative measurement (2026-08-09, importing the real compiled patterns
from the package — NOT re-typed literals):

  * ``search._CJK_RE`` (shared by anchors.py) matches 38960 codepoints.
  * ``db._CJK_CHAR_RE`` matches 39024 codepoints.
  * ``db._CJK_CHAR_RE`` is a strict SUPERSET; the ONLY difference is the range
    **U+4DC0–U+4DFF** (64 Yijing hexagram / legacy symbols), which db matches
    and search does not.

Because the match sets differ, the two regexes MUST be kept as two named
constants. A naive "use the broadest" merge would flip ``search``'s behavior
for those 64 codepoints (and ``db``'s if merged the other way), changing FTS
sanitize / subject tokenization for any query containing them.

These tests assert the CURRENT difference. A future deliberate unification must
update them intentionally, not as a drive-by in a move commit.
"""
from __future__ import annotations

import pytest

from memory_arbiter.db import _CJK_CHAR_RE, _subject_tokens
from memory_arbiter.search import _CJK_RE


@pytest.mark.parametrize("cp", [
    0x3400,  # 㐀 Ext-A first
    0x4DBF,  # 䶿 Ext-A last
    0x4E00,  # 一 CJK Unified first
    0x9FFF,  # 鿿 CJK Unified last
    0xF900,  # 豈 Compat first
    0x3042,  # あ Hiragana
    0x30A2,  # ア Katakana
    0xAC00,  # 가 Hangul first
])
def test_shared_core_codepoints_match_both(cp: int) -> None:
    """Codepoints both regexes agree on (the CJK core)."""
    ch = chr(cp)
    assert _CJK_RE.search(ch), f"search._CJK_RE should match U+{cp:04X}"
    assert _CJK_CHAR_RE.search(ch), f"db._CJK_CHAR_RE should match U+{cp:04X}"


@pytest.mark.parametrize("cp", [0x4DC0, 0x4DFF])
def test_only_difference_is_yijing_range(cp: int) -> None:
    """Lock the CURRENT sole divergence: U+4DC0-U+4DFF matches db, not search.

    (Probing the endpoints; the full-range subset proof below covers the rest.)"""
    ch = chr(cp)
    assert _CJK_CHAR_RE.search(ch), f"db._CJK_CHAR_RE should match U+{cp:04X}"
    assert not _CJK_RE.search(ch), (
        f"search._CJK_RE currently does NOT match U+{cp:04X}; if it now does, "
        "the regexes were unified and this characterization test must be updated "
        "deliberately (not as a drive-by)"
    )


def test_db_is_strict_superset_of_search() -> None:
    """Whole-range proof: search ⊂ db and the delta is exactly U+4DC0-4DFF."""
    search_hits = {cp for cp in range(0x2000, 0xDB00) if _CJK_RE.search(chr(cp))}
    db_hits = {cp for cp in range(0x2000, 0xDB00) if _CJK_CHAR_RE.search(chr(cp))}
    assert search_hits < db_hits, "search set must be a strict subset of db set"
    assert (db_hits - search_hits) == set(range(0x4DC0, 0x4E00))
    assert not (search_hits - db_hits), "search must not match anything db misses"


def test_ascii_and_digits_match_neither() -> None:
    for ch in "aZ019_":
        assert not _CJK_RE.search(ch)
        assert not _CJK_CHAR_RE.search(ch)


def test_subject_tokens_uses_two_char_sliding_windows() -> None:
    """db._subject_tokens splits a CJK run into 2-char sliding windows."""
    assert _subject_tokens("金营项目") == ["金营", "营项", "项目"]
    assert _subject_tokens("金营") == ["金营"]
    assert _subject_tokens("金") == []  # a lone CJK char forms no 2-char window


def test_subject_tokens_keeps_ascii_tokens_whole() -> None:
    tokens = _subject_tokens("release 金营项目 v0.12")
    assert "release" in tokens
    assert "金营" in tokens and "项目" in tokens
