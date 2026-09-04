"""Deterministic token-count estimation for response size reporting.

Pure-Python bucket table calibrated against a Qwen2.5 tokenizer ablation on
real mema records (2026-09-02, .workbuddy/token_probe*.py): five real records
landed within ±7% (+6.6/-2.6/+1.2/-4.4/+2.1%). Known skew, documented in the
memory tool's help (find_size_metering) and the CHANGELOG: continuous Chinese
prose overestimates ~30% (the 0.6/char deep-merge rate degrades to 0.77 when
mixed with other scripts) and pure English prose ~17%. The estimate and the
estimated share one yardstick, so "saved versus pasting the full text"
comparisons stay meaningful.
"""
from __future__ import annotations

import json
import re
from typing import Any

TOKEN_ESTIMATE_BASIS = "heuristic_v1(qwen2.5-calibrated)"

_WORD_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RE = re.compile(r"[0-9]+")

# Per-unit costs by bucket; see module docstring for calibration notes.
_CJK_PER_CHAR = 0.77            # CJK unified ideographs
_CJK_PUNCT_PER_CHAR = 0.85      # CJK punctuation + fullwidth forms
_WORD_PER_TOKEN = 1.15          # ASCII word (Qwen vocab is rich; long words are single tokens)
_DIGIT_PER_CHAR = 1.15          # versions/dates: nearly every digit is a token
_ASCII_PUNCT_PER_CHAR = 0.6     # . , : ; ( ) etc.
_MARKDOWN_PER_CHAR = 0.9        # # * _ ` > [ ] | ~ structural markup
_NEWLINE_PER_CHAR = 1.0         # structure carries hidden cost
_SPACE_PER_CHAR = 0.15
_OTHER_PER_CHAR = 0.9           # remaining scripts/symbols

_MARKDOWN_CHARS = set("#*_`>[]|~=+-")

_CJK_RANGES = (
    (0x3400, 0x9FFF),    # CJK ext-A + unified
    (0xF900, 0xFAFF),    # compatibility ideographs
    (0x20000, 0x2FA1F),  # ext-B..: outside BMP, handled via ord()
)

_CJK_PUNCT_RANGES = (
    (0x3000, 0x303F),    # CJK symbols and punctuation
    (0xFF01, 0xFF60),    # fullwidth forms
)


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= code <= high for low, high in ranges)


def estimate_tokens(text: str) -> int:
    """Estimate LLM token count for one string (deterministic, zero deps)."""
    if not text:
        return 0
    total = 0.0
    consumed = [False] * len(text)  # words/digits claimed by regex passes
    for match in _WORD_RE.finditer(text):
        total += _WORD_PER_TOKEN
        for index in range(match.start(), match.end()):
            consumed[index] = True
    for match in _DIGIT_RE.finditer(text):
        total += _DIGIT_PER_CHAR * (match.end() - match.start())
        for index in range(match.start(), match.end()):
            consumed[index] = True
    for index, char in enumerate(text):
        if consumed[index]:
            continue
        if char == "\n":
            total += _NEWLINE_PER_CHAR
        elif char == " ":
            total += _SPACE_PER_CHAR
        elif char in _MARKDOWN_CHARS:
            total += _MARKDOWN_PER_CHAR
        elif char.isascii():
            total += _ASCII_PUNCT_PER_CHAR
        else:
            code = ord(char)
            if _in_ranges(code, _CJK_RANGES):
                total += _CJK_PER_CHAR
            elif _in_ranges(code, _CJK_PUNCT_RANGES):
                total += _CJK_PUNCT_PER_CHAR
            else:
                total += _OTHER_PER_CHAR
    return int(round(total))


def meter_payloads(payloads: list[Any]) -> dict[str, int]:
    """Meter response payloads for the shared size block.

    One yardstick for every recall surface (find/read/expired/history):
    each item is dumped exactly as it serializes into the response and
    estimated with the same bucket table, so cross-surface token reports
    ("find page ~120 + read #123 ~450") stay comparable.
    """
    dumps = [
        json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)
        for p in payloads
    ]
    return {
        "returned_chars": sum(len(p) for p in dumps),
        "returned_count": len(dumps),
        "tokens_estimate": sum(estimate_tokens(p) for p in dumps),
    }
