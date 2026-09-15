"""Equivalence gate for validate_product_payload.

The corpus in tests/golden/validation.json was generated from the pre-refactor
implementation by scripts/gen_golden_validation.py. Every record pins four
observable outcomes at once -- the reported error, the appended warnings, the
in-place payload mutations callers depend on, and (for the boundary calls that
currently raise) the exception type. Splitting the function into ordered
validators must reproduce all four byte for byte.

Regenerating the corpus is how a *deliberate* behaviour change is recorded; it
must be its own commit with the diff reviewed line by line.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.validation import validate_product_payload

GOLDEN_PATH = Path(__file__).parent / "golden" / "validation.json"
GOLDEN: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _expand(obj: Any) -> Any:
    """Undo the corpus' giant-literal placeholders (see gen_golden_validation).

    Reconstruction is exact string multiplication -- microseconds even for the
    megabyte cases, so the slimming costs nothing at test time.
    """
    if isinstance(obj, dict):
        if {"$rep", "$n"} <= set(obj):
            return obj["$rep"] * obj["$n"]
        return {key: _expand(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand(item) for item in obj]
    return obj


def test_hash_randomization_disabled() -> None:
    """difflib.get_close_matches iterates a set, so did_you_mean is seed-bound.

    This must fail rather than skip: a silently seed-random CI run would be
    exercising a gate whose expected values were recorded under a different
    iteration order.
    """
    assert sys.flags.hash_randomization == 0, (
        "run pytest with PYTHONHASHSEED=0 (see scripts/gen_golden_validation.py)"
    )


def test_corpus_is_non_trivial() -> None:
    """Guard against an empty or truncated corpus silently disarming the gate."""
    assert len(GOLDEN) > 800
    assert sum(1 for case in GOLDEN if case["error"]) > 400
    assert sum(1 for case in GOLDEN if case["warnings"]) >= 5
    # Invariant since the 0.16.6 boundary fix: the validator NEVER raises on
    # hostile input. These cases used to escape as OverflowError/TypeError; a
    # non-empty raises column means the boundary is leaking exceptions again.
    assert sum(1 for case in GOLDEN if case["raises"]) == 0


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: str(case["case_id"]))
def test_validation_matches_golden(case: dict[str, Any]) -> None:
    payload = copy.deepcopy(_expand(case["payload_in"]))
    if case["raises"] is not None:
        with pytest.raises(BaseException) as exc_info:
            validate_product_payload(case["surface"], case["operation"], payload)
        assert type(exc_info.value).__name__ == case["raises"]["type"]
        assert str(exc_info.value)[:40] == case["raises"]["msg_prefix"]
        # Mutations applied before the raise are observable too: a validator
        # reordered ahead of the raising one would otherwise slip through.
        assert payload == _expand(case["payload_out"])
        return

    result = validate_product_payload(case["surface"], case["operation"], payload)
    assert result.error == case["error"]
    assert list(result.warnings) == case["warnings"]  # order-sensitive, never sorted
    assert payload == _expand(case["payload_out"])


def test_unhashable_values_never_escape_the_boundary() -> None:
    """Direct, because JSON cannot carry these shapes -- only in-process
    callers (pipeline/write, backup replay) can. Before the boundary fix each
    of these escaped as TypeError/OverflowError through three call layers."""
    surrogate = "a\ud800b"  # legal JSON string, unencodable as UTF-8
    probes = [
        ("memory", "batch_read", {"memory_ids": [1], "content_mode": {"preview"}}),
        ("memory", "remember", {"content": "a", "subject": "b", "status": {"a": 1}}),
        ("memory", "remember", {"content": "a", "subject": "b", "confidence": 10 ** 400}),
        ("memory_repair", "semantic_control", {"action": "status", "timeout": 10 ** 400}),
        ("memory", "find", {"query": "q", "query_embedding": [10 ** 400]}),
        # Beyond Python 3.11's 4300-digit int(str) cap.
        ("memory", "read", {"id": "9" * 5000}),
        ("memory_review", "expired", {"limit": "9" * 5000}),
        ("memory", "batch_read", {"memory_ids": ["9" * 5000], "content_mode": "preview"}),
        # Lone surrogates: byte-check and every bounded/tag string now refuse
        # them at the boundary instead of exploding at the sqlite bind.
        ("memory", "remember", {"content": surrogate, "subject": "b"}),
        ("memory", "remember", {"content": "a", "subject": "s\udfffubject"}),
        ("memory", "remember", {"content": "a", "subject": "b", "tags": [surrogate]}),
        ("memory", "update", {"id": 1, "old_text": surrogate, "new_text": "x"}),
        # In-process-only shapes: JSON cannot express any of these keys/values.
        ("memory", "remember", {"content": "a", "subject": "b", "source_type": 10 ** 5000}),
    ]
    for surface, operation, payload in probes:
        result = validate_product_payload(surface, operation, dict(payload))
        assert result.error is not None, f"{surface}/{operation} leaked {payload} through"
        assert result.error["error"] == "invalid_input"



class _Unserialisable:
    """json.dumps refuses this, which is how the _json_size guards are reached."""


# Field -> (surface, operation, payload builder). These guards sit behind
# _json_size and therefore need a value JSON cannot represent, so they can
# never appear in the golden corpus. Covered directly instead.
_JSON_SIZE_GUARDS = [
    ("queries", "memory", "batch_find",
     lambda bad: {"queries": [{"query": "x", "id": bad}]}),
    ("metadata", "memory", "remember",
     lambda bad: {"content": "a", "subject": "b", "metadata": {"k": bad}}),
    ("members", "memory_repair", "record_conflict",
     lambda bad: {"members": [{"k": bad}], "authorized": True}),
    ("candidate_key", "memory_repair", "record_conflict",
     lambda bad: {"candidate_key": {"k": bad}, "authorized": True}),
    ("slot_key", "memory_repair", "record_conflict",
     lambda bad: {"slot_key": {"k": bad}, "authorized": True}),
]


@pytest.mark.parametrize(
    ("field", "surface", "operation", "build"), _JSON_SIZE_GUARDS,
    ids=[guard[0] for guard in _JSON_SIZE_GUARDS],
)
def test_non_serialisable_values_are_refused(
    field: str, surface: str, operation: str, build: Any,
) -> None:
    result = validate_product_payload(surface, operation, build(_Unserialisable()))
    assert result.error is not None, f"{field} accepted a non-serialisable value"
    assert result.error["error"] in {"invalid_input", "resource_limit_exceeded"}
