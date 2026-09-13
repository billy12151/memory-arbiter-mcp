"""Generate the validate_product_payload equivalence corpus.

Run once against the *unmodified* implementation; the snapshot it writes is
what ``tests/test_golden_validation.py`` then holds the refactor to.

    PYTHONHASHSEED=0 python scripts/gen_golden_validation.py

PYTHONHASHSEED must be 0: ``difflib.get_close_matches`` iterates ``allowed``,
which is a ``set``. The current registry has no ratio ties (verified by
exhaustive probing across five seeds), but a future field could introduce one
and then ``did_you_mean`` would become order-dependent.

Each record captures four observable outcomes, because the refactor must
preserve all of them:

* ``error``      -- which field is reported first (block order is behaviour)
* ``warnings``   -- appended, order-sensitive
* ``payload_out``-- the in-place mutations callers depend on
* ``raises``     -- three float() sites do not catch OverflowError; that is
                    current behaviour and has to stay pinned
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.validation import (  # noqa: E402
    MAX_BATCH_FIND_QUERIES,
    MAX_BATCH_IDS,
    MAX_CONFLICT_MEMBERS,
    MAX_METADATA_BYTES,
    MAX_QUERY_CHARS,
    MAX_REPLACEMENT_TEXT_CHARS,
    MAX_SUBJECT_CHARS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_TEXT_FIELD_CHARS,
    PRODUCT_FIELD_REGISTRY,
    validate_product_payload,
)

# Probe the interpreter, not os.environ: PYTHONHASHSEED is read once at startup,
# so a parent process writing os.environ in-flight would satisfy an env check
# while randomisation stays on.
assert sys.flags.hash_randomization == 0, "run with PYTHONHASHSEED=0"

OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "validation.json"

# --------------------------------------------------------------------------
# Field kinds. Drives which injections are meaningful per field.
#
#   none      -- no validation rule at all; only record that it is accepted
#   str_len   -- bounded string
#   bytes     -- bounded by UTF-8 byte length
#   list_str  -- bounded list of bounded strings
#   list_obj  -- bounded list of JSON objects
#   int_range -- bounded integer
#   int_id    -- positive integer id
#   enum      -- fixed value set
#   shape     -- structural (dict / nested object)
#   iso8601   -- timestamp; a "boundary legal" value is unconstructible because
#                the rule is len<=128 AND parseable, and ISO8601 tops out ~35
# --------------------------------------------------------------------------
FIELD_KIND: dict[str, str] = {
    "action": "none", "add_tags": "list_str", "after_time": "iso8601", "alias": "none",
    "anchor_memory_id": "none", "apply_plan": "list_obj", "audit_id": "none",
    "authorized": "none", "batch": "none", "batch_size": "int_range",
    "before_time": "iso8601", "candidate_key": "shape", "canonical": "str_len",
    "chosen_value": "str_len", "clear": "none", "confidence": "float_unit",
    "conflict_id": "int_id", "conflict_point": "str_len", "content": "bytes",
    "content_hash": "none", "content_mode": "none", "debug_ranking": "none",
    "decided_by": "none", "decisions": "none", "deduplicate": "none", "deep": "none",
    "default_fallback": "none", "detector_version": "str_len", "dry_run": "none",
    "entity": "str_len", "event_time": "iso8601", "expected_content_hash": "none",
    "expected_revision": "int_range", "expected_version": "int_range",
    "from": "str_len", "id": "int_id", "include_check": "none",
    "include_conflict_signal": "none", "include_duplicates": "none",
    "include_linked_open_items": "none", "include_quotes": "none",
    "include_size": "none", "include_unassigned": "none", "ingest_time": "iso8601",
    "k": "none", "limit": "int_range", "limit_per_query": "int_range",
    "loser_ids": "none", "max_distance": "none", "max_memories": "none",
    "members": "list_obj", "memory_id": "int_id", "memory_ids": "manual",
    "merged_content": "none", "metadata": "shape", "neighbor_k": "none",
    "new": "str_len", "new_content": "bytes", "new_subject": "str_len",
    "new_tags": "list_str", "new_text": "str_len", "new_workspace": "str_len",
    "notice_id": "int_id", "offset": "int_range", "old": "str_len",
    "old_text": "str_len", "older_than_days": "int_range", "page_size": "none",
    "page_token": "none", "patches": "manual", "prompt_version": "str_len",
    "protection_level": "enum", "queries": "manual", "query": "str_len",
    "query_embedding": "shape", "reason": "str_len", "ref": "str_len",
    "remove_tags": "list_str", "resolution_memory_id": "none", "scope": "str_len",
    "slot_key": "shape", "source": "str_len", "source_ref": "str_len",
    "source_type": "enum", "span": "none", "spans": "shape", "status": "manual",
    "subject": "str_len", "superseded_by": "int_id", "survivor_id": "none",
    "tags": "list_str", "tags_filter": "list_str", "tags_only": "none",
    "task": "none", "time_budget_s": "none", "timeout": "float_timeout",
    "to": "str_len", "topic": "none", "value_groups": "list_obj",
    "view": "none", "workspace": "str_len", "workspaces": "manual",
}

STR_LEN_CAP: dict[str, int] = {
    "subject": MAX_SUBJECT_CHARS, "new_subject": MAX_SUBJECT_CHARS,
    "query": MAX_QUERY_CHARS,
    "old_text": MAX_REPLACEMENT_TEXT_CHARS, "new_text": MAX_REPLACEMENT_TEXT_CHARS,
}

INT_RANGE: dict[str, tuple[int, int]] = {
    "limit": (1, 100), "offset": (0, 10_000), "batch_size": (1, 500),
    "older_than_days": (0, 365_000), "expected_version": (1, 2_147_483_647),
    "expected_revision": (1, 2_147_483_647), "limit_per_query": (1, 20),
}

LEGAL_SAMPLE: dict[str, Any] = {
    "action": "page", "add_tags": ["a"], "after_time": "2026-01-01T00:00:00+00:00",
    "alias": "alias-x", "anchor_memory_id": 1, "apply_plan": [{"memory_id": 1}],
    "audit_id": 1, "authorized": True, "batch": 10, "batch_size": 10,
    "before_time": "2026-12-31T00:00:00+00:00", "candidate_key": {"k": "v"},
    "canonical": "ws", "chosen_value": "v", "clear": False, "confidence": 0.5,
    "conflict_id": 1, "conflict_point": "port", "content": "body text",
    "content_hash": "deadbeef", "content_mode": "preview", "debug_ranking": False,
    "decided_by": "owner", "decisions": [], "deduplicate": True, "deep": False,
    "default_fallback": False, "detector_version": "v1", "dry_run": True,
    "entity": "ent", "event_time": "2026-01-01T00:00:00+00:00",
    "expected_content_hash": "cafe", "expected_revision": 1, "expected_version": 1,
    "from": "ws-a", "id": 1, "include_check": False, "include_conflict_signal": False,
    "include_duplicates": False, "include_linked_open_items": False,
    "include_quotes": False, "include_size": False, "include_unassigned": False,
    "ingest_time": "2026-01-01T00:00:00+00:00", "k": 5, "limit": 10,
    "limit_per_query": 5, "loser_ids": [2], "max_distance": 0.5, "max_memories": 10,
    "members": [{"memory_id": 1}], "memory_id": 1, "memory_ids": [1, 2],
    "merged_content": "merged", "metadata": {"entity": "e"}, "neighbor_k": 5,
    "new": "ws-b", "new_content": "new body", "new_subject": "new subj",
    "new_tags": ["t"], "new_text": "after", "new_workspace": "ws-c", "notice_id": 1,
    "offset": 0, "old": "ws-a", "old_text": "before", "older_than_days": 30,
    "page_size": 20, "page_token": "tok",
    "patches": [{"old_text": "a", "new_text": "b"}], "prompt_version": "p1",
    "protection_level": "normal", "queries": [{"id": "q1", "query": "hello"}],
    "query": "hello", "query_embedding": [0.1, 0.2], "reason": "because",
    "ref": "chat", "remove_tags": ["t"], "resolution_memory_id": 1, "scope": "sc",
    "slot_key": {"s": "v"}, "source": "scan", "source_ref": "ref",
    "source_type": "agent_generated", "span": {"start": 0}, "spans": {"1": {"start": 0}},
    "status": "active", "subject": "subj", "superseded_by": 2, "survivor_id": 1,
    "tags": ["t"], "tags_filter": ["t"], "tags_only": False, "task": "page",
    "time_budget_s": 5, "timeout": 1.0, "to": "ws-b", "topic": "scheduled_tasks",
    "value_groups": [{"value": "v"}], "view": "overview", "workspace": "ws",
    "workspaces": ["ws"],
}

_MISSING = sorted(set(FIELD_KIND) ^ set(LEGAL_SAMPLE))
assert not _MISSING, f"kind/sample tables disagree on: {_MISSING}"


def run_case(case_id: str, surface: str, operation: str, payload_in: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(payload_in)
    error: dict[str, Any] | None = None
    warnings: list[str] = []
    raises: dict[str, str] | None = None
    try:
        result = validate_product_payload(surface, operation, payload)
        error, warnings = result.error, list(result.warnings)
    except BaseException as exc:  # noqa: BLE001 - OverflowError must land in the corpus
        raises = {"type": type(exc).__name__, "msg_prefix": str(exc)[:40]}
    return {
        "case_id": case_id, "surface": surface, "operation": operation,
        "payload_in": payload_in, "error": error, "warnings": warnings,
        "payload_out": payload, "raises": raises,
    }


def legal_payload(surface: str, operation: str) -> dict[str, Any]:
    fields = PRODUCT_FIELD_REGISTRY[(surface, operation)]
    payload = {f: copy.deepcopy(LEGAL_SAMPLE[f]) for f in sorted(fields)}
    # batch_read caps ids per content_mode; keep the all-fields payload legal.
    if (surface, operation) == ("memory", "batch_read"):
        payload["content_mode"] = "preview"
    return payload


def _injections(field: str) -> list[tuple[str, Any]]:
    """(suffix, value) pairs worth recording for this field's kind."""
    kind = FIELD_KIND[field]
    if kind == "manual":
        return []
    if kind == "none":
        # No rule exists. Record that a hostile value is accepted, so a future
        # refactor that accidentally *adds* a rule shows up as a diff.
        return [("wrongtype", {"unexpected": "object"})]
    if kind == "str_len":
        cap = STR_LEN_CAP.get(field, MAX_TEXT_FIELD_CHARS)
        return [("wrongtype", 123), ("over", "x" * (cap + 1)), ("atcap", "x" * cap)]
    if kind == "bytes":
        return [("wrongtype", 123), ("over", "\u4e00" * 800_000)]
    if kind == "list_str":
        return [
            ("wrongtype", "notalist"),
            ("overcount", ["t"] * (MAX_TAGS + 1)),
            ("overitem", ["x" * (MAX_TAG_CHARS + 1)]),
            ("atcap", ["t"] * MAX_TAGS),
        ]
    if kind == "list_obj":
        return [
            ("wrongtype", "notalist"),
            ("nonobject", ["x"]),
            ("overcount", [{"a": 1}] * (MAX_CONFLICT_MEMBERS + 1)),
        ]
    if kind == "int_id":
        return [("wrongtype", "abc"), ("zero", 0), ("negative", -1), ("float", 1.5), ("bool", True)]
    if kind == "int_range":
        low, high = INT_RANGE[field]
        return [
            ("wrongtype", "abc"), ("under", low - 1), ("over", high + 1),
            ("atmin", low), ("atmax", high),
        ]
    if kind == "float_unit":
        return [("wrongtype", "abc"), ("over", 1.5), ("under", -0.1), ("atmax", 1.0), ("bool", True)]
    if kind == "float_timeout":
        return [("wrongtype", "abc"), ("over", 601.0), ("under", -1.0), ("atmax", 600.0)]
    if kind == "enum":
        return [("badvalue", "definitely-not-valid")]
    if kind == "shape":
        if field == "query_embedding":
            return [("wrongtype", "notalist"), ("empty", []), ("nonnumeric", ["a"]), ("nonfinite", [float("inf")])]
        if field == "metadata":
            return [("wrongtype", "notadict"), ("oversize", {"k": "x" * (MAX_METADATA_BYTES + 10)})]
        if field == "spans":
            return [
                ("wrongtype", "notadict"), ("badkey", {"abc": {"start": 0}}),
                ("badspan", {"1": {"start": -1}}), ("badend", {"1": {"start": 5, "end": 5}}),
                ("unknownkey", {"1": {"start": 0, "bogus": 1}}),
            ]
        return [("wrongtype", "notadict")]
    if kind == "iso8601":
        # No "boundary legal" case: the rule is len<=128 AND ISO8601-parseable,
        # and a parseable timestamp never approaches 128 chars.
        return [("wrongtype", 123), ("unparseable", "not-a-timestamp"), ("toolong", "x" * 129)]
    raise AssertionError(f"unhandled kind {kind!r} for {field}")


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    keys = sorted(PRODUCT_FIELD_REGISTRY)

    # --- A: every registry key with every allowed field at a legal value ----
    for surface, operation in keys:
        cases.append(run_case(f"A/{surface}.{operation}", surface, operation, legal_payload(surface, operation)))

    # --- B: one hostile value at a time, per field kind ---------------------
    for surface, operation in keys:
        base = legal_payload(surface, operation)
        for field in sorted(PRODUCT_FIELD_REGISTRY[(surface, operation)]):
            for suffix, value in _injections(field):
                payload = copy.deepcopy(base)
                payload[field] = value
                cases.append(run_case(f"B/{surface}.{operation}/{field}/{suffix}", surface, operation, payload))

    # --- C: boundary calls that raise instead of returning invalid_input ----
    # Current behaviour, pinned deliberately (see the plan's appendix B).
    # Where the operation has a field an earlier validator rewrites in place,
    # the case carries it so the golden test's payload_out assertion on the
    # raising branch is not a no-op. semantic_control has no such field --
    # there the assertion still pins "timeout was NOT written back".
    cases.append(run_case("C/timeout_overflow", "memory_repair", "semantic_control",
                          {"action": "status", "timeout": 10 ** 400, "workspace": "ws"}))
    cases.append(run_case("C/confidence_overflow", "memory_govern", "confirm",
                          {"id": "7", "confidence": 10 ** 400, "authorized": True}))
    cases.append(run_case("C/embedding_overflow", "memory", "find",
                          {"query": "q", "query_embedding": [10 ** 400], "limit": "5"}))
    # Unhashable content_mode hits `x not in {...}` before any type check.
    cases.append(run_case("C/content_mode_unhashable_dict", "memory", "batch_read",
                          {"memory_ids": ["1"], "content_mode": {"a": 1}}))
    cases.append(run_case("C/content_mode_unhashable_list", "memory", "batch_read",
                          {"memory_ids": ["1"], "content_mode": ["full"]}))

    cases.extend(build_combination_cases())
    return cases


def build_combination_cases() -> list[dict[str, Any]]:
    """D: one case per cross-block constraint the refactor could break."""
    out: list[dict[str, Any]] = []
    add = lambda cid, s, o, p: out.append(run_case(cid, s, o, p))  # noqa: E731

    # Block order == which error wins.
    add("D/priority_id_before_content", "memory", "update",
        {"id": "abc", "new_content": "\u4e00" * 800_000})
    # The unknown-field pop decides whether a later block runs at all.
    add("D/pop_suppresses_confidence", "memory", "find", {"query": "x", "confidence": "garbage"})
    add("D/no_pop_reports_confidence", "memory", "remember",
        {"content": "a", "subject": "b", "confidence": "garbage"})
    # A warning and an error can coexist; the warning is still recorded.
    add("D/warning_plus_error", "memory", "remember",
        {"content": "a", "subject": "b", "bogus_field": 1, "confidence": 5})
    # Multiple unknown fields: warning order follows payload iteration order,
    # and every one of them is popped before the later blocks run.
    add("D/multiple_unknown_fields", "memory", "find",
        {"query": "x", "zzz_unknown": 1, "aaa_unknown": 2, "mmm_unknown": 3})
    add("D/unknown_field_only", "memory_review", "overview", {"bogus": "v"})
    # The non-string-key branch (validation.py "field names must be strings")
    # cannot live here: JSON has no integer keys, so a round-trip would silently
    # turn 7 into "7" and test a different branch. It is asserted directly in
    # tests/test_golden_validation.py instead.
    # judge skips the id block; its dispatcher reports missing receipt fields.
    add("D/judge_id_skipped", "memory", "judge", {"id": "abc", "authorized": True})
    # id is renamed to memory_id for exactly seven operations.
    for surface, operation in (
        ("memory", "read"), ("memory", "update"), ("memory_review", "history"),
        ("memory_repair", "set_entity"), ("memory_repair", "activate_pending"),
        ("memory_repair", "cleanup_history"), ("memory_govern", "confirm_pending_workspace"),
    ):
        add(f"D/id_rename/{surface}.{operation}", surface, operation, {"id": "abc"})
    # Unregistered (surface, operation): the unknown-field block is skipped.
    add("D/unregistered_combo", "memory", "nosuchaction", {"whatever": object.__doc__, "id": "abc"})
    # Removed field and remember-only aliases fail loudly with a migration hint.
    add("D/include_content_removed", "memory", "find", {"query": "x", "include_content": True})
    add("D/batch_find_include_content", "memory", "batch_find",
        {"queries": [{"query": "x"}], "include_content": True})
    for alias in ("content", "subject", "tags"):
        add(f"D/update_alias_{alias}", "memory", "update", {"id": 1, alias: "v"})
    # Near-miss on a protected field name suggests it instead of ignoring it.
    add("D/did_you_mean_workspace", "memory", "find", {"query": "x", "workspac": "ws"})
    add("D/did_you_mean_memory_id", "memory", "read", {"memory_i": 1})
    # memory_ids is owned by two blocks: batch_read caps the *raw* list first.
    add("D/batch_read_cap_before_coercion", "memory", "batch_read",
        {"memory_ids": ["1"] * 300, "content_mode": "full"})
    add("D/batch_read_string_ids_coerced", "memory", "batch_read",
        {"memory_ids": ["1", "2"], "content_mode": "full"})
    add("D/memory_ids_max_batch", "memory_govern", "move_memories_workspace",
        {"memory_ids": [1] * (MAX_BATCH_IDS + 1), "new_workspace": "ws", "authorized": True})
    # patches has two independent ceilings (count and bytes).
    add("D/patches_over_count", "memory", "update",
        {"id": 1, "patches": [{"old_text": "a", "new_text": "b"}] * 9})
    add("D/patches_over_bytes", "memory", "update",
        {"id": 1, "patches": [{"old_text": "x" * 900_000, "new_text": "y" * 900_000}] * 3})
    # Literal dict write order is the in-block error priority.
    add("D/order_bounded_strings", "memory_govern", "apply_conflict_action",
        {"reason": 123, "old_text": 456})
    add("D/order_structured_limits", "memory_repair", "record_conflict",
        {"members": "notalist", "value_groups": "notalist"})
    add("D/order_integer_limits", "memory_review", "expired", {"limit": "abc", "offset": "abc"})
    add("D/order_enums", "memory", "remember",
        {"content": "a", "subject": "b", "source_type": "bogus", "protection_level": "bogus"})
    # integer_limits bounds depend on (surface, operation) -- replay_backup differs.
    add("D/limit_replay_backup_high", "memory_repair", "replay_backup", {"limit": 9_000, "authorized": True})
    add("D/limit_other_surface_high", "memory_review", "expired", {"limit": 9_000})
    # remember refuses lifecycle statuses as write inputs.
    add("D/remember_status_lifecycle", "memory", "remember",
        {"content": "a", "subject": "b", "status": "superseded"})
    add("D/remember_status_pending", "memory", "remember",
        {"content": "a", "subject": "b", "status": "pending"})

    # --- batch_find: a single registry key guards a whole validation block ---
    q = lambda **kw: {"queries": [kw]}  # noqa: E731
    add("D/bf_not_a_list", "memory", "batch_find", {"queries": "nope"})
    add("D/bf_empty_list", "memory", "batch_find", {"queries": []})
    add("D/bf_too_many", "memory", "batch_find",
        {"queries": [{"query": f"q{i}"} for i in range(MAX_BATCH_FIND_QUERIES + 1)]})
    add("D/bf_total_bytes", "memory", "batch_find",
        {"queries": [{"id": f"i{i}", "query": "x" * 30_000} for i in range(8)]})
    add("D/bf_item_not_object", "memory", "batch_find", {"queries": ["plain string"]})
    add("D/bf_item_unknown_field", "memory", "batch_find", {"queries": [{"query": "x", "bogus": 1}]})
    add("D/bf_query_missing", "memory", "batch_find", {"queries": [{"id": "a"}]})
    add("D/bf_query_blank", "memory", "batch_find", {"queries": [{"query": "   "}]})
    add("D/bf_query_too_long", "memory", "batch_find", {"queries": [q(query="x" * (MAX_QUERY_CHARS + 1))["queries"][0]]})
    add("D/bf_id_not_string", "memory", "batch_find", {"queries": [{"id": 7, "query": "x"}]})
    add("D/bf_id_too_long", "memory", "batch_find", {"queries": [{"id": "i" * 65, "query": "x"}]})
    add("D/bf_duplicate_id", "memory", "batch_find",
        {"queries": [{"id": "same", "query": "a"}, {"id": "same", "query": "b"}]})
    add("D/bf_duplicate_query", "memory", "batch_find",
        {"queries": [{"id": "a", "query": "dup"}, {"id": "b", "query": "dup"}]})
    add("D/bf_id_defaults_to_query", "memory", "batch_find",
        {"queries": [{"query": "alpha"}, {"query": "beta"}]})
    add("D/bf_limit_per_query_bad", "memory", "batch_find",
        {"queries": [{"query": "x"}], "limit_per_query": 999})

    # --- batch_read spans / content_mode caps -------------------------------
    add("D/br_ids_not_list", "memory", "batch_read", {"memory_ids": "nope"})
    add("D/br_ids_empty", "memory", "batch_read", {"memory_ids": []})
    add("D/br_content_mode_bad", "memory", "batch_read", {"memory_ids": [1], "content_mode": "bogus"})
    add("D/br_hits_cap", "memory", "batch_read", {"memory_ids": list(range(1, 60)), "content_mode": "hits"})

    # --- patches item-level branches ----------------------------------------
    add("D/patches_not_list", "memory", "update", {"id": 1, "patches": "nope"})
    add("D/patches_empty", "memory", "update", {"id": 1, "patches": []})
    add("D/patches_non_object", "memory", "update", {"id": 1, "patches": ["x"]})
    add("D/patches_extra_key", "memory", "update",
        {"id": 1, "patches": [{"old_text": "a", "new_text": "b", "extra": 1}]})
    add("D/patches_missing_key", "memory", "update", {"id": 1, "patches": [{"old_text": "a"}]})
    add("D/patches_old_text_empty", "memory", "update",
        {"id": 1, "patches": [{"old_text": "", "new_text": "b"}]})
    add("D/patches_old_text_type", "memory", "update",
        {"id": 1, "patches": [{"old_text": 1, "new_text": "b"}]})
    add("D/patches_new_text_type", "memory", "update",
        {"id": 1, "patches": [{"old_text": "a", "new_text": 1}]})
    add("D/patches_item_too_long", "memory", "update",
        {"id": 1, "patches": [{"old_text": "x" * (MAX_REPLACEMENT_TEXT_CHARS + 1), "new_text": "b"}]})
    add("D/patches_new_text_empty_ok", "memory", "update",
        {"id": 1, "patches": [{"old_text": "a", "new_text": ""}]})

    # --- byte-ceiling branches on structured fields -------------------------
    big = "x" * 200_000
    add("D/metadata_oversize", "memory", "remember",
        {"content": "a", "subject": "b", "metadata": {"k": big + big}})
    add("D/members_oversize", "memory_repair", "record_conflict",
        {"members": [{"note": big}, {"note": big}], "authorized": True})
    add("D/value_groups_oversize", "memory_repair", "record_conflict",
        {"value_groups": [{"note": big}], "authorized": True})
    add("D/apply_plan_oversize", "memory", "judge",
        {"conflict_id": 1, "apply_plan": [{"note": big}], "authorized": True})
    add("D/candidate_key_oversize", "memory_repair", "record_conflict",
        {"candidate_key": {"k": "x" * 70_000}, "authorized": True})
    add("D/candidate_key_wrongtype", "memory_repair", "record_conflict",
        {"candidate_key": "notadict", "authorized": True})
    add("D/slot_key_oversize", "memory_repair", "record_conflict",
        {"slot_key": {"k": "x" * 5_000}, "authorized": True})
    add("D/slot_key_wrongtype", "memory_repair", "record_conflict",
        {"slot_key": "notadict", "authorized": True})

    # --- memory_ids element coercion / workspaces list ----------------------
    add("D/memory_ids_bad_element", "memory_govern", "move_memories_workspace",
        {"memory_ids": [1, "abc"], "new_workspace": "ws", "authorized": True})
    add("D/memory_ids_zero_element", "memory_govern", "move_memories_workspace",
        {"memory_ids": [0], "new_workspace": "ws", "authorized": True})
    add("D/workspaces_wrongtype", "memory_govern", "confirm_workspaces",
        {"workspaces": "nope", "authorized": True})
    add("D/workspaces_too_many", "memory_govern", "confirm_workspaces",
        {"workspaces": [f"w{i}" for i in range(101)], "authorized": True})
    add("D/workspaces_item_too_long", "memory_govern", "confirm_workspaces",
        {"workspaces": ["x" * (MAX_TEXT_FIELD_CHARS + 1)], "authorized": True})

    # --- id-family edge values ----------------------------------------------
    add("D/id_numeric_string_coerced", "memory", "read", {"id": "42"})
    add("D/id_float_rejected", "memory", "read", {"id": 1.5})
    add("D/id_bool_rejected", "memory", "read", {"id": True})
    add("D/id_beyond_sqlite_max", "memory", "read", {"id": 2 ** 63})
    add("D/id_empty_string", "memory", "read", {"id": ""})
    add("D/id_non_ascii_digits", "memory", "read", {"id": "\uff11\uff12\uff13"})
    add("D/id_plus_sign_accepted", "memory", "read", {"id": "+42"})
    add("D/id_negative_string", "memory", "read", {"id": "-42"})
    add("D/superseded_by_null_allowed", "memory_govern", "retire",
        {"id": 1, "superseded_by": None, "authorized": True})

    # Idempotence: memory_write re-validates a payload surfaces already mutated.
    first_in = {"content": "a", "subject": "b", "confidence": "0.5", "bogus": 1,
                "event_time": "2026-01-01T00:00:00+00:00"}
    first = run_case("D/idempotent#1", "memory", "remember", first_in)
    out.append(first)
    out.append(run_case("D/idempotent#2", "memory", "remember", copy.deepcopy(first["payload_out"])))
    second_in = {"id": "7", "patches": [{"old_text": "a", "new_text": "b"}], "expected_version": "3"}
    second = run_case("D/idempotent_update#1", "memory", "update", second_in)
    out.append(second)
    out.append(run_case("D/idempotent_update#2", "memory", "update", copy.deepcopy(second["payload_out"])))
    return out


def main() -> int:
    cases = build_cases()
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys must stay off: payload key order drives the order in which
    # unknown fields are warned about, and the corpus has to replay it exactly.
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    raising = sum(1 for c in cases if c["raises"])
    erroring = sum(1 for c in cases if c["error"])
    warning = sum(1 for c in cases if c["warnings"])
    print(f"wrote {len(cases)} cases to {OUT}")
    print(f"  errors={erroring}  warnings={warning}  raises={raising}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
