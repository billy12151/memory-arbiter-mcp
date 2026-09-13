"""Narrow validation boundary for the four product MCP surfaces."""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import MemoryStatus, ProtectionLevel, SourceType
from .timeutil import parse_iso8601_utc

MAX_CONTENT_BYTES = 2 * 1024 * 1024
MAX_SQLITE_INTEGER = 2**63 - 1
MAX_SUBJECT_CHARS = 2_000
MAX_QUERY_CHARS = 32_000
MAX_TAGS = 100
MAX_TAG_CHARS = 256
MAX_METADATA_BYTES = 256 * 1024
MAX_TEXT_FIELD_CHARS = 2_000
MAX_REPLACEMENT_TEXT_CHARS = 1_000_000
MAX_UPDATE_PATCHES = 8
MAX_UPDATE_PATCHES_BYTES = 2 * 1024 * 1024
MAX_BATCH_IDS = 1_000
MAX_CONFLICT_MEMBERS = 256
MAX_CONFLICT_MEMBERS_BYTES = 256 * 1024
MAX_CONFLICT_VALUE_GROUPS_BYTES = 128 * 1024
MAX_CONFLICT_CANDIDATE_KEY_BYTES = 64 * 1024
MAX_CONFLICT_SLOT_KEY_BYTES = 4 * 1024
MAX_APPLY_PLAN_ITEMS = 256
MAX_APPLY_PLAN_BYTES = 128 * 1024
MAX_REVISION = 2_147_483_647
SEMANTIC_CONTROL_MAX_TIMEOUT = 600.0
MAX_RESULT_LIMIT = 100
MAX_OFFSET = 10_000
# v0.15.9 batch_find bounds (single source: constants, re-exported for the
# narrow validation boundary)
from .constants import (  # noqa: E402
    BATCH_FIND_MAX_LIMIT_PER_QUERY,
    BATCH_FIND_TOTAL_BYTES,
    BATCH_READ_MAX_FULL,
    BATCH_READ_MAX_HITS,
    BATCH_READ_MAX_PREVIEW,
    MAX_BATCH_FIND_QUERIES,
)

_SENSITIVE_FIELDS = {
    "authorized", "workspace", "memory_id", "conflict_id", "notice_id",
    "source_type", "protection_level", "status", "content", "new_content",
    "event_time", "ingest_time", "confidence", "dry_run",
}

_COMMON = {"topic", "workspace"}
PRODUCT_FIELD_REGISTRY: dict[tuple[str, str], set[str]] = {
    ("memory", "help"): {"topic", "action"},
    ("memory", "status"): {"workspace"},
    ("memory", "remember"): {
        "content", "workspace", "tags", "source_type", "source_ref",
        "event_time", "ingest_time", "confidence", "protection_level", "status",
        "subject", "metadata",
    },
    ("memory", "find"): {
        "query", "workspace", "tags", "limit", "offset", "debug_ranking",
        "query_embedding", "tags_filter", "after_time", "before_time",
        "source_type", "include_linked_open_items", "include_conflict_signal",
        "include_size", "content_mode",
    },
    ("memory", "batch_find"): {
        "queries", "workspace", "tags_filter", "after_time", "before_time",
        "source_type", "limit_per_query", "content_mode", "deduplicate",
    },
    ("memory", "read"): {"id", "memory_id", "span", "content_mode", "workspace"},
    ("memory", "batch_read"): {"memory_ids", "content_mode", "spans", "workspace"},
    ("memory", "update"): {
        "id", "memory_id", "new_content", "old_text", "new_text", "patches",
        "new_subject", "new_tags", "reason", "authorized", "tags_only", "add_tags",
        "remove_tags", "expected_version", "expected_content_hash", "content_hash",
        "workspace",
    },
    ("memory", "judge"): {
        "id", "conflict_id", "expected_revision", "chosen_value", "decided_by",
        "ref", "reason", "apply_plan", "resolution_memory_id", "authorized",
        "workspace",
    },
    ("memory_review", "overview"): _COMMON,
    ("memory_review", "doctor"): {"deep", "workspace"},
    ("memory_review", "audit"): {"workspace"},
    ("memory_review", "conflicts"): {"status", "limit", "source", "workspace"},
    ("memory_review", "conflict_detail"): {"id", "conflict_id", "workspace"},
    ("memory_review", "history"): {"id", "memory_id", "workspace"},
    ("memory_review", "expired"): {
        "query", "workspace", "tags", "limit", "offset", "debug_ranking",
        "query_embedding", "tags_filter", "after_time", "before_time",
        "source_type", "include_conflict_signal",
    },
    ("memory_review", "entities"): {"limit", "include_unassigned", "workspace"},
    ("memory_review", "help"): {"topic", "view"},
    ("memory_govern", "retire"): {"id", "memory_id", "reason", "superseded_by", "authorized", "workspace"},
    ("memory_govern", "merge_memories"): {
        "survivor_id", "loser_ids", "merged_content", "reason", "authorized", "workspace",
    },
    ("memory_govern", "resolve_conflict"): {
        "id", "conflict_id", "expected_revision", "reason", "authorized", "workspace",
    },
    ("memory_govern", "apply_conflict_action"): {
        "id", "conflict_id", "expected_revision", "memory_id", "action", "content",
        "old_text", "new_text", "reason", "authorized", "workspace",
    },
    ("memory_govern", "replan_conflict"): {
        "id", "conflict_id", "expected_revision", "apply_plan", "chosen_value",
        "resolution_memory_id", "authorized", "workspace",
    },
    ("memory_govern", "confirm"): {"id", "memory_id", "source_ref", "confidence", "authorized", "workspace"},
    ("memory_govern", "rename_workspace_canonical"): {"old", "new", "reason", "authorized"},
    ("memory_govern", "migrate_workspace"): {"from", "to", "reason", "authorized"},
    ("memory_govern", "move_memories_workspace"): {"memory_ids", "new_workspace", "reason", "authorized", "workspace", "default_fallback"},
    ("memory_govern", "rollback_auto_move"): {"audit_id", "reason", "authorized", "workspace"},
    ("memory_govern", "confirm_pending_workspace"): {"id", "memory_id", "canonical", "reason", "authorized", "workspace"},
    ("memory_govern", "confirm_workspaces"): {"workspaces", "reason", "authorized"},
    ("memory_govern", "separate_workspace_alias"): {"alias", "canonical", "reason", "authorized", "workspace"},
    ("memory_govern", "help"): {"topic", "action"},
    ("memory_repair", "rebuild_evidence"): {"memory_ids", "dry_run", "batch_size", "workspace"},
    ("memory_repair", "cleanup_history"): {"id", "memory_id", "older_than_days", "authorized", "workspace"},
    ("memory_repair", "set_entity"): {"id", "memory_id", "entity", "scope", "clear", "authorized", "workspace"},
    ("memory_repair", "activate_pending"): {"id", "memory_id", "authorized", "workspace"},
    ("memory_repair", "semantic_control"): {"action", "timeout", "workspace"},
    ("memory_repair", "notice"): {"action", "status", "limit", "id", "notice_id", "reason", "workspace"},
    ("memory_repair", "scan_pipeline"): {"action", "max_memories", "time_budget_s", "neighbor_k", "workspace"},
    ("memory_repair", "scan_queue"): {"action", "page_size", "page_token", "decisions", "workspace"},
    ("memory_repair", "scan_candidates"): {
        "anchor_memory_id", "batch", "k", "include_check", "max_distance",
        "include_duplicates", "include_quotes", "workspace",
    },
    ("memory_repair", "scan_duplicates"): {"include_quotes", "workspace"},
    ("memory_repair", "scan_workspace_anomalies"): {"workspace"},
    ("memory_repair", "record_conflict"): {
        "slot_key", "members", "value_groups", "candidate_key", "status",
        "detector_version", "prompt_version", "source", "reason", "conflict_point",
        "expected_revision", "authorized", "workspace",
    },
    ("memory_repair", "replay_backup"): {"dry_run", "authorized", "limit", "offset"},
    ("memory_repair", "normalize_workspaces"): {"dry_run", "authorized"},
    ("memory_repair", "help"): {"topic", "task"},
}


@dataclass
class ValidationResult:
    warnings: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None


def _error(field_name: str, reason: str, **detail: Any) -> dict[str, Any]:
    return {"error": "invalid_input", "field": field_name, "reason": reason, **detail}


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _controlled_integer(value: Any) -> int | None:
    """Accept JSON integers and canonical decimal strings, never floats/bools."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.isascii():
            return None
        digits = stripped[1:] if stripped[:1] in {"+", "-"} else stripped
        if not digits.isdigit():
            return None
        return int(stripped)
    return None


# --------------------------------------------------------------------------
# Ordered validators.
#
# Each one reports the first problem it finds as an error dict, or returns None
# to let the next one run. They deliberately share a single payload reference:
# the in-place normalisation (popped unknown keys, coerced ids, normalised
# patches, defaulted content_mode) is part of the contract -- surfaces hands the
# same dict straight to dispatch. Never copy the payload.
#
# They must also be idempotent: memory_write re-validates a payload that
# surfaces already normalised, so every remember is validated twice.
# --------------------------------------------------------------------------
_Validator = Callable[[str, str, dict[str, Any], ValidationResult], dict[str, Any] | None]


def _v_unknown_fields(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """Registry gate: removed-field migrations, update aliases, near-miss
    protected names, then drop whatever is left over.

    The pop is a precondition for every later validator: a field that is not in
    this operation's allow-list never reaches the type and range checks below,
    so this decides *whether* a bad value is reported at all, not merely which
    error wins.
    """
    allowed = PRODUCT_FIELD_REGISTRY.get((surface, operation))
    if allowed is None:
        return None
    unknown_keys: list[str] = []
    for key in payload:
        if not isinstance(key, str):
            return _error(str(key), "field names must be strings")
        if key in allowed:
            continue
        if (
            key == "include_content"
            and (surface, operation) in (("memory", "find"), ("memory", "batch_find"))
        ):
            # v0.15.10 breaking: silently ignoring the removed boolean would
            # look like a working call returning previews — fail loudly with
            # the migration pointer instead.
            return _error(
                key,
                'include_content was removed in v0.15.10; use '
                'content_mode="full" for full text or content_mode="hits" '
                "for vector-hit spans",
                did_you_mean="content_mode",
            )
        update_aliases = {
            "content": "new_content",
            "subject": "new_subject",
            "tags": "new_tags",
        }
        if (surface, operation) == ("memory", "update") and key in update_aliases:
            return _error(
                key,
                "remember field is not valid for update",
                did_you_mean=update_aliases[key],
            )
        suggestion = difflib.get_close_matches(key, allowed, n=1, cutoff=0.78)
        if suggestion and suggestion[0] in _SENSITIVE_FIELDS:
            return _error(key, "unknown field resembles a protected field", did_you_mean=suggestion[0])
        result.warnings.append(f"unknown field ignored: {key}")
        unknown_keys.append(key)
    for key in unknown_keys:
        payload.pop(key, None)
    return None


def _v_batch_find_queries(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """batch_find query list: shape, bounds, duplicate ids/queries, default ids."""
    if (surface, operation) != ("memory", "batch_find"):
        return None
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        return _error("queries", "must be a non-empty list of {id?, query} objects")
    if len(queries) > MAX_BATCH_FIND_QUERIES:
        return _error("queries", f"must contain at most {MAX_BATCH_FIND_QUERIES} items")
    try:
        queries_bytes = _json_size(queries)
    except (TypeError, ValueError, RecursionError):
        return _error("queries", "must contain only JSON-serializable values")
    if queries_bytes > BATCH_FIND_TOTAL_BYTES:
        return {
            "error": "resource_limit_exceeded", "field": "queries",
            "actual_bytes": queries_bytes, "max_bytes": BATCH_FIND_TOTAL_BYTES,
        }
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for item in queries:
        if not isinstance(item, dict):
            return _error("queries", "every item must be a JSON object with a non-empty query")
        unknown = set(item) - {"id", "query"}
        if unknown:
            return _error("queries", f"unknown item field(s): {', '.join(sorted(unknown))}")
        q = item.get("query")
        if not isinstance(q, str) or not q.strip():
            return _error("queries", "every item requires a non-empty string query")
        if len(q) > MAX_QUERY_CHARS:
            return _error("query", f"must be a string of at most {MAX_QUERY_CHARS} characters")
        qid = item.get("id")
        defaulted = qid is None
        if defaulted:
            qid = q
        if not isinstance(qid, str) or not qid or len(qid) > 64:
            return _error("queries", "id must be a non-empty string of at most 64 characters")
        if qid in seen_ids:
            return _error("queries", f"duplicate query id: {qid}")
        if q in seen_queries:
            return _error("queries", f"duplicate query: {q}")
        seen_ids.add(qid)
        seen_queries.add(q)
        # Normalize defaulted ids in place so dispatch sees explicit ids.
        item["id"] = qid
    limit_per_query = payload.get("limit_per_query")
    if limit_per_query is not None:
        parsed_limit = _controlled_integer(limit_per_query)
        if parsed_limit is None or not 1 <= parsed_limit <= BATCH_FIND_MAX_LIMIT_PER_QUERY:
            return _error(
                "limit_per_query",
                f"must be an integer between 1 and {BATCH_FIND_MAX_LIMIT_PER_QUERY}",
            )
        payload["limit_per_query"] = parsed_limit
    return None


def _v_remember_required(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """remember refuses to store a memory with no body or no subject."""
    if (surface, operation) != ("memory", "remember"):
        return None
    for key in ("content", "subject"):
        if key not in payload or not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            return _error(key, "is required and must be a non-empty string")
    return None


def _v_batch_read(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """batch_read id count, content_mode enum and per-mode cap, span shapes.

    The cap is measured on the *raw* list, before _v_memory_ids coerces string
    ids to integers. Merging the two memory_ids owners would change which error
    a caller sees for an over-cap batch, so they stay separate on purpose.
    """
    if (surface, operation) != ("memory", "batch_read"):
        return None
    memory_ids = payload.get("memory_ids")
    if not isinstance(memory_ids, list) or not memory_ids:
        return _error("memory_ids", "must be a non-empty list of positive integer ids")
    content_mode = payload.get("content_mode") or "preview"
    if content_mode not in {"preview", "hits", "full"}:
        return _error(
            "content_mode",
            'must be one of "preview" | "hits" | "full" (default "preview")',
        )
    cap = {"preview": BATCH_READ_MAX_PREVIEW, "hits": BATCH_READ_MAX_HITS, "full": BATCH_READ_MAX_FULL}[content_mode]
    if len(memory_ids) > cap:
        return _error(
            "memory_ids",
            f'content_mode="{content_mode}" accepts at most {cap} ids per call',
            cap=cap,
        )
    spans = payload.get("spans")
    if spans is not None:
        if not isinstance(spans, dict):
            return _error("spans", "must be an object keyed by memory_id: {start, end}")
        for key, span in spans.items():
            if _controlled_integer(key) is None:
                return _error("spans", "keys must be integer memory ids")
            if not isinstance(span, dict) or set(span) - {"start", "end"}:
                return _error("spans", "each span must be an object with optional integer start/end")
            raw_start = span.get("start", 0)
            raw_end = span.get("end")
            if isinstance(raw_start, bool) or not isinstance(raw_start, int) or raw_start < 0:
                return _error("spans", "span start must be a non-negative integer")
            if raw_end is not None and (isinstance(raw_end, bool) or not isinstance(raw_end, int) or raw_end <= raw_start):
                return _error("spans", "span end must be an integer greater than start")
    payload["content_mode"] = content_mode
    return None


def _v_id_fields(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """Positive-integer id family, coerced in place to the value that passed."""
    for key in ("id", "memory_id", "conflict_id", "notice_id", "superseded_by", "suggested_winner"):
        if key not in payload:
            continue
        if key in {"id", "conflict_id"} and (surface, operation) == ("memory", "judge"):
            # Their dispatchers intentionally report all missing receipt fields
            # before coercing the primary id. The dispatcher uses the same strict
            # integer policy, so skipping here does not admit floats.
            continue
        value = payload[key]
        if value is None and key in {"superseded_by", "suggested_winner"}:
            continue
        parsed_id = _controlled_integer(value)
        if parsed_id is None:
            field_name = "memory_id" if key == "id" and operation in {"read", "update", "history", "set_entity", "activate_pending", "cleanup_history", "confirm_pending_workspace"} else key
            return _error(field_name, "must be a positive integer")
        if parsed_id <= 0 or parsed_id > MAX_SQLITE_INTEGER:
            field_name = "memory_id" if key == "id" and operation in {"read", "update", "history", "set_entity", "activate_pending", "cleanup_history", "confirm_pending_workspace"} else key
            return _error(field_name, "must be a positive integer")
        # Preserve controlled numeric-string compatibility, but make the value
        # consumed by dispatch exactly the value that passed validation.
        payload[key] = parsed_id
    return None


def _v_content_bytes(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """Memory bodies are bounded by UTF-8 byte length, not character count."""
    for key in ("content", "new_content"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str):
                return _error(key, "must be a string")
            actual = len(value.encode("utf-8"))
            if actual > MAX_CONTENT_BYTES:
                return {"error": "resource_limit_exceeded", "field": key, "actual_bytes": actual, "max_bytes": MAX_CONTENT_BYTES}
    return None


# Write order is the in-block error priority: the first offending field in this
# mapping is the one reported. Do not sort it and do not turn it into a set.
_BOUNDED_STRINGS: dict[str, int] = {
    "subject": MAX_SUBJECT_CHARS,
    "new_subject": MAX_SUBJECT_CHARS,
    "query": MAX_QUERY_CHARS,
    "old_text": MAX_REPLACEMENT_TEXT_CHARS,
    "new_text": MAX_REPLACEMENT_TEXT_CHARS,
    "workspace": MAX_TEXT_FIELD_CHARS,
    "source_ref": MAX_TEXT_FIELD_CHARS,
    "agent_id": MAX_TEXT_FIELD_CHARS,
    "client": MAX_TEXT_FIELD_CHARS,
    "reason": MAX_TEXT_FIELD_CHARS,
    "ref": MAX_TEXT_FIELD_CHARS,
    "chosen_value": MAX_TEXT_FIELD_CHARS,
    "detector_version": MAX_TEXT_FIELD_CHARS,
    "prompt_version": MAX_TEXT_FIELD_CHARS,
    "canonical": MAX_TEXT_FIELD_CHARS,
    "old": MAX_TEXT_FIELD_CHARS,
    "new": MAX_TEXT_FIELD_CHARS,
    "from": MAX_TEXT_FIELD_CHARS,
    "to": MAX_TEXT_FIELD_CHARS,
    "new_workspace": MAX_TEXT_FIELD_CHARS,
    "entity": MAX_TEXT_FIELD_CHARS,
    "scope": MAX_TEXT_FIELD_CHARS,
    # Governance/scan metadata is echoed by review surfaces; unbounded
    # values would allow single-call storage amplification.
    "conflict_type": MAX_TEXT_FIELD_CHARS,
    "conflict_point": MAX_TEXT_FIELD_CHARS,
    "scan_prompt_version": MAX_TEXT_FIELD_CHARS,
    "scan_model": MAX_TEXT_FIELD_CHARS,
    "confidence_hint": MAX_TEXT_FIELD_CHARS,
    "judge_ref": MAX_TEXT_FIELD_CHARS,
    "usage_context": MAX_TEXT_FIELD_CHARS,
    "source": MAX_TEXT_FIELD_CHARS,
}


def _v_bounded_strings(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    for key, maximum in _BOUNDED_STRINGS.items():
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > maximum):
            return _error(key, f"must be a string of at most {maximum} characters")
    return None


def _v_tag_lists(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    for key in ("tags", "tags_filter", "new_tags", "add_tags", "remove_tags"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > MAX_TAGS or any(not isinstance(tag, str) or len(tag) > MAX_TAG_CHARS for tag in value):
            return _error(key, f"must be a list of at most {MAX_TAGS} strings, each at most {MAX_TAG_CHARS} characters")
    return None


def _v_metadata(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    metadata = payload.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return _error("metadata", "must be a JSON object")
    try:
        metadata_size = _json_size(metadata)
    except (TypeError, ValueError, RecursionError):
        return _error("metadata", "must contain only JSON-serializable values")
    if metadata_size > MAX_METADATA_BYTES:
        return {"error": "resource_limit_exceeded", "field": "metadata", "max_bytes": MAX_METADATA_BYTES}
    return None


# Write order is the in-block error priority; see _BOUNDED_STRINGS.
_STRUCTURED_LIMITS: dict[str, tuple[int, int]] = {
    "members": (MAX_CONFLICT_MEMBERS, MAX_CONFLICT_MEMBERS_BYTES),
    "value_groups": (MAX_CONFLICT_MEMBERS, MAX_CONFLICT_VALUE_GROUPS_BYTES),
    "apply_plan": (MAX_APPLY_PLAN_ITEMS, MAX_APPLY_PLAN_BYTES),
}


def _v_structured_limits(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    for key, (max_items, max_bytes) in _STRUCTURED_LIMITS.items():
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > max_items:
            return _error(key, f"must be a list with at most {max_items} items")
        if any(not isinstance(item, dict) for item in value):
            return _error(key, "must contain only JSON objects")
        try:
            actual_bytes = _json_size(value)
        except (TypeError, ValueError, RecursionError):
            return _error(key, "must contain only JSON-serializable values")
        if actual_bytes > max_bytes:
            return {
                "error": "resource_limit_exceeded", "field": key,
                "actual_bytes": actual_bytes, "max_bytes": max_bytes,
            }
    return None


def _v_candidate_key(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    candidate_key = payload.get("candidate_key")
    if candidate_key is None:
        return None
    if not isinstance(candidate_key, dict):
        return _error("candidate_key", "must be a JSON object")
    try:
        candidate_bytes = _json_size(candidate_key)
    except (TypeError, ValueError, RecursionError):
        return _error("candidate_key", "must contain only JSON-serializable values")
    if candidate_bytes > MAX_CONFLICT_CANDIDATE_KEY_BYTES:
        return {
            "error": "resource_limit_exceeded", "field": "candidate_key",
            "actual_bytes": candidate_bytes,
            "max_bytes": MAX_CONFLICT_CANDIDATE_KEY_BYTES,
        }
    return None


def _v_slot_key(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    slot_key = payload.get("slot_key")
    if slot_key is None:
        return None
    if not isinstance(slot_key, dict):
        return _error("slot_key", "must be a JSON object or null")
    try:
        slot_bytes = _json_size(slot_key)
    except (TypeError, ValueError, RecursionError):
        return _error("slot_key", "must contain only JSON-serializable values")
    if slot_bytes > MAX_CONFLICT_SLOT_KEY_BYTES:
        return {
            "error": "resource_limit_exceeded", "field": "slot_key",
            "actual_bytes": slot_bytes, "max_bytes": MAX_CONFLICT_SLOT_KEY_BYTES,
        }
    return None


def _v_memory_ids(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """Coerce the id list. batch_read already capped the raw list length above."""
    for key in ("memory_ids",):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > MAX_BATCH_IDS:
            return _error(key, f"must be a list with at most {MAX_BATCH_IDS} items")
        parsed_items: list[int] = []
        for item in value:
            parsed_item = _controlled_integer(item)
            if parsed_item is None or parsed_item <= 0 or parsed_item > MAX_SQLITE_INTEGER:
                return _error(key, "must contain positive integer ids")
            parsed_items.append(parsed_item)
        payload[key] = parsed_items
    return None


def _v_workspaces_list(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    # confirm_workspaces list: bounded like tags (count) and text fields (per-
    # item length) so one authorized call cannot persist an unbounded sidecar.
    workspaces_value = payload.get("workspaces")
    if workspaces_value is None:
        return None
    if (
        not isinstance(workspaces_value, list)
        or len(workspaces_value) > 100
        or any(not isinstance(item, str) or len(item) > MAX_TEXT_FIELD_CHARS for item in workspaces_value)
    ):
        return _error(
            "workspaces",
            f"must be a list of at most 100 strings, each at most {MAX_TEXT_FIELD_CHARS} characters",
        )
    return None


def _v_patches(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    # update patches (v0.15.12): [{old_text, new_text}, ...] — sequential
    # partial replacements applied atomically in one write transaction.
    # Guard here at the boundary (shape/count/size) so a malformed batch
    # never reaches the edit loop; the db layer re-checks defensively.
    patches_value = payload.get("patches")
    if patches_value is None:
        return None
    if not isinstance(patches_value, list) or not 1 <= len(patches_value) <= MAX_UPDATE_PATCHES:
        return _error("patches", f"must be a list of 1..{MAX_UPDATE_PATCHES} patch objects")
    if any(not isinstance(item, dict) for item in patches_value):
        return _error("patches", "must contain only JSON objects")
    normalized_patches: list[dict[str, Any]] = []
    for item in patches_value:
        if set(item) != {"old_text", "new_text"}:
            return _error(
                "patches",
                "each patch must have exactly the keys old_text and new_text",
            )
        old_text_value, new_text_value = item["old_text"], item["new_text"]
        if not isinstance(old_text_value, str) or not old_text_value:
            return _error("patches.old_text", "must be a non-empty string")
        if not isinstance(new_text_value, str):
            return _error("patches.new_text", "must be a string (empty deletes the fragment)")
        if len(old_text_value) > MAX_REPLACEMENT_TEXT_CHARS or len(new_text_value) > MAX_REPLACEMENT_TEXT_CHARS:
            return _error(
                "patches", f"old_text/new_text must each be at most {MAX_REPLACEMENT_TEXT_CHARS} characters",
            )
        normalized_patches.append({"old_text": old_text_value, "new_text": new_text_value})
    try:
        patches_bytes = _json_size(normalized_patches)
    except (TypeError, ValueError, RecursionError):  # pragma: no cover - every value is a checked str by here
        return _error("patches", "must contain only JSON-serializable values")
    if patches_bytes > MAX_UPDATE_PATCHES_BYTES:
        return {
            "error": "resource_limit_exceeded", "field": "patches",
            "actual_bytes": patches_bytes, "max_bytes": MAX_UPDATE_PATCHES_BYTES,
        }
    payload["patches"] = normalized_patches
    return None


def _v_integer_limits(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    """Bounded integers. The bounds depend on (surface, operation) -- replay_backup
    pages far deeper than a product read -- so this table cannot be hoisted to
    module scope. Write order is the in-block error priority."""
    integer_limits = {
        "limit": (1, 10_000 if (surface, operation) == ("memory_repair", "replay_backup") else MAX_RESULT_LIMIT),
        "offset": (0, MAX_REVISION if (surface, operation) == ("memory_repair", "replay_backup") else MAX_OFFSET),
        "batch_size": (1, 500),
        "older_than_days": (0, 365_000),
        "expected_version": (1, MAX_REVISION),
        "expected_revision": (1, MAX_REVISION),
    }
    for key, (minimum, maximum) in integer_limits.items():
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        parsed = _controlled_integer(value)
        if parsed is None:
            return _error(key, f"must be an integer between {minimum} and {maximum}")
        if parsed < minimum or parsed > maximum:
            return _error(key, f"must be an integer between {minimum} and {maximum}")
        payload[key] = parsed
    return None


def _v_timeout(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    if "timeout" not in payload:
        return None
    value = payload["timeout"]
    try:
        parsed_timeout = float(value)
    except (TypeError, ValueError):
        return _error("timeout", f"must be a finite number between 0 and {SEMANTIC_CONTROL_MAX_TIMEOUT:g}")
    if isinstance(value, bool) or not math.isfinite(parsed_timeout) or not 0.0 <= parsed_timeout <= SEMANTIC_CONTROL_MAX_TIMEOUT:
        return _error("timeout", f"must be a finite number between 0 and {SEMANTIC_CONTROL_MAX_TIMEOUT:g}")
    payload["timeout"] = parsed_timeout
    return None


def _v_embedding(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    embedding = payload.get("query_embedding")
    if embedding is None:
        embedding = payload.get("embedding")
    if embedding is None:
        return None
    # Shape-only since 0.15.0: there is no configured vec.dim to compare
    # against at the API boundary (the active dim is a per-library fact,
    # discovered from the model). A wrong-length embedding is rejected
    # where it is used — the vec index — rather than here, so a fresh
    # library plus a non-default-dim model is never wrongly refused.
    if not isinstance(embedding, list) or not embedding:
        return _error("embedding", "must be a non-empty list of numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in embedding):
        return _error("embedding", "all values must be finite numbers")
    return None


def _v_confidence(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    if "confidence" not in payload:
        return None
    value = payload["confidence"]
    try:
        parsed_confidence = float(value)
    except (TypeError, ValueError):
        return _error("confidence", "must be a finite number between 0 and 1")
    if isinstance(value, bool) or not math.isfinite(parsed_confidence) or not 0.0 <= parsed_confidence <= 1.0:
        return _error("confidence", "must be a finite number between 0 and 1")
    payload["confidence"] = parsed_confidence
    return None


def _v_time_fields(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    for key in ("event_time", "ingest_time", "after_time", "before_time"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 128 or parse_iso8601_utc(value) is None:
                return _error(key, "must be a valid ISO 8601 timestamp of at most 128 characters")
    return None


def _v_enums(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    # Write order is the in-block error priority; see _BOUNDED_STRINGS.
    enums = {
        "source_type": {item.value for item in SourceType},
        "protection_level": {item.value for item in ProtectionLevel},
    }
    for key, choices in enums.items():
        value = payload.get(key)
        if value is not None and str(value) not in choices:
            return _error(key, "invalid enum value", allowed=sorted(choices))
    return None


def _v_remember_status(
    surface: str, operation: str, payload: dict[str, Any], result: ValidationResult,
) -> dict[str, Any] | None:
    if (surface, operation) != ("memory", "remember") or payload.get("status") is None:
        return None
    # superseded/conflicted/deleted are lifecycle outcomes owned by
    # govern/repair operations; they are never caller-supplied write inputs.
    status_value = payload.get("status")
    if status_value not in {MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value}:
        return _error(
            "status",
            "must be 'active' (default) or 'pending'; superseded/conflicted/deleted are lifecycle outcomes, not write inputs",
        )
    return None


# Order is product behaviour twice over: it decides which error a caller sees
# when several fields are invalid, and -- because _v_unknown_fields pops keys
# that are not in the operation's allow-list -- whether later validators run on
# a given field at all. Reordering this tuple changes observable output.
_VALIDATORS: tuple[_Validator, ...] = (
    _v_unknown_fields,
    _v_batch_find_queries,
    _v_remember_required,
    _v_batch_read,
    _v_id_fields,
    _v_content_bytes,
    _v_bounded_strings,
    _v_tag_lists,
    _v_metadata,
    _v_structured_limits,
    _v_candidate_key,
    _v_slot_key,
    _v_memory_ids,
    _v_workspaces_list,
    _v_patches,
    _v_integer_limits,
    _v_timeout,
    _v_embedding,
    _v_confidence,
    _v_time_fields,
    _v_enums,
    _v_remember_status,
)


def validate_product_payload(surface: str, operation: str, payload: dict[str, Any]) -> ValidationResult:
    result = ValidationResult()
    for validator in _VALIDATORS:
        error = validator(surface, operation, payload, result)
        if error is not None:
            result.error = error
            return result
    return result
