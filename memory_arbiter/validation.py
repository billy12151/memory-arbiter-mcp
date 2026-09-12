"""Narrow validation boundary for the four product MCP surfaces."""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from typing import Any

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
    ("memory_govern", "move_memories_workspace"): {"memory_ids", "new_workspace", "reason", "authorized", "workspace"},
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


def validate_product_payload(surface: str, operation: str, payload: dict[str, Any]) -> ValidationResult:
    result = ValidationResult()
    allowed = PRODUCT_FIELD_REGISTRY.get((surface, operation))
    if allowed is not None:
        unknown_keys: list[str] = []
        for key in payload:
            if not isinstance(key, str):
                result.error = _error(str(key), "field names must be strings")
                return result
            if key in allowed:
                continue
            if (
                key == "include_content"
                and (surface, operation) in (("memory", "find"), ("memory", "batch_find"))
            ):
                # v0.15.10 breaking: silently ignoring the removed boolean would
                # look like a working call returning previews — fail loudly with
                # the migration pointer instead.
                result.error = _error(
                    key,
                    'include_content was removed in v0.15.10; use '
                    'content_mode="full" for full text or content_mode="hits" '
                    "for vector-hit spans",
                    did_you_mean="content_mode",
                )
                return result
            update_aliases = {
                "content": "new_content",
                "subject": "new_subject",
                "tags": "new_tags",
            }
            if (surface, operation) == ("memory", "update") and key in update_aliases:
                result.error = _error(
                    key,
                    "remember field is not valid for update",
                    did_you_mean=update_aliases[key],
                )
                return result
            suggestion = difflib.get_close_matches(key, allowed, n=1, cutoff=0.78)
            if suggestion and suggestion[0] in _SENSITIVE_FIELDS:
                result.error = _error(key, "unknown field resembles a protected field", did_you_mean=suggestion[0])
                return result
            result.warnings.append(f"unknown field ignored: {key}")
            unknown_keys.append(key)
        for key in unknown_keys:
            payload.pop(key, None)

    if (surface, operation) == ("memory", "batch_find"):
        queries = payload.get("queries")
        if not isinstance(queries, list) or not queries:
            result.error = _error("queries", "must be a non-empty list of {id?, query} objects")
            return result
        if len(queries) > MAX_BATCH_FIND_QUERIES:
            result.error = _error("queries", f"must contain at most {MAX_BATCH_FIND_QUERIES} items")
            return result
        try:
            queries_bytes = _json_size(queries)
        except (TypeError, ValueError, RecursionError):
            result.error = _error("queries", "must contain only JSON-serializable values")
            return result
        if queries_bytes > BATCH_FIND_TOTAL_BYTES:
            result.error = {
                "error": "resource_limit_exceeded", "field": "queries",
                "actual_bytes": queries_bytes, "max_bytes": BATCH_FIND_TOTAL_BYTES,
            }
            return result
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        for item in queries:
            if not isinstance(item, dict):
                result.error = _error("queries", "every item must be a JSON object with a non-empty query")
                return result
            unknown = set(item) - {"id", "query"}
            if unknown:
                result.error = _error("queries", f"unknown item field(s): {', '.join(sorted(unknown))}")
                return result
            q = item.get("query")
            if not isinstance(q, str) or not q.strip():
                result.error = _error("queries", "every item requires a non-empty string query")
                return result
            if len(q) > MAX_QUERY_CHARS:
                result.error = _error("query", f"must be a string of at most {MAX_QUERY_CHARS} characters")
                return result
            qid = item.get("id")
            defaulted = qid is None
            if defaulted:
                qid = q
            if not isinstance(qid, str) or not qid or len(qid) > 64:
                result.error = _error("queries", "id must be a non-empty string of at most 64 characters")
                return result
            if qid in seen_ids:
                result.error = _error("queries", f"duplicate query id: {qid}")
                return result
            if q in seen_queries:
                result.error = _error("queries", f"duplicate query: {q}")
                return result
            seen_ids.add(qid)
            seen_queries.add(q)
            # Normalize defaulted ids in place so dispatch sees explicit ids.
            item["id"] = qid
        limit_per_query = payload.get("limit_per_query")
        if limit_per_query is not None:
            parsed_limit = _controlled_integer(limit_per_query)
            if parsed_limit is None or not 1 <= parsed_limit <= BATCH_FIND_MAX_LIMIT_PER_QUERY:
                result.error = _error(
                    "limit_per_query",
                    f"must be an integer between 1 and {BATCH_FIND_MAX_LIMIT_PER_QUERY}",
                )
                return result
            payload["limit_per_query"] = parsed_limit

    if (surface, operation) == ("memory", "remember"):
        for key in ("content", "subject"):
            if key not in payload or not isinstance(payload.get(key), str) or not str(payload[key]).strip():
                result.error = _error(key, "is required and must be a non-empty string")
                return result

    if (surface, operation) == ("memory", "batch_read"):
        memory_ids = payload.get("memory_ids")
        if not isinstance(memory_ids, list) or not memory_ids:
            result.error = _error("memory_ids", "must be a non-empty list of positive integer ids")
            return result
        content_mode = payload.get("content_mode") or "preview"
        if content_mode not in {"preview", "hits", "full"}:
            result.error = _error(
                "content_mode",
                'must be one of "preview" | "hits" | "full" (default "preview")',
            )
            return result
        cap = {"preview": BATCH_READ_MAX_PREVIEW, "hits": BATCH_READ_MAX_HITS, "full": BATCH_READ_MAX_FULL}[content_mode]
        if len(memory_ids) > cap:
            result.error = _error(
                "memory_ids",
                f'content_mode="{content_mode}" accepts at most {cap} ids per call',
                cap=cap,
            )
            return result
        spans = payload.get("spans")
        if spans is not None:
            if not isinstance(spans, dict):
                result.error = _error("spans", "must be an object keyed by memory_id: {start, end}")
                return result
            for key, span in spans.items():
                if _controlled_integer(key) is None:
                    result.error = _error("spans", "keys must be integer memory ids")
                    return result
                if not isinstance(span, dict) or set(span) - {"start", "end"}:
                    result.error = _error("spans", "each span must be an object with optional integer start/end")
                    return result
                raw_start = span.get("start", 0)
                raw_end = span.get("end")
                if isinstance(raw_start, bool) or not isinstance(raw_start, int) or raw_start < 0:
                    result.error = _error("spans", "span start must be a non-negative integer")
                    return result
                if raw_end is not None and (isinstance(raw_end, bool) or not isinstance(raw_end, int) or raw_end <= raw_start):
                    result.error = _error("spans", "span end must be an integer greater than start")
                    return result
        payload["content_mode"] = content_mode

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
            result.error = _error(field_name, "must be a positive integer")
            return result
        if parsed_id <= 0 or parsed_id > MAX_SQLITE_INTEGER:
            field_name = "memory_id" if key == "id" and operation in {"read", "update", "history", "set_entity", "activate_pending", "cleanup_history", "confirm_pending_workspace"} else key
            result.error = _error(field_name, "must be a positive integer")
            return result
        # Preserve controlled numeric-string compatibility, but make the value
        # consumed by dispatch exactly the value that passed validation.
        payload[key] = parsed_id

    for key in ("content", "new_content"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str):
                result.error = _error(key, "must be a string")
                return result
            actual = len(value.encode("utf-8"))
            if actual > MAX_CONTENT_BYTES:
                result.error = {"error": "resource_limit_exceeded", "field": key, "actual_bytes": actual, "max_bytes": MAX_CONTENT_BYTES}
                return result
    bounded_strings = {
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
    for key, maximum in bounded_strings.items():
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > maximum):
            result.error = _error(key, f"must be a string of at most {maximum} characters")
            return result

    for key in ("tags", "tags_filter", "new_tags", "add_tags", "remove_tags"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > MAX_TAGS or any(not isinstance(tag, str) or len(tag) > MAX_TAG_CHARS for tag in value):
            result.error = _error(key, f"must be a list of at most {MAX_TAGS} strings, each at most {MAX_TAG_CHARS} characters")
            return result
    metadata = payload.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            result.error = _error("metadata", "must be a JSON object")
            return result
        try:
            metadata_size = _json_size(metadata)
        except (TypeError, ValueError, RecursionError):
            result.error = _error("metadata", "must contain only JSON-serializable values")
            return result
        if metadata_size > MAX_METADATA_BYTES:
            result.error = {"error": "resource_limit_exceeded", "field": "metadata", "max_bytes": MAX_METADATA_BYTES}
            return result

    structured_limits = {
        "members": (MAX_CONFLICT_MEMBERS, MAX_CONFLICT_MEMBERS_BYTES),
        "value_groups": (MAX_CONFLICT_MEMBERS, MAX_CONFLICT_VALUE_GROUPS_BYTES),
        "apply_plan": (MAX_APPLY_PLAN_ITEMS, MAX_APPLY_PLAN_BYTES),
    }
    for key, (max_items, max_bytes) in structured_limits.items():
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > max_items:
            result.error = _error(key, f"must be a list with at most {max_items} items")
            return result
        if any(not isinstance(item, dict) for item in value):
            result.error = _error(key, "must contain only JSON objects")
            return result
        try:
            actual_bytes = _json_size(value)
        except (TypeError, ValueError, RecursionError):
            result.error = _error(key, "must contain only JSON-serializable values")
            return result
        if actual_bytes > max_bytes:
            result.error = {
                "error": "resource_limit_exceeded", "field": key,
                "actual_bytes": actual_bytes, "max_bytes": max_bytes,
            }
            return result
    candidate_key = payload.get("candidate_key")
    if candidate_key is not None:
        if not isinstance(candidate_key, dict):
            result.error = _error("candidate_key", "must be a JSON object")
            return result
        try:
            candidate_bytes = _json_size(candidate_key)
        except (TypeError, ValueError, RecursionError):
            result.error = _error("candidate_key", "must contain only JSON-serializable values")
            return result
        if candidate_bytes > MAX_CONFLICT_CANDIDATE_KEY_BYTES:
            result.error = {
                "error": "resource_limit_exceeded", "field": "candidate_key",
                "actual_bytes": candidate_bytes,
                "max_bytes": MAX_CONFLICT_CANDIDATE_KEY_BYTES,
            }
            return result
    slot_key = payload.get("slot_key")
    if slot_key is not None:
        if not isinstance(slot_key, dict):
            result.error = _error("slot_key", "must be a JSON object or null")
            return result
        try:
            slot_bytes = _json_size(slot_key)
        except (TypeError, ValueError, RecursionError):
            result.error = _error("slot_key", "must contain only JSON-serializable values")
            return result
        if slot_bytes > MAX_CONFLICT_SLOT_KEY_BYTES:
            result.error = {
                "error": "resource_limit_exceeded", "field": "slot_key",
                "actual_bytes": slot_bytes, "max_bytes": MAX_CONFLICT_SLOT_KEY_BYTES,
            }
            return result

    for key in ("memory_ids",):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > MAX_BATCH_IDS:
            result.error = _error(key, f"must be a list with at most {MAX_BATCH_IDS} items")
            return result
        parsed_items: list[int] = []
        for item in value:
            parsed_item = _controlled_integer(item)
            if parsed_item is None or parsed_item <= 0 or parsed_item > MAX_SQLITE_INTEGER:
                result.error = _error(key, "must contain positive integer ids")
                return result
            parsed_items.append(parsed_item)
        payload[key] = parsed_items

    # confirm_workspaces list: bounded like tags (count) and text fields (per-
    # item length) so one authorized call cannot persist an unbounded sidecar.
    workspaces_value = payload.get("workspaces")
    if workspaces_value is not None:
        if (
            not isinstance(workspaces_value, list)
            or len(workspaces_value) > 100
            or any(not isinstance(item, str) or len(item) > MAX_TEXT_FIELD_CHARS for item in workspaces_value)
        ):
            result.error = _error(
                "workspaces",
                f"must be a list of at most 100 strings, each at most {MAX_TEXT_FIELD_CHARS} characters",
            )
            return result

    # update patches (v0.15.12): [{old_text, new_text}, ...] — sequential
    # partial replacements applied atomically in one write transaction.
    # Guard here at the boundary (shape/count/size) so a malformed batch
    # never reaches the edit loop; the db layer re-checks defensively.
    patches_value = payload.get("patches")
    if patches_value is not None:
        if not isinstance(patches_value, list) or not 1 <= len(patches_value) <= MAX_UPDATE_PATCHES:
            result.error = _error("patches", f"must be a list of 1..{MAX_UPDATE_PATCHES} patch objects")
            return result
        if any(not isinstance(item, dict) for item in patches_value):
            result.error = _error("patches", "must contain only JSON objects")
            return result
        normalized_patches: list[dict[str, Any]] = []
        for item in patches_value:
            if set(item) != {"old_text", "new_text"}:
                result.error = _error(
                    "patches",
                    "each patch must have exactly the keys old_text and new_text",
                )
                return result
            old_text_value, new_text_value = item["old_text"], item["new_text"]
            if not isinstance(old_text_value, str) or not old_text_value:
                result.error = _error("patches.old_text", "must be a non-empty string")
                return result
            if not isinstance(new_text_value, str):
                result.error = _error("patches.new_text", "must be a string (empty deletes the fragment)")
                return result
            if len(old_text_value) > MAX_REPLACEMENT_TEXT_CHARS or len(new_text_value) > MAX_REPLACEMENT_TEXT_CHARS:
                result.error = _error(
                    "patches", f"old_text/new_text must each be at most {MAX_REPLACEMENT_TEXT_CHARS} characters",
                )
                return result
            normalized_patches.append({"old_text": old_text_value, "new_text": new_text_value})
        try:
            patches_bytes = _json_size(normalized_patches)
        except (TypeError, ValueError, RecursionError):
            result.error = _error("patches", "must contain only JSON-serializable values")
            return result
        if patches_bytes > MAX_UPDATE_PATCHES_BYTES:
            result.error = {
                "error": "resource_limit_exceeded", "field": "patches",
                "actual_bytes": patches_bytes, "max_bytes": MAX_UPDATE_PATCHES_BYTES,
            }
            return result
        payload["patches"] = normalized_patches

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
            result.error = _error(key, f"must be an integer between {minimum} and {maximum}")
            return result
        if parsed < minimum or parsed > maximum:
            result.error = _error(key, f"must be an integer between {minimum} and {maximum}")
            return result
        payload[key] = parsed

    if "timeout" in payload:
        value = payload["timeout"]
        try:
            parsed_timeout = float(value)
        except (TypeError, ValueError):
            result.error = _error("timeout", f"must be a finite number between 0 and {SEMANTIC_CONTROL_MAX_TIMEOUT:g}")
            return result
        if isinstance(value, bool) or not math.isfinite(parsed_timeout) or not 0.0 <= parsed_timeout <= SEMANTIC_CONTROL_MAX_TIMEOUT:
            result.error = _error("timeout", f"must be a finite number between 0 and {SEMANTIC_CONTROL_MAX_TIMEOUT:g}")
            return result
        payload["timeout"] = parsed_timeout

    embedding = payload.get("query_embedding")
    if embedding is None:
        embedding = payload.get("embedding")
    if embedding is not None:
        # Shape-only since 0.15.0: there is no configured vec.dim to compare
        # against at the API boundary (the active dim is a per-library fact,
        # discovered from the model). A wrong-length embedding is rejected
        # where it is used — the vec index — rather than here, so a fresh
        # library plus a non-default-dim model is never wrongly refused.
        if not isinstance(embedding, list) or not embedding:
            result.error = _error("embedding", "must be a non-empty list of numbers")
            return result
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in embedding):
            result.error = _error("embedding", "all values must be finite numbers")
            return result

    if "confidence" in payload:
        value = payload["confidence"]
        try:
            parsed_confidence = float(value)
        except (TypeError, ValueError):
            result.error = _error("confidence", "must be a finite number between 0 and 1")
            return result
        if isinstance(value, bool) or not math.isfinite(parsed_confidence) or not 0.0 <= parsed_confidence <= 1.0:
            result.error = _error("confidence", "must be a finite number between 0 and 1")
            return result
        payload["confidence"] = parsed_confidence
    for key in ("event_time", "ingest_time", "after_time", "before_time"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 128 or parse_iso8601_utc(value) is None:
                result.error = _error(key, "must be a valid ISO 8601 timestamp of at most 128 characters")
                return result
    enums = {
        "source_type": {item.value for item in SourceType},
        "protection_level": {item.value for item in ProtectionLevel},
    }
    for key, choices in enums.items():
        value = payload.get(key)
        if value is not None and str(value) not in choices:
            result.error = _error(key, "invalid enum value", allowed=sorted(choices))
            return result
    if (surface, operation) == ("memory", "remember") and payload.get("status") is not None:
        # superseded/conflicted/deleted are lifecycle outcomes owned by
        # govern/repair operations; they are never caller-supplied write inputs.
        status_value = payload.get("status")
        if status_value not in {MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value}:
            result.error = _error(
                "status",
                "must be 'active' (default) or 'pending'; superseded/conflicted/deleted are lifecycle outcomes, not write inputs",
            )
            return result
    return result
