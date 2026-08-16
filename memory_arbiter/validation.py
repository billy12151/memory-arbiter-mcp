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
MAX_SUBJECT_CHARS = 2_000
MAX_QUERY_CHARS = 32_000
MAX_TAGS = 100
MAX_TAG_CHARS = 256
MAX_METADATA_BYTES = 256 * 1024
MAX_TEXT_FIELD_CHARS = 2_000
MAX_BATCH_IDS = 1_000
MAX_RESULT_LIMIT = 100
MAX_OFFSET = 10_000

_SENSITIVE_FIELDS = {
    "authorized", "workspace", "memory_id", "conflict_id", "notice_id",
    "source_type", "protection_level", "status", "content", "new_content",
    "event_time", "ingest_time", "confidence", "dry_run",
}

_COMMON = {"topic", "workspace"}
_ALLOWED: dict[tuple[str, str], set[str]] = {
    ("memory", "help"): {"topic", "action"},
    ("memory", "status"): set(),
    ("memory", "remember"): {
        "content", "agent_id", "workspace", "tags", "source_type", "source_ref",
        "event_time", "ingest_time", "confidence", "protection_level", "status",
        "subject", "metadata", "client",
    },
    ("memory", "find"): {
        "query", "workspace", "tags", "limit", "offset", "debug_ranking",
        "query_embedding", "tags_filter", "after_time", "before_time",
        "source_type", "include_linked_open_items", "include_conflict_signal",
    },
    ("memory", "read"): {"id", "memory_id", "sections", "section_ids", "workspace"},
    ("memory", "update"): {
        "id", "memory_id", "new_content", "old_text", "new_text", "new_subject",
        "new_tags", "reason", "authorized", "tags_only", "add_tags", "remove_tags",
        "expected_version", "expected_content_hash", "content_hash", "workspace",
    },
    ("memory", "judge"): {
        "id", "conflict_id", "expected_left_version", "expected_right_version",
        "expected_left_claim_revision", "expected_right_claim_revision", "verdict",
        "recommended_use", "suggested_winner", "confidence_hint", "reason",
        "affects_current_output", "usage_context", "judge_ref", "resolution_kind",
        "conflict_scope", "workspace",
    },
    ("memory_review", "overview"): _COMMON,
    ("memory_review", "doctor"): {"deep", "workspace"},
    ("memory_review", "audit"): {"workspace"},
    ("memory_review", "conflicts"): {"status", "limit", "source", "workspace"},
    ("memory_review", "conflict_detail"): {"id", "conflict_id", "limit", "workspace"},
    ("memory_review", "judgments"): {"id", "conflict_id", "workspace"},
    ("memory_review", "history"): {"id", "memory_id", "workspace"},
    ("memory_review", "expired"): {
        "query", "workspace", "tags", "limit", "offset", "debug_ranking",
        "query_embedding", "tags_filter", "after_time", "before_time",
        "source_type", "include_conflict_signal",
    },
    ("memory_review", "entities"): {"limit", "include_unassigned", "workspace"},
    ("memory_review", "help"): {"topic", "view"},
    ("memory_govern", "retire"): {"id", "memory_id", "reason", "superseded_by", "authorized", "workspace"},
    ("memory_govern", "resolve_conflict"): {"id", "conflict_id", "reason", "status", "authorized", "workspace"},
    ("memory_govern", "confirm"): {"id", "memory_id", "source_ref", "confidence", "authorized", "workspace"},
    ("memory_govern", "correct_judgment"): {
        "id", "conflict_id", "verdict", "recommended_use", "suggested_winner", "reason",
        "expected_judgment_id", "expected_left_version", "expected_right_version",
        "expected_left_claim_revision", "expected_right_claim_revision", "authorized",
        "judge_ref", "resolution_kind", "conflict_scope", "workspace",
    },
    ("memory_govern", "accept_workspace_alias"): {"alias", "canonical", "relation", "reason", "source", "authorized"},
    ("memory_govern", "reject_workspace_alias"): {"alias", "canonical", "reason", "source", "authorized"},
    ("memory_govern", "rename_workspace_canonical"): {"old", "new", "reason", "authorized"},
    ("memory_govern", "migrate_workspace"): {"from", "to", "reason", "authorized"},
    ("memory_govern", "confirm_pending_workspace"): {"id", "memory_id", "canonical", "reason", "authorized", "workspace"},
    ("memory_govern", "help"): {"topic", "action"},
    ("memory_repair", "split"): {
        "id", "memory_id", "split_decision", "decision_content_hash",
        "decision_memory_version", "decision_split_status", "decision_split_revision",
        "sections", "workspace",
    },
    ("memory_repair", "rebuild_claims"): {"memory_ids", "dry_run", "batch_size", "workspace"},
    ("memory_repair", "rebuild_embeddings"): {"memory_ids", "dry_run", "batch_size", "workspace"},
    ("memory_repair", "cleanup_history"): {"id", "memory_id", "older_than_days", "authorized", "workspace"},
    ("memory_repair", "cleanup_vectors"): {"dry_run", "authorized", "workspace"},
    ("memory_repair", "resync_vectors"): {"dry_run", "authorized", "workspace"},
    ("memory_repair", "set_entity"): {"id", "memory_id", "entity", "scope", "clear", "authorized", "workspace"},
    ("memory_repair", "activate_pending"): {"id", "memory_id", "authorized", "workspace"},
    ("memory_repair", "semantic_control"): {"action", "timeout"},
    ("memory_repair", "notice"): {"action", "limit", "id", "notice_id", "reason", "authorized"},
    ("memory_repair", "replay_backup"): {"dry_run", "authorized", "limit", "offset"},
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


def validate_product_payload(surface: str, operation: str, payload: dict[str, Any], *, vec_dim: int) -> ValidationResult:
    result = ValidationResult()
    allowed = _ALLOWED.get((surface, operation))
    if allowed is not None:
        unknown_keys: list[str] = []
        for key in payload:
            if not isinstance(key, str):
                result.error = _error(str(key), "field names must be strings")
                return result
            if key in allowed:
                continue
            suggestion = difflib.get_close_matches(key, allowed, n=1, cutoff=0.78)
            if suggestion and suggestion[0] in _SENSITIVE_FIELDS:
                result.error = _error(key, "unknown field resembles a protected field", did_you_mean=suggestion[0])
                return result
            result.warnings.append(f"unknown field ignored: {key}")
            unknown_keys.append(key)
        for key in unknown_keys:
            payload.pop(key, None)

    if (surface, operation) == ("memory", "remember"):
        for key in ("content", "subject"):
            if key not in payload or not isinstance(payload.get(key), str) or not str(payload[key]).strip():
                result.error = _error(key, "is required and must be a non-empty string")
                return result

    for key in ("id", "memory_id", "conflict_id", "notice_id", "superseded_by"):
        if key not in payload:
            continue
        if key in {"id", "conflict_id"} and (
            (surface, operation) == ("memory", "judge")
            or (surface, operation) == ("memory_govern", "correct_judgment")
        ):
            # Their dispatchers intentionally report all missing receipt fields
            # before coercing the primary id.
            continue
        value = payload[key]
        try:
            parsed_id = int(value)
        except (TypeError, ValueError):
            field_name = "memory_id" if key == "id" and operation in {"read", "update", "history", "split", "set_entity", "activate_pending", "cleanup_history", "confirm_pending_workspace"} else key
            result.error = _error(field_name, "must be a positive integer")
            return result
        if isinstance(value, bool) or parsed_id <= 0:
            field_name = "memory_id" if key == "id" and operation in {"read", "update", "history", "split", "set_entity", "activate_pending", "cleanup_history", "confirm_pending_workspace"} else key
            result.error = _error(field_name, "must be a positive integer")
            return result

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
    for key, maximum in (("subject", MAX_SUBJECT_CHARS), ("new_subject", MAX_SUBJECT_CHARS), ("query", MAX_QUERY_CHARS)):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > maximum):
            result.error = _error(key, f"must be a string of at most {maximum} characters")
            return result
    for key in ("workspace", "source_ref", "agent_id", "client"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > MAX_TEXT_FIELD_CHARS):
            result.error = _error(key, f"must be a string of at most {MAX_TEXT_FIELD_CHARS} characters")
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

    for key in ("memory_ids", "section_ids"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) > MAX_BATCH_IDS:
            result.error = _error(key, f"must be a list with at most {MAX_BATCH_IDS} items")
            return result
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
            result.error = _error(key, "must contain positive integer ids")
            return result

    integer_limits = {
        "limit": (1, 10_000 if (surface, operation) == ("memory_repair", "replay_backup") else MAX_RESULT_LIMIT),
        "offset": (0, MAX_OFFSET),
        "batch_size": (1, 500),
        "older_than_days": (0, 365_000),
        "expected_version": (1, 2_147_483_647),
    }
    for key, (minimum, maximum) in integer_limits.items():
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            result.error = _error(key, f"must be an integer between {minimum} and {maximum}")
            return result
        if isinstance(value, bool) or parsed < minimum or parsed > maximum:
            result.error = _error(key, f"must be an integer between {minimum} and {maximum}")
            return result
    embedding = payload.get("query_embedding")
    if embedding is None:
        embedding = payload.get("embedding")
    if embedding is not None:
        if not isinstance(embedding, list) or len(embedding) != int(vec_dim):
            result.error = _error("embedding", f"must contain exactly {int(vec_dim)} numbers")
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
        if not math.isfinite(parsed_confidence) or not 0.0 <= parsed_confidence <= 1.0:
            result.error = _error("confidence", "must be a finite number between 0 and 1")
            return result
    for key in ("event_time", "ingest_time", "after_time", "before_time"):
        value = payload.get(key)
        if value is not None and parse_iso8601_utc(value) is None:
            result.error = _error(key, "must be a valid ISO 8601 timestamp")
            return result
    enums = {
        "source_type": {item.value for item in SourceType},
        "protection_level": {item.value for item in ProtectionLevel},
    }
    if (surface, operation) == ("memory", "remember"):
        enums["status"] = {item.value for item in MemoryStatus}
    for key, choices in enums.items():
        value = payload.get(key)
        if value is not None and str(value) not in choices:
            result.error = _error(key, "invalid enum value", allowed=sorted(choices))
            return result
    return result
