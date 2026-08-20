"""Workspace read ACL helpers.

Shared by the product pipeline, DB-facing read helpers, and the console API.
This module deliberately adds ACL-specific helpers instead of changing the
semantics of general-purpose DB reads such as ``get_memory``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


WORKSPACE_EXPR = "COALESCE(NULLIF(workspace_canonical, ''), workspace)"


@dataclass(frozen=True)
class CallerWorkspace:
    isolation: str
    workspace: Optional[str]
    canonical: Optional[str]
    source: str
    warnings: tuple[str, ...] = ()

    @property
    def strict(self) -> bool:
        return self.isolation == "strict"

    def response_fields(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "caller_workspace": self.workspace,
            "workspace_source": self.source,
        }
        if self.canonical is not None:
            data["caller_workspace_canonical"] = self.canonical
        return data


def raw_workspace(record: Optional[dict[str, Any]]) -> str:
    if not record:
        return ""
    return str((record.get("workspace_canonical") or record.get("workspace") or "")).strip()


def visible_memory(record: Optional[dict[str, Any]], canonical: Optional[str]) -> bool:
    if canonical is None:
        return record is not None
    return record is not None and raw_workspace(record) == canonical


def forbidden_payload(kind: str, *, workspace: CallerWorkspace, reason: str = "workspace_acl") -> dict[str, Any]:
    payload = {
        "error": "forbidden_strict_workspace",
        "reason": reason,
        "resource": kind,
    }
    payload.update(workspace.response_fields())
    return payload


def memory_public_stub(memory_id: Any, *, visible: bool, memory: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if visible:
        return {
            "visible": True,
            "memory_id": int(memory_id) if memory_id is not None else None,
            "memory": memory,
            "redaction_reason": None,
        }
    return {
        "visible": False,
        "memory_id": None,
        "memory": None,
        "redaction_reason": "workspace_acl",
    }


def redacted_conflict_shell(conflict: dict[str, Any], left_visible: bool, right_visible: bool) -> dict[str, Any]:
    """Return a conflict row safe for a caller that may see only one side.

    Raw conflict fields such as reason/structured_details/conflict_point may embed
    hidden-side evidence, so strict partial visibility exposes only neutral shell
    fields and a public summary.
    """
    winner = conflict.get("suggested_winner") or conflict.get("winner_id") or conflict.get("judgment_suggested_winner")
    winner_side: Optional[str] = None
    try:
        winner_int = int(winner) if winner is not None else None
        left_id = conflict.get("left_id")
        right_id = conflict.get("right_id")
        left_id_int = int(left_id) if left_id is not None else None
        right_id_int = int(right_id) if right_id is not None else None
        if winner_int is not None and left_visible and winner_int == left_id_int:
            winner_side = "left"
        elif winner_int is not None and right_visible and winner_int == right_id_int:
            winner_side = "right"
        elif winner_int is not None:
            winner_side = "unknown"
    except (TypeError, ValueError):
        winner_side = None
    redacted_fields = [
        name for name in (
            "subject", "reason", "conflict_point", "structured_details",
            "judgment_reason", "confidence_hint", "suggested_winner", "winner_id",
            "judgment_suggested_winner",
        )
        if conflict.get(name) is not None
    ]
    return {
        "id": conflict.get("id"),
        "status": conflict.get("status"),
        "created_at": conflict.get("created_at"),
        "resolved_at": conflict.get("resolved_at"),
        "conflict_type": conflict.get("conflict_type"),
        "source": conflict.get("source"),
        "detection_channel": conflict.get("detection_channel"),
        "judgment_status": conflict.get("judgment_status"),
        "active_judgment_id": conflict.get("active_judgment_id"),
        "resolution_kind": conflict.get("resolution_kind") or conflict.get("judgment_resolution_kind"),
        "conflict_scope": conflict.get("conflict_scope") or conflict.get("judgment_conflict_scope"),
        "public_conflict_summary": "Conflict is visible because at least one side is in the caller workspace; hidden-side evidence is redacted.",
        "redacted_fields": redacted_fields,
        "winner_side": winner_side,
    }


def redact_judgment(row: dict[str, Any], *, visible: bool) -> dict[str, Any]:
    if visible:
        return {**row, "visible": True, "redacted_fields": []}
    return {
        "visible": False,
        "id": row.get("id"),
        "conflict_id": row.get("conflict_id"),
        "verdict": None,
        "recommended_use": None,
        "suggested_winner": None,
        "reason": None,
        "redaction_reason": "hidden_side_evidence",
        "redacted_fields": [
            "verdict", "recommended_use", "suggested_winner", "confidence_hint",
            "reason", "judge_ref", "resolution_kind", "conflict_scope",
        ],
    }
