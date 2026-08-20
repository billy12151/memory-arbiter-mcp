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


def redacted_conflict_shell(conflict: dict[str, Any]) -> dict[str, Any]:
    """Return a conflict-group shell safe when any member is hidden.

    Group fields are correlated: slot, values, evidence, decisions, and apply
    results can all reveal a hidden member.  Partial strict visibility therefore
    exposes only lifecycle metadata and explicit redaction markers.
    """
    sensitive_fields = (
        "slot_key", "candidate_key", "conflict_point", "member_versions",
        "member_fingerprint", "value_groups", "detection_reason", "chosen_value",
        "resolution_memory_id", "resolution_memory_version", "decided_by",
        "decided_ref", "decision_reason", "decided_at", "apply_summary",
    )
    return {
        "id": conflict.get("id"),
        "revision": conflict.get("revision"),
        "status": conflict.get("status"),
        "source": conflict.get("source"),
        "detector_version": conflict.get("detector_version"),
        "prompt_version": conflict.get("prompt_version"),
        "overflow": bool(conflict.get("overflow")),
        "created_at": conflict.get("created_at"),
        "refreshed_at": conflict.get("refreshed_at"),
        "resolved_at": conflict.get("resolved_at"),
        "public_conflict_summary": (
            "Conflict group is linked to the caller workspace; fields correlated "
            "with hidden members are redacted."
        ),
        "redacted_fields": [name for name in sensitive_fields if conflict.get(name) is not None],
    }
