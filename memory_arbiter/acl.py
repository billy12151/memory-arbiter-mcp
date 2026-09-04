"""Workspace read ACL helpers.

Shared by the product pipeline, DB-facing read helpers, and the console API.
This module deliberately adds ACL-specific helpers instead of changing the
semantics of general-purpose DB reads such as ``get_memory``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Collection


WORKSPACE_EXPR = "COALESCE(NULLIF(workspace_canonical, ''), workspace)"

#: A workspace scope is either one canonical name or the admitted set.
WorkspaceScope = str | Sequence[str] | None


def scope_names(scope: WorkspaceScope) -> list[str]:
    """Normalize a scope (None / one name / a set) to a deduped name list."""
    if scope is None:
        return []
    values: Sequence[Any] = [scope] if isinstance(scope, str) else scope
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        names.append(text)
    return names


def workspace_scope_sql(expr: str, scope: WorkspaceScope) -> tuple[str, list[str]]:
    """Build a workspace-scoping SQL fragment over an admitted canonical set.

    strict scoping is an admitted-set membership, not a single equality.
    A single-element scope produces ``EXPR = ?`` — byte-identical to the previous
    equality filter — so with vector admission off (the admitted set collapses
    to just the caller's own canonical) every call site keeps its old plan.
    Multiple canonicals produce ``EXPR IN (?, ?, ...)``. Empty scope →
    ``("", [])`` so the caller adds no clause.
    """
    names = scope_names(scope)
    if not names:
        return "", []
    if len(names) == 1:
        return f"{expr} = ?", [names[0]]
    placeholders = ",".join("?" for _ in names)
    return f"{expr} IN ({placeholders})", names


def workspace_exclusion_sql(
    exclude: "Collection[str] | None",
    expr_m: str = "COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace)",
    expr_plain: str = "COALESCE(NULLIF(workspace_canonical, ''), workspace)",
) -> tuple[str, str, list[str]]:
    """Recall-blacklist NOT IN conditions (v0.15.5). Returns
    ``(cond_m, cond_plain, params)`` without a leading AND so callers can
    compose it either as ``f" AND {cond}"`` or a ``' AND '.join`` clause list.

    Matches *both* the canonical-with-raw-fallback expression and the raw
    workspace column: a blacklisted name must also catch rows whose canonical
    drifted (e.g. the twin-dryrun normalization era left workspace=mema-twin
    rows canonicalized elsewhere). Empty/absent input → no-ops.

    ``params`` carries ONE cond's worth (``names * 2``): each embedding SQL
    uses exactly one of the two conds, so a call site extends its parameter
    list once per embedded cond — never a shared 4x list, which would
    over-bind and get silently swallowed by per-channel try/except.
    """
    names = sorted({str(n).strip() for n in (exclude or []) if str(n or "").strip()})
    if not names:
        return "", "", []
    placeholders = ",".join("?" for _ in names)
    cond_m = (
        f"({expr_m} NOT IN ({placeholders}) AND m.workspace NOT IN ({placeholders}))"
    )
    cond_plain = (
        f"({expr_plain} NOT IN ({placeholders}) AND workspace NOT IN ({placeholders}))"
    )
    return cond_m, cond_plain, names * 2


@dataclass(frozen=True)
class CallerWorkspace:
    isolation: str
    workspace: str | None
    canonical: str | None
    source: str
    warnings: tuple[str, ...] = ()
    #: canonicals a strict caller may read/act on — its own plus any
    #: within the recall cutoff (vector admission). Always contains ``canonical``
    #: when set. Empty for non-strict callers (they never hard-scope by it).
    admitted: tuple[str, ...] = ()

    @property
    def strict(self) -> bool:
        return self.isolation == "strict"

    def scope_canonicals(self) -> tuple[str, ...]:
        """Admitted set for strict SQL/visibility scoping.

        Falls back to ``(canonical,)`` so a caller constructed without an
        explicit admitted set (legacy call sites, tests) behaves exactly like
        the single-canonical equality filter.
        """
        if self.admitted:
            return self.admitted
        return (self.canonical,) if self.canonical else ()

    def response_fields(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "caller_workspace": self.workspace,
            "workspace_source": self.source,
        }
        if self.canonical is not None:
            data["caller_workspace_canonical"] = self.canonical
        return data


def raw_workspace(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return str((record.get("workspace_canonical") or record.get("workspace") or "")).strip()


def visible_memory(
    record: dict[str, Any] | None,
    canonical: str | None,
    admitted: WorkspaceScope = None,
) -> bool:
    """Strict read-ACL predicate.

    when an ``admitted`` scope is supplied the record is visible if its
    workspace is any admitted canonical (vector-admission neighbourhood);
    otherwise the single-canonical-canonical check applies. ``canonical=None``
    means no ACL (non-strict).
    """
    if canonical is None:
        return record is not None
    if record is None:
        return False
    allowed = scope_names(admitted)
    if allowed:
        return raw_workspace(record) in set(allowed)
    return raw_workspace(record) == canonical


def forbidden_payload(kind: str, *, workspace: CallerWorkspace, reason: str = "workspace_acl") -> dict[str, Any]:
    payload = {
        "error": "forbidden_strict_workspace",
        "reason": reason,
        "resource": kind,
    }
    payload.update(workspace.response_fields())
    return payload


def memory_public_stub(memory_id: Any, *, visible: bool, memory: dict[str, Any] | None = None) -> dict[str, Any]:
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
