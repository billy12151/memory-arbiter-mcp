"""conflicts 规范化 helper mixin（从 conflicts.py 搬出，拆分批 ②c 纯移动）。"""
from __future__ import annotations

import sqlite3
from typing import Any, TYPE_CHECKING

from ..degrade import DegradeState
from ..models import ConflictMember, ConflictValueGroup

from ._conflicts_helpers import (
    _MAX_FIELD_CHARS,
    _MAX_SLOT_JSON,
    _canonical_json,
    _member_ref,
)
from ..acl import WorkspaceScope, scope_names
from ..semantic_conflict import normalize_value
from ..text import canon_entity, canon_scope

if TYPE_CHECKING:
    from .core import MemoryDB


class _ConflictsNormMixin:
    """conflicts 规范化 helper mixin（从 conflicts.py 搬出，拆分批 ②c 纯移动）。5 个 @staticmethod 原样保留 static 形态（mixin 内经 self 或类名均可达）。"""

    # 拆分批 ②c：声明式注解（mypy strict；形态对齐主类）
    if TYPE_CHECKING:
        _db: "MemoryDB"
        from typing import Any as _Any
        @property
        def _db_available(self) -> bool: ...
        @property
        def state(self) -> "DegradeState": ...
        def connection(self) -> "_Any": ...
        def write_transaction(self) -> "_Any": ...

    @staticmethod
    def _normalize_slot(slot_key: dict[str, Any] | None) -> dict[str, str] | None:
        if slot_key is None:
            return None
        if set(slot_key) != {"entity", "attribute", "scope"}:
            raise ValueError("slot_key must contain exactly entity, attribute, and scope")
        normalized = {key: str(slot_key[key]).strip() for key in ("entity", "attribute", "scope")}
        # Storage-side canonicalisation (B-C4): entity/scope are stored in
        # canon form so slot identity matches the comparison side's canonical
        # matching; attribute keeps its raw-stripped form (detector-owned).
        normalized["entity"] = canon_entity(normalized["entity"])
        normalized["scope"] = canon_scope(normalized["scope"])
        if not all(normalized.values()) or any(
            value.casefold() in {"unknown", "__unknown__"} for value in normalized.values()
        ):
            raise ValueError("slot_key entity, attribute, and scope must be reliable and non-empty")
        if any(len(value) > _MAX_FIELD_CHARS for value in normalized.values()):
            raise ValueError("slot_key field exceeds size bound")
        if len(_canonical_json(normalized)) > _MAX_SLOT_JSON:
            raise ValueError("slot_key exceeds size bound")
        return normalized

    @staticmethod
    def _normalize_members(members: list[dict[str, Any] | ConflictMember]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in members:
            member = raw.to_dict() if isinstance(raw, ConflictMember) else dict(raw)
            required = {
                "memory_id", "version", "attribute_raw", "value_raw",
                "normalized_attribute", "normalized_value", "evidence_quote",
                "evidence_span", "content_hash", "direction", "prompt_version",
                "detector_version",
            }
            missing = required - member.keys()
            if missing:
                raise ValueError(f"member missing required fields: {', '.join(sorted(missing))}")
            member["memory_id"] = int(member["memory_id"])
            member["version"] = int(member["version"])
            span = member["evidence_span"]
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                raise ValueError("evidence_span must be [start, end]")
            member["evidence_span"] = [int(span[0]), int(span[1])]
            unit = member.get("evidence_unit")
            member["evidence_unit"] = None if unit is None else int(unit)
            if member["memory_id"] <= 0 or member["version"] <= 0:
                raise ValueError("member memory_id and version must be positive")
            if member["evidence_span"][0] < 0 or member["evidence_span"][1] < member["evidence_span"][0]:
                raise ValueError("evidence_span must be ordered and non-negative")
            if member["evidence_unit"] is not None and member["evidence_unit"] < 0:
                raise ValueError("evidence_unit must be non-negative")
            if len(str(member["content_hash"])) != 64:
                raise ValueError("member content_hash must be 64 characters")
            # D1 (#970): the stored normalized_value is every later gate's
            # anchor (judge canonicalization, D1 group validation). A
            # paraphrased value_raw (an agent's retelling in neither the
            # mechanical nor the verbatim form) would silently poison all of
            # them, so reject it at intake. Dual-channel (0.17.1, owner
            # 选项 1): a member value is legitimate in exactly two shapes —
            # the mechanical normalization of value_raw, or the §3.4
            # snapshot-verbatim form (judged notices carry row text as both
            # fields, so escalate/promote can file them). The gate validates
            # the RELATION between the two fields, never the content.
            # Unenhanced scan candidates legitimately carry no values
            # (value_raw=None from the deterministic route); those are not
            # D1's concern.
            raw_value = member["value_raw"]
            if raw_value is not None and str(raw_value) != "":
                normalized_member_value = str(member["normalized_value"])
                if normalized_member_value != str(raw_value) and normalized_member_value != normalize_value(str(raw_value)):
                    raise ValueError(
                        "member normalized_value must equal normalize_value(value_raw) or value_raw verbatim"
                    )
            for key, value in member.items():
                if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
                    raise ValueError(f"member field {key} exceeds size bound")
            ref = _member_ref(member)
            if ref in seen:
                raise ValueError("members must contain each memory@version exactly once")
            normalized.append(member)
            seen.add(ref)
        normalized.sort(key=lambda item: (item["memory_id"], item["version"]))
        return normalized

    @staticmethod
    def _normalize_value_groups(
        groups: list[dict[str, Any] | ConflictValueGroup], members: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        member_values = {_member_ref(member): str(member["normalized_value"]) for member in members}
        normalized: list[dict[str, Any]] = []
        seen_values: set[str] = set()
        covered: set[str] = set()
        for raw in groups:
            group = raw.to_dict() if isinstance(raw, ConflictValueGroup) else dict(raw)
            if set(group) != {"normalized_value", "display_value", "members"}:
                raise ValueError("value group must contain normalized_value, display_value, members")
            value = str(group["normalized_value"])
            display = str(group["display_value"])
            raw_refs = group["members"]
            if not isinstance(raw_refs, (list, tuple)):
                raise ValueError("value group members must be an array")
            refs = [str(ref) for ref in raw_refs]
            if len(refs) != len(set(refs)):
                raise ValueError("value groups must contain each member exactly once")
            refs.sort()
            if not value or value in seen_values or not refs or not set(refs) <= set(member_values):
                raise ValueError("invalid value group membership or duplicate normalized value")
            if covered.intersection(refs):
                raise ValueError("value groups must contain each member exactly once")
            if any(member_values[ref] != value for ref in refs):
                raise ValueError("value group normalized_value must match every member")
            if len(value) > _MAX_FIELD_CHARS or len(display) > _MAX_FIELD_CHARS:
                raise ValueError("value group field exceeds size bound")
            normalized.append({"normalized_value": value, "display_value": display, "members": refs})
            seen_values.add(value)
            covered.update(refs)
        if covered != set(member_values):
            raise ValueError("value groups must cover every member exactly once")
        normalized.sort(key=lambda item: item["normalized_value"])
        return normalized

    @staticmethod
    def _candidate_key(
        detector_version: str, members: list[dict[str, Any]], candidate_key: dict[str, Any] | None
    ) -> dict[str, Any]:
        expected_evidence = [{
            "member": _member_ref(member),
            "unit": member.get("evidence_unit"),
            "span": member["evidence_span"],
            "hash": member["content_hash"],
        } for member in members]
        expected = {
            "detector_version": detector_version,
            "members": [_member_ref(member) for member in members],
            "evidence": expected_evidence,
        }
        if candidate_key is None:
            return expected
        key = dict(candidate_key)
        if set(key) != {"detector_version", "members", "evidence"}:
            raise ValueError("candidate_key must contain exactly detector_version, members, and evidence")
        try:
            normalized = {
                "detector_version": str(key["detector_version"]),
                "members": [str(ref) for ref in key["members"]],
                "evidence": [
                    {
                        "member": str(item["member"]),
                        "unit": None if item["unit"] is None else int(item["unit"]),
                        "span": [int(item["span"][0]), int(item["span"][1])],
                        "hash": str(item["hash"]),
                    }
                    for item in key["evidence"]
                ],
            }
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError("candidate_key has invalid member evidence") from exc
        if normalized != expected:
            raise ValueError("candidate_key does not match detector and sorted member evidence")
        if len(_canonical_json(normalized)) > 65_536:
            raise ValueError("candidate_key exceeds size bound")
        return normalized

    @staticmethod
    def _active_members_match_workspace(
        conn: sqlite3.Connection, conflict: dict[str, Any],
        caller_workspace: "WorkspaceScope" = None,
    ) -> bool:
        """Revalidate every current member against the conflict and strict caller scope.

        The group's own ``workspace_canonical`` is authoritative: every live
        member must still sit in it. Strict admission widens the CALLER side only — a
        strict caller may act on a group whose canonical is any of its admitted
        canonicals (its own plus in-radius neighbours). With vector admission
        off the scope is the single caller canonical, i.e. the single-name equality.
        """
        expected_workspace = str(conflict.get("workspace_canonical") or "").strip()
        allowed = set(scope_names(caller_workspace))
        if not expected_workspace or (allowed and expected_workspace not in allowed):
            return False
        members = conflict.get("member_versions") or []
        if not members:
            return False
        for member in members:
            current = conn.execute(
                "SELECT status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                "FROM memories WHERE id=?", (int(member["memory_id"]),),
            ).fetchone()
            if (
                current is None or current["status"] != "active"
                or str(current["workspace"] or "").strip() != expected_workspace
            ):
                return False
        return True
