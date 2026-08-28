"""Revisioned conflict-group persistence.

A row is one immutable detection event whose open snapshot may only grow by
CAS-guarded member append.  Decisions and per-member application results live
on that same row; there is no judgment side table.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Optional, TYPE_CHECKING

from ..acl import WorkspaceScope, scope_names, workspace_scope_sql
from ..models import ConflictMember, ConflictValueGroup, utc_now_iso
from ..semantic_conflict import normalize_value
from ..text import canon_entity, canon_scope

if TYPE_CHECKING:
    from .core import MemoryDB

_MAX_MEMBERS = 256
_MAX_FIELD_CHARS = 16_384
_MAX_MEMBER_JSON = 262_144
_MAX_VALUE_JSON = 131_072
_MAX_APPLY_PLAN_JSON = 131_072
_MAX_SLOT_JSON = 4_096
_ALLOWED_ACTIONS = {
    "update_current_claim", "append_superseded_context",
    "preserve_historical_record", "use_as_resolution", "needs_authorization",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _member_ref(member: dict[str, Any]) -> str:
    return f"{int(member['memory_id'])}@{int(member['version'])}"


def _decode_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("slot_key", "candidate_key", "member_versions", "value_groups", "apply_summary"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    data["overflow"] = bool(data.get("overflow"))
    return data


class ConflictStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    @staticmethod
    def _normalize_slot(slot_key: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
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
        detector_version: str, members: list[dict[str, Any]], candidate_key: Optional[dict[str, Any]]
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

    def get_conflict(self, conflict_id: int) -> Optional[dict[str, Any]]:
        if not self._db_available:
            return None
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)).fetchone()
        return _decode_row(row) if row else None

    def list_conflicts(
        self,
        status: str = "open",
        limit: int = 50,
        source: Optional[str] = None,
        workspace: "WorkspaceScope" = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        sql = "SELECT * FROM conflicts WHERE status=?"
        params: list[Any] = [status]
        if source is not None:
            sql += " AND source=?"
            params.append(source)
        scope_sql, scope_params = workspace_scope_sql("workspace_canonical", workspace)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        sql += " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self.connection() as conn:
            return [_decode_row(row) for row in conn.execute(sql, params).fetchall()]

    def record_conflict_group(
        self, *, workspace_canonical: str, slot_key: Optional[dict[str, Any]],
        members: list[dict[str, Any] | ConflictMember],
        value_groups: list[dict[str, Any] | ConflictValueGroup],
        detection_reason: str, source: str, detector_version: str,
        conflict_point: Optional[str] = None, prompt_version: Optional[str] = None,
        candidate_key: Optional[dict[str, Any]] = None, status: str = "open",
        expected_revision: Optional[int] = None,
    ) -> dict[str, Any]:
        if status not in {"open", "not_a_conflict"}:
            return {"outcome": "invalid_status"}
        if status == "open" and slot_key is None:
            return {"outcome": "invalid_slot_key", "error": "open conflicts require a complete slot_key"}
        try:
            slot = self._normalize_slot(slot_key)
            normalized_members = self._normalize_members(members)
            if not normalized_members:
                raise ValueError("at least one member is required")
            if any(str(member["detector_version"]) != str(detector_version) for member in normalized_members):
                raise ValueError("member detector_version must match detector_version")
            refs = {_member_ref(member) for member in normalized_members}
            groups = (
                self._normalize_value_groups(value_groups, normalized_members)
                if status == "open" or value_groups else []
            )
            candidate = self._candidate_key(detector_version, normalized_members, candidate_key)
            candidate_hash = _hash_json(candidate)
            slot_hash = _hash_json(slot) if slot is not None else None
        except (TypeError, ValueError) as exc:
            return {"outcome": "invalid_input", "error": str(exc)}
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable"}
        now = utc_now_iso()
        with self.write_transaction() as conn:
            # Derive the workspace from current pinned member rows inside the
            # same transaction; caller labels cannot create cross-workspace groups.
            member_workspaces: set[str] = set()
            for member in normalized_members:
                memory = conn.execute(
                    "SELECT version,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?", (int(member["memory_id"]),),
                ).fetchone()
                if memory is None or int(memory["version"] or 0) != int(member["version"]):
                    return {"outcome": "stale_snapshot"}
                member_workspaces.add(str(memory["workspace"] or "").strip())
            if len(member_workspaces) != 1:
                return {"outcome": "workspace_mismatch"}
            derived_workspace = next(iter(member_workspaces))
            if not derived_workspace or (
                workspace_canonical and str(workspace_canonical).strip() != derived_workspace
            ):
                return {"outcome": "workspace_mismatch"}
            workspace_canonical = derived_workspace
            active = None
            if slot_hash is not None:
                active = conn.execute(
                    "SELECT * FROM conflicts WHERE workspace_canonical=? AND slot_key_hash=? "
                    "AND status IN ('open','applying')", (workspace_canonical, slot_hash),
                ).fetchone()
            if active:
                current = _decode_row(active)
                if status == "not_a_conflict":
                    # The caller's disposition must never be silently inverted
                    # into a member append on an existing formal group.
                    return {"outcome": "open_group_exists", "conflict_id": current["id"], "revision": current["revision"]}
                # Update calls are CAS operations.  Check the revision before
                # candidate/member dedupe so replaying an old exact payload is
                # observably stale rather than appearing successful.
                if expected_revision is not None and int(expected_revision) != int(current["revision"]):
                    return {"outcome": "stale_conflict", "conflict_id": current["id"], "revision": current["revision"]}
                if current["status"] != "open":
                    return {"outcome": "applying", "conflict_id": current["id"], "revision": current["revision"]}
                if expected_revision is None:
                    duplicate = conn.execute(
                        "SELECT * FROM conflicts WHERE candidate_key_hash=?", (candidate_hash,)
                    ).fetchone()
                    if duplicate:
                        row = _decode_row(duplicate)
                        if row["candidate_key"] == candidate:
                            return {"outcome": "deduped", "conflict_id": row["id"], "revision": row["revision"]}
                        return {"outcome": "identity_collision"}
                    return {"outcome": "stale_conflict", "conflict_id": current["id"], "revision": current["revision"]}
                existing_refs = {_member_ref(member) for member in current["member_versions"]}
                additions = [member for member in normalized_members if _member_ref(member) not in existing_refs]
                if not additions:
                    return {"outcome": "deduped", "conflict_id": current["id"], "revision": current["revision"]}
                combined = current["member_versions"] + additions
                merged_groups = {group["normalized_value"]: dict(group) for group in current["value_groups"]}
                for group in groups:
                    found = merged_groups.setdefault(group["normalized_value"], dict(group))
                    found["members"] = sorted(set(found["members"]) | set(group["members"]))
                merged = sorted(merged_groups.values(), key=lambda item: item["normalized_value"])
                members_json, groups_json = _canonical_json(combined), _canonical_json(merged)
                overflow = len(combined) > _MAX_MEMBERS or len(members_json) > _MAX_MEMBER_JSON or len(groups_json) > _MAX_VALUE_JSON
                if overflow:
                    conn.execute(
                        "UPDATE conflicts SET overflow=1,revision=revision+1,refreshed_at=? WHERE id=? AND revision=?",
                        (now, current["id"], current["revision"]),
                    )
                    return {"outcome": "overflow", "conflict_id": current["id"], "revision": current["revision"] + 1}
                fingerprint = _hash_json(sorted(_member_ref(member) for member in combined))
                # The row's candidate identity must describe the row's whole
                # member snapshot after the append, not just the incoming batch.
                combined_candidate = self._candidate_key(detector_version, combined, None)
                combined_hash = _hash_json(combined_candidate)
                collision = conn.execute(
                    "SELECT * FROM conflicts WHERE candidate_key_hash=? AND id<>?",
                    (combined_hash, current["id"]),
                ).fetchone()
                if collision is not None:
                    collision_row = _decode_row(collision)
                    if collision_row["candidate_key"] == combined_candidate:
                        return {"outcome": "deduped", "conflict_id": collision_row["id"], "revision": collision_row["revision"]}
                    return {"outcome": "identity_collision", "conflict_id": collision_row["id"]}
                try:
                    cur = conn.execute(
                        "UPDATE conflicts SET revision=revision+1,member_versions=?,member_fingerprint=?,"
                        "value_groups=?,candidate_key=?,candidate_key_hash=?,refreshed_at=? "
                        "WHERE id=? AND status='open' AND revision=?",
                        (members_json, fingerprint, groups_json, _canonical_json(combined_candidate), combined_hash,
                         now, current["id"], current["revision"]),
                    )
                except sqlite3.IntegrityError:
                    # The event-snapshot index (workspace_canonical, slot_key_hash,
                    # member_fingerprint) excludes detector_version, so a resolved /
                    # not_a_conflict row recorded under a different detector can own
                    # the combined fingerprint. Return a structured outcome rather
                    # than leaking a raw IntegrityError through the tool layer.
                    snapshot_row = conn.execute(
                        "SELECT id,revision FROM conflicts WHERE workspace_canonical=? "
                        "AND slot_key_hash=? AND member_fingerprint=? AND id<>?",
                        (workspace_canonical, slot_hash, fingerprint, current["id"]),
                    ).fetchone()
                    if snapshot_row is not None:
                        return {"outcome": "duplicate_event", "conflict_id": int(snapshot_row["id"]),
                                "revision": int(snapshot_row["revision"])}
                    raise
                if cur.rowcount != 1:
                    return {"outcome": "stale_conflict", "conflict_id": current["id"]}
                return {"outcome": "appended", "conflict_id": current["id"], "revision": current["revision"] + 1}
            duplicate = conn.execute(
                "SELECT * FROM conflicts WHERE candidate_key_hash=?", (candidate_hash,)
            ).fetchone()
            if duplicate:
                duplicate_row = _decode_row(duplicate)
                promotable_notice = (
                    status == "open"
                    and duplicate_row.get("status") == "candidate"
                    and duplicate_row.get("notice_type") is not None
                    and duplicate_row.get("slot_key_hash") == slot_hash
                )
                if not promotable_notice:
                    if duplicate_row["candidate_key"] == candidate:
                        return {
                            "outcome": "deduped", "conflict_id": duplicate_row["id"],
                            "revision": duplicate_row["revision"],
                        }
                    return {"outcome": "identity_collision"}
            if expected_revision is not None:
                return {"outcome": "stale_conflict"}
            if status == "open" and len(groups) < 2:
                # Creation only: a fresh open event needs two value groups. An
                # append into an existing open group may carry a single new
                # member/value group as long as it covers the incoming members.
                return {"outcome": "invalid_input", "error": "open conflicts require at least two value groups"}
            members_json, groups_json = _canonical_json(normalized_members), _canonical_json(groups)
            if len(normalized_members) > _MAX_MEMBERS or len(members_json) > _MAX_MEMBER_JSON or len(groups_json) > _MAX_VALUE_JSON:
                return {"outcome": "overflow", "error": "initial conflict snapshot exceeds bounds"}
            fingerprint = _hash_json(sorted(refs))
            # A delivered notice may already own this exact frozen event
            # snapshot. Promote that row instead of creating a parallel formal
            # conflict or replacing its immutable member/value evidence.
            if status == "open" and slot_hash is not None:
                notice_row = conn.execute(
                    "SELECT * FROM conflicts WHERE workspace_canonical=? AND slot_key_hash=? "
                    "AND member_fingerprint=? AND status='candidate' AND notice_type IS NOT NULL",
                    (workspace_canonical, slot_hash, fingerprint),
                ).fetchone()
                if notice_row:
                    frozen = _decode_row(notice_row)
                    frozen_members = sorted(
                        frozen["member_versions"], key=lambda item: (item["memory_id"], item["version"]),
                    )
                    frozen_groups = sorted(
                        frozen["value_groups"], key=lambda item: item["normalized_value"],
                    )
                    if frozen_members != normalized_members or frozen_groups != groups:
                        return {"outcome": "snapshot_mismatch", "conflict_id": frozen["id"]}
                    cur = conn.execute(
                        "UPDATE conflicts SET status='open',revision=revision+1,conflict_point=?,"
                        "detection_reason=?,source=?,detector_version=?,prompt_version=?,refreshed_at=? "
                        "WHERE id=? AND status='candidate' AND revision=?",
                        (conflict_point, detection_reason, source, detector_version, prompt_version,
                         now, frozen["id"], frozen["revision"]),
                    )
                    if cur.rowcount != 1:
                        return {"outcome": "stale_conflict", "conflict_id": frozen["id"]}
                    return {
                        "outcome": "deduped", "conflict_id": frozen["id"],
                        "revision": int(frozen["revision"]) + 1,
                    }
            try:
                cur = conn.execute(
                    """INSERT INTO conflicts(
                       revision,workspace_canonical,slot_key,slot_key_hash,candidate_key,candidate_key_hash,
                       conflict_point,status,member_versions,member_fingerprint,value_groups,detection_reason,
                       source,detector_version,prompt_version,created_at,refreshed_at)
                       VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (workspace_canonical, _canonical_json(slot) if slot else None, slot_hash,
                     _canonical_json(candidate), candidate_hash, conflict_point, status, members_json,
                     fingerprint, groups_json, detection_reason, source, detector_version, prompt_version,
                     now, now),
                )
            except sqlite3.IntegrityError:
                duplicate = conn.execute(
                    "SELECT * FROM conflicts WHERE candidate_key_hash=?", (candidate_hash,),
                ).fetchone()
                if duplicate is not None:
                    existing = _decode_row(duplicate)
                    if existing["candidate_key"] == candidate:
                        return {
                            "outcome": "deduped", "conflict_id": int(existing["id"]),
                            "revision": int(existing["revision"]),
                        }
                else:
                    # The event-snapshot index spans all statuses: a resolved or
                    # not_a_conflict row that already owns this exact
                    # workspace+slot+member fingerprint turns a re-record under
                    # a changed detector/candidate identity into a structured
                    # duplicate instead of an uncaught IntegrityError.
                    if slot_hash is not None:
                        snapshot_row = conn.execute(
                            "SELECT id,revision FROM conflicts WHERE workspace_canonical=? "
                            "AND slot_key_hash=? AND member_fingerprint=?",
                            (workspace_canonical, slot_hash, fingerprint),
                        ).fetchone()
                        if snapshot_row is not None:
                            return {
                                "outcome": "duplicate_event",
                                "conflict_id": int(snapshot_row["id"]),
                                "revision": int(snapshot_row["revision"]),
                            }
                raise
            return {"outcome": "inserted", "conflict_id": int(cur.lastrowid), "revision": 1}

    def escalate_structured_notice(
        self, notice_id: int, *, workspace_canonical: "WorkspaceScope", reason: str,
    ) -> dict[str, Any]:
        """Atomically validate and promote a complete frozen notice snapshot."""
        now = utc_now_iso()
        with self.write_transaction() as conn:
            sql = "SELECT * FROM conflicts WHERE id=? AND notice_type IS NOT NULL"
            args: list[Any] = [int(notice_id)]
            if workspace_canonical is not None:
                scope_sql, scope_params = workspace_scope_sql("workspace_canonical", workspace_canonical)
                if scope_sql:
                    sql += f" AND {scope_sql}"
                    args.extend(scope_params)
            row = conn.execute(sql, args).fetchone()
            if row is None:
                return {"outcome": "not_found"}
            notice = _decode_row(row)
            delivery = str(notice.get("notice_delivery_status") or "")
            try:
                slot = self._normalize_slot(notice.get("slot_key"))
                members = self._normalize_members(notice.get("member_versions") or [])
                groups = self._normalize_value_groups(notice.get("value_groups") or [], members)
                if slot is None or len(members) < 2 or len(groups) < 2:
                    raise ValueError("complete slot, members, and at least two value groups are required")
            except (TypeError, ValueError) as exc:
                return {"outcome": "structured_group_required", "error": str(exc)}
            checks: list[dict[str, Any]] = []
            notice_workspace = str(notice.get("workspace_canonical") or "").strip()
            for member in members:
                current = conn.execute(
                    "SELECT version,status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?",
                    (member["memory_id"],),
                ).fetchone()
                member_workspace = str(current["workspace"] or "").strip() if current else None
                fresh = bool(
                    current and current["status"] == "active"
                    and int(current["version"]) == int(member["version"])
                    and notice_workspace
                    and member_workspace == notice_workspace
                )
                checks.append({
                    "memory_id": member["memory_id"],
                    "expected_version": member["version"],
                    "current_version": current["version"] if current else None,
                    "status": current["status"] if current else None,
                    "expected_workspace": notice_workspace,
                    "workspace": member_workspace,
                    "fresh": fresh,
                })
            if not all(check["fresh"] for check in checks):
                return {"outcome": "stale_snapshot", "freshness": {"fresh": False, "checks": checks}}
            if delivery not in {"pending", "delivered"} or notice.get("status") != "candidate":
                return {"outcome": "already_terminal", "status": delivery}
            slot_hash = _hash_json(slot)
            occupied = conn.execute(
                "SELECT * FROM conflicts WHERE workspace_canonical=? AND slot_key_hash=? "
                "AND status IN ('open','applying') AND id<>?",
                (str(notice["workspace_canonical"] or ""), slot_hash, int(notice_id)),
            ).fetchone()
            if occupied is not None:
                # Spec §12: escalate 创建/追加. When a formal group already owns
                # the slot, append the frozen notice members into it under CAS
                # and link the notice row; never let the partial unique index
                # raise through the tool layer.
                group = _decode_row(occupied)
                if group["status"] != "open":
                    return {"outcome": "applying_group_exists", "conflict_id": group["id"], "revision": group["revision"]}
                existing_refs = {_member_ref(member) for member in group["member_versions"]}
                additions = [member for member in members if _member_ref(member) not in existing_refs]
                if not additions:
                    conn.execute(
                        "UPDATE conflicts SET notice_delivery_status='resolved',"
                        "notice_resolution_reason=?,refreshed_at=? WHERE id=? "
                        "AND notice_delivery_status IN ('pending','delivered') AND revision=?",
                        (f"escalated_to_conflict: {reason}", now, int(notice_id), int(notice["revision"])),
                    )
                    return {"outcome": "linked", "conflict_id": group["id"], "revision": group["revision"],
                            "notice_conflict_id": int(notice_id)}
                combined = group["member_versions"] + additions
                merged_groups = {found["normalized_value"]: dict(found) for found in group["value_groups"]}
                for found_group in groups:
                    merged = merged_groups.setdefault(found_group["normalized_value"], dict(found_group))
                    merged["members"] = sorted(set(merged["members"]) | set(found_group["members"]))
                merged_list = sorted(merged_groups.values(), key=lambda item: item["normalized_value"])
                members_json, groups_json = _canonical_json(combined), _canonical_json(merged_list)
                if len(combined) > _MAX_MEMBERS or len(members_json) > _MAX_MEMBER_JSON or len(groups_json) > _MAX_VALUE_JSON:
                    conn.execute(
                        "UPDATE conflicts SET overflow=1,revision=revision+1,refreshed_at=? WHERE id=? AND revision=?",
                        (now, group["id"], group["revision"]),
                    )
                    return {"outcome": "overflow", "conflict_id": group["id"], "revision": group["revision"] + 1}
                combined_fingerprint = _hash_json(sorted(_member_ref(member) for member in combined))
                combined_candidate = self._candidate_key(str(group.get("detector_version") or "attribute-value-v1"), combined, None)
                try:
                    cur = conn.execute(
                        "UPDATE conflicts SET revision=revision+1,member_versions=?,member_fingerprint=?,"
                        "value_groups=?,candidate_key=?,candidate_key_hash=?,refreshed_at=? "
                        "WHERE id=? AND status='open' AND revision=?",
                        (members_json, combined_fingerprint, groups_json,
                         _canonical_json(combined_candidate), _hash_json(combined_candidate),
                         now, group["id"], group["revision"]),
                    )
                except sqlite3.IntegrityError:
                    return {"outcome": "identity_collision", "conflict_id": group["id"]}
                if cur.rowcount != 1:
                    return {"outcome": "stale_conflict", "conflict_id": group["id"]}
                conn.execute(
                    "UPDATE conflicts SET notice_delivery_status='resolved',notice_resolution_reason=?,"
                    "revision=revision+1,refreshed_at=? WHERE id=? "
                    "AND notice_delivery_status IN ('pending','delivered') AND revision=?",
                    (f"escalated_to_conflict: {reason}", now, int(notice_id), int(notice["revision"])),
                )
                return {
                    "outcome": "appended", "conflict_id": group["id"],
                    "revision": group["revision"] + 1, "notice_conflict_id": int(notice_id),
                    "member_versions": combined, "value_groups": merged_list, "slot_key": slot,
                }
            cur = conn.execute(
                """UPDATE conflicts SET status='open',revision=revision+1,source='semantic_notice',
                   detection_reason=?,notice_delivery_status='resolved',notice_resolution_reason=?,
                   refreshed_at=? WHERE id=? AND status='candidate'
                   AND notice_delivery_status IN ('pending','delivered') AND revision=?""",
                (reason, f"escalated_to_conflict: {reason}", now, int(notice_id), int(notice["revision"])),
            )
            if cur.rowcount != 1:
                return {"outcome": "stale_conflict"}
            return {
                "outcome": "promoted", "conflict_id": int(notice_id),
                "revision": int(notice["revision"]) + 1,
                "member_versions": members, "value_groups": groups, "slot_key": slot,
            }

    def judge_conflict(
        self, conflict_id: int, *, expected_revision: int, chosen_value: str,
        decided_by: str, decided_ref: Optional[str], decision_reason: str,
        apply_plan: list[dict[str, Any]], resolution_memory_id: Optional[int] = None,
        strict_workspace: "WorkspaceScope" = None,
    ) -> dict[str, Any]:
        if decided_by not in {"user", "agent"} or not chosen_value:
            return {"outcome": "invalid_input"}
        if not isinstance(apply_plan, list) or len(apply_plan) > _MAX_MEMBERS or any(
            not isinstance(item, dict) for item in apply_plan
        ):
            return {"outcome": "invalid_plan"}
        with self.write_transaction() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)).fetchone()
            if not row:
                return {"outcome": "not_found"}
            conflict = _decode_row(row)
            if conflict["status"] != "open":
                return {"outcome": "not_open", "status": conflict["status"]}
            if int(conflict["revision"]) != int(expected_revision):
                return {"outcome": "stale_conflict", "revision": conflict["revision"]}
            if strict_workspace is not None and not self._active_members_match_workspace(
                conn, conflict, strict_workspace,
            ):
                return {"outcome": "workspace_mismatch"}
            normalized_chosen = normalize_value(str(chosen_value))
            existing_values = {
                normalize_value(str(group.get("normalized_value") or "")): str(group.get("normalized_value") or "")
                for group in conflict.get("value_groups") or []
            }
            if not normalized_chosen or normalized_chosen not in existing_values:
                return {"outcome": "invalid_chosen_value"}
            chosen_value = existing_values[normalized_chosen]
            member_versions = {_member_ref(member): member for member in conflict["member_versions"]}
            plan: list[dict[str, Any]] = []
            seen: set[int] = set()
            for raw in apply_plan:
                item = dict(raw)
                target = int(item.get("memory_id", 0))
                action = str(item.get("action", ""))
                matches = [member for member in member_versions.values() if int(member["memory_id"]) == target]
                if action not in _ALLOWED_ACTIONS or not matches or target in seen:
                    return {"outcome": "invalid_plan"}
                current = conn.execute(
                    "SELECT version,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=? AND status='active'", (target,),
                ).fetchone()
                current_version = int(current["version"] or 0) if current else 0
                # An open group may legitimately hold several versions of one
                # memory (append after an external edit). Pin the stored member
                # entry that matches the memory's CURRENT version; only a target
                # whose current version was never recorded is stale.
                match = next((member for member in matches if int(member["version"]) == current_version), None)
                if (
                    current is None
                    or match is None
                    or str(current["workspace"] or "") != str(conflict["workspace_canonical"] or "")
                ):
                    return {"outcome": "stale_member"}
                plan.append({"memory_id": target, "expected_version": int(match["version"]),
                             "action": action, "status": "pending", "result_version": None,
                             "result_hash": None, "error": None})
                seen.add(target)
            resolution_version = None
            if resolution_memory_id is not None:
                resolution = conn.execute(
                    "SELECT version,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=? AND status='active'", (int(resolution_memory_id),)
                ).fetchone()
                if (
                    not resolution
                    or str(resolution["workspace"] or "") != str(conflict["workspace_canonical"] or "")
                ):
                    return {"outcome": "invalid_resolution_memory"}
                resolution_version = int(resolution["version"])
            now = utc_now_iso()
            summary_json = _canonical_json({"plan": plan})
            if len(summary_json.encode("utf-8")) > _MAX_APPLY_PLAN_JSON:
                return {"outcome": "invalid_plan"}
            cur = conn.execute(
                """UPDATE conflicts SET status='applying',revision=revision+1,chosen_value=?,
                   resolution_memory_id=?,resolution_memory_version=?,decided_by=?,decided_ref=?,
                   decision_reason=?,decided_at=?,apply_summary=?,refreshed_at=?
                   WHERE id=? AND status='open' AND revision=?""",
                (chosen_value, resolution_memory_id, resolution_version, decided_by, decided_ref,
                 decision_reason, now, summary_json, now, int(conflict_id), int(expected_revision)),
            )
            if cur.rowcount != 1:
                return {"outcome": "stale_conflict"}
            return {"outcome": "applying", "conflict_id": int(conflict_id),
                    "revision": int(expected_revision) + 1, "apply_summary": {"plan": plan}}

    def replan_conflict(
        self, conflict_id: int, *, expected_revision: int,
        apply_plan: list[dict[str, Any]], resolution_memory_id: Optional[int] = None,
        strict_workspace: "WorkspaceScope" = None,
    ) -> dict[str, Any]:
        """CAS-reset an applying plan while retaining every prior plan snapshot."""
        if not isinstance(apply_plan, list) or len(apply_plan) > _MAX_MEMBERS or any(
            not isinstance(item, dict) for item in apply_plan
        ):
            return {"outcome": "invalid_plan"}
        with self.write_transaction() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)).fetchone()
            if not row:
                return {"outcome": "not_found"}
            conflict = _decode_row(row)
            if conflict["status"] != "applying":
                return {"outcome": "not_applying"}
            if int(conflict["revision"]) != int(expected_revision):
                return {"outcome": "stale_conflict", "revision": conflict["revision"]}
            if strict_workspace is not None and not self._active_members_match_workspace(
                conn, conflict, strict_workspace,
            ):
                return {"outcome": "workspace_mismatch"}
            members = {int(member["memory_id"]): member for member in conflict["member_versions"]}
            plan: list[dict[str, Any]] = []
            seen: set[int] = set()
            for raw in apply_plan:
                target = int(dict(raw).get("memory_id", 0))
                action = str(dict(raw).get("action", ""))
                member = members.get(target)
                current = conn.execute(
                    "SELECT version,status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?", (target,),
                ).fetchone()
                if (
                    action not in _ALLOWED_ACTIONS or member is None or target in seen
                    or current is None or current["status"] != "active"
                    or str(current["workspace"] or "") != str(conflict["workspace_canonical"] or "")
                ):
                    return {"outcome": "invalid_plan"}
                plan.append({"memory_id": target, "expected_version": int(current["version"]),
                             "action": action, "status": "pending", "result_version": None,
                             "result_hash": None, "error": None})
                seen.add(target)
            resolution_id = (
                int(resolution_memory_id) if resolution_memory_id is not None
                else conflict.get("resolution_memory_id")
            )
            resolution_version = None
            if resolution_id is not None:
                resolution = conn.execute(
                    "SELECT version,status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?", (int(resolution_id),),
                ).fetchone()
                if (
                    resolution is None or resolution["status"] != "active"
                    or str(resolution["workspace"] or "") != str(conflict["workspace_canonical"] or "")
                ):
                    return {"outcome": "invalid_resolution_memory"}
                resolution_version = int(resolution["version"])
            previous = conflict.get("apply_summary") or {"plan": []}
            history = list(previous.get("history") or [])
            history.append({"revision": int(expected_revision), "plan": previous.get("plan") or []})
            summary = {"plan": plan, "history": history}
            summary_json = _canonical_json(summary)
            if len(summary_json.encode("utf-8")) > _MAX_APPLY_PLAN_JSON:
                return {"outcome": "invalid_plan"}
            now = utc_now_iso()
            cur = conn.execute(
                "UPDATE conflicts SET revision=revision+1,apply_summary=?,resolution_memory_id=?,"
                "resolution_memory_version=?,refreshed_at=? WHERE id=? AND status='applying' AND revision=?",
                (summary_json, resolution_id, resolution_version, now,
                 int(conflict_id), int(expected_revision)),
            )
            if cur.rowcount != 1:
                return {"outcome": "stale_conflict"}
            return {"outcome": "replanned", "conflict_id": int(conflict_id),
                    "revision": int(expected_revision) + 1, "apply_summary": summary}

    def resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved", *,
        expected_revision: Optional[int] = None, strict_workspace: "WorkspaceScope" = None,
    ) -> dict[str, Any]:
        if status != "resolved":
            return {"outcome": "invalid_status", "conflict_id": int(conflict_id)}
        with self.write_transaction() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)).fetchone()
            if not row:
                return {"outcome": "not_found"}
            conflict = _decode_row(row)
            if conflict["status"] != "applying":
                return {"outcome": "not_applying", "status": conflict["status"]}
            if expected_revision is None or int(conflict["revision"]) != int(expected_revision):
                return {"outcome": "stale_conflict", "revision": conflict["revision"]}
            if strict_workspace is not None and not self._active_members_match_workspace(
                conn, conflict, strict_workspace,
            ):
                return {"outcome": "workspace_mismatch"}
            plan = conflict["apply_summary"].get("plan", [])
            if any(item.get("status") != "completed" for item in plan):
                return {"outcome": "apply_incomplete", "apply_summary": conflict["apply_summary"]}
            resolution_id = conflict.get("resolution_memory_id")
            resolution_version = conflict.get("resolution_memory_version")
            if resolution_id is not None:
                resolution = conn.execute(
                    "SELECT version,status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?", (int(resolution_id),),
                ).fetchone()
                if (
                    resolution is None or resolution["status"] != "active"
                    or int(resolution["version"] or 0) != int(resolution_version or 0)
                    or str(resolution["workspace"] or "") != str(conflict["workspace_canonical"] or "")
                ):
                    return {"outcome": "stale_resolution_memory"}
            now = utc_now_iso()
            cur = conn.execute(
                "UPDATE conflicts SET status='resolved',revision=revision+1,decision_reason=CASE WHEN ?='' "
                "THEN decision_reason ELSE ? END,resolved_at=?,refreshed_at=? WHERE id=? AND revision=?",
                (reason, reason, now, now, int(conflict_id), int(expected_revision)),
            )
            if cur.rowcount != 1:
                return {"outcome": "stale_conflict"}
            return {"outcome": "resolved", "conflict_id": int(conflict_id), "revision": int(expected_revision) + 1}

    def list_open_conflicts_for_memory_ids(
        self, memory_ids: list[int], *, include_applying: bool = False,
    ) -> list[dict[str, Any]]:
        wanted = sorted({int(value) for value in memory_ids})
        if not wanted or not self._db_available:
            return []
        statuses = "('open','applying')" if include_applying else "('open')"
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT c.* FROM conflicts AS c "
                "JOIN json_each(c.member_versions) AS member "
                "JOIN json_each(?) AS wanted "
                "ON CAST(json_extract(member.value,'$.memory_id') AS INTEGER)=CAST(wanted.value AS INTEGER) "
                f"WHERE c.status IN {statuses} ORDER BY c.created_at DESC,c.id DESC",
                (_canonical_json(wanted),),
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def resolve_conflicts_for_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> int:
        # Generic memory mutation cannot complete a revisioned application plan.
        return 0

    def resolve_conflicts_for(self, memory_id: int, *, conn: Optional[sqlite3.Connection] = None) -> int:
        return 0

    def get_memory_version(self, memory_id: int) -> Optional[int]:
        memory = self._db.get_memory(int(memory_id))
        return int(memory["version"]) if memory else None

    def dismissed_pairs_snapshot(self) -> set[tuple[int, int]]:
        if not self._db_available:
            return set()
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT c.id,CAST(json_extract(member.value,'$.memory_id') AS INTEGER) AS memory_id "
                "FROM conflicts AS c JOIN json_each(c.member_versions) AS member "
                "WHERE c.status='not_a_conflict' ORDER BY c.id,memory_id"
            ).fetchall()
        by_conflict: dict[int, set[int]] = {}
        for row in rows:
            by_conflict.setdefault(int(row["id"]), set()).add(int(row["memory_id"]))
        return {
            tuple(sorted(ids))  # type: ignore[misc]
            for ids in by_conflict.values() if len(ids) == 2
        }

    def is_pair_dismissed(self, left_id: int, right_id: int) -> bool:
        pair = sorted({int(left_id), int(right_id)})
        if len(pair) != 2 or not self._db_available:
            return False
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM conflicts AS c WHERE c.status='not_a_conflict' "
                "AND json_array_length(c.member_versions)=2 "
                "AND EXISTS (SELECT 1 FROM json_each(c.member_versions) WHERE json_extract(value,'$.memory_id')=?) "
                "AND EXISTS (SELECT 1 FROM json_each(c.member_versions) WHERE json_extract(value,'$.memory_id')=?) LIMIT 1",
                (pair[0], pair[1]),
            ).fetchone()
        return row is not None

    def dismissed_pairs_for(self, memory_ids: list[int]) -> set[tuple[int, int]]:
        wanted = sorted({int(value) for value in memory_ids})
        if not wanted or not self._db_available:
            return set()
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT c.id,CAST(json_extract(member.value,'$.memory_id') AS INTEGER) AS memory_id "
                "FROM conflicts AS c JOIN json_each(c.member_versions) AS member "
                "WHERE c.status='not_a_conflict' AND json_array_length(c.member_versions)=2 "
                "AND EXISTS (SELECT 1 FROM json_each(c.member_versions) AS linked "
                "JOIN json_each(?) AS wanted ON json_extract(linked.value,'$.memory_id')=wanted.value) "
                "ORDER BY c.id,memory_id",
                (_canonical_json(wanted),),
            ).fetchall()
        by_conflict: dict[int, set[int]] = {}
        for row in rows:
            by_conflict.setdefault(int(row["id"]), set()).add(int(row["memory_id"]))
        return {
            tuple(sorted(ids))  # type: ignore[misc]
            for ids in by_conflict.values() if len(ids) == 2
        }
