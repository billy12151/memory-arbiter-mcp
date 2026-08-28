"""Advisory notice lifecycle stored on unified conflict-domain rows."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, cast, Optional, TYPE_CHECKING

from ..acl import WorkspaceScope, scope_names, workspace_scope_sql
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB

NOTICE_STATUSES = frozenset({"open", "dismissed", "resolved", "stale"})
NOTICE_SCAN_LIMIT = 25


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class SemanticNoticeStore:
    """Expose the notice API over candidate/open snapshots in ``conflicts``."""

    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    @staticmethod
    def _decode(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        conflict = dict(row)
        for key in ("slot_key", "candidate_key", "member_versions", "value_groups", "apply_summary", "notice_payload", "notice_slot_provenance"):
            if isinstance(conflict.get(key), str):
                try:
                    conflict[key] = json.loads(conflict[key])
                except (TypeError, json.JSONDecodeError):
                    conflict[key] = {} if key not in {"member_versions", "value_groups"} else []
        members = conflict.get("member_versions") or []
        ids = [int(member["memory_id"]) for member in members if member.get("memory_id") is not None]
        versions = [int(member["version"]) for member in members if member.get("version") is not None]
        delivery = str(conflict.get("notice_delivery_status") or "not_applicable")
        status = "open" if delivery in {"pending", "delivered"} else delivery
        return {
            **conflict,
            "conflict_id": int(conflict["id"]),
            "notice_id": int(conflict["id"]),
            "status": status,
            "severity": conflict.get("notice_severity"),
            "notice_type": conflict.get("notice_type"),
            "title": conflict.get("notice_title"),
            "message": conflict.get("notice_message"),
            "payload": conflict.get("notice_payload") or {},
            "dedupe_key": conflict.get("notice_dedupe_key"),
            "delivered_at": conflict.get("notice_delivered_at"),
            "resolution_reason": conflict.get("notice_resolution_reason"),
            "memory_id": ids[0] if ids else None,
            "peer_id": ids[1] if len(ids) > 1 else None,
            "left_version": versions[0] if versions else None,
            "right_version": versions[1] if len(versions) > 1 else None,
        }

    @staticmethod
    def _workspace_clause(workspace: "WorkspaceScope") -> tuple[str, list[Any]]:
        """Scope notices to the caller's admitted canonical set.

        ``None`` means unscoped. A single name yields the single-name equality; a
        set yields IN (...). Legacy rows with a NULL/empty workspace_canonical
        stay out of every scoped read, exactly as before.
        """
        if workspace is None:
            return ("", [])
        scope_sql, params = workspace_scope_sql("workspace_canonical", workspace)
        if not scope_sql:
            # An empty scope must not silently widen to "all notices".
            return (" AND 1=0", [])
        return (f" AND {scope_sql}", list(params))

    @staticmethod
    def _snapshot(payload: dict[str, Any], memory_id: int, peer_id: int, left_version: int, right_version: int, detector_version: str, prompt_version: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw_members_value = payload.get("member_versions")
        raw_members = raw_members_value if isinstance(raw_members_value, list) else []
        qwen_value = payload.get("qwen_signal")
        qwen = qwen_value if isinstance(qwen_value, dict) else {}
        fallback = [
            {"memory_id": memory_id, "version": left_version, "value": None, "evidence": payload.get("left_evidence") or {}, "content_hash": payload.get("left_content_hash")},
            {"memory_id": peer_id, "version": right_version, "value": None, "evidence": payload.get("right_evidence") or {}, "content_hash": payload.get("right_content_hash")},
        ]
        source = raw_members if len(raw_members) >= 2 else fallback
        members: list[dict[str, Any]] = []
        attribute = str((payload.get("slot_key") or {}).get("attribute") or qwen.get("attribute") or "")
        for index, raw in enumerate(source[:256]):
            evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
            value = str(raw.get("normalized_value") or raw.get("value_raw") or raw.get("value") or "")
            raw_span = raw.get("evidence_span")
            if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
                evidence_span = [int(raw_span[0]), int(raw_span[1])]
            else:
                evidence_span = [
                    int(evidence.get("start", evidence.get("start_offset", 0)) or 0),
                    int(evidence.get("end", evidence.get("end_offset", 0)) or 0),
                ]
            members.append({
                "memory_id": int(raw.get("memory_id") or (memory_id if index == 0 else peer_id)),
                "version": int(raw.get("version") or (left_version if index == 0 else right_version)),
                "attribute_raw": str(raw.get("attribute_raw") or attribute),
                "value_raw": str(raw.get("value_raw") or value),
                "normalized_attribute": str(raw.get("normalized_attribute") or attribute),
                "normalized_value": value,
                "evidence_quote": str(raw.get("evidence_quote") or evidence.get("quote") or evidence.get("text") or ""),
                "evidence_span": evidence_span,
                "content_hash": str(raw.get("content_hash") or (payload.get("left_content_hash") if index == 0 else payload.get("right_content_hash")) or ""),
                "direction": str(raw.get("direction") or ("a_to_b" if index == 0 else "b_to_a")),
                "prompt_version": raw.get("prompt_version", prompt_version),
                "detector_version": str(raw.get("detector_version") or detector_version),
                "evidence_unit": raw.get("evidence_unit"),
            })
        groups_value = payload.get("value_groups")
        groups = groups_value if isinstance(groups_value, list) else []
        return members, [dict(group) for group in groups]

    def record_semantic_notice(self, *, memory_id: int, peer_id: int | None, severity: str, notice_type: str, title: str, message: str, payload: dict[str, Any], dedupe_key: str | None = None, conflict_id: int | None = None, left_version: int | None = None, right_version: int | None = None, source: str = "semantic_evidence") -> dict[str, Any]:
        db = self._db
        if not db.db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        if peer_id is None or left_version is None or right_version is None:
            return {"outcome": "invalid_snapshot"}
        payload = dict(payload or {})
        detector_version = str((payload.get("candidate_key") or {}).get("detector_version") or "semantic-evidence-v1")
        prompt_version = payload.get("prompt_version")
        members, groups = self._snapshot(payload, int(memory_id), int(peer_id), int(left_version), int(right_version), detector_version, prompt_version)
        candidate_value = payload.get("candidate_key")
        candidate = cast(dict[str, Any], candidate_value) if isinstance(candidate_value, dict) else {
            "detector_version": detector_version,
            "members": sorted(f"{member['memory_id']}@{member['version']}" for member in members),
        }
        # Notice delivery is deduped by its task/delivery identity.  Keep that
        # identity in the candidate snapshot so a later, structurally complete
        # notice never aliases an earlier incomplete advisory snapshot.
        candidate = {
            **candidate,
            "detector_version": detector_version,
            "notice_task_id": str(payload.get("task_id") or dedupe_key or ""),
            "notice_dedupe_key": dedupe_key,
        }
        slot = payload.get("slot_key") if isinstance(payload.get("slot_key"), dict) else None
        workspace = ""
        with db.connection() as conn:
            member_ids = [int(member["memory_id"]) for member in members]
            if not member_ids:
                return {"outcome": "invalid_snapshot"}
            placeholders = ",".join("?" for _ in member_ids)
            rows = conn.execute(
                "SELECT id,version,status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                f"FROM memories WHERE id IN ({placeholders})",
                member_ids,
            ).fetchall()
            current = {int(row["id"]): row for row in rows}
            member_workspaces: set[str] = set()
            for member in members:
                row = current.get(int(member["memory_id"]))
                if row is None:
                    return {"outcome": "invalid_snapshot"}
                member_workspaces.add(str(row["workspace"] or "").strip())
            # A notice must be promotable to one formal conflict group. Mixed-
            # workspace snapshots cannot satisfy that invariant and, under
            # strict isolation, would leak a hidden member through the visible
            # notice row. Reject them before persistence.
            if len(member_workspaces) != 1 or not next(iter(member_workspaces), ""):
                return {"outcome": "workspace_mismatch"}
            workspace = next(iter(member_workspaces))
        now = utc_now_iso()
        candidate_hash = _sha(candidate)
        slot_hash = _sha(slot) if slot else None
        fingerprint = _sha(sorted(f"{member['memory_id']}@{member['version']}" for member in members))
        task_id = str(payload.get("task_id") or dedupe_key or candidate_hash)
        try:
            with db.write_transaction() as conn:
                cur = conn.execute(
                    """INSERT INTO conflicts(
                       revision,workspace_canonical,slot_key,slot_key_hash,candidate_key,candidate_key_hash,
                       conflict_point,status,member_versions,member_fingerprint,value_groups,detection_reason,
                       source,detector_version,prompt_version,notice_severity,notice_type,notice_title,
                       notice_message,notice_payload,notice_task_id,notice_dedupe_key,notice_delivery_status,
                       notice_slot_provenance,created_at,refreshed_at)
                       VALUES(1,?,?,?,?,?,?,'candidate',?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?,?)""",
                    (workspace, _canonical_json(slot) if slot else None, slot_hash, _canonical_json(candidate), candidate_hash,
                     str(payload.get("conflict_point") or (slot or {}).get("attribute") or title),
                     _canonical_json(members), fingerprint, _canonical_json(groups), str(payload.get("reason") or message),
                     str(source), detector_version, prompt_version, str(severity or "normal"), str(notice_type or "semantic_evidence"),
                     str(title), str(message), _canonical_json(payload), task_id, dedupe_key,
                     _canonical_json(payload.get("slot_provenance") or {}), now, now),
                )
                row_id = cast(int, cur.lastrowid)
                return {"outcome": "created", "notice_id": int(row_id), "conflict_id": int(row_id)}
        except sqlite3.IntegrityError:
            with db.connection() as conn:
                row = conn.execute("SELECT id FROM conflicts WHERE candidate_key_hash=? OR notice_dedupe_key=? LIMIT 1", (candidate_hash, dedupe_key)).fetchone()
            return {"outcome": "deduped", "notice_id": int(row["id"]), "conflict_id": int(row["id"])} if row else {"outcome": "error", "reason": "integrity_constraint"}

    @staticmethod
    def _freshness(conn: sqlite3.Connection, notice: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        fresh = True
        expected_workspace = str(notice.get("workspace_canonical") or "").strip()
        for member in notice.get("member_versions") or []:
            row = conn.execute(
                "SELECT version,status,COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                "FROM memories WHERE id=?",
                (int(member["memory_id"]),),
            ).fetchone()
            actual_workspace = str(row["workspace"] or "").strip() if row else None
            ok = bool(
                row and row["status"] == "active"
                and int(row["version"]) == int(member["version"])
                and expected_workspace
                and actual_workspace == expected_workspace
            )
            checks.append({
                "memory_id": member["memory_id"],
                "expected_version": member["version"],
                "current_version": row["version"] if row else None,
                "status": row["status"] if row else None,
                "expected_workspace": expected_workspace,
                "workspace": actual_workspace,
                "fresh": ok,
            })
            fresh = fresh and ok
        return {"fresh": fresh, "checks": checks}

    @staticmethod
    def _mark_stale(conn: sqlite3.Connection, notice: dict[str, Any]) -> bool:
        if notice.get("notice_delivery_status") not in {"pending", "delivered"}:
            return False
        if SemanticNoticeStore._freshness(conn, notice)["fresh"]:
            return False
        now = utc_now_iso()
        cur = conn.execute(
            "UPDATE conflicts SET notice_delivery_status='stale',refreshed_at=? "
            "WHERE id=? AND notice_delivery_status IN ('pending','delivered')",
            (now, int(notice["id"])),
        )
        return bool(cur.rowcount)

    @staticmethod
    def _members_visible_in_scope(
        conn: sqlite3.Connection,
        notice: dict[str, Any],
        workspace_canonical: "WorkspaceScope",
    ) -> bool:
        """Whether every current member remains readable by a scoped caller.

        A stale notice may still be inspected when only a member version/status
        changed, but never after a member moved outside the caller's admitted
        workspace set: the frozen values, evidence, and member IDs are a
        correlated snapshot and must fail closed like conflict detail.
        """
        if workspace_canonical is None:
            return True
        allowed = set(scope_names(workspace_canonical))
        members = notice.get("member_versions") or []
        if not allowed or not members:
            return False
        for member in members:
            row = conn.execute(
                "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                "FROM memories WHERE id=?",
                (int(member["memory_id"]),),
            ).fetchone()
            if row is None or str(row["workspace"] or "").strip() not in allowed:
                return False
        return True

    @staticmethod
    def _echo_workspace(workspace_canonical: "WorkspaceScope") -> str | None:
        """The single workspace string to echo back in a suggested call.

        A scope may be the caller's admitted set ; the caller's own
        canonical is always its first element, and that is what a retry call
        must carry — a set is not a valid ``workspace`` payload value.
        """
        names = scope_names(workspace_canonical)
        return names[0] if names else None

    @staticmethod
    def _read_calls(notice: dict[str, Any], workspace_canonical: "WorkspaceScope") -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        echo = SemanticNoticeStore._echo_workspace(workspace_canonical)
        for member in notice.get("member_versions") or []:
            data: dict[str, Any] = {"memory_id": int(member["memory_id"])}
            if echo is not None:
                data["workspace"] = echo
            calls.append({"tool": "memory", "action": "read", "data": data})
        return calls

    def claim_next_semantic_notice(self, workspace_canonical: "WorkspaceScope" = None) -> dict[str, Any] | None:
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        with self._db.write_transaction() as conn:
            while True:
                rows = conn.execute(
                    "SELECT * FROM conflicts WHERE notice_delivery_status='pending'" + workspace_sql +
                    " ORDER BY CASE lower(notice_severity) WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'warning' THEN 2 WHEN 'normal' THEN 3 ELSE 4 END,created_at,id LIMIT ?",
                    (*args, NOTICE_SCAN_LIMIT),
                ).fetchall()
                if not rows:
                    return None
                transitioned = False
                for row in rows:
                    notice = self._decode(row)
                    if not self._members_visible_in_scope(conn, notice, workspace_canonical):
                        transitioned = self._mark_stale(conn, notice) or transitioned
                        continue
                    if self._mark_stale(conn, notice):
                        transitioned = True
                        continue
                    now = utc_now_iso()
                    cur = conn.execute(
                        "UPDATE conflicts SET notice_delivery_status='delivered',notice_delivered_at=?,refreshed_at=? "
                        "WHERE id=? AND notice_delivery_status='pending'",
                        (now, now, notice["id"]),
                    )
                    if cur.rowcount:
                        data: dict[str, Any] = {"action": "read", "notice_id": notice["id"]}
                        echo = self._echo_workspace(workspace_canonical)
                        if echo is not None:
                            data["workspace"] = echo
                        return {
                            "notice_id": notice["id"], "severity": notice["severity"],
                            "type": notice["notice_type"],
                            "action_required": "read_semantic_notice",
                            "read_call": {"tool": "memory_repair", "task": "notice", "data": data},
                        }
                # Rows marked stale leave the pending query; retry immediately
                # so a valid notice behind a full stale/hidden page is delivered
                # on this same product call, not a later one.
                if not transitioned:
                    return None

    def read_semantic_notice(self, notice_id: int, workspace_canonical: "WorkspaceScope" = None) -> dict[str, Any] | None:
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        with self._db.write_transaction() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=? AND notice_type IS NOT NULL" + workspace_sql, (int(notice_id), *args)).fetchone()
            if row is None:
                return None
            notice = self._decode(row)
            if not self._members_visible_in_scope(conn, notice, workspace_canonical):
                self._mark_stale(conn, notice)
                return None
            if self._mark_stale(conn, notice):
                notice["notice_delivery_status"] = "stale"
                notice["status"] = "stale"
            notice["freshness"] = self._freshness(conn, notice)
            read_calls = self._read_calls(notice, workspace_canonical)
            notice["read_calls"] = read_calls
            if len(read_calls) == 2:
                notice["left_read_call"], notice["right_read_call"] = read_calls
            notice["agent_instruction"] = (
                "First verify freshness.fresh is true, then execute every read_calls entry and read each "
                "complete memory before triage. Do not triage from evidence quotes, spans, or a partial subset."
            )
            return notice

    def list_semantic_notices(self, status: str = "open", limit: int = 10, workspace_canonical: "WorkspaceScope" = None) -> list[dict[str, Any]]:
        status = str(status).lower()
        if status not in NOTICE_STATUSES:
            raise ValueError("invalid_notice_status")
        delivery = {"open": ("pending", "delivered"), "dismissed": ("dismissed",), "resolved": ("resolved",), "stale": ("stale",)}[status]
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        placeholders = ",".join("?" for _ in delivery)
        with self._db.write_transaction() as conn:
            if status in {"open", "stale"}:
                last_id = 0
                while True:
                    sweep_rows = conn.execute(
                        "SELECT * FROM conflicts WHERE notice_delivery_status IN ('pending','delivered') "
                        "AND id>?" + workspace_sql + " ORDER BY id LIMIT ?",
                        (last_id, *args, NOTICE_SCAN_LIMIT),
                    ).fetchall()
                    if not sweep_rows:
                        break
                    for sweep_row in sweep_rows:
                        self._mark_stale(conn, self._decode(sweep_row))
                    last_id = int(sweep_rows[-1]["id"])
                    if len(sweep_rows) < NOTICE_SCAN_LIMIT:
                        break
            requested = max(1, min(100, int(limit)))
            notices: list[dict[str, Any]] = []
            offset = 0
            page_size = max(NOTICE_SCAN_LIMIT, requested)
            while len(notices) < requested:
                rows = conn.execute(
                    "SELECT * FROM conflicts WHERE notice_delivery_status IN (" + placeholders + ")"
                    + workspace_sql + " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                    (*delivery, *args, page_size, offset),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    notice = self._decode(row)
                    if not self._members_visible_in_scope(conn, notice, workspace_canonical):
                        continue
                    notice["freshness"] = self._freshness(conn, notice)
                    notices.append(notice)
                    if len(notices) >= requested:
                        break
                offset += len(rows)
                if len(rows) < page_size:
                    break
            return notices

    def semantic_notice_counts(self, workspace_canonical: "WorkspaceScope" = None) -> dict[str, int]:
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        if workspace_canonical is None:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT notice_delivery_status,COUNT(*) AS count FROM conflicts "
                    "WHERE notice_type IS NOT NULL GROUP BY notice_delivery_status"
                ).fetchall()
            result: dict[str, int] = {}
            for row in rows:
                key = "open" if row["notice_delivery_status"] in {"pending", "delivered"} else str(row["notice_delivery_status"])
                result[key] = result.get(key, 0) + int(row["count"])
            return result
        with self._db.write_transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM conflicts WHERE notice_type IS NOT NULL" + workspace_sql,
                args,
            ).fetchall()
            result = {}
            for row in rows:
                notice = self._decode(row)
                visible = self._members_visible_in_scope(conn, notice, workspace_canonical)
                if notice.get("notice_delivery_status") in {"pending", "delivered"}:
                    if self._mark_stale(conn, notice):
                        notice["notice_delivery_status"] = "stale"
                if not visible:
                    continue
                delivery = str(notice.get("notice_delivery_status") or "")
                key = "open" if delivery in {"pending", "delivered"} else delivery
                result[key] = result.get(key, 0) + 1
            return result

    def is_semantic_pair_closed(self, left_id: int, right_id: int, left_version: int | None = None, right_version: int | None = None, notice_type: str = "semantic_evidence") -> bool:
        for notice in self.list_semantic_notices("dismissed", 10000) + self.list_semantic_notices("resolved", 10000):
            if notice.get("notice_type") != notice_type:
                continue
            pins = {(int(member["memory_id"]), int(member["version"])) for member in notice.get("member_versions") or []}
            if left_version is None or right_version is None:
                if {left_id, right_id} <= {pin[0] for pin in pins}:
                    return True
            elif {(int(left_id), int(left_version)), (int(right_id), int(right_version))} <= pins:
                return True
        return False

    def update_semantic_notice_status(self, notice_id: int, status: str, reason: str = "", workspace_canonical: "WorkspaceScope" = None, conflict_id: int | None = None) -> dict[str, Any]:
        status = str(status).lower()
        if status not in {"dismissed", "resolved"}:
            return {"outcome": "invalid_status"}
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        with self._db.write_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM conflicts WHERE id=? AND notice_type IS NOT NULL" + workspace_sql,
                (int(notice_id), *args),
            ).fetchone()
            if row is None:
                return {"outcome": "not_found"}
            notice = self._decode(row)
            if not self._members_visible_in_scope(conn, notice, workspace_canonical):
                self._mark_stale(conn, notice)
                return {"outcome": "not_found"}
            if self._mark_stale(conn, notice):
                return {"outcome": "stale_snapshot", "status": "stale"}
            if notice["notice_delivery_status"] not in {"pending", "delivered"}:
                return {"outcome": "already_terminal", "status": notice["notice_delivery_status"]}
            # Use the raw conflict status, not the decoded one: decoding maps
            # delivery pending/delivered to "open", which would turn a resolved
            # notice into a formal open conflict and wedge the per-slot unique
            # index. A notice stays a terminal candidate unless escalated.
            conflict_status = "not_a_conflict" if status == "dismissed" else str(row["status"] or "candidate")
            now = utc_now_iso()
            try:
                cur = conn.execute("UPDATE conflicts SET status=?,notice_delivery_status=?,notice_resolution_reason=?,revision=revision+1,refreshed_at=?,resolved_at=CASE WHEN ?='not_a_conflict' THEN ? ELSE resolved_at END WHERE id=? AND notice_delivery_status IN ('pending','delivered')", (conflict_status, status, str(reason), now, conflict_status, now, int(notice_id)))
            except sqlite3.IntegrityError:
                return {"outcome": "slot_occupied", "status": status}
        return {"outcome": "updated" if cur.rowcount else "already_terminal", "status": status}
