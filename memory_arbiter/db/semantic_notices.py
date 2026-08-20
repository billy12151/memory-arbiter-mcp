"""Persistence for advisory local-text conflict notices."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


NOTICE_STATUSES = frozenset({"open", "dismissed", "resolved", "stale"})
NOTICE_SCAN_LIMIT = 25


class SemanticNoticeStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    def record_semantic_notice(
        self,
        *,
        memory_id: int,
        peer_id: Optional[int],
        severity: str,
        notice_type: str,
        title: str,
        message: str,
        payload: dict[str, Any],
        dedupe_key: Optional[str] = None,
        conflict_id: Optional[int] = None,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        source: str = "semantic_evidence",
    ) -> dict[str, Any]:
        db = self._db
        if not db.db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        if peer_id is None or left_version is None or right_version is None:
            return {"outcome": "invalid_snapshot"}
        try:
            with db.write_transaction() as conn:
                cur = conn.execute(
                    """INSERT INTO semantic_notices(
                         created_at,status,severity,source,memory_id,peer_id,
                         conflict_id,notice_type,title,message,payload,dedupe_key,
                         left_version,right_version
                       ) VALUES (?,'open',?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        utc_now_iso(), str(severity or "normal"), str(source),
                        int(memory_id), int(peer_id),
                        int(conflict_id) if conflict_id is not None else None,
                        str(notice_type or "semantic_evidence"), str(title),
                        str(message), json.dumps(payload or {}, ensure_ascii=False),
                        dedupe_key, int(left_version), int(right_version),
                    ),
                )
                if cur.lastrowid is None:
                    raise sqlite3.Error("semantic notice insert did not return an id")
                return {"outcome": "created", "notice_id": int(cur.lastrowid)}
        except sqlite3.IntegrityError:
            if not dedupe_key:
                return {"outcome": "error", "reason": "integrity_constraint"}
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT id FROM semantic_notices WHERE dedupe_key=?", (dedupe_key,)
                ).fetchone()
            return (
                {"outcome": "deduped", "notice_id": int(row["id"])}
                if row else {"outcome": "error", "reason": "integrity_constraint"}
            )

    @staticmethod
    def _decode(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        notice = dict(row)
        try:
            notice["payload"] = json.loads(notice.get("payload") or "{}")
        except (TypeError, json.JSONDecodeError):
            notice["payload"] = {}
        return notice

    @staticmethod
    def _workspace_clause(workspace: Optional[str]) -> tuple[str, list[Any]]:
        if workspace is None:
            return "", []
        visible = "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)=?"
        return (
            " AND EXISTS(SELECT 1 FROM memories m WHERE m.id=semantic_notices.memory_id AND "
            + visible + ") AND EXISTS(SELECT 1 FROM memories m WHERE m.id=semantic_notices.peer_id AND "
            + visible + ")",
            [workspace, workspace],
        )

    @staticmethod
    def _memories(conn: sqlite3.Connection, notice: dict[str, Any]) -> dict[int, dict[str, Any]]:
        ids = [int(notice[key]) for key in ("memory_id", "peer_id") if notice.get(key) is not None]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id,status,version,workspace,workspace_canonical FROM memories WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {int(row["id"]): dict(row) for row in rows}

    @classmethod
    def _freshness(cls, conn: sqlite3.Connection, notice: dict[str, Any]) -> dict[str, Any]:
        memories = cls._memories(conn, notice)
        checks: list[dict[str, Any]] = []
        fresh = True
        for side, id_key, version_key in (
            ("left", "memory_id", "left_version"),
            ("right", "peer_id", "right_version"),
        ):
            memory_id = notice.get(id_key)
            memory = memories.get(int(memory_id)) if memory_id is not None else None
            expected = notice.get(version_key)
            side_fresh = bool(
                memory and memory.get("status") == "active" and expected is not None
                and int(memory.get("version") or 0) == int(expected)
            )
            checks.append({
                "side": side, "memory_id": memory_id,
                "expected_version": expected,
                "current_version": memory.get("version") if memory else None,
                "status": memory.get("status") if memory else None,
                "fresh": side_fresh,
            })
            fresh = fresh and side_fresh
        return {"fresh": fresh, "checks": checks}

    @staticmethod
    def _read_call(memory: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if memory is None:
            return None
        data: dict[str, Any] = {"memory_id": int(memory["id"])}
        workspace = str(memory.get("workspace_canonical") or memory.get("workspace") or "").strip()
        if workspace:
            data["workspace"] = workspace
        return {"tool": "memory", "action": "read", "data": data}

    def _with_review_calls(self, conn: sqlite3.Connection, notice: dict[str, Any]) -> dict[str, Any]:
        memories = self._memories(conn, notice)
        notice["left_read_call"] = self._read_call(memories.get(int(notice["memory_id"])))
        peer_id = notice.get("peer_id")
        notice["right_read_call"] = self._read_call(memories.get(int(peer_id))) if peer_id is not None else None
        notice["agent_instruction"] = (
            "Read both full memories before deciding. Dismiss false positives, "
            "resolve handled ones, and escalate a credible contradiction you "
            "have verified against both complete memories into a formal "
            "conflict; do not treat this advisory notice as confirmed by itself."
        )
        return notice

    def _mark_stale(self, conn: sqlite3.Connection, notice_id: int) -> None:
        conn.execute(
            "UPDATE semantic_notices SET status='stale',resolved_at=? "
            "WHERE id=? AND status='open' AND delivered_at IS NULL",
            (utc_now_iso(), int(notice_id)),
        )

    def claim_next_semantic_notice(self, workspace_canonical: Optional[str] = None) -> Optional[dict[str, Any]]:
        db = self._db
        if not db.db_available or not db.state.sqlite_writable:
            return None
        workspace_sql, workspace_args = self._workspace_clause(workspace_canonical)
        where = " WHERE status='open' AND delivered_at IS NULL" + workspace_sql
        order = (
            " ORDER BY CASE lower(severity) WHEN 'critical' THEN 0 WHEN 'high' THEN 1 "
            "WHEN 'warning' THEN 2 WHEN 'normal' THEN 3 WHEN 'info' THEN 4 ELSE 5 END,created_at,id LIMIT ?"
        )
        with db.connection() as conn:
            if conn.execute("SELECT id FROM semantic_notices" + where + order, (*workspace_args, 1)).fetchone() is None:
                return None
        with db.write_transaction() as conn:
            rows = conn.execute(
                "SELECT id,severity,notice_type,memory_id,peer_id,left_version,right_version "
                "FROM semantic_notices" + where + order,
                (*workspace_args, NOTICE_SCAN_LIMIT),
            ).fetchall()
            for row in rows:
                notice = dict(row)
                if not self._freshness(conn, notice)["fresh"]:
                    self._mark_stale(conn, int(notice["id"]))
                    continue
                cur = conn.execute(
                    "UPDATE semantic_notices SET delivered_at=? WHERE id=? AND status='open' AND delivered_at IS NULL",
                    (utc_now_iso(), int(notice["id"])),
                )
                if cur.rowcount != 1:
                    continue
                data: dict[str, Any] = {"action": "read", "notice_id": int(notice["id"])}
                if workspace_canonical is not None:
                    data["workspace"] = workspace_canonical
                return {
                    "notice_id": int(notice["id"]), "severity": notice["severity"],
                    "type": notice["notice_type"], "action_required": "read_semantic_notice",
                    "read_call": {"tool": "memory_repair", "task": "notice", "data": data},
                }
        return None

    def read_semantic_notice(self, notice_id: int, workspace_canonical: Optional[str] = None) -> Optional[dict[str, Any]]:
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_notices WHERE id=?" + workspace_sql,
                (int(notice_id), *args),
            ).fetchone()
            if row is None:
                return None
            notice = self._decode(row)
            notice["freshness"] = self._freshness(conn, notice)
            return self._with_review_calls(conn, notice)

    def list_semantic_notices(self, status: str = "open", limit: int = 10, workspace_canonical: Optional[str] = None) -> list[dict[str, Any]]:
        status = str(status).lower()
        if status not in NOTICE_STATUSES:
            raise ValueError("invalid_notice_status")
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_notices WHERE status=?" + workspace_sql
                + " ORDER BY created_at DESC,id DESC LIMIT ?",
                (status, *args, max(1, min(100, int(limit)))),
            ).fetchall()
            notices = [self._decode(row) for row in rows]
            for notice in notices:
                notice["freshness"] = self._freshness(conn, notice)
            return notices

    def semantic_notice_counts(self, workspace_canonical: Optional[str] = None) -> dict[str, int]:
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT status,COUNT(*) AS count FROM semantic_notices WHERE 1=1"
                    + workspace_sql + " GROUP BY status", args,
                ).fetchall()
            return {str(row["status"]): int(row["count"]) for row in rows}
        except sqlite3.Error:
            return {}

    def is_semantic_pair_closed(
        self, left_id: int, right_id: int, left_version: Optional[int] = None,
        right_version: Optional[int] = None, notice_type: str = "semantic_evidence",
    ) -> bool:
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """SELECT memory_id,left_version,right_version FROM semantic_notices
                       WHERE status IN ('dismissed','resolved') AND notice_type=?
                         AND ((memory_id=? AND peer_id=?) OR (memory_id=? AND peer_id=?))""",
                    (notice_type, int(left_id), int(right_id), int(right_id), int(left_id)),
                ).fetchall()
            for row in rows:
                if left_version is None or right_version is None:
                    return True
                if int(row["memory_id"]) == int(left_id):
                    matched = int(row["left_version"]) == int(left_version) and int(row["right_version"]) == int(right_version)
                else:
                    matched = int(row["left_version"]) == int(right_version) and int(row["right_version"]) == int(left_version)
                if matched:
                    return True
        except sqlite3.Error:
            return False
        return False

    def update_semantic_notice_status(self, notice_id: int, status: str, reason: str = "", workspace_canonical: Optional[str] = None, conflict_id: Optional[int] = None) -> dict[str, Any]:
        status = str(status).lower()
        if status not in {"dismissed", "resolved"}:
            return {"outcome": "invalid_status"}
        workspace_sql, args = self._workspace_clause(workspace_canonical)
        column = "dismissed_at" if status == "dismissed" else "resolved_at"
        with self._db.write_transaction() as conn:
            row = conn.execute(
                "SELECT status FROM semantic_notices WHERE id=?" + workspace_sql,
                (int(notice_id), *args),
            ).fetchone()
            if row is None:
                return {"outcome": "not_found"}
            if row["status"] != "open":
                return {"outcome": "already_terminal", "status": row["status"]}
            # conflict_id backfills the link when an open notice is escalated
            # into a formal conflict row (the column stays NULL otherwise).
            conflict_sql = ", conflict_id=?" if conflict_id is not None else ""
            conflict_args: list[Any] = [int(conflict_id)] if conflict_id is not None else []
            cur = conn.execute(
                f"UPDATE semantic_notices SET status=?,{column}=?,resolution_reason=?{conflict_sql} "
                "WHERE id=? AND status='open'",
                (status, utc_now_iso(), str(reason), *conflict_args, int(notice_id)),
            )
        return {"outcome": "updated" if cur.rowcount else "already_terminal", "status": status}
