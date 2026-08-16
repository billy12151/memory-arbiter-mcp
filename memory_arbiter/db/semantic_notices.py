"""Semantic notice persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


NOTICE_STATUSES = frozenset({"open", "dismissed", "resolved", "stale"})
CLAIM_SCAN_LIMIT = 25


class SemanticNoticeStore:
    def __init__(self, db: "MemoryDB"):
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
        left_claim_revision: Optional[int] = None,
        right_claim_revision: Optional[int] = None,
        source: str = "semantic_write_gate",
    ) -> dict[str, Any]:
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        now = utc_now_iso()
        with db.write_transaction() as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO semantic_notices(
                        created_at, status, severity, source, memory_id, peer_id,
                        conflict_id, notice_type, title, message, payload, dedupe_key,
                        left_version, right_version, left_claim_revision, right_claim_revision
                    ) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now, str(severity or "normal"), str(source or "semantic_write_gate"),
                        int(memory_id), int(peer_id) if peer_id is not None else None,
                        int(conflict_id) if conflict_id is not None else None,
                        str(notice_type or "semantic_candidate"), str(title or "Semantic notice"),
                        str(message or ""), json.dumps(payload or {}, ensure_ascii=False),
                        dedupe_key, left_version, right_version,
                        left_claim_revision, right_claim_revision,
                    ),
                )
                assert cur.lastrowid is not None
                return {"outcome": "created", "notice_id": int(cur.lastrowid)}
            except sqlite3.IntegrityError:
                if not dedupe_key:
                    return {"outcome": "error", "reason": "integrity_constraint"}
                row = conn.execute(
                    "SELECT id FROM semantic_notices WHERE dedupe_key=?",
                    (dedupe_key,),
                ).fetchone()
                if row is None:
                    return {"outcome": "error", "reason": "integrity_constraint"}
                return {"outcome": "deduped", "notice_id": int(row["id"])}

    @staticmethod
    def _decode_notice(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        notice = dict(row)
        try:
            notice["payload"] = json.loads(notice.get("payload") or "{}")
        except Exception:
            notice["payload"] = {}
        return notice

    @staticmethod
    def _freshness(notice: dict[str, Any], memories: dict[int, dict[str, Any]]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        pairs = (
            ("left", notice.get("memory_id"), notice.get("left_version"), notice.get("left_claim_revision")),
            ("right", notice.get("peer_id"), notice.get("right_version"), notice.get("right_claim_revision")),
        )
        fresh = True
        for side, memory_id, expected_version, expected_revision in pairs:
            memory = memories.get(int(memory_id)) if memory_id is not None else None
            side_fresh = bool(
                memory is not None
                and memory.get("status") == "active"
                and expected_version is not None
                and expected_revision is not None
                and int(memory.get("version") or 0) == int(expected_version)
                and int(memory.get("claim_revision") or 0) == int(expected_revision)
            )
            checks.append({
                "side": side,
                "memory_id": memory_id,
                "exists": memory is not None,
                "status": memory.get("status") if memory else None,
                "expected_version": expected_version,
                "current_version": memory.get("version") if memory else None,
                "expected_claim_revision": expected_revision,
                "current_claim_revision": memory.get("claim_revision") if memory else None,
                "fresh": side_fresh,
            })
            fresh = fresh and side_fresh
        return {"fresh": fresh, "checks": checks}

    @staticmethod
    def _notice_memories_in_connection(
        conn: sqlite3.Connection, notice: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        ids = [int(value) for value in (notice.get("memory_id"), notice.get("peer_id")) if value is not None]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id, status, version, claim_revision, workspace, workspace_canonical "
            f"FROM memories WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {int(row["id"]): dict(row) for row in rows}

    def _freshness_in_connection(self, conn: sqlite3.Connection, notice: dict[str, Any]) -> dict[str, Any]:
        return self._freshness(notice, self._notice_memories_in_connection(conn, notice))

    @staticmethod
    def _read_call(memory: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if memory is None:
            return None
        data: dict[str, Any] = {"memory_id": int(memory["id"])}
        workspace = str(memory.get("workspace_canonical") or memory.get("workspace") or "").strip()
        if workspace:
            data["workspace"] = workspace
        return {"tool": "memory", "action": "read", "data": data}

    def _with_review_calls(
        self, conn: sqlite3.Connection, notice: dict[str, Any],
    ) -> dict[str, Any]:
        memories = self._notice_memories_in_connection(conn, notice)
        notice["left_read_call"] = self._read_call(memories.get(int(notice["memory_id"])))
        peer_id = notice.get("peer_id")
        notice["right_read_call"] = self._read_call(
            memories.get(int(peer_id)) if peer_id is not None else None
        )
        notice["agent_instruction"] = (
            "Execute left_read_call and right_read_call and read both full memories. "
            "Only after both reads succeed, assess the advisory candidate and tell the user if it appears credible. "
            "Do not present it as a confirmed conflict. Dismiss false positives or resolve notices already handled."
        )
        return notice

    @staticmethod
    def _workspace_clause(workspace_canonical: Optional[str], alias: str = "semantic_notices") -> tuple[str, list[Any]]:
        if workspace_canonical is None:
            return "", []
        visible = "COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
        return (
            f" AND EXISTS (SELECT 1 FROM memories m WHERE m.id={alias}.memory_id AND {visible})"
            f" AND EXISTS (SELECT 1 FROM memories m WHERE m.id={alias}.peer_id AND {visible})",
            [workspace_canonical, workspace_canonical],
        )

    def _mark_stale_in_connection(self, conn: sqlite3.Connection, notice_id: int) -> bool:
        """Internal-only freshness transition; public status updates cannot set stale."""
        cur = conn.execute(
            "UPDATE semantic_notices SET status='stale', resolved_at=? "
            "WHERE id=? AND status='open' AND delivered_at IS NULL",
            (utc_now_iso(), int(notice_id)),
        )
        return cur.rowcount == 1

    def claim_next_semantic_notice(self, workspace_canonical: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Atomically claim one fresh notice, scanning at most CLAIM_SCAN_LIMIT rows."""
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return None
        workspace_sql, workspace_args = self._workspace_clause(workspace_canonical)
        from_sql = (
            " FROM semantic_notices INDEXED BY idx_semantic_notices_open_undelivered_priority "
            "WHERE status='open' AND delivered_at IS NULL"
            + workspace_sql
        )
        order_sql = (
            " ORDER BY CASE lower(severity) "
            "WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'warning' THEN 2 "
            "WHEN 'normal' THEN 3 WHEN 'info' THEN 4 ELSE 5 END, created_at ASC, id ASC LIMIT ?"
        )
        # Empty queues are the common path. Probe only the indexed identity with
        # a reader so successful product calls avoid both row payload reads and
        # serialization behind BEGIN IMMEDIATE.
        with db.connection() as probe_conn:
            if probe_conn.execute(
                "SELECT id" + from_sql + order_sql,
                (*workspace_args, 1),
            ).fetchone() is None:
                return None
        with db.write_transaction() as conn:
            # Re-read only claim/freshness fields under the writer lock:
            # concurrent callers may have claimed the probed row, and only this
            # snapshot is eligible for CAS delivery.
            rows = conn.execute(
                "SELECT id, severity, notice_type, memory_id, peer_id, "
                "left_version, right_version, left_claim_revision, right_claim_revision"
                + from_sql + order_sql,
                (*workspace_args, CLAIM_SCAN_LIMIT),
            ).fetchall()
            for row in rows:
                notice = self._decode_notice(row)
                if not self._freshness_in_connection(conn, notice)["fresh"]:
                    self._mark_stale_in_connection(conn, int(notice["id"]))
                    continue
                now = utc_now_iso()
                cur = conn.execute(
                    "UPDATE semantic_notices SET delivered_at=? "
                    "WHERE id=? AND status='open' AND delivered_at IS NULL",
                    (now, int(notice["id"])),
                )
                if cur.rowcount != 1:
                    continue
                read_data: dict[str, Any] = {
                    "action": "read",
                    "notice_id": int(notice["id"]),
                }
                if workspace_canonical is not None:
                    read_data["workspace"] = workspace_canonical
                return {
                    "notice_id": int(notice["id"]),
                    "severity": notice["severity"],
                    "type": notice["notice_type"],
                    "action_required": "read_semantic_notice",
                    "agent_instruction": (
                        "Read this notice explicitly. After reviewing both sides, if the candidate seems credible, "
                        "tell the user clearly; dismiss false positives and resolve notices already handled. "
                        "Do not claim this notice is a confirmed conflict."
                    ),
                    "read_call": {
                        "tool": "memory_repair",
                        "task": "notice",
                        "data": read_data,
                    },
                }
        return None

    def read_semantic_notice(
        self, notice_id: int, workspace_canonical: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        db = self._db
        if not db._db_available:
            return None
        workspace_sql, workspace_args = self._workspace_clause(workspace_canonical)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_notices WHERE id=?" + workspace_sql,
                (int(notice_id), *workspace_args),
            ).fetchone()
            if row is None:
                return None
            notice = self._decode_notice(row)
            notice["freshness"] = self._freshness_in_connection(conn, notice)
            return self._with_review_calls(conn, notice)

    def list_semantic_notices(
        self, status: str = "open", limit: int = 10, workspace_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        db = self._db
        if not db._db_available:
            return []
        normalized = str(status or "").strip().lower()
        if normalized not in NOTICE_STATUSES:
            raise ValueError("invalid_notice_status")
        workspace_sql, workspace_args = self._workspace_clause(workspace_canonical)
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_notices WHERE status=?" + workspace_sql
                + " ORDER BY created_at DESC, id DESC LIMIT ?",
                (normalized, *workspace_args, max(1, min(100, int(limit)))),
            ).fetchall()
            notices = [self._decode_notice(row) for row in rows]
            for notice in notices:
                notice["freshness"] = self._freshness_in_connection(conn, notice)
            return notices

    def semantic_notice_counts(
        self, workspace_canonical: Optional[str] = None,
    ) -> dict[str, int]:
        db = self._db
        if not db._db_available:
            return {}
        workspace_sql, workspace_args = self._workspace_clause(workspace_canonical)
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS count FROM semantic_notices WHERE 1=1"
                    + workspace_sql + " GROUP BY status",
                    workspace_args,
                ).fetchall()
            return {str(row["status"]): int(row["count"] or 0) for row in rows}
        except Exception:
            return {}

    def is_semantic_pair_closed(
        self,
        left_id: int,
        right_id: int,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        notice_type: str = "semantic_pair",
        left_claim_revision: Optional[int] = None,
        right_claim_revision: Optional[int] = None,
    ) -> bool:
        db = self._db
        if not db._db_available:
            return False
        a, b = sorted([int(left_id), int(right_id)])
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM semantic_notices
                    WHERE status IN ('dismissed','resolved')
                      AND notice_type=?
                      AND ((memory_id=? AND peer_id=?) OR (memory_id=? AND peer_id=?))
                    """,
                    (notice_type, a, b, b, a),
                ).fetchall()
            for row in rows:
                notice = dict(row)
                lv = notice.get("left_version")
                rv = notice.get("right_version")
                lcr = notice.get("left_claim_revision")
                rcr = notice.get("right_claim_revision")
                if left_version is None or right_version is None:
                    if lv is not None and rv is not None and lcr is not None and rcr is not None:
                        return True
                    continue
                if lv is None or rv is None or lcr is None or rcr is None:
                    continue
                if left_claim_revision is None or right_claim_revision is None:
                    continue
                if int(notice.get("memory_id")) == int(left_id):
                    if (
                        int(lv) == int(left_version)
                        and int(rv) == int(right_version)
                        and int(lcr) == int(left_claim_revision)
                        and int(rcr) == int(right_claim_revision)
                    ):
                        return True
                elif (
                    int(lv) == int(right_version)
                    and int(rv) == int(left_version)
                    and int(lcr) == int(right_claim_revision)
                    and int(rcr) == int(left_claim_revision)
                ):
                    return True
        except Exception:
            return False
        return False

    def update_semantic_notice_status(
        self,
        notice_id: int,
        status: str,
        reason: str = "",
        workspace_canonical: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        status = str(status or "").strip().lower()
        if status not in {"dismissed", "resolved"}:
            return {"outcome": "invalid_status"}
        workspace_sql, workspace_args = self._workspace_clause(workspace_canonical)
        column = "dismissed_at" if status == "dismissed" else "resolved_at"
        now = utc_now_iso()
        with db.write_transaction() as conn:
            row = conn.execute(
                "SELECT status FROM semantic_notices WHERE id=?" + workspace_sql,
                (int(notice_id), *workspace_args),
            ).fetchone()
            if row is None:
                return {"outcome": "not_found"}
            current = str(row["status"])
            if current != "open":
                return {"outcome": "already_terminal", "status": current}
            cur = conn.execute(
                f"UPDATE semantic_notices SET status=?, {column}=?, resolution_reason=? "
                "WHERE id=? AND status='open'",
                (status, now, str(reason or ""), int(notice_id)),
            )
        return {"outcome": "updated" if cur.rowcount else "already_terminal", "status": status}
