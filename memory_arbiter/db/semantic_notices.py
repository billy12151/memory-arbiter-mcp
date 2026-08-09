"""Semantic notice persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


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

    def list_semantic_notices(self, status: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        db = self._db
        if not db._db_available:
            return []
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM semantic_notices WHERE status=? ORDER BY created_at DESC, id DESC LIMIT ?",
                    (str(status or "open"), max(1, min(100, int(limit)))),
                ).fetchall()
            notices = [dict(row) for row in rows]
            for notice in notices:
                try:
                    notice["payload"] = json.loads(notice.get("payload") or "{}")
                except Exception:
                    notice["payload"] = {}
            return notices
        except Exception:
            return []

    def semantic_notice_counts(self) -> dict[str, int]:
        db = self._db
        if not db._db_available:
            return {}
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS count FROM semantic_notices GROUP BY status"
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
                if left_version is None or right_version is None:
                    if lv is not None and rv is not None:
                        return True
                    continue
                if lv is None or rv is None:
                    continue
                if notice.get("memory_id") == left_id:
                    if lv == left_version and rv == right_version:
                        return True
                else:
                    if lv == right_version and rv == left_version:
                        return True
        except Exception:
            return False
        return False

    def update_semantic_notice_status(self, notice_id: int, status: str, reason: str = "") -> dict[str, Any]:
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        status = str(status or "").strip().lower()
        if status not in {"open", "delivered", "dismissed", "resolved", "stale"}:
            return {"outcome": "invalid_status"}
        column = {
            "delivered": "delivered_at",
            "dismissed": "dismissed_at",
            "resolved": "resolved_at",
            "stale": "resolved_at",
        }.get(status)
        now = utc_now_iso()
        with db.write_transaction() as conn:
            if column:
                cur = conn.execute(
                    f"UPDATE semantic_notices SET status=?, {column}=? WHERE id=?",
                    (status, now, int(notice_id)),
                )
            else:
                cur = conn.execute(
                    "UPDATE semantic_notices SET status=? WHERE id=?",
                    (status, int(notice_id)),
                )
        return {"outcome": "updated" if cur.rowcount else "not_found", "reason": reason}
