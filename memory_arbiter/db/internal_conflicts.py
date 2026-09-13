"""Same-memory internal contradiction storage (0.16.0 plan §6⑳).

The ``conflicts`` table's pair/slot invariants reject a single memory@version
appearing twice, and the write-time KNN path excludes the self memory — so
internal contradictions had NO carrier. This separate structure holds them
without touching those invariants. Deduped by (memory, version, unit pair);
an edit (version lift) naturally stales old rows.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("span_a", "span_b"):
        if isinstance(data.get(key), str):
            try:
                data[key] = json.loads(data[key])
            except (TypeError, json.JSONDecodeError):
                data[key] = None
    return data


class InternalConflictStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    def create(
        self, *, memory_id: int, memory_version: int, unit_a: int, unit_b: int,
        quote_a: str, quote_b: str, span_a: list[int], span_b: list[int],
        reason: str, detector_version: str,
        status: str = "pending", decided_reason: "str | None" = None,
    ) -> bool:
        """Insert one internal contradiction.

        ``status`` defaults to ``pending``; the write-time Qwen veto uses
        ``dismissed`` (0.16.2): a definitive semantic negative lands as a
        decided row so the scan-side re-examination's ``exists()`` probe
        cannot resurrect the pair — the veto must outlive the write.
        """
        if status not in {"pending", "dismissed"}:
            status = "pending"
        if not self._db._db_available or not self._db.state.sqlite_writable:
            return False
        now = utc_now_iso()
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO internal_conflicts(
                         memory_id,memory_version,status,unit_a,unit_b,
                         quote_a,quote_b,span_a,span_b,reason,detector_version,
                         decided_reason,decided_at,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?, ?,?, ?,?)""",
                    (
                        int(memory_id), int(memory_version), status, int(unit_a), int(unit_b),
                        quote_a, quote_b,
                        json.dumps(span_a), json.dumps(span_b),
                        reason, detector_version,
                        decided_reason, (now if decided_reason else None), now, now,
                    ),
                )
                return bool(cur.rowcount)
        except sqlite3.Error as exc:
            self._db.state.warn(f"internal_conflict create failed: {exc}")
            return False

    def exists(self, memory_id: int, memory_version: int, unit_a: int, unit_b: int) -> bool:
        if not self._db._db_available:
            return True  # degrade closed: a down DB must not duplicate rows
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM internal_conflicts WHERE memory_id=? AND memory_version=? "
                    "AND unit_a=? AND unit_b=? AND status!='stale' LIMIT 1",
                    (int(memory_id), int(memory_version), int(unit_a), int(unit_b)),
                ).fetchone()
            return row is not None
        except sqlite3.Error:
            return True

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._db._db_available:
            return []
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """SELECT i.* FROM internal_conflicts i
                       JOIN memories m ON m.id=i.memory_id
                       WHERE i.status='pending' AND m.version=i.memory_version
                         AND m.status='active'
                       ORDER BY i.created_at, i.id LIMIT ?""",
                    (max(1, int(limit)),),
                ).fetchall()
            return [_decode_row(row) for row in rows]
        except sqlite3.Error:
            return []

    def decide(self, internal_id: int, status: str, *, reason: str = "") -> dict[str, Any]:
        if status not in {"dismissed", "resolved", "stale"}:
            return {"outcome": "invalid_status"}
        if not self._db._db_available or not self._db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        now = utc_now_iso()
        with self._db.write_transaction() as conn:
            cur = conn.execute(
                "UPDATE internal_conflicts SET status=?, decided_reason=?, decided_at=?, "
                "updated_at=? WHERE id=? AND status='pending'",
                (status, reason, now, now, int(internal_id)),
            )
            if not cur.rowcount:
                return {"outcome": "not_found"}
        return {"outcome": "updated", "status": status}

    def expire_stale(self) -> int:
        """Version-lifted pending rows can no longer be judged — mark stale
        (opportunistic sweep; counts() then reports honestly)."""
        if not self._db._db_available or not self._db.state.sqlite_writable:
            return 0
        now = utc_now_iso()
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    """UPDATE internal_conflicts SET status='stale', updated_at=?
                       WHERE status='pending'
                         AND EXISTS(SELECT 1 FROM memories m
                                    WHERE m.id=internal_conflicts.memory_id
                                      AND (m.version != internal_conflicts.memory_version
                                           OR m.status != 'active'))""",
                    (now,),
                )
                return int(cur.rowcount or 0)
        except sqlite3.Error:
            return 0

    def counts(self) -> dict[str, int]:
        if not self._db._db_available:
            return {}
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS c FROM internal_conflicts GROUP BY status"
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {str(row["status"]): int(row["c"]) for row in rows}
