"""Write-time conflict candidate backlog (0.17.0 P2-4, owner design).

Contract (plan §6 + review A7):
- The ONLY writer is process_conflicts' truncation early-exit: unexamined
  candidates (by-peer leftovers + collected-but-unvisited rows) land here
  instead of being silently dropped; the truncation receipt reports
  ``backlogged: N``.
- ``candidate_key_hash`` includes the CONFLICT_DETECTOR_VERSION plus both
  members@version and their row anchors — a detector bump (threshold
  recalibration) or a member edit invalidates the frozen pair (A7: stale
  extractions must never be replayed under new gates).
- 500-row cap with lowest-pair_score eviction: under a sustained write burst
  the backlog starvation goes from silent-and-unbounded to visible-and-
  bounded (owner intent). Evictions are counted, never hidden.
- The semantic worker consumes pending rows ONLY when the job queue is empty
  (new writes always win); a stored extraction replays through the
  deterministic gates without a fresh Qwen call.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB

CONFLICT_BACKLOG_MAX = 500  # owner-fixed cap; eviction counter reports overflow

BACKLOG_STATUSES = ("pending", "done", "stale")


def conflict_backlog_ddl() -> str:
    return """
CREATE TABLE IF NOT EXISTS conflict_backlog (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_key_hash TEXT NOT NULL UNIQUE,
  left_memory_id INTEGER NOT NULL, left_version INTEGER NOT NULL,
  right_memory_id INTEGER NOT NULL, right_version INTEGER NOT NULL,
  left_text TEXT NOT NULL, right_text TEXT NOT NULL,
  pair_score REAL NOT NULL DEFAULT 0,
  extraction TEXT,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK(status IN ('pending','done','stale')),
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conflict_backlog_status_score_idx
  ON conflict_backlog(status, pair_score DESC);
"""


class ConflictBacklogStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    @property
    def _db_available(self) -> bool:
        return self._db._db_available

    def enqueue(
        self,
        *,
        candidate_key_hash: str,
        left_memory_id: int,
        left_version: int,
        right_memory_id: int,
        right_version: int,
        left_text: str,
        right_text: str,
        pair_score: float,
        extraction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Idempotent insert; evicts the lowest-score pending row past cap."""
        if not self._db_available or not self._db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        now = utc_now_iso()
        evicted = 0
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO conflict_backlog(
                         candidate_key_hash,left_memory_id,left_version,
                         right_memory_id,right_version,left_text,right_text,
                         pair_score,extraction,status,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?)""",
                    (
                        candidate_key_hash,
                        int(left_memory_id),
                        int(left_version),
                        int(right_memory_id),
                        int(right_version),
                        left_text,
                        right_text,
                        float(pair_score),
                        json.dumps(extraction, ensure_ascii=False)
                        if extraction
                        else None,
                        now,
                        now,
                    ),
                )
                outcome = "queued" if cur.rowcount else "duplicate"
                # Cap enforcement AFTER insert: evict the lowest-score pending
                # rows beyond the cap (the freshest candidate CAN lose the
                # race if it scores lowest — that loss is counted, not hidden).
                cur = conn.execute(
                    """DELETE FROM conflict_backlog WHERE status='pending' AND id IN (
                         SELECT id FROM conflict_backlog WHERE status='pending'
                         ORDER BY pair_score ASC, id ASC
                         LIMIT max(0, (SELECT COUNT(*) FROM conflict_backlog
                                       WHERE status='pending') - ?)
                       )""",
                    (CONFLICT_BACKLOG_MAX,),
                )
                evicted = int(cur.rowcount or 0)
                return {"outcome": outcome, "evicted": evicted}
        except sqlite3.IntegrityError:
            return {"outcome": "duplicate", "evicted": 0}
        except sqlite3.Error as exc:
            return {"outcome": "error", "error": str(exc), "evicted": 0}

    def take_next(self, exclude_ids: "list[int] | None" = None) -> dict[str, Any] | None:
        """Highest-score pending row (worker idle path); marks nothing —
        the caller completes or re-queues it. ``exclude_ids`` lets an idle
        drain SKIP entries it cannot process this pass (no backend) without
        freezing on the queue head (adversarial review P2-4)."""
        if not self._db_available:
            return None
        exclusion = ""
        params: list[Any] = []
        if exclude_ids:
            marks = ",".join("?" for _ in exclude_ids)
            exclusion = f" AND id NOT IN ({marks})"
            params.extend(int(i) for i in exclude_ids)
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    f"""SELECT * FROM conflict_backlog
                       WHERE status='pending'{exclusion}
                       ORDER BY pair_score DESC, id ASC LIMIT 1""",
                    params,
                ).fetchone()
                if row is None:
                    return None
                data = dict(row)
                if isinstance(data.get("extraction"), str):
                    try:
                        data["extraction"] = json.loads(data["extraction"])
                    except (TypeError, json.JSONDecodeError):
                        data["extraction"] = None
                return data
        except sqlite3.Error:
            return None

    def complete(self, backlog_id: int) -> bool:
        if not self._db_available or not self._db.state.sqlite_writable:
            return False
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    "UPDATE conflict_backlog SET status='done', updated_at=? WHERE id=?",
                    (utc_now_iso(), int(backlog_id)),
                )
                return bool(cur.rowcount)
        except sqlite3.Error:
            return False

    def refresh_stale(self) -> int:
        """Expire pending rows whose pinned member versions drifted (edit or
        status change on either side) — mirror of scan_queue.refresh_stale_pins."""
        if not self._db_available or not self._db.state.sqlite_writable:
            return 0
        now = utc_now_iso()
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    """UPDATE conflict_backlog SET status='stale', updated_at=?
                       WHERE status='pending' AND EXISTS (
                         SELECT 1 FROM memories m
                         WHERE (m.id = conflict_backlog.left_memory_id
                                AND (m.version != conflict_backlog.left_version
                                     OR m.status != 'active'))
                            OR (m.id = conflict_backlog.right_memory_id
                                AND (m.version != conflict_backlog.right_version
                                     OR m.status != 'active'))
                       )""",
                    (now,),
                )
                return int(cur.rowcount or 0)
        except sqlite3.Error:
            return 0

    def counts(self) -> dict[str, int]:
        if not self._db_available:
            return {}
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS c FROM conflict_backlog GROUP BY status"
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {str(row["status"]): int(row["c"]) for row in rows}
