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

    def list_pending_grouped(
        self, limit_memories: int = 3, pairs_cap: int = 8,
    ) -> list[dict[str, Any]]:
        """0.16.4 §3: per-memory aggregation for the judgment page.

        The judgment unit is the MEMORY (one disposition clears a whole
        memory), so the page shows one aggregated item per memory: a capped
        pair preview, the FULL pair_count, and the reason distribution.
        Current-version active memories only — the same JOIN predicate as
        list_pending; drifted rows never render.
        """
        if not self._db._db_available:
            return []
        try:
            with self._db.connection() as conn:
                mem_rows = conn.execute(
                    """SELECT i.memory_id, COUNT(*) AS pair_count, MIN(i.created_at) AS first_at
                       FROM internal_conflicts i JOIN memories m ON m.id=i.memory_id
                       WHERE i.status='pending' AND m.version=i.memory_version
                         AND m.status='active'
                       GROUP BY i.memory_id ORDER BY first_at, i.memory_id LIMIT ?""",
                    (max(1, int(limit_memories)),),
                ).fetchall()
                groups: list[dict[str, Any]] = []
                for mem in mem_rows:
                    mid = int(mem["memory_id"])
                    rows = conn.execute(
                        """SELECT * FROM internal_conflicts
                           WHERE memory_id=? AND status='pending'
                             AND memory_version=(SELECT version FROM memories WHERE id=?)
                           ORDER BY created_at, id LIMIT ?""",
                        (mid, mid, max(1, int(pairs_cap))),
                    ).fetchall()
                    dist = conn.execute(
                        """SELECT reason, COUNT(*) AS c FROM internal_conflicts
                           WHERE memory_id=? AND status='pending'
                             AND memory_version=(SELECT version FROM memories WHERE id=?)
                           GROUP BY reason""",
                        (mid, mid),
                    ).fetchall()
                    groups.append({
                        "memory_id": mid,
                        "pair_count": int(mem["pair_count"]),
                        "pairs": [_decode_row(row) for row in rows],
                        "reasons_summary": {str(r["reason"]): int(r["c"]) for r in dist},
                    })
            return groups
        except sqlite3.Error:
            return []

    def pending_pair_count(self, memory_id: int) -> int:
        """Current-version pending pair count for one memory (guard input)."""
        if not self._db._db_available:
            return 0
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    """SELECT COUNT(*) FROM internal_conflicts i
                       JOIN memories m ON m.id=i.memory_id
                       WHERE i.memory_id=? AND i.status='pending'
                         AND m.version=i.memory_version AND m.status='active'""",
                    (int(memory_id),),
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def decide_memory(self, memory_id: int, status: str, *, reason: str = "") -> dict[str, Any]:
        """0.16.4 §3: judge one memory's WHOLE pending set in one call.

        The version guard updates only current-version active rows — drifted
        rows stay for expire_stale (decided vs stale audit separation). A
        dismissed/resolved set then blocks scan re-examination via exists().
        """
        if status not in {"dismissed", "resolved"}:
            return {"outcome": "invalid_status"}
        if not self._db._db_available or not self._db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        now = utc_now_iso()
        with self._db.write_transaction() as conn:
            cur = conn.execute(
                """UPDATE internal_conflicts SET status=?, decided_reason=?, decided_at=?, updated_at=?
                   WHERE memory_id=? AND status='pending'
                     AND EXISTS(SELECT 1 FROM memories m
                                WHERE m.id=internal_conflicts.memory_id
                                  AND m.version=internal_conflicts.memory_version
                                  AND m.status='active')""",
                (status, reason, now, now, int(memory_id)),
            )
            updated = int(cur.rowcount or 0)
        if not updated:
            try:
                with self._db.connection() as conn:
                    drifted = conn.execute(
                        "SELECT 1 FROM internal_conflicts WHERE memory_id=? AND status='pending' LIMIT 1",
                        (int(memory_id),),
                    ).fetchone()
            except sqlite3.Error:
                drifted = None
            if drifted is not None:
                return {"outcome": "stale_snapshot", "updated": 0}
            return {"outcome": "not_found", "updated": 0}
        return {"outcome": status, "updated": updated}

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
