"""Independent judgment queue for scan-suspected items (0.16.0 plan §6㉑).

Design contract (owner-ratified, seven-round ruling):
- ALL scan-suspected items live here — never in ``conflicts`` (whose pair/slot
  invariants cannot host value-less candidates and whose ``candidate`` rows
  became ghost notifications), never in the notice channel, never in any
  user-facing conflict list.
- Visibility: agent judgment pages only. doctor/console report a backlog
  COUNT and a "let your agent read the queue" hint — nothing more.
- Dismissal dedupe is structural: a dismissed pair@version lands in
  ``conflicts`` as ``not_a_conflict`` (the suppression source), after which
  the scan never re-enqueues the same identity (candidate_key_hash UNIQUE
  here + the suppression lookup in the pairing engine).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB

QUEUE_STATUSES = ("pending", "in_review", "confirmed", "dismissed", "voided", "expired")
QUEUE_KINDS = ("conflict", "internal", "workspace")


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("member_versions", "evidence", "detail"):
        if isinstance(data.get(key), str):
            try:
                data[key] = json.loads(data[key])
            except (TypeError, json.JSONDecodeError):
                data[key] = None
    return data


class ScanQueueStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    @property
    def _db_available(self) -> bool:
        return self._db._db_available

    def enqueue(
        self,
        *,
        kind: str,
        workspace_canonical: str,
        candidate_key_hash: str,
        member_versions: list[dict[str, Any]],
        evidence: list[dict[str, Any]] | None,
        reason: str,
        severity: str | None,
        source: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert one suspected item; ``candidate_key_hash`` UNIQUE makes
        re-enumeration idempotent (a rescan of the same pair@version is a
        no-op, not a duplicate row)."""
        if kind not in QUEUE_KINDS:
            return {"outcome": "invalid_kind"}
        if not self._db_available or not self._db.state.sqlite_writable:
            return {"outcome": "unavailable"}
        now = utc_now_iso()
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO scan_queue(
                         kind,workspace_canonical,status,candidate_key_hash,member_versions,
                         evidence,reason,severity,source,detail,created_at,updated_at)
                       VALUES(?,?,'pending',?,?,?,?,?,?,?,?,?)""",
                    (
                        kind, workspace_canonical, candidate_key_hash,
                        json.dumps(member_versions, ensure_ascii=False),
                        json.dumps(evidence or [], ensure_ascii=False),
                        reason, severity, source,
                        json.dumps(detail, ensure_ascii=False) if detail else None,
                        now, now,
                    ),
                )
                if cur.rowcount:
                    return {"outcome": "queued", "queue_id": int(cur.lastrowid or 0)}
                row = conn.execute(
                    "SELECT id,status FROM scan_queue WHERE candidate_key_hash=?",
                    (candidate_key_hash,),
                ).fetchone()
                return {
                    "outcome": "duplicate",
                    "queue_id": int(row["id"]) if row else None,
                    "status": str(row["status"]) if row else None,
                }
        except sqlite3.IntegrityError:
            return {"outcome": "duplicate"}
        except sqlite3.Error as exc:
            return {"outcome": "error", "error": str(exc)}

    def counts(self) -> dict[str, int]:
        if not self._db_available:
            return {}
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS c FROM scan_queue GROUP BY status"
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {str(row["status"]): int(row["c"]) for row in rows}

    def backlog(self) -> int:
        counts = self.counts()
        return counts.get("pending", 0) + counts.get("in_review", 0)

    def refresh_stale_pins(self) -> int:
        """Expire rows whose pinned member versions no longer match reality.

        A pinned version drifting (memory edited since enqueue) means the
        frozen evidence no longer describes the live memory — the row is
        marked ``expired`` and the pairing engine re-enqueues the CURRENT
        identity on its next pass (version lift ⇒ new candidate hash).
        """
        if not self._db_available or not self._db.state.sqlite_writable:
            return 0
        now = utc_now_iso()
        try:
            with self._db.write_transaction() as conn:
                cur = conn.execute(
                    """UPDATE scan_queue SET status='expired', updated_at=?, decided_at=?
                       WHERE status IN ('pending','in_review')
                         AND EXISTS (
                           SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                           JOIN memories AS mem
                             ON mem.id = CAST(json_extract(m.value,'$.memory_id') AS INTEGER)
                           WHERE mem.version != CAST(json_extract(m.value,'$.version') AS INTEGER)
                              OR mem.status != 'active'
                         )""",
                    (now, now),
                )
                return int(cur.rowcount or 0)
        except sqlite3.Error:
            return 0
