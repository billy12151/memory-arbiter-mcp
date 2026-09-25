"""Structured claims storage (0.17.0 P2-5, plan §7).

Contract (owner decisions #6 + review):
- ``claims`` is a required remember field (empty array = explicit "no
  claims"); rows land atomically with the write, pinned to the memory's
  ``memory_version`` — an edit bumps the version and the old claims stop
  matching (they are kept for audit, filtered out by every live query).
- attr vectors live ONLY in the ``memory_claim_vec`` vec0 table (created at
  wiring alongside the other vec tables, dim = embedder.dim); no JSON-float
  mirror column — review R1-7: dual storage invites drift, and no existing
  vec store keeps a text copy.
- The zero-Qwen claims channel compares value_norm AFTER the attr vector
  gate; normalization (attr_norm/value_norm) is computed by the pipeline
  using the same normalize_attribute/normalize_value as the text channel,
  so 半秒 vs 500ms folds identically in both channels.
- Coexistence (review A4): a side whose OWN claims carry one attr_norm with
  multiple different value_norms (day/night dual values) never fires — a
  self-coexisting attribute is not an opposing claim.
"""

from __future__ import annotations

import sqlite3
from typing import Any, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB

CLAIMS_MAX_PER_MEMORY = 20
CLAIM_SOURCES = ("agent", "backfill")


def claims_ddl() -> str:
    return """
CREATE TABLE IF NOT EXISTS memory_claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  memory_version INTEGER NOT NULL,
  attr TEXT NOT NULL, attr_norm TEXT NOT NULL,
  value TEXT NOT NULL, value_norm TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'agent' CHECK(source IN ('agent','backfill')),
  created_at TEXT NOT NULL,
  UNIQUE(memory_id, memory_version, attr_norm, value_norm)
);
CREATE INDEX IF NOT EXISTS memory_claims_attr_norm_idx
  ON memory_claims(attr_norm);
"""


class ClaimsStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    @property
    def _db_available(self) -> bool:
        return self._db._db_available

    def insert(
        self,
        *,
        memory_id: int,
        memory_version: int,
        claims: list[dict[str, Any]],
        conn: "sqlite3.Connection | None" = None,
    ) -> dict[str, Any]:
        """Insert normalized claim rows; UNIQUE dedupes exact repeats.

        ``claims`` items carry attr/attr_norm/value/value_norm and an optional
        ``source`` ("agent" default, coerced on invalid values). Pass
        ``conn`` to run INSIDE an existing write transaction (pipeline
        path); a standalone BEGIN IMMEDIATE transaction otherwise. The
        caller is responsible for capping ``claims`` at
        CLAIMS_MAX_PER_MEMORY — this method inserts what it is given.
        """
        if not self._db_available or not self._db.state.sqlite_writable:
            return {"outcome": "unavailable", "written": 0}
        now = utc_now_iso()
        written = 0

        def _insert(target: sqlite3.Connection) -> None:
            nonlocal written
            for claim in claims:
                source = claim.get("source") or "agent"
                if source not in CLAIM_SOURCES:
                    source = "agent"
                cur = target.execute(
                    """INSERT OR IGNORE INTO memory_claims(
                         memory_id, memory_version, attr, attr_norm,
                         value, value_norm, source, created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        int(memory_id),
                        int(memory_version),
                        str(claim["attr"]),
                        str(claim["attr_norm"]),
                        str(claim["value"]),
                        str(claim["value_norm"]),
                        source,
                        now,
                    ),
                )
                written += int(cur.rowcount or 0)

        try:
            if conn is not None:
                _insert(conn)
                return {"outcome": "ok", "written": written}
            with self._db.write_transaction() as own:
                _insert(own)
                return {"outcome": "ok", "written": written}
        except sqlite3.Error as exc:
            return {"outcome": "error", "error": str(exc), "written": 0}

    def current_claims(self, memory_id: int) -> list[dict[str, Any]]:
        """Claims pinned to the memory's CURRENT version (active only)."""
        if not self._db_available:
            return []
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """SELECT c.* FROM memory_claims c
                       JOIN memories m ON m.id = c.memory_id
                       WHERE c.memory_id = ?
                         AND c.memory_version = m.version
                         AND m.status = 'active'
                       ORDER BY c.id""",
                    (int(memory_id),),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def claims_for_version(self, memory_id: int, memory_version: int) -> list[dict[str, Any]]:
        """Claims pinned to an EXACT version (used by edit-time inheritance)."""
        if not self._db_available:
            return []
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT c.* FROM memory_claims c"
                    " WHERE c.memory_id = ? AND c.memory_version = ?"
                    " ORDER BY c.id",
                    (int(memory_id), int(memory_version)),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def attr_conflict_candidates(
        self,
        *,
        attr_norm: str,
        exclude_memory_id: int,
        limit: int = 100,
        conn: "sqlite3.Connection | None" = None,
    ) -> list[dict[str, Any]]:
        """Other active memories' current claims on the same attr_norm.

        Exact-key deterministic lane for the claims channel
        (check_claims_conflicts): peers whose attr_norm is IDENTICAL need
        no vector gate — unlike the KNN leg this is not bounded by the
        k=10 window and does not require sqlite-vec. Still bounded (hot
        attrs on large libraries); callers detect truncation via
        ``len(rows) == limit`` and must surface it (never silent).
        Caller may pass ``conn`` to reuse an open read connection
        (hot-loop discipline).
        """
        if not self._db_available:
            return []

        def _query(target: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = target.execute(
                """SELECT c.* FROM memory_claims c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE c.attr_norm = ?
                     AND c.memory_id != ?
                     AND c.memory_version = m.version
                     AND m.status = 'active'
                   ORDER BY c.memory_id, c.id
                   LIMIT ?""",
                (attr_norm, int(exclude_memory_id), int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]

        try:
            if conn is not None:
                return _query(conn)
            with self._db.connection() as own:
                return _query(own)
        except sqlite3.Error:
            return []

    def coexisting_values(self, memory_id: int, attr_norm: str) -> list[str]:
        """Review A4: distinct value_norms the memory ITSELF declares for one
        attr — more than one means self-coexistence, not opposition."""
        claims = self.current_claims(memory_id)
        values: list[str] = []
        for claim in claims:
            if claim["attr_norm"] == attr_norm and claim["value_norm"] not in values:
                values.append(str(claim["value_norm"]))
        return values

    def coverage(self) -> dict[str, int]:
        """Doctor hook: active memories with current-version claims."""
        if not self._db_available:
            return {"with_claims": 0, "active_memories": 0}
        try:
            with self._db.connection() as conn:
                active = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE status='active'"
                ).fetchone()
                with_claims = conn.execute(
                    """SELECT COUNT(DISTINCT c.memory_id) AS c FROM memory_claims c
                       JOIN memories m ON m.id = c.memory_id
                       WHERE m.status='active' AND c.memory_version = m.version"""
                ).fetchone()
                return {
                    "with_claims": int(with_claims["c"]) if with_claims else 0,
                    "active_memories": int(active["c"]) if active else 0,
                }
        except sqlite3.Error:
            return {"with_claims": 0, "active_memories": 0}
