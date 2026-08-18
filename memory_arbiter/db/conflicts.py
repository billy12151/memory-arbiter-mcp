"""Conflict-row persistence and dismissal helpers for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional, TYPE_CHECKING

from ..models import utc_now_iso
from ..text import canon_entity as _canon_entity, canon_scope as _canon_scope

if TYPE_CHECKING:
    from .core import MemoryDB


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata", "structured_details"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


class ConflictStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    def record_conflict_on_conn(
        self,
        conn: sqlite3.Connection,
        left_id: int,
        right_id: int,
        subject: Optional[str],
        reason: str,
        winner_id: Optional[int],
        status: str = "open",
    ) -> int:
        cur = conn.execute(
            """
            INSERT INTO conflicts(left_id, right_id, subject, status, reason, winner_id, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (left_id, right_id, subject, status, reason, winner_id, utc_now_iso(), utc_now_iso() if status != "open" else None),
        )
        if cur.lastrowid is None:
            raise sqlite3.Error("conflict insert did not return an id")
        return int(cur.lastrowid)

    def record_conflict(
        self,
        left_id: int,
        right_id: int,
        subject: Optional[str],
        reason: str,
        winner_id: Optional[int],
        status: str = "open",
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[int]:
        if conn is not None:
            return self.record_conflict_on_conn(conn, left_id, right_id, subject, reason, winner_id, status)
        if not self._db_available or not self.state.sqlite_writable:
            return None
        try:
            with self.connection() as txn_conn:
                conflict_id = self.record_conflict_on_conn(txn_conn, left_id, right_id, subject, reason, winner_id, status)
                txn_conn.commit()
                return conflict_id
        except sqlite3.Error:
            return None

    def resolve_conflicts_for_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> int:
        cur = conn.execute(
            "UPDATE conflicts SET status='resolved', resolved_at=? "
            "WHERE status='open' AND (left_id=? OR right_id=?)",
            (utc_now_iso(), memory_id, memory_id),
        )
        # v0.8.8: a memory resolved-away (supersede) also obsoletes any
        # not_a_conflict (advisory dismissal) rows touching it — the
        # dismissal has no referent once the memory is superseded.
        conn.execute(
            "DELETE FROM conflicts WHERE status='not_a_conflict' "
            "AND (left_id=? OR right_id=?)",
            (memory_id, memory_id),
        )
        return int(cur.rowcount)

    def resolve_conflicts_for(
        self,
        memory_id: int,
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        if conn is not None:
            return self.resolve_conflicts_for_on_conn(conn, memory_id)
        if not self._db_available or not self.state.sqlite_writable:
            return 0
        try:
            with self.connection() as txn_conn:
                resolved = self.resolve_conflicts_for_on_conn(txn_conn, memory_id)
                txn_conn.commit()
                return resolved
        except sqlite3.Error:
            return 0

    def list_conflicts(self, status: str = "open", limit: int = 50, source: Optional[str] = None) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        with self.connection() as conn:
            select = (
                "SELECT c.*, j.verdict AS judgment_verdict, "
                "j.recommended_use AS judgment_recommended_use, "
                "j.suggested_winner AS judgment_suggested_winner, "
                "j.confidence_hint AS judgment_confidence_hint, "
                "j.reason AS judgment_reason, j.judge_type AS judgment_judge_type, "
                "j.judge_ref AS judgment_judge_ref, j.resolution_kind AS judgment_resolution_kind, "
                "j.conflict_scope AS judgment_conflict_scope, j.created_at AS judged_at "
                "FROM conflicts c LEFT JOIN conflict_judgments j "
                "ON j.id=c.active_judgment_id "
            )
            if source is None:
                rows = conn.execute(
                    select + "WHERE c.status = ? ORDER BY c.created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    select + "WHERE c.status = ? AND c.source = ? "
                    "ORDER BY c.created_at DESC LIMIT ?",
                    (status, source, limit),
                ).fetchall()
            return [_row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # v0.7.6: batch conflict-signal helpers for search attachment.
    # Both are read-only, chunked to respect SQLite's parameter limit,
    # and never raise (callers treat failure as empty).
    # ------------------------------------------------------------------

    def list_open_conflicts_for_memory_ids(
        self, memory_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Batch-fetch all open conflicts where either side is in *memory_ids*.

        Returns row_to_dict rows. Single SQL per chunk (no N+1). DB-unavailable
        or empty input → [].

        Note (v0.10.2): the resolved branch surfaces reusable *guidance* only
        for evolution/compatible verdicts with a live active judgment.
        Search must not re-litigate a pair as guidance; a contradiction-resolved
        pair is left ringable for a second look.
        """
        if not memory_ids or not self._db_available:
            return []
        unique_ids = sorted(set(int(i) for i in memory_ids if i is not None))
        if not unique_ids:
            return []
        results: list[dict[str, Any]] = []
        try:
            with self.connection() as conn:
                # chunk=250 because the query binds each id twice (left_id IN + right_id IN);
                # 2×250=500 stays under SQLite's default SQLITE_MAX_VARIABLE_NUMBER=999.
                for chunk_start in range(0, len(unique_ids), 250):
                    chunk = unique_ids[chunk_start:chunk_start + 250]
                    ph = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT c.*, j.verdict AS judgment_verdict, "
                        f"j.recommended_use AS judgment_recommended_use, "
                        f"j.suggested_winner AS judgment_suggested_winner, "
                        f"j.confidence_hint AS judgment_confidence_hint, "
                        f"j.reason AS judgment_reason, j.judge_type AS judgment_judge_type, "
                        f"j.judge_ref AS judgment_judge_ref, j.resolution_kind AS judgment_resolution_kind, "
                        f"j.conflict_scope AS judgment_conflict_scope, j.created_at AS judged_at "
                        f"FROM conflicts c LEFT JOIN conflict_judgments j ON j.id=c.active_judgment_id "
                        f"WHERE ("
                        f"(c.status='open' AND (c.left_version IS NULL OR ("
                        f"c.left_version=(SELECT version FROM memories WHERE id=c.left_id) AND "
                        f"c.right_version=(SELECT version FROM memories WHERE id=c.right_id)))) "
                        f"OR (c.status='resolved' AND c.active_judgment_id IS NOT NULL "
                        f"AND j.verdict IN ('evolution','compatible') "
                        f"AND c.left_version IS NOT NULL AND c.right_version IS NOT NULL "
                        f"AND c.left_version=(SELECT version FROM memories WHERE id=c.left_id) "
                        f"AND c.right_version=(SELECT version FROM memories WHERE id=c.right_id))"
                        f") AND (c.left_id IN ({ph}) OR c.right_id IN ({ph}))",
                        (*chunk, *chunk),
                    ).fetchall()
                    results.extend(_row_to_dict(r) for r in rows)
        except sqlite3.Error:
            return []
        return results

    def get_memory_summaries(
        self, memory_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        return self._db.audit.get_memory_summaries(memory_ids)

    # ------------------------------------------------------------------
    # Explicit conflict persistence. Candidate discovery remains advisory in
    # semantic_notices until an Agent or user decides governance is needed.
    # ------------------------------------------------------------------

    def record_conflict_enriched(
        self,
        left_id: int,
        right_id: int,
        conflict_type: Optional[str],
        conflict_point: Optional[str],
        reason: str,
        suggested_winner: Optional[int] = None,
        confidence_hint: Optional[str] = None,
        source: Optional[str] = None,
        status: str = "open",
        refresh: bool = False,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        judgment_status: Optional[str] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Insert a conflict row carrying scan-enrichment fields.

        Pairs are canonicalised to ``left_id < right_id``. Idempotent: if an
        open conflict on the same (left, right) pair already exists, no new
        row is written and ``deduped`` is returned — *unless* ``refresh=True``,
        in which case the existing row's enrichment fields are UPDATEd in place
        and ``refreshed`` is returned (``created_at`` is preserved).
        """
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable"}
        raw_left, raw_right = int(left_id), int(right_id)
        if raw_left <= raw_right:
            a, b = raw_left, raw_right
        else:
            a, b = raw_right, raw_left
            left_version, right_version = right_version, left_version
        subject = conflict_point or reason
        now = utc_now_iso()
        with self.write_transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM conflicts WHERE status='open' AND left_id=? AND right_id=?",
                (a, b),
            ).fetchone()
            if existing:
                existing_row = _row_to_dict(existing)
                priority = {
                    "metadata_write_hint": 10,
                    "llm_informed": 30,
                    "policy_informed": 35,
                    "human_confirmed": 40,
                    None: 30,
                }
                incoming_priority = priority.get(source, 25)
                existing_priority = priority.get(existing_row.get("source"), 25)
                preserve_judgment_projection = existing_row.get("active_judgment_id") is not None
                effective_left_version = left_version
                effective_right_version = right_version
                pins_changed = (
                    effective_left_version != existing_row.get("left_version")
                    or effective_right_version != existing_row.get("right_version")
                )
                reset_judgment = pins_changed and existing_row.get("active_judgment_id") is not None
                should_update = (
                    incoming_priority > existing_priority
                    or (refresh and incoming_priority >= existing_priority)
                    or reset_judgment
                )
                if should_update:
                    effective_judgment_status = (
                        "pending_llm" if reset_judgment else
                        (judgment_status if judgment_status is not None else existing_row.get("judgment_status"))
                    )
                    active_judgment_id = None if reset_judgment else existing_row.get("active_judgment_id")
                    effective_resolution_kind = None if reset_judgment else existing_row.get("resolution_kind")
                    effective_conflict_scope = None if reset_judgment else existing_row.get("conflict_scope")
                    effective_reason = (
                        existing_row.get("reason")
                        if preserve_judgment_projection else reason
                    )
                    effective_winner = (
                        existing_row.get("suggested_winner")
                        if preserve_judgment_projection else suggested_winner
                    )
                    effective_confidence = (
                        existing_row.get("confidence_hint")
                        if preserve_judgment_projection else confidence_hint
                    )
                    effective_source = (
                        existing_row.get("source")
                        if preserve_judgment_projection else source
                    )
                    cur = conn.execute(
                        """
                        UPDATE conflicts SET
                            conflict_type=?, conflict_point=?, reason=?,
                            winner_id=?, suggested_winner=?, confidence_hint=?,
                            source=?, left_version=?, right_version=?,
                            judgment_status=?, active_judgment_id=?,
                            resolution_kind=?, conflict_scope=?,
                            scan_prompt_version=?, scan_model=?, refreshed_at=?
                        WHERE id=?
                        """,
                        (
                            conflict_type, conflict_point, effective_reason,
                            effective_winner, effective_winner, effective_confidence,
                            effective_source, effective_left_version, effective_right_version,
                            effective_judgment_status, active_judgment_id,
                            effective_resolution_kind, effective_conflict_scope,
                            scan_prompt_version, scan_model, now,
                            int(existing["id"]),
                        ),
                    )
                    if cur.rowcount == 0:
                        return {"outcome": "not_open", "conflict_id": int(existing["id"])}
                    return {"outcome": "refreshed", "conflict_id": int(existing["id"])}
                return {"outcome": "deduped", "conflict_id": int(existing["id"])}
            cur = conn.execute(
                """
                INSERT INTO conflicts(
                    left_id, right_id, subject, status, reason, winner_id,
                    created_at, resolved_at,
                    conflict_type, conflict_point, suggested_winner,
                    confidence_hint, source,
                    left_version, right_version, scan_prompt_version, scan_model,
                    judgment_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a, b, subject, status, reason, suggested_winner,
                    now, now if status != "open" else None,
                    conflict_type, conflict_point, suggested_winner,
                    confidence_hint, source,
                    left_version, right_version, scan_prompt_version, scan_model,
                    judgment_status,
                ),
            )
            return {"outcome": "inserted", "conflict_id": int(cur.lastrowid)}

    def resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved",
    ) -> dict[str, Any]:
        """Close a single open conflict by id (status -> resolved or not_a_conflict).

        ``status='not_a_conflict'`` records that the pair was judged NOT a real
        conflict (advisory dismissal): write/search then skip it (Layer 0) until
        a version change invalidates the row. ``status`` must be 'resolved' or
        'not_a_conflict'. Unlike ``resolve_conflicts_for`` (which closes *all*
        conflicts touching a memory), this targets exactly one row.
        """
        if status not in ("resolved", "not_a_conflict"):
            return {"outcome": "invalid_status", "conflict_id": int(conflict_id)}
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable"}
        with self.connection() as conn:
            cur = conn.execute(
                "UPDATE conflicts SET status=?, resolved_at=?, reason=? "
                "WHERE id=? AND status='open'",
                (status, utc_now_iso(), reason, int(conflict_id)),
            )
            conn.commit()
            if cur.rowcount == 0:
                return {"outcome": "not_open", "conflict_id": int(conflict_id)}
            return {"outcome": status, "conflict_id": int(conflict_id)}

    def is_pair_dismissed(self, left_id: int, right_id: int) -> bool:
        """v0.8.8: True if (left, right) has a ``not_a_conflict`` row whose pinned
        ``left_version``/``right_version`` still match the memories' current
        versions (neither edited since dismissal). One correlated query; never
        raises (best-effort: on error returns False → fail-open, re-ring).
        """
        if not self._db_available:
            return False
        a, b = sorted((int(left_id), int(right_id)))
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM conflicts c "
                    "WHERE c.status='not_a_conflict' AND c.left_id=? AND c.right_id=? "
                    "AND (c.left_version IS NULL OR c.left_version = (SELECT version FROM memories WHERE id=c.left_id)) "
                    "AND (c.right_version IS NULL OR c.right_version = (SELECT version FROM memories WHERE id=c.right_id)) "
                    "LIMIT 1",
                    (a, b),
                ).fetchone()
                return row is not None
        except sqlite3.Error:
            return False

    def get_memory_version(self, memory_id: int) -> Optional[int]:
        """v0.8.8: current version of a memory (for conflict-row version pinning)."""
        if not self._db_available:
            return None
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT version FROM memories WHERE id=?", (int(memory_id),)
                ).fetchone()
                return int(row["version"]) if row else None
        except sqlite3.Error:
            return None


    def dismissed_pairs_for(self, memory_ids: list[int]) -> set[tuple[int, int]]:
        """v0.8.8: canonical ``(a, b)`` pairs (a<b) that are currently dismissed
        — a ``not_a_conflict`` row whose pinned versions still match the
        memories' current versions. Restricted to pairs touching *memory_ids*.
        For Layer 0 gating of the computed-overlap advisory path. Best-effort.
        """
        if not memory_ids or not self._db_available:
            return set()
        ids = sorted(set(int(i) for i in memory_ids if i is not None))
        if not ids:
            return set()
        out: set[tuple[int, int]] = set()
        try:
            with self.connection() as conn:
                ph = ",".join("?" * len(ids))
                rows = conn.execute(
                    "SELECT left_id, right_id FROM conflicts "
                    "WHERE status='not_a_conflict' "
                    f"AND (left_id IN ({ph}) OR right_id IN ({ph})) "
                    "AND (left_version IS NULL OR left_version = (SELECT version FROM memories WHERE id=conflicts.left_id)) "
                    "AND (right_version IS NULL OR right_version = (SELECT version FROM memories WHERE id=conflicts.right_id))",
                    (*ids, *ids),
                ).fetchall()
                for r in rows:
                    a, b = int(r["left_id"]), int(r["right_id"])
                    out.add((min(a, b), max(a, b)))
        except sqlite3.Error:
            return set()
        return out
