"""Vector storage and KNN operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
import struct
from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..degrade import DegradeState
    from .core import MemoryDB


class VectorStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    @property
    def state(self) -> "DegradeState":
        return self._db.state

    @property
    def _db_available(self) -> bool:
        return self._db._db_available

    def store_embedding(self, memory_id: int, embedding: list[float]) -> Tuple[bool, list[str]]:
        warnings: list[str] = []
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; embedding not stored."]
        if not self.state.sqlite_vec_available:
            return False, ["sqlite-vec unavailable; embedding not stored."]
        if not embedding:
            return False, ["embedding is empty (encode failed); not stored."]
        try:
            with self._db.connection() as conn:
                # v0.9.4: look up parent status from memories table (N8: COALESCE for deleted)
                parent_status = conn.execute(
                    "SELECT COALESCE(status, 'deleted') AS status FROM memories WHERE id = ?",
                    (int(memory_id),),
                ).fetchone()
                parent_status = parent_status["status"] if parent_status else "deleted"
                conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
                conn.execute(
                    "INSERT INTO memories_vec(id, parent_status, embedding) VALUES (?, ?, ?)",
                    (memory_id, parent_status, json.dumps(embedding)),
                )
                conn.commit()
            return True, []
        except sqlite3.Error as exc:
            warnings.append(f"store_embedding failed: {exc}")
            return False, warnings

    def delete_embedding(self, memory_id: int) -> Tuple[bool, list[str]]:
        try:
            with self._db.connection() as conn:
                conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
                conn.commit()
            return True, []
        except sqlite3.Error as exc:
            return False, [f"delete_embedding failed: {exc}"]

    def delete_vectors_for_memory(self, memory_id: int) -> Tuple[bool, list[str]]:
        """Hard-delete memory-level AND section-level vec rows for a memory (v0.9.2).

        Purges vectors from both ``memories_vec`` and ``memory_sections_vec``.
        v0.9.4: this is now ONLY called for hard-delete paths (e.g. edit-failure
        rollback, memory_cleanup_history cascade).  For supersede/arbitrate loser
        paths use ``mark_vectors_for_memory(memory_id, 'superseded')`` instead,
        which retains vectors with parent_status='superseded' for audit/history
        recall via ``memory_search_expired``.

        The memory's content and FTS rows are kept for audit; vectors are a
        pure derivative and can always be recomputed from content.
        """
        warnings: list[str] = []
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; vectors not deleted."]
        if not self.state.sqlite_vec_available:
            return False, ["sqlite-vec unavailable; vectors not deleted."]
        try:
            with self._db.write_transaction() as conn:
                conn.execute("DELETE FROM memories_vec WHERE id = ?", (int(memory_id),))
                conn.execute(
                    "DELETE FROM memory_sections_vec WHERE id IN "
                    "(SELECT id FROM memory_sections WHERE memory_id = ?)",
                    (int(memory_id),),
                )
            return True, []
        except sqlite3.Error as exc:
            return False, [f"delete_vectors_for_memory failed: {exc}"]

    def mark_vectors_for_memory(self, memory_id: int, new_status: str) -> Tuple[bool, list[str]]:
        """UPDATE parent_status for a memory's vec rows (v0.9.4).

        Called during supersede/arbitrate: marks vectors as 'superseded' so
        they remain available for ``memory_search_expired`` vec-hybrid recall
        but are excluded from active searches (``parent_status='active'``).
        Unlike ``delete_vectors_for_memory`` this does NOT physically purge
        rows — it only flips a short text column in-place.
        """
        warnings: list[str] = []
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; vectors not marked."]
        if not self.state.sqlite_vec_available:
            return False, ["sqlite-vec unavailable; vectors not marked."]
        try:
            with self._db.write_transaction() as conn:
                conn.execute(
                    "UPDATE memories_vec SET parent_status = ? WHERE id = ?",
                    (str(new_status), int(memory_id)),
                )
                conn.execute(
                    "UPDATE memory_sections_vec SET parent_status = ? WHERE id IN "
                    "(SELECT id FROM memory_sections WHERE memory_id = ?)",
                    (str(new_status), int(memory_id)),
                )
            return True, []
        except sqlite3.Error as exc:
            return False, [f"mark_vectors_for_memory failed: {exc}"]

    def _purge_inactive_vectors(self) -> Tuple[dict[str, int], list[str]]:
        """Physically delete only orphan vec rows (v0.9.4).

        v0.9.2-v0.9.3 deleted ALL inactive-vector rows (superseded + deleted +
        orphan).  That was too aggressive: superseded vectors should be kept
        for ``memory_search_expired`` vec-hybrid recall.  This method now only
        removes true orphans — rows whose parent memory/section row no longer
        exists — and leaves superseded vectors untouched.
        """
        warnings: list[str] = []
        if not self._db_available or not self.state.sqlite_writable:
            return {}, ["SQLite write unavailable; orphan vectors not purged."]
        if not self.state.sqlite_vec_available:
            return {}, ["sqlite-vec unavailable; orphan vectors not purged."]
        try:
            with self._db.write_transaction() as conn:
                mem_cur = conn.execute(
                    "DELETE FROM memories_vec WHERE id NOT IN "
                    "(SELECT id FROM memories)"
                )
                sec_cur = conn.execute(
                    "DELETE FROM memory_sections_vec WHERE id NOT IN "
                    "(SELECT id FROM memory_sections)"
                )
            counts = {
                "purged_memory_orphans": max(0, mem_cur.rowcount),
                "purged_section_orphans": max(0, sec_cur.rowcount),
            }
            return counts, []
        except sqlite3.Error as exc:
            return {}, [f"_purge_inactive_vectors failed: {exc}"]

    def _count_vec_parent_status_mismatch(self) -> dict[str, int]:
        """Count rows where vec.parent_status != memories.status (v0.9.4 doctor).

        Uses INNER JOIN, so orphan vec rows (no parent in memories/
        memory_sections) are NOT counted here — they are handled separately by
        ``_purge_inactive_vectors``. This keeps the dry_run mismatch count
        aligned with what ``_resync_vec_parent_status`` can actually repair
        (resync only touches rows with a joinable parent).
        """
        try:
            with self._db.connection() as conn:
                mem_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories_vec v "
                    "JOIN memories m ON m.id = v.id "
                    "WHERE v.parent_status IS NULL "
                    "OR v.parent_status != COALESCE(m.status, 'deleted')"
                ).fetchone()["c"]
                sec_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_sections_vec v "
                    "JOIN memory_sections s ON s.id = v.id "
                    "JOIN memories m ON m.id = s.memory_id "
                    "WHERE v.parent_status IS NULL "
                    "OR v.parent_status != COALESCE(m.status, 'deleted')"
                ).fetchone()["c"]
            return {"memory_vec_mismatch": max(0, int(mem_count)),
                    "section_vec_mismatch": max(0, int(sec_count))}
        except sqlite3.Error:
            return {}

    def _resync_vec_parent_status(self) -> dict[str, int | str]:
        """Repair mismatched vec.parent_status to match memories.status (v0.9.4)."""
        try:
            with self._db.write_transaction() as conn:
                mem_cur = conn.execute(
                    "UPDATE memories_vec SET parent_status = COALESCE(m.status, 'deleted') "
                    "FROM memories m WHERE m.id = memories_vec.id "
                    "AND (memories_vec.parent_status IS NULL "
                    "OR memories_vec.parent_status != COALESCE(m.status, 'deleted'))"
                )
                sec_cur = conn.execute(
                    "UPDATE memory_sections_vec SET parent_status = COALESCE(m.status, 'deleted') "
                    "FROM memory_sections s JOIN memories m ON m.id = s.memory_id "
                    "WHERE s.id = memory_sections_vec.id "
                    "AND (memory_sections_vec.parent_status IS NULL "
                    "OR memory_sections_vec.parent_status != COALESCE(m.status, 'deleted'))"
                )
            return {"resynced_memory_vecs": max(0, mem_cur.rowcount),
                    "resynced_section_vecs": max(0, sec_cur.rowcount)}
        except sqlite3.Error as exc:
            return {"error": str(exc)}

    def vec_knn(
        self,
        query_embedding: list[float],
        k: int = 10,
        parent_status_filter: str = "active",
        ws_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """v0.9.4: single-stage KNN with metadata-predicate pre-filter.

        ``parent_status_filter`` selects which vec rows compete for top-k:
          - ``"active"`` (default): only ``parent_status='active'`` rows —
            active recall (``memory_search``).
          - ``"expired"``: ``parent_status NOT IN ('active','deleted')`` —
            superseded + conflicted + pending, for history/audit recall
            (``memory_search_expired``).
          - ``"all"``: ``parent_status != 'deleted'`` — every non-deleted row,
            matching the FTS ``status_filter="all"`` semantics.

        Fallback: if the vec table lacks a ``parent_status`` column (e.g.
        after a failed migration), falls back to the old two-stage probe +
        exact-L2 scan with JOIN-based status filter.
        """
        if not self._db_available or not self.state.sqlite_vec_available:
            return []
        requested = max(0, int(k))
        if requested == 0:
            return []
        # Build the parent_status predicate from the controlled enum. Values
        # are whitelisted here (not interpolated from arbitrary caller input),
        # so SQL interpolation is injection-safe.
        if parent_status_filter == "expired":
            parent_predicate = "AND v.parent_status NOT IN ('active','deleted')"
            eligible = "COALESCE(m.status, 'deleted') NOT IN ('active','deleted')"
        elif parent_status_filter == "all":
            parent_predicate = "AND v.parent_status != 'deleted'"
            eligible = "COALESCE(m.status, 'deleted') != 'deleted'"
        else:  # "active" (default; unknown values fall back to active-only)
            parent_predicate = "AND v.parent_status = 'active'"
            eligible = "COALESCE(m.status, 'deleted') = 'active'"
        workspace_predicate = ""
        workspace_params: list[Any] = []
        if ws_canonical:
            workspace_predicate = "AND COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
            workspace_params.append(ws_canonical)
        query_json = json.dumps(query_embedding)
        try:
            with self._db.connection() as conn:
                conn.execute("BEGIN")
                # Try metadata-predicate fast path
                try:
                    rows = conn.execute(
                        f"""SELECT v.id AS id,
                               vec_distance_L2(v.embedding, ?) AS distance,
                               m.workspace AS workspace, m.workspace_canonical AS workspace_canonical, m.agent_id AS agent_id,
                               m.status AS status, m.subject AS subject,
                               m.tags AS tags, m.content AS content,
                               m.source_type AS source_type, m.confidence AS confidence,
                               m.protection_level AS protection_level,
                               m.event_time AS event_time, m.ingest_time AS ingest_time,
                               m.metadata AS metadata, m.split_status AS split_status
                        FROM memories_vec v
                        JOIN memories m ON m.id = v.id
                        WHERE v.embedding MATCH ? AND k = ?
                          {parent_predicate}
                          AND {eligible}
                          {workspace_predicate}
                        ORDER BY distance
                        """,
                        (query_json, query_json, requested, *workspace_params),
                    ).fetchall()
                    conn.execute("COMMIT")
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    # Fallback: parent_status column missing — use two-stage
                    # probe + exact-L2 scan (pre-v0.9.4 path). ROLLBACK the
                    # failed fast-path statement, then re-BEGIN so the probe
                    # and KNN share one transaction (N6: prevents TOCTOU between
                    # the excluded-count probe and the KNN scan).
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    conn.execute("BEGIN")
                    # D3: eligible mirrors the main-path parent_status_filter —
                    # expired/all/superseded leak the right rows into each
                    # channel. The old hardcoded "!= 'deleted'" let active rows
                    # leak into memory_search_expired's vec channel.
                    excluded = int(conn.execute(
                        f"SELECT COUNT(*) AS excluded FROM memories_vec v "
                        f"LEFT JOIN memories m ON m.id=v.id"
                        f" WHERE NOT ({eligible})"
                    ).fetchone()["excluded"] or 0)
                    if excluded:
                        rows = conn.execute(
                            f"""SELECT v.id AS id,
                                vec_distance_L2(v.embedding, ?) AS distance,
                                m.workspace AS workspace, m.workspace_canonical AS workspace_canonical, m.agent_id AS agent_id,
                                m.status AS status, m.subject AS subject,
                                m.tags AS tags, m.content AS content,
                                m.source_type AS source_type, m.confidence AS confidence,
                                m.protection_level AS protection_level,
                                m.event_time AS event_time, m.ingest_time AS ingest_time,
                                m.metadata AS metadata, m.split_status AS split_status
                            FROM memories_vec v
                            JOIN memories m ON m.id=v.id
                            WHERE {eligible}
                              {workspace_predicate}
                            ORDER BY distance
                            LIMIT ?""",
                            (query_json, *workspace_params, requested),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            f"""SELECT v.id AS id, v.distance AS distance,
                                m.workspace AS workspace, m.workspace_canonical AS workspace_canonical, m.agent_id AS agent_id,
                                m.status AS status, m.subject AS subject,
                                m.tags AS tags, m.content AS content,
                                m.source_type AS source_type, m.confidence AS confidence,
                                m.protection_level AS protection_level,
                                m.event_time AS event_time, m.ingest_time AS ingest_time,
                                m.metadata AS metadata, m.split_status AS split_status
                            FROM memories_vec v
                            JOIN memories m ON m.id=v.id
                            WHERE v.embedding MATCH ? AND k = ?
                              {workspace_predicate}
                            ORDER BY v.distance""",
                            (query_json, requested, *workspace_params),
                        ).fetchall()
                    conn.execute("COMMIT")
                    return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def section_vec_knn(
        self,
        query_embedding: list[float],
        k: int = 10,
        parent_status_filter: str = "active",
        ws_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """v0.9.4: single-stage section KNN with metadata-predicate pre-filter.

        ``parent_status_filter`` selects which section-vec rows compete for
        top-k (same enum as ``vec_knn``): ``"active"`` (default),
        ``"expired"`` (superseded+conflicted+pending), or ``"all"``
        (non-deleted).

        Unlike ``vec_knn`` this does NOT select ``m.content`` — Channel 6
        candidates score via the vec floor and get their content re-fetched
        by ``_attach_sections`` from ``current_mem_map``.

        Fallback: if the vec table lacks a ``parent_status`` column, falls
        back to the old two-stage probe + exact-L2 scan with JOIN-based
        status filter.
        """
        if not self._db_available or not self.state.sqlite_vec_available:
            return []
        requested = max(0, int(k))
        if requested == 0:
            return []
        # Build the parent_status predicate from the controlled enum (see
        # vec_knn for the injection-safety rationale).
        if parent_status_filter == "expired":
            parent_predicate = "AND v.parent_status NOT IN ('active','deleted')"
            eligible = "COALESCE(m.status, 'deleted') NOT IN ('active','deleted')"
        elif parent_status_filter == "all":
            parent_predicate = "AND v.parent_status != 'deleted'"
            eligible = "COALESCE(m.status, 'deleted') != 'deleted'"
        else:  # "active" (default; unknown values fall back to active-only)
            parent_predicate = "AND v.parent_status = 'active'"
            eligible = "COALESCE(m.status, 'deleted') = 'active'"
        workspace_predicate = ""
        workspace_params: list[Any] = []
        if ws_canonical:
            workspace_predicate = "AND COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
            workspace_params.append(ws_canonical)
        query_json = json.dumps(query_embedding)
        try:
            with self._db.connection() as conn:
                conn.execute("BEGIN")
                # Try metadata-predicate fast path
                try:
                    rows = conn.execute(
                        f"""SELECT s.memory_id AS memory_id,
                            s.id AS section_id,
                            vec_distance_L2(v.embedding, ?) AS distance,
                            s.title AS section_title,
                            s.title_path AS section_title_path,
                            m.workspace AS workspace, m.workspace_canonical AS workspace_canonical, m.status AS status,
                            m.subject AS subject, m.tags AS tags,
                            m.source_type AS source_type,
                            m.confidence AS confidence,
                            m.protection_level AS protection_level,
                            m.event_time AS event_time,
                            m.ingest_time AS ingest_time,
                            m.metadata AS metadata,
                            m.split_status AS split_status
                        FROM memory_sections_vec v
                        JOIN memory_sections s ON s.id = v.id
                        JOIN memories m ON m.id = s.memory_id
                        WHERE v.embedding MATCH ? AND k = ?
                          {parent_predicate}
                          AND {eligible}
                          {workspace_predicate}
                        ORDER BY distance
                        """,
                        (query_json, query_json, requested, *workspace_params),
                    ).fetchall()
                    conn.execute("COMMIT")
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    # Fallback: parent_status column missing — use two-stage
                    # probe + exact-L2 scan (pre-v0.9.4 path). ROLLBACK the
                    # failed fast-path statement, then re-BEGIN so the probe
                    # and KNN share one transaction (N6: prevents TOCTOU).
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    conn.execute("BEGIN")
                    # D3: eligible mirrors the main-path parent_status_filter
                    # (not the old hardcoded "!= 'deleted'").
                    excluded = int(conn.execute(
                        f"SELECT COUNT(*) AS excluded FROM memory_sections_vec v "
                        f"LEFT JOIN memory_sections s ON s.id=v.id "
                        f"LEFT JOIN memories m ON m.id=s.memory_id"
                        f" WHERE NOT ({eligible})"
                    ).fetchone()["excluded"] or 0)
                    if excluded:
                        rows = conn.execute(
                            f"""SELECT s.memory_id AS memory_id,
                                s.id AS section_id,
                                vec_distance_L2(v.embedding, ?) AS distance,
                                s.title AS section_title,
                                s.title_path AS section_title_path,
                                m.workspace AS workspace, m.workspace_canonical AS workspace_canonical, m.status AS status,
                                m.subject AS subject, m.tags AS tags,
                                m.source_type AS source_type,
                                m.confidence AS confidence,
                                m.protection_level AS protection_level,
                                m.event_time AS event_time,
                                m.ingest_time AS ingest_time,
                                m.metadata AS metadata,
                                m.split_status AS split_status
                            FROM memory_sections_vec v
                            JOIN memory_sections s ON s.id=v.id
                            JOIN memories m ON m.id=s.memory_id
                            WHERE {eligible}
                              {workspace_predicate}
                            ORDER BY distance
                            LIMIT ?""",
                            (query_json, *workspace_params, requested),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            f"""SELECT s.memory_id AS memory_id,
                                s.id AS section_id, v.distance AS distance,
                                s.title AS section_title,
                                s.title_path AS section_title_path,
                                m.workspace AS workspace, m.workspace_canonical AS workspace_canonical, m.status AS status,
                                m.subject AS subject, m.tags AS tags,
                                m.source_type AS source_type,
                                m.confidence AS confidence,
                                m.protection_level AS protection_level,
                                m.event_time AS event_time,
                                m.ingest_time AS ingest_time,
                                m.metadata AS metadata,
                                m.split_status AS split_status
                            FROM memory_sections_vec v
                            JOIN memory_sections s ON s.id=v.id
                            JOIN memories m ON m.id=s.memory_id
                            WHERE v.embedding MATCH ? AND k = ?
                              {workspace_predicate}
                            ORDER BY v.distance""",
                            (query_json, requested, *workspace_params),
                        ).fetchall()
                    conn.execute("COMMIT")
                    return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    # ------------------------------------------------------------------
    #  Memory CRUD
    # ------------------------------------------------------------------

    def get_embedding(self, memory_id: int) -> Optional[list[float]]:
        """Read back a memory's embedding vector as a list of floats.

        sqlite-vec stores embeddings internally as packed float32 little-endian
        bytes even though ``store_embedding`` writes JSON — vec0 converts on
        INSERT and returns binary on SELECT. So we ``struct.unpack`` here, not
        ``json.loads``. Returns ``None`` if vec is unavailable, the DB is
        unavailable, or the memory has no embedding row.
        """
        if not self._db_available or not self.state.sqlite_vec_available:
            return None
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT embedding FROM memories_vec WHERE id = ?",
                    (int(memory_id),),
                ).fetchone()
            if not row or row["embedding"] is None:
                return None
            raw = row["embedding"]
            if isinstance(raw, (bytes, bytearray)):
                n = len(raw) // 4
                if n == 0:
                    return None
                return list(struct.unpack(f"<{n}f", raw))
            # Legacy / forward-compat: if a future vec build returns JSON or a
            # list, accept it without crashing.
            if isinstance(raw, (list, tuple)):
                return list(raw)
            import json as _json
            return list(_json.loads(raw))
        except sqlite3.Error:
            return None
