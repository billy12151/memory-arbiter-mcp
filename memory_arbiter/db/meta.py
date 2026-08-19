"""Vector-index metadata persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Optional, TYPE_CHECKING

from ..evidence import INDEXABLE_PREFILTER_SQL, has_indexable_text

if TYPE_CHECKING:
    from .core import MemoryDB


class MetaStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    @staticmethod
    def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute("SELECT value FROM _vec_index_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    @staticmethod
    def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO _vec_index_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def delete_meta(conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM _vec_index_meta WHERE key = ?", (key,))

    def get_vec_index_state(self) -> dict[str, Any]:
        db = self._db
        if not db._db_available:
            return {"state": "unmanaged"}
        with db.connection() as conn:
            rows = conn.execute("SELECT key, value FROM _vec_index_meta").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in rows}
        result: dict[str, Any] = {
            "state": meta.get("state", "unmanaged"),
            "active_space_id": meta.get("active_space_id"),
            "target_space_id": meta.get("target_space_id"),
            "migration_cursor": int(meta["migration_cursor"]) if "migration_cursor" in meta else None,
            "migration_epoch": meta.get("migration_epoch"),
            "last_error": meta.get("last_error"),
        }
        return result

    def mark_space_rebuild_started(self) -> None:
        """Record the evidence-id epoch of an embedding-space rebuild (idempotent).

        memory_evidence ids are AUTOINCREMENT-monotonic, so every row with
        id > epoch was written after this mark — in the target embedding
        space for single-process flows. The vec channel flips back to ready
        only once every non-deleted memory has fresh-version evidence above
        the epoch (timestamp comparison is deliberately avoided: second
        granularity would let same-second old-space rows count as rebuilt)."""
        db = self._db
        if not db._db_available:
            return
        with db.write_transaction() as conn:
            if self.get_meta(conn, "space_rebuild_evidence_id") is None:
                epoch = int(
                    conn.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM memory_evidence"
                    ).fetchone()[0]
                )
                self.set_meta(conn, "space_rebuild_evidence_id", str(epoch))

    @staticmethod
    def _space_rebuild_pending_sql() -> str:
        # INDEXABLE_PREFILTER_SQL is a cheap over-inclusive screen; rows it
        # leaks (exotic-whitespace legacy artifacts that publish zero
        # evidence units) are dropped by the has_indexable_text check in
        # space_rebuild_pending_ids / maybe_complete_space_rebuild, so they
        # can never block the mismatch->ready flip.
        return f"""SELECT m.id AS id, m.subject AS subject, m.content AS content
               FROM memories m
               WHERE m.status!='deleted'
                 AND {INDEXABLE_PREFILTER_SQL}
                 AND NOT EXISTS(
                   SELECT 1 FROM memory_evidence e
                   WHERE e.memory_id=m.id AND e.memory_version=m.version
                     AND e.id > ?
                 )"""

    def space_rebuild_pending_ids(self, limit: int) -> list[int]:
        """Ids still needing republish in the active space rebuild (paged).

        Without a persisted epoch (dry-run preview before any execute) the
        current MAX(id) stands in read-only, which yields the same starting
        set the first execute would mark (modulo rows written in between).
        Rows surviving the SQL screen but rejected by has_indexable_text are
        skipped and the page advances, so zero-unit memories neither consume
        batch slots nor reappear on every call."""
        db = self._db
        if not db._db_available:
            return []
        wanted = max(1, int(limit))
        ids: list[int] = []
        with db.connection() as conn:
            epoch = self.get_meta(conn, "space_rebuild_evidence_id")
            if epoch is None:
                epoch = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM memory_evidence"
                ).fetchone()[0]
            last_id = 0
            while len(ids) < wanted:
                rows = conn.execute(
                    self._space_rebuild_pending_sql()
                    + " AND m.id > ? ORDER BY m.id LIMIT ?",
                    (int(epoch), last_id, wanted),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    last_id = int(row["id"])
                    if has_indexable_text(
                        str(row["subject"] or ""), str(row["content"] or "")
                    ):
                        ids.append(last_id)
                        if len(ids) >= wanted:
                            break
        return ids

    def stale_index_ids(self, limit: int, workspace: Optional[str] = None) -> list[int]:
        """Ready-mode candidates: indexable memories lacking current-version evidence.

        Same screen-then-verify shape as space_rebuild_pending_ids so a
        zero-unit legacy row is never queued (it would be re-enqueued on
        every call forever). Strict-isolation callers pass their workspace
        and only see their own rows."""
        db = self._db
        if not db._db_available:
            return []
        workspace_sql = ""
        params: list[Any] = []
        if workspace:
            workspace_sql = "AND COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)=? "
            params.append(workspace)
        wanted = max(1, int(limit))
        ids: list[int] = []
        with db.connection() as conn:
            last_id = 0
            while len(ids) < wanted:
                rows = conn.execute(
                    f"""SELECT m.id AS id, m.subject AS subject, m.content AS content
                        FROM memories m
                        WHERE m.status!='deleted' AND m.id>? {workspace_sql}
                          AND {INDEXABLE_PREFILTER_SQL}
                          AND NOT EXISTS(SELECT 1 FROM memory_evidence e
                                         WHERE e.memory_id=m.id AND e.memory_version=m.version)
                        ORDER BY m.id LIMIT ?""",
                    (last_id, *params, wanted),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    last_id = int(row["id"])
                    if has_indexable_text(
                        str(row["subject"] or ""), str(row["content"] or "")
                    ):
                        ids.append(last_id)
                        if len(ids) >= wanted:
                            break
        return ids

    def maybe_complete_space_rebuild(self, embedding_space_id: str) -> bool:
        """Flip mismatch -> ready when the whole index is in the target space."""
        db = self._db
        if not db._db_available:
            return False
        with db.write_transaction() as conn:
            state = self.get_meta(conn, "state")
            target = self.get_meta(conn, "target_space_id")
            epoch = self.get_meta(conn, "space_rebuild_evidence_id")
            if state != "mismatch" or not target or epoch is None or target != embedding_space_id:
                return False
            cursor = conn.execute(
                self._space_rebuild_pending_sql(), (int(epoch),)
            )
            for row in cursor:
                if has_indexable_text(
                    str(row["subject"] or ""), str(row["content"] or "")
                ):
                    return False
            self.set_meta(conn, "state", "ready")
            self.set_meta(conn, "active_space_id", embedding_space_id)
            for key in (
                "target_space_id", "space_rebuild_evidence_id",
                "migration_cursor", "migration_epoch",
                "migration_lease_owner", "migration_lease_expires_at",
                "last_error",
            ):
                self.delete_meta(conn, key)
            return True

    def init_vec_index_state(
        self,
        embedding_space_id: Optional[str],
        has_managed_embedder: bool,
    ) -> None:
        db = self._db
        if not db._db_available:
            return
        with db.write_transaction() as conn:
            if not has_managed_embedder or embedding_space_id is None:
                self.set_meta(conn, "state", "unmanaged")
                return

            rows = conn.execute("SELECT key, value FROM _vec_index_meta").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in rows}
            state = meta.get("state")
            active_space_id = meta.get("active_space_id")
            target_space_id = meta.get("target_space_id")

            if active_space_id == embedding_space_id:
                epoch = meta.get("space_rebuild_evidence_id")
                try:
                    epoch_int = int(epoch) if epoch is not None else None
                except ValueError:
                    # Hand-edited/corrupt epoch: flipping to ready and
                    # clearing the key below still applies; only the purge
                    # (which depends on the epoch bound) is skipped.
                    epoch_int = None
                if epoch_int is not None:
                    # Reverting the embedding model mid-rebuild lands here.
                    # Every evidence row above the epoch was published by the
                    # aborted rebuild in the foreign space, and no later path
                    # would reselect it (the rows match current versions), so
                    # purge them to keep the ready channel single-space.
                    # Memories whose only rows were written during the aborted
                    # rebuild lose all coverage and resurface as stale
                    # candidates for the next rebuild_evidence run.
                    foreign_ids = [
                        int(row["id"]) for row in conn.execute(
                            "SELECT id FROM memory_evidence WHERE id > ?",
                            (epoch_int,),
                        ).fetchall()
                    ]
                    # Chunked: unbounded IN (...) lists exceed the variable
                    # cap on SQLite builds compiled with the classic limits.
                    for start in range(0, len(foreign_ids), 500):
                        chunk = foreign_ids[start:start + 500]
                        placeholders = ",".join("?" for _ in chunk)
                        conn.execute(
                            f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})",
                            chunk,
                        )
                    conn.execute(
                        "DELETE FROM memory_evidence WHERE id > ?", (epoch_int,)
                    )
                self.set_meta(conn, "state", "ready")
                for key in (
                    "target_space_id", "space_rebuild_evidence_id",
                    "migration_cursor", "migration_epoch",
                    "migration_lease_owner", "migration_lease_expires_at",
                    "last_error",
                ):
                    self.delete_meta(conn, key)
                return

            if state in {"mismatch", "failed"} and target_space_id == embedding_space_id:
                return

            try:
                mem_vec_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_evidence_vec"
                ).fetchone()["c"]
            except sqlite3.Error:
                mem_vec_count = 0
            sec_vec_count = 0

            if not active_space_id and mem_vec_count == 0 and sec_vec_count == 0:
                self.set_meta(conn, "state", "ready")
                self.set_meta(conn, "active_space_id", embedding_space_id)
                self.delete_meta(conn, "target_space_id")
            else:
                self.set_meta(conn, "state", "mismatch")
                self.set_meta(conn, "target_space_id", embedding_space_id)
                self.set_meta(conn, "migration_epoch", uuid.uuid4().hex)
                for key in (
                    "space_rebuild_evidence_id", "migration_cursor",
                    "migration_lease_owner",
                    "migration_lease_expires_at", "last_error",
                ):
                    self.delete_meta(conn, key)
