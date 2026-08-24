"""Vector-index metadata persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Optional, TYPE_CHECKING

from ..acl import WorkspaceScope, workspace_scope_sql
from ..constants import is_default_workspace_term
from ..evidence import INDEXABLE_PREFILTER_SQL, has_indexable_text


def active_scan_boundary_on_connection(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return a stable identity for the active set a rebuild scan must cover."""
    import hashlib

    digest = hashlib.sha256()
    count = 0
    max_memory_id = 0
    for row in conn.execute(
        "SELECT id,version FROM memories WHERE status='active' ORDER BY id"
    ):
        memory_id = int(row["id"])
        version = int(row["version"] or 1)
        count += 1
        max_memory_id = memory_id
        digest.update(f"{memory_id}@{version}\n".encode("ascii"))
    return {
        "max_memory_id": max_memory_id,
        "active_count": count,
        "active_set_digest": digest.hexdigest(),
    }


def canonical_scan_boundary(boundary: dict[str, Any]) -> str:
    return json.dumps(boundary, sort_keys=True, separators=(",", ":"))

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

    @staticmethod
    def _migration_state(conn: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key,value FROM migration_state")
        }

    def conflict_scan_state(self) -> dict[str, Any]:
        if not self._db._db_available:
            return {"required": False}
        with self._db.connection() as conn:
            state = self._migration_state(conn)
        boundary_raw = state.get("conflict_scan_boundary")
        try:
            boundary = json.loads(boundary_raw) if boundary_raw else None
        except json.JSONDecodeError:
            boundary = boundary_raw
        progress_raw = state.get("conflict_scan_progress")
        try:
            progress = json.loads(progress_raw) if progress_raw else None
        except json.JSONDecodeError:
            progress = None
        return {
            "required": state.get("conflict_scan_required") == "true",
            "epoch": state.get("conflict_scan_epoch"),
            "detector_version": state.get("conflict_scan_detector_version"),
            "boundary": boundary,
            "progress": progress,
        }

    def rearm_conflict_scan_if_drifted(self) -> bool:
        """Re-baseline a required scan whose live set or detector drifted.

        A memory write between the upgrade and scan completion changes the
        live active-set boundary, so pages recorded against the persisted
        boundary can never validate again. Separately, a detector-version
        change between releases leaves a stale ``conflict_scan_detector_version``
        persisted that no running scan can match. Either would wedge
        ``conflict_scan_required`` forever. Both are resolved by atomically
        persisting the CURRENT live boundary and running detector under a fresh
        epoch and resetting progress, so one subsequent full scan clears it.
        """
        import uuid

        from ..db_generation import CONFLICT_DETECTOR_VERSION

        with self._db.write_transaction() as conn:
            state = self._migration_state(conn)
            if state.get("conflict_scan_required") != "true":
                return False
            persisted = state.get("conflict_scan_boundary") or ""
            live = canonical_scan_boundary(active_scan_boundary_on_connection(conn))
            detector_drift = state.get("conflict_scan_detector_version") != CONFLICT_DETECTOR_VERSION
            if persisted == live and not detector_drift:
                return False
            conn.execute(
                """INSERT INTO migration_state(key,value,updated_at)
                   VALUES('conflict_scan_epoch',?,CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP""",
                (uuid.uuid4().hex,),
            )
            conn.execute(
                """INSERT INTO migration_state(key,value,updated_at)
                   VALUES('conflict_scan_boundary',?,CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP""",
                (live,),
            )
            conn.execute(
                """INSERT INTO migration_state(key,value,updated_at)
                   VALUES('conflict_scan_detector_version',?,CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP""",
                (CONFLICT_DETECTOR_VERSION,),
            )
            conn.execute("DELETE FROM migration_state WHERE key='conflict_scan_progress'")
            return True

    def record_conflict_scan_page(
        self,
        *,
        epoch: str,
        detector_version: str,
        boundary: dict[str, Any],
        after_memory_id: int,
        next_anchor_memory_id: int | None,
        anchors_scanned: int,
        workspace: Any = None,
    ) -> bool:
        """Persist one contiguous server-enumerated page of a rebuild full scan.

        A workspace-scoped scan (strict caller, any scope shape) can never
        advance the global rebuild progress — it saw only part of the library.
        """
        if workspace:
            return False
        boundary_json = canonical_scan_boundary(boundary)
        with self._db.write_transaction() as conn:
            state = self._migration_state(conn)
            if not (
                state.get("conflict_scan_required") == "true"
                and state.get("conflict_scan_epoch") == epoch
                and state.get("conflict_scan_detector_version") == detector_version
                and state.get("conflict_scan_boundary") == boundary_json
                and canonical_scan_boundary(active_scan_boundary_on_connection(conn)) == boundary_json
            ):
                return False
            progress_raw = state.get("conflict_scan_progress")
            try:
                progress = json.loads(progress_raw) if progress_raw else None
            except json.JSONDecodeError:
                return False
            if progress is not None and str(progress.get("epoch") or "") != str(epoch):
                # Stale progress from a superseded epoch (e.g. a completed row
                # copied by side-by-side migration) must not wedge the new one.
                progress = None
            expected_after = 0 if progress is None else int(progress.get("next_after_memory_id", -1))
            if progress is not None and bool(progress.get("complete")):
                return False
            if int(after_memory_id) != expected_after or int(anchors_scanned) < 0:
                return False
            terminal = next_anchor_memory_id is None
            if terminal:
                if int(anchors_scanned) == 0 and int(after_memory_id) < int(boundary.get("max_memory_id", 0)):
                    return False
                next_after = int(boundary.get("max_memory_id", 0))
            else:
                if next_anchor_memory_id is None:
                    return False
                next_after = int(next_anchor_memory_id)
                if int(anchors_scanned) <= 0 or next_after <= int(after_memory_id):
                    return False
            pages = (int(progress.get("pages", 0)) if progress else 0) + 1
            scanned = (int(progress.get("anchors_scanned", 0)) if progress else 0) + int(anchors_scanned)
            payload = canonical_scan_boundary({
                "epoch": epoch,
                "detector_version": detector_version,
                "boundary": boundary,
                "pages": pages,
                "anchors_scanned": scanned,
                "next_after_memory_id": next_after,
                "complete": terminal,
            })
            conn.execute(
                """INSERT INTO migration_state(key,value,updated_at)
                   VALUES('conflict_scan_progress',?,CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP""",
                (payload,),
            )
            return True

    def complete_conflict_scan(
        self,
        *,
        epoch: str,
        detector_version: str,
        boundary: dict[str, Any],
    ) -> bool:
        """CAS-clear only after persisted server-validated full-scan progress."""
        boundary_json = canonical_scan_boundary(boundary)
        with self._db.write_transaction() as conn:
            state = self._migration_state(conn)
            try:
                progress = json.loads(state.get("conflict_scan_progress", ""))
            except (TypeError, json.JSONDecodeError):
                return False
            live_boundary = canonical_scan_boundary(active_scan_boundary_on_connection(conn))
            if not (
                state.get("conflict_scan_required") == "true"
                and state.get("conflict_scan_epoch") == epoch
                and state.get("conflict_scan_detector_version") == detector_version
                and state.get("conflict_scan_boundary") == boundary_json
                and live_boundary == boundary_json
                and progress.get("epoch") == epoch
                and progress.get("detector_version") == detector_version
                and canonical_scan_boundary(progress.get("boundary") or {}) == boundary_json
                and progress.get("complete") is True
                and int(progress.get("anchors_scanned", -1)) == int(boundary.get("active_count", -2))
                and int(progress.get("next_after_memory_id", -1)) == int(boundary.get("max_memory_id", -2))
            ):
                return False
            row = conn.execute(
                """UPDATE migration_state SET value='false', updated_at=CURRENT_TIMESTAMP
                   WHERE key='conflict_scan_required' AND value='true'
                     AND EXISTS(SELECT 1 FROM migration_state WHERE key='conflict_scan_epoch' AND value=?)
                     AND EXISTS(SELECT 1 FROM migration_state WHERE key='conflict_scan_detector_version' AND value=?)
                     AND EXISTS(SELECT 1 FROM migration_state WHERE key='conflict_scan_boundary' AND value=?)""",
                (epoch, detector_version, boundary_json),
            )
            return row.rowcount == 1

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
            "space_rebuild_active": "space_rebuild_evidence_id" in meta,
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

    def require_space_rebuild(self, embedding_space_id: str, reason: str) -> None:
        """Force an explicit full rebuild after derived-table loss/corruption."""
        with self._db.write_transaction() as conn:
            self.set_meta(conn, "state", "mismatch")
            self.set_meta(conn, "target_space_id", embedding_space_id)
            self.set_meta(conn, "migration_epoch", uuid.uuid4().hex)
            self.set_meta(conn, "last_error", reason)
            for key in (
                "space_rebuild_evidence_id", "migration_cursor",
                "migration_lease_owner", "migration_lease_expires_at",
                "workspace_rebuild_space_id",
            ):
                self.delete_meta(conn, key)

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

    def stale_index_ids(self, limit: int, workspace: WorkspaceScope = None) -> list[int]:
        """Ready-mode candidates: indexable memories lacking current-version evidence.

        Same screen-then-verify shape as space_rebuild_pending_ids so a
        zero-unit legacy row is never queued (it would be re-enqueued on
        every call forever). Strict callers pass their admitted canonical set;
        with admission off this collapses to the previous single equality.
        """
        db = self._db
        if not db._db_available:
            return []
        scope_sql, params = workspace_scope_sql(
            "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
        )
        workspace_sql = f"AND {scope_sql} " if scope_sql else ""
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
            if self.get_meta(conn, "workspace_rebuild_space_id") != embedding_space_id:
                return False
            expected_workspace_ids = {
                int(row["id"])
                for row in conn.execute("SELECT id,name FROM workspace_canonicals")
                if not is_default_workspace_term(str(row["name"] or ""))
            }
            try:
                workspace_vector_ids = {
                    int(row["id"])
                    for row in conn.execute("SELECT id FROM workspace_canonicals_vec")
                }
            except sqlite3.Error:
                return False
            if workspace_vector_ids != expected_workspace_ids:
                return False
            cursor = conn.execute(
                self._space_rebuild_pending_sql(), (int(epoch),)
            )
            for row in cursor:
                if has_indexable_text(
                    str(row["subject"] or ""), str(row["content"] or "")
                ):
                    return False
            obsolete_ids = [
                int(row["id"])
                for row in conn.execute(
                    "SELECT e.id FROM memory_evidence e "
                    "LEFT JOIN memories m ON m.id=e.memory_id "
                    "WHERE m.id IS NULL OR m.status='deleted'"
                )
            ]
            for start in range(0, len(obsolete_ids), 500):
                chunk = obsolete_ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})",
                    chunk,
                )
            if obsolete_ids:
                conn.execute(
                    "DELETE FROM memory_evidence WHERE id IN ("
                    "SELECT e.id FROM memory_evidence e LEFT JOIN memories m "
                    "ON m.id=e.memory_id WHERE m.id IS NULL OR m.status='deleted')"
                )
            orphan_vector_ids = [
                int(row["id"])
                for row in conn.execute(
                    "SELECT v.id FROM memory_evidence_vec v "
                    "LEFT JOIN memory_evidence e ON e.id=v.id WHERE e.id IS NULL"
                )
            ]
            for start in range(0, len(orphan_vector_ids), 500):
                chunk = orphan_vector_ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})",
                    chunk,
                )
            evidence_ids = {
                int(row["id"]) for row in conn.execute("SELECT id FROM memory_evidence")
            }
            vector_ids = {
                int(row["id"])
                for row in conn.execute("SELECT id FROM memory_evidence_vec")
            }
            if vector_ids != evidence_ids:
                return False
            self.set_meta(conn, "state", "ready")
            self.set_meta(conn, "active_space_id", embedding_space_id)
            for key in (
                "target_space_id", "space_rebuild_evidence_id",
                "migration_cursor", "migration_epoch",
                "migration_lease_owner", "migration_lease_expires_at",
                "last_error", "workspace_rebuild_space_id",
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
                    "last_error", "workspace_rebuild_space_id",
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
                    "workspace_rebuild_space_id",
                ):
                    self.delete_meta(conn, key)
