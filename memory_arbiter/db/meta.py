"""Vector-index metadata persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from typing import Any, TYPE_CHECKING

from ..acl import WorkspaceScope, workspace_scope_sql
from ..constants import is_default_workspace_term
from ..evidence import INDEXABLE_PREFILTER_SQL, has_indexable_text

# _vec_index_meta key holding the library's active embedding dimension. The
# model that produced it is the fact source (there is no configured vec.dim
# since 0.15.0); the key is written when a managed embedder loads and
# back-filled from an existing vec0 table's CREATE SQL for legacy libraries.
ACTIVE_DIM_META_KEY = "active_dim"


def vec_table_dimension(conn: sqlite3.Connection, table: str) -> int | None:
    """Parse a vec0 table's float[N] dimension from its CREATE SQL."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None:
        return None
    match = re.search(r"embedding\s+float\[(\d+)]", str(row[0] or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def active_dim_on_connection(conn: sqlite3.Connection) -> int | None:
    """Read-only active dim: _vec_index_meta key, else vec0 table SQL parse.

    No backfill — safe on read-only/diagnostic connections. Callers that own
    a writable database use MetaStore.get_active_dim, which persists the
    parsed value so it survives a later table drop.
    """
    try:
        row = conn.execute(
            "SELECT value FROM _vec_index_meta WHERE key = ?", (ACTIVE_DIM_META_KEY,)
        ).fetchone()
        if row is not None:
            # Positional access: diagnostic/read-only connections may not use
            # the sqlite3.Row factory.
            try:
                return int(row[0])
            except (TypeError, ValueError):
                pass
    except sqlite3.Error:
        pass
    return vec_table_dimension(conn, "memory_evidence_vec")


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
    def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
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
            "active_dim": int(meta[ACTIVE_DIM_META_KEY]) if ACTIVE_DIM_META_KEY in meta else None,
        }
        return result

    def get_active_dim(self) -> int | None:
        """The library's active embedding dimension (fact source since 0.15.0).

        Reads the persisted meta key; a legacy library without one is
        back-filled from the existing vec0 table's CREATE SQL and persisted.
        Returns None when neither exists (fresh library, no embedder yet).
        """
        db = self._db
        if not db._db_available:
            return None
        with db.connection() as conn:
            dim = active_dim_on_connection(conn)
        if dim is None:
            return None
        with db.write_transaction() as conn:
            if self.get_meta(conn, ACTIVE_DIM_META_KEY) is None:
                self.set_meta(conn, ACTIVE_DIM_META_KEY, str(dim))
        return dim

    def set_active_dim(self, dim: int) -> None:
        """Record the dimension of the currently active managed embedder."""
        if not self._db._db_available:
            return
        with self._db.write_transaction() as conn:
            self.set_meta(conn, ACTIVE_DIM_META_KEY, str(int(dim)))

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
        embedding_space_id: str | None,
        has_managed_embedder: bool,
        active_dim: int | None = None,
    ) -> None:
        db = self._db
        if not db._db_available:
            return
        with db.write_transaction() as conn:
            if not has_managed_embedder or embedding_space_id is None:
                self.set_meta(conn, "state", "unmanaged")
                return

            # The dimension of the embedder being activated is a library fact;
            # record it in the same transaction as the state flip.
            if active_dim is not None:
                self.set_meta(conn, ACTIVE_DIM_META_KEY, str(int(active_dim)))

            rows = conn.execute("SELECT key, value FROM _vec_index_meta").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in rows}
            state = meta.get("state")
            active_space_id = meta.get("active_space_id")
            target_space_id = meta.get("target_space_id")

            # A rebuild already armed toward this space (e.g. the dim-swap
            # revert below) must stay armed: repeated init calls are
            # idempotent no-ops until the rebuild flow completes. Checked
            # before the revert branch so the same-id flip cannot bypass it.
            if state in {"mismatch", "failed"} and target_space_id == embedding_space_id:
                return

            if active_space_id == embedding_space_id:
                # Same space id can still sit on foreign-dim tables: swapping
                # the model to dim B and back to A keeps the A id active in
                # meta while the forward flip already rebuilt the tables at B.
                # The tables are rebuilt at the model's dim either way, but a
                # rebuild that happened (existing dim differed) wiped every
                # vector — going ready would strand surviving evidence rows
                # without vectors behind a passing coverage gate. That case
                # arms the standard mismatch rebuild instead.
                dim_rebuilt = False
                if active_dim is not None:
                    existing_dim = vec_table_dimension(conn, "memory_evidence_vec")
                    if existing_dim is not None and existing_dim != int(active_dim):
                        self._db.schema.rebuild_vec_tables(conn, int(active_dim))
                        dim_rebuilt = True
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
                if dim_rebuilt:
                    # The table rebuild above wiped all vectors (both spaces');
                    # the ready flip's coverage gate (maybe_complete_space_
                    # rebuild) never runs on this path, so surviving evidence
                    # rows and the workspace vector pool would lose coverage
                    # silently. Arm the standard full rebuild toward this
                    # space instead of flipping ready.
                    self.set_meta(conn, "state", "mismatch")
                    self.set_meta(conn, "target_space_id", embedding_space_id)
                    self.set_meta(conn, "migration_epoch", uuid.uuid4().hex)
                    for key in (
                        "space_rebuild_evidence_id", "migration_cursor",
                        "migration_lease_owner", "migration_lease_expires_at",
                        "workspace_rebuild_space_id",
                    ):
                        self.delete_meta(conn, key)
                    return
                self.set_meta(conn, "state", "ready")
                for key in (
                    "target_space_id", "space_rebuild_evidence_id",
                    "migration_cursor", "migration_epoch",
                    "migration_lease_owner", "migration_lease_expires_at",
                    "last_error", "workspace_rebuild_space_id",
                ):
                    self.delete_meta(conn, key)
                return

            # Space change (or first activation over unknown history): the
            # duplicate-hint vectors live in the OLD embedding space. A dim
            # swap below rebuilds the tables anyway, but a SAME-dim model
            # swap would otherwise keep stale-space rows that the
            # missing-rows backfill never replaces — clear them here so the
            # startup backfill republishes in the current space.
            try:
                conn.execute("DELETE FROM subject_tags_vec")
            except sqlite3.Error:
                pass

            # Model swap to a different output dim: the existing vec0 tables
            # are unusable, and IF-NOT-EXISTS creation can never re-create
            # them at the new dim — drop them (plus any vec0 shadow leftovers,
            # which normally die with the main table) and re-create empty at
            # the new dim, atomically with the mismatch flip, so the rebuild
            # flow can republish into fresh tables.
            if active_dim is not None:
                existing_dim = vec_table_dimension(conn, "memory_evidence_vec")
                if existing_dim is not None and existing_dim != int(active_dim):
                    self._db.schema.rebuild_vec_tables(conn, int(active_dim))

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
