"""Memory row CRUD, filters, edit/history operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
import sys

from ..evidence import INDEXABLE_PREFILTER_SQL
import struct
from typing import Any, TYPE_CHECKING

from ..config import Settings
from ..degrade import DegradeState

from ..constants import VEC0_MAX_K

if TYPE_CHECKING:
    from .core import MemoryDB
class _MemVectorsMixin:
    """memories 向量/摘要索引面（从 memories.py 搬出，拆分批 ②b 纯移动）。"""

    # 拆分批 ②b：声明式注解（mypy strict；形态对齐主类）
    if TYPE_CHECKING:
        _db: "MemoryDB"
        from typing import Any as _Any
        @property
        def _db_available(self) -> bool: ...
        @property
        def settings(self) -> "Settings": ...
        @property
        def state(self) -> "DegradeState": ...
        def connection(self) -> "_Any": ...
        def write_transaction(self) -> "_Any": ...
    def active_subject_tag_rows(
        self, exclude_memory_id: int, workspace_canonical: str | None,
        *, limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Lightweight id/subject/tags rows for the write-time duplicate hint.

        Same-workspace active memories only: the hint must never leak a
        subject the caller could not read, and cross-workspace near-duplicates
        are not this check's business. ``limit`` caps the scan fallback used
        when no embedder/index is available (ORDER BY id keeps the cap
        deterministic); the primary recall path is memory_summary_knn.
        """
        if not self._db_available:
            return []
        cap_sql = ""
        cap_params: list[Any] = []
        if limit is not None:
            cap_sql = " ORDER BY id LIMIT ?"
            cap_params.append(max(1, int(limit)))
        with self.connection() as conn:
            # Sargable form of COALESCE(NULLIF(workspace_canonical,''),workspace) = ?:
            # the OR lets SQLite serve the canonical branch from
            # idx_memories_canonical instead of scanning the whole table.
            rows = conn.execute(
                "SELECT id, subject, tags, event_time, ingest_time, content FROM memories "
                "WHERE status = 'active' AND id != ? "
                "AND (workspace_canonical = ? "
                "OR ((workspace_canonical IS NULL OR workspace_canonical = '') AND workspace = ?))"
                + cap_sql,
                (int(exclude_memory_id), workspace_canonical, workspace_canonical, *cap_params),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except (TypeError, ValueError):
                tags = []
            out.append({
                "id": int(row["id"]),
                "subject": str(row["subject"] or ""),
                "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
                "event_time": row["event_time"],
                "content": str(row["content"] or ""),
            })
        return out

    def memory_summary_knn(
        self,
        query_embedding: list[float],
        *,
        k: int,
        exclude_memory_id: int,
        workspace_canonical: str | None,
        conn: sqlite3.Connection | None = None,
    ) -> list[dict[str, Any]]:
        """0.17.0 P2-7: write-time duplicate-hint recall over memory_summary_vec.

        Same rowid-IN pre-filter contract as the other vec KNN readers;
        rows exist only for ACTIVE memories (the summary index tracks the
        active set, refresh_summary_vector deletes on exit). The extra embed
        per write buys retitle-tolerant recall — the subject gate alone
        killed retitle near-duplicates. (0.17.1 review: the params list kept
        a third element from the retired subject_tags_knn shape after the
        SQL collapsed to one COALESCE placeholder, so every call hit
        sqlite3.Error and silently returned [] — the scoped-recall test now
        pins the working path.)
        """
        if (
            not self._db_available or not self.state.sqlite_vec_available
            or not query_embedding or not str(workspace_canonical or "").strip()
        ):
            return []
        # A1（0.17.1 修复批）：同 row_knn 的 vec0 k 硬上限——仓内当前唯一
        # 生产调用传 k=10（write 去重），属 API 层防御；超限同样会被下方
        # except sqlite3.Error 吞成空结果。
        requested_k = max(1, int(k))
        if requested_k > VEC0_MAX_K:
            print(
                f"summary_knn: k={requested_k} exceeds the vec0 limit; "
                f"clamped to {VEC0_MAX_K}",
                file=sys.stderr,
            )
            requested_k = VEC0_MAX_K
        eligible_params: list[Any] = [
            int(exclude_memory_id), workspace_canonical,
        ]
        try:
            query = """SELECT v.id AS id, m.subject AS subject, m.tags AS tags,
                               m.event_time AS event_time, m.content AS content
                        FROM memory_summary_vec v
                        JOIN memories m ON m.id=v.id
                        WHERE v.embedding MATCH ? AND k=?
                          AND v.id IN (
                            SELECT m2.id FROM memories m2
                            WHERE m2.status='active' AND m2.id != ?
                              AND COALESCE(NULLIF(m2.workspace_canonical,''),m2.workspace) = ?
                          )
                        ORDER BY v.distance"""
            params = [json.dumps(query_embedding), requested_k, *eligible_params]
            if conn is not None:
                rows = conn.execute(query, params).fetchall()
            else:
                with self._db.connection() as owned:
                    rows = owned.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def upsert_subject_tags_vector(self, memory_id: int, embedding: list[float]) -> bool:
        """Publish/refresh one memory's subject+tags vector (write-time hint index).

        sqlite-vec 0.1.x vec0 rejects every conflict clause (even OR IGNORE
        raises on a PK conflict), so an upsert is DELETE+INSERT inside one
        transaction. The memory's active status is re-checked under the same
        write lock: a retire that commits between the caller's check and this
        publish must not leave a stale vector behind.
        """
        if not self._db_available or not self.state.sqlite_writable or not embedding:
            return False
        try:
            with self.write_transaction() as conn:
                status_row = conn.execute(
                    "SELECT status FROM memories WHERE id = ?", (int(memory_id),)
                ).fetchone()
                if status_row is None or str(status_row["status"]) != "active":
                    conn.execute(
                        "DELETE FROM subject_tags_vec WHERE id = ?", (int(memory_id),)
                    )
                    return False
                conn.execute(
                    "DELETE FROM subject_tags_vec WHERE id = ?", (int(memory_id),)
                )
                conn.execute(
                    "INSERT INTO subject_tags_vec(id, embedding) VALUES (?, ?)",
                    (int(memory_id), json.dumps([float(x) for x in embedding])),
                )
            return True
        except sqlite3.Error:
            return False

    def delete_subject_tags_vector(self, memory_id: int) -> bool:
        """Drop one memory's hint vector (status left active / hard delete)."""
        if not self._db_available or not self.state.sqlite_writable:
            return False
        try:
            with self.write_transaction() as conn:
                conn.execute(
                    "DELETE FROM subject_tags_vec WHERE id = ?", (int(memory_id),)
                )
            return True
        except sqlite3.Error:
            return False

    def subject_tags_vectors(self, memory_ids: list[int]) -> dict[int, list[float]]:
        """Point-read several memories' subject+tags vectors in one connection.

        Soft ordering (v0.15.12 C4) consumes these to score pair overlap. The
        vec0 point lookup (``WHERE id IN (...)``) was probe-verified to return
        the float32 blob and no row for a missing id; ids absent from the
        table are simply omitted so callers score them 0. One connection for
        the whole batch — a per-id helper would open/close one connection per
        pair and the scan path feeds hundreds of pairs.
        """
        if not self._db_available or not self.state.sqlite_vec_available or not memory_ids:
            return {}
        wanted = sorted({int(mid) for mid in memory_ids if int(mid) > 0})
        if not wanted:
            return {}
        found: dict[int, list[float]] = {}
        try:
            with self.connection() as conn:
                # Point reads, not a KNN match: the vec0 virtual table still
                # honors plain rowid membership, but keep the batch bounded so
                # the IN list stays sane even for a large pool.
                for start in range(0, len(wanted), 500):
                    chunk = wanted[start:start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT id, embedding FROM subject_tags_vec WHERE id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        if row["embedding"] is None:
                            continue
                        try:
                            blob = bytes(row["embedding"])
                            found[int(row["id"])] = list(
                                struct.unpack(f"{len(blob) // 4}f", blob)
                            )
                        except (struct.error, TypeError):
                            continue
        except sqlite3.Error:
            return {}
        return found

    def missing_subject_tags_rows(self) -> list[dict[str, Any]]:
        """Active memories whose hint vector is absent (startup backfill set).

        Vec-id membership is computed in Python from one full scan per side:
        backfill runs once per process, and plain id scans avoid any reliance
        on vec0 point-lookup planning inside a correlated subquery.
        """
        if not self._db_available:
            return []
        try:
            with self.connection() as conn:
                vector_ids = {
                    int(row["id"]) for row in conn.execute("SELECT id FROM subject_tags_vec")
                }
                rows = conn.execute(
                    "SELECT id, subject, tags FROM memories WHERE status='active' "
                    "ORDER BY id"
                ).fetchall()
        except sqlite3.Error:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if int(row["id"]) in vector_ids:
                continue
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except (TypeError, ValueError):
                tags = []
            out.append({
                "id": int(row["id"]),
                "subject": str(row["subject"] or ""),
                "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
            })
        return out

    def missing_row_vector_rows(self) -> list[dict[str, Any]]:
        """C6 (owner ruling 2026-09-23 #2): non-deleted memories that pass the
        indexable prefilter but have no row segments — the row backfill's
        pending set. The old unit-table EXISTS precondition is gone (it dies
        with the tables and silently starved the backfill); the expired
        family is INCLUDED so the delete-table guard can ever be satisfied
        and memory_search_expired keeps its vector channel on old rows."""
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    f"""SELECT m.id, m.version, m.subject, m.content,
                              COALESCE(m.content_sha,'') AS content_sha
                       FROM memories m
                       WHERE m.status!='deleted' AND {INDEXABLE_PREFILTER_SQL}
                         AND NOT EXISTS(SELECT 1 FROM memory_row r WHERE r.memory_id=m.id)
                       ORDER BY m.id"""
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            # C6: a silent empty return starved the backfill forever (the
            # review P1-2 trap); surface the failure through doctor instead.
            import sys
            print(f"missing_row_vector_rows failed: {exc}", file=sys.stderr)
            return []

    def missing_summary_vec_rows(self) -> list[dict[str, Any]]:
        """Active memories whose summary vector is absent (C3a backfill set).

        Mirrors missing_subject_tags_rows: vec-id membership in Python, one
        plain scan per side, content included so the caller can build the
        summary text without a second query.
        """
        if not self._db_available:
            return []
        try:
            with self.connection() as conn:
                vector_ids = {
                    int(row["id"]) for row in conn.execute("SELECT id FROM memory_summary_vec")
                }
                rows = conn.execute(
                    "SELECT id, subject, tags, content FROM memories WHERE status='active' "
                    "ORDER BY id"
                ).fetchall()
        except sqlite3.Error:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if int(row["id"]) in vector_ids:
                continue
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except (TypeError, ValueError):
                tags = []
            out.append({
                "id": int(row["id"]),
                "subject": str(row["subject"] or ""),
                "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
                "content": str(row["content"] or ""),
            })
        return out

    def all_summary_vectors(self) -> dict[int, tuple[str, list[float]]]:
        """Single-trip read of every active memory's summary vector (C3a).

        The anomaly check needs the FULL matrix — KNN per row would be N
        queries and cannot vote on neighbourhood composition. One SELECT
        over the vec table joined to the active set; the ~770-row library
        is ~2.3MB of float32. Returns {memory_id: (workspace, vector)}.
        """
        if not self._db_available or not self.state.sqlite_vec_available:
            return {}
        found: dict[int, tuple[str, list[float]]] = {}
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    "SELECT v.id AS id, v.embedding AS embedding, "
                    "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace) AS workspace "
                    "FROM memory_summary_vec v "
                    "JOIN memories m ON m.id=v.id AND m.status='active'"
                ).fetchall()
                for row in rows:
                    if row["embedding"] is None:
                        continue
                    try:
                        blob = bytes(row["embedding"])
                        found[int(row["id"])] = (
                            str(row["workspace"] or ""),
                            list(struct.unpack(f"{len(blob) // 4}f", blob)),
                        )
                    except (struct.error, TypeError):
                        continue
        except sqlite3.Error:
            return {}
        return found

    def upsert_summary_vector(self, memory_id: int, embedding: list[float]) -> bool:
        """Publish/refresh one memory's summary vector (C3a ownership index).

        Same vec0 conflict-clause constraint as subject_tags_vec: DELETE+INSERT
        in one transaction, active status re-checked under the write lock.
        """
        if not self._db_available or not self.state.sqlite_writable or not embedding:
            return False
        try:
            with self.write_transaction() as conn:
                status_row = conn.execute(
                    "SELECT status FROM memories WHERE id = ?", (int(memory_id),)
                ).fetchone()
                if status_row is None or str(status_row["status"]) != "active":
                    conn.execute(
                        "DELETE FROM memory_summary_vec WHERE id = ?", (int(memory_id),)
                    )
                    return False
                conn.execute(
                    "DELETE FROM memory_summary_vec WHERE id = ?", (int(memory_id),)
                )
                conn.execute(
                    "INSERT INTO memory_summary_vec(id, embedding) VALUES (?, ?)",
                    (int(memory_id), json.dumps([float(x) for x in embedding])),
                )
            return True
        except sqlite3.Error:
            return False

    def delete_summary_vector(self, memory_id: int) -> bool:
        """Drop one memory's summary vector (status left active / hard delete)."""
        if not self._db_available or not self.state.sqlite_writable:
            return False
        try:
            with self.write_transaction() as conn:
                conn.execute(
                    "DELETE FROM memory_summary_vec WHERE id = ?", (int(memory_id),)
                )
            return True
        except sqlite3.Error:
            return False
