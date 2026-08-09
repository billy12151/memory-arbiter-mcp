from __future__ import annotations

import json
import re
import sqlite3
import struct
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, cast

from ..claims_db import StructuredClaimStore
from ..config import Settings
from ..conflict_judgments import ConflictJudgmentStore
from ..degrade import DegradeState
from ..models import MemoryRecord, utc_now_iso
from .sections_store import SectionStore
from .semantic_notices import SemanticNoticeStore
from .audit import AuditStore
from .meta import MetaStore
from .schema import SchemaStore
from .vectors import VectorStore
from .workspaces import WorkspaceStore, _coerce_ws, _normalize_alias_key
from .conflicts import ConflictStore

# Explicit export list. The pre-split db.py surfaced its top-level imports
# (json/re/sqlite3/…) as module attributes; the package facade re-exports them
# for attribute/snapshot parity (R8), and listing them here satisfies mypy
# strict's "explicit export" rule for that facade re-export.
__all__ = [
    "MemoryDB",
    "row_to_dict",
    "_BUSY_TIMEOUT_MS",
    "_CJK_CHAR_RE",
    "_canon_entity",
    "_canon_scope",
    "_coerce_tags_db",
    "_coerce_ws",
    "_normalize_alias_key",
    "_subject_tokens",
    "Any",
    "ConflictJudgmentStore",
    "DegradeState",
    "Iterator",
    "MemoryRecord",
    "Optional",
    "Path",
    "Settings",
    "StructuredClaimStore",
    "Tuple",
    "contextmanager",
    "datetime",
    "json",
    "re",
    "sqlite3",
    "struct",
    "time",
    "timezone",
    "utc_now_iso",
    "uuid",
]

_BUSY_TIMEOUT_MS = 5000


class MemoryDB:
    """SQLite-backed memory store with per-operation connections.

    v0.6.0 refactor: the old shared ``self.conn`` is replaced by a connection
    factory.  Each tool call / transaction gets its own connection via the
    ``connection()`` or ``write_transaction()`` context manager.  Schema
    migration and feature probing happen once on a dedicated init connection
    before the server accepts any tool calls.

    Design doc §1.1c — SQLite transactions are connection-scoped; sharing a
    single long-lived connection across concurrent MCP calls risks nested
    transactions and cross-call commit/rollback.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = DegradeState()
        self._db_available = False
        self._sqlite_vec_loadable = False
        # Public sub-store accessors (Phase 3). ``claims``/``judgments`` are the
        # canonical handles; the underscore aliases are kept for back-compat with
        # existing internal references.
        self.claims = StructuredClaimStore(self)
        self.judgments = ConflictJudgmentStore(self)
        self.sections = SectionStore(self)
        self.semantic_notices = SemanticNoticeStore(self)
        self.audit = AuditStore(self)
        self.meta = MetaStore(self)
        self.schema = SchemaStore(self)
        self.vectors = VectorStore(self)
        self.workspaces = WorkspaceStore(self)
        self.conflicts = ConflictStore(self)
        self._claim_store = self.claims
        self._judgment_store = self.judgments
        self._init_database()

    # ------------------------------------------------------------------
    #  Connection factory + context managers
    # ------------------------------------------------------------------

    def _new_connection(self, *, init: bool = False) -> sqlite3.Connection:
        """Create a properly configured one-shot connection."""
        conn = sqlite3.connect(
            str(self.settings.db_path),
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        if init:
            conn.execute("PRAGMA journal_mode=WAL")
        if self._sqlite_vec_loadable:
            conn.enable_load_extension(True)
            import sqlite_vec  # type: ignore

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        return conn

    @property
    def db_available(self) -> bool:
        """Whether the database file can be opened for read/write."""
        return self._db_available

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a short-lived connection for a single read or write.

        The caller is responsible for ``commit()`` / ``rollback()``.
        The connection is always closed when the context exits.
        """
        if not self._db_available:
            raise sqlite3.Error("Database not available")
        conn = self._new_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection wrapped in ``BEGIN IMMEDIATE`` … ``COMMIT``.

        On any exception the transaction is rolled back.  Use this for
        atomic multi-statement writes (CAS, section publish, etc.).
        """
        if not self._db_available:
            raise sqlite3.Error("Database not available")
        conn = self._new_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def diagnostic_connection(self) -> Iterator[sqlite3.Connection]:
        """Read-only connection for doctor diagnostics (design doc §11.1).

        Opens with ``mode=ro`` via URI so the connection can never write, even
        if buggy check SQL ever tried.  Loads sqlite-vec when loadable so check
        SQL referencing the vec0 virtual tables can run.  Safe to run
        concurrently with MCP tool calls: it never takes the write lock.
        """
        if not self._db_available:
            raise sqlite3.Error("Database not available")
        conn = sqlite3.connect(
            f"file:{self.settings.db_path}?mode=ro", uri=True,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        conn.row_factory = sqlite3.Row
        if self._sqlite_vec_loadable:
            conn.enable_load_extension(True)
            import sqlite_vec  # type: ignore

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    #  One-time init (runs before any tool call)
    # ------------------------------------------------------------------

    def _init_database(self) -> None:
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = self._new_connection(init=True)
            try:
                self._init_schema(conn)
                self._probe_features(conn)
                self._db_available = True
            finally:
                conn.close()
        except sqlite3.Error as exc:
            self._db_available = False
            self.state.sqlite_writable = False
            self.state.mode = "jsonl_backup"
            self.state.jsonl_backup_active = True
            self.state.warn(
                f"SQLite unavailable or not writable: {exc}. "
                "Using JSONL append-only backup when possible."
            )

    # ------------------------------------------------------------------
    #  Schema
    # ------------------------------------------------------------------

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        return self.schema._init_schema(conn)

    def _migrate_v090_claims(self, conn: sqlite3.Connection) -> None:
        return self.schema._migrate_v090_claims(conn)

    @staticmethod
    def _migrate_add_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        decl: str,
    ) -> None:
        return SchemaStore._migrate_add_column(conn, table, column, decl)

    def _probe_features(self, conn: sqlite3.Connection) -> None:
        return self.schema._probe_features(conn)

    def _probe_sqlite_vec_loadable(self) -> Optional[bool]:
        return self.schema._probe_sqlite_vec_loadable()

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        return self.schema._rebuild_fts(conn)

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        return self.schema._ensure_fts(conn)

    def _ensure_vec_table(self, conn: sqlite3.Connection) -> None:
        return self.schema._ensure_vec_table(conn)

    def _ensure_section_vec_table(self, conn: sqlite3.Connection) -> None:
        return self.schema._ensure_section_vec_table(conn)

    def _ensure_workspace_vec_table(self, conn: sqlite3.Connection) -> None:
        return self.schema._ensure_workspace_vec_table(conn)

    def _migrate_vec_parent_status(self, conn: sqlite3.Connection) -> None:
        return self.schema._migrate_vec_parent_status(conn)

    # ------------------------------------------------------------------
    #  Embedding operations
    # ------------------------------------------------------------------

    def store_embedding(self, memory_id: int, embedding: list[float]) -> Tuple[bool, list[str]]:
        return self.vectors.store_embedding(memory_id, embedding)

    def resolve_workspace_canonical(
        self,
        ws_raw: Optional[str],
        embedder: Any = None,
        *,
        match_distance: Optional[float] = None,
        register_new: bool = True,
    ) -> dict[str, Any]:
        return self.workspaces.resolve_workspace_canonical(
            ws_raw, embedder, match_distance=match_distance, register_new=register_new,
        )

    def upsert_workspace_alias(
        self,
        alias: str,
        canonical: str,
        *,
        relation: str = "alias",
        status: str = "confirmed",
        source: str = "user",
        action: str = "accept",
        judge_type: str = "user",
        reason: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[bool, list[str]]:
        return self.workspaces.upsert_workspace_alias(
            alias, canonical,
            relation=relation,
            status=status,
            source=source,
            action=action,
            judge_type=judge_type,
            reason=reason,
            force=force,
        )

    def get_workspace_alias(self, alias: str) -> Optional[dict[str, Any]]:
        return self.workspaces.get_workspace_alias(alias)

    def list_workspace_alias_events(
        self, alias: Optional[str] = None, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        return self.workspaces.list_workspace_alias_events(alias, limit=limit)

    def rename_workspace_canonical(
        self, old: str, new: str, *, judge_type: str = "user", reason: Optional[str] = None
    ) -> Tuple[int, list[str]]:
        return self.workspaces.rename_workspace_canonical(
            old, new, judge_type=judge_type, reason=reason,
        )

    def migrate_workspace(
        self, from_ws: str, to_ws: str, *, judge_type: str = "user",
        reason: Optional[str] = None, embedder: Any = None,
    ) -> Tuple[int, list[str]]:
        return self.workspaces.migrate_workspace(
            from_ws, to_ws, judge_type=judge_type, reason=reason, embedder=embedder,
        )

    def set_memory_workspace_canonical(
        self,
        memory_id: int,
        canonical: str,
        embedder: Any = None,
    ) -> Tuple[bool, list[str]]:
        return self.workspaces.set_memory_workspace_canonical(memory_id, canonical, embedder)

    def delete_embedding(self, memory_id: int) -> Tuple[bool, list[str]]:
        return self.vectors.delete_embedding(memory_id)

    def delete_vectors_for_memory(self, memory_id: int) -> Tuple[bool, list[str]]:
        return self.vectors.delete_vectors_for_memory(memory_id)

    def mark_vectors_for_memory(self, memory_id: int, new_status: str) -> Tuple[bool, list[str]]:
        return self.vectors.mark_vectors_for_memory(memory_id, new_status)

    def _purge_inactive_vectors(self) -> Tuple[dict[str, int], list[str]]:
        return self.vectors._purge_inactive_vectors()

    def _count_vec_parent_status_mismatch(self) -> dict[str, int]:
        return self.vectors._count_vec_parent_status_mismatch()

    def _resync_vec_parent_status(self) -> dict[str, int]:
        return cast(dict[str, int], self.vectors._resync_vec_parent_status())

    def vec_knn(
        self,
        query_embedding: list[float],
        k: int = 10,
        parent_status_filter: str = "active",
        ws_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return self.vectors.vec_knn(query_embedding, k, parent_status_filter, ws_canonical)

    def section_vec_knn(
        self,
        query_embedding: list[float],
        k: int = 10,
        parent_status_filter: str = "active",
        ws_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return self.vectors.section_vec_knn(query_embedding, k, parent_status_filter, ws_canonical)

    def insert_memory(
        self, record: MemoryRecord, workspace_canonical: Optional[str] = None
    ) -> Tuple[Optional[int], list[str]]:
        warnings: list[str] = []
        if not record.content:
            raise ValueError("content is required")
        if not record.subject or not str(record.subject).strip():
            raise ValueError("subject is required")
        if not self._db_available or not self.state.sqlite_writable:
            self._append_backup(record)
            warnings.append("SQLite write unavailable; wrote append-only JSONL backup.")
            return None, warnings
        # Double-store: raw workspace stays in `workspace`; resolved canonical
        # (from tools-side alias resolution) lands in `workspace_canonical`.
        # Fall back to the raw workspace so the column is never NULL on new rows.
        canonical = (workspace_canonical or record.workspace or "").strip() or record.workspace
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO memories
                (content, agent_id, workspace, workspace_canonical, tags, source_type, source_ref,
                 event_time, ingest_time, confidence, protection_level, status, subject, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.content,
                    record.agent_id,
                    record.workspace,
                    canonical,
                    json.dumps(record.tags, ensure_ascii=False),
                    record.source_type,
                    record.source_ref,
                    record.event_time,
                    record.ingest_time,
                    record.confidence,
                    record.protection_level,
                    record.status,
                    record.subject,
                    json.dumps(record.metadata, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )
            memory_id = int(cur.lastrowid)
            if self.state.fts5_available:
                conn.execute(
                    "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                    (memory_id, record.content, " ".join(record.tags), record.subject or ""),
                )
            conn.commit()
        return memory_id, warnings

    def _append_backup(self, record: MemoryRecord) -> None:
        self.settings.backup_jsonl.parent.mkdir(parents=True, exist_ok=True)
        payload = record.__dict__.copy()
        payload["backup_written_at"] = utc_now_iso()
        with self.settings.backup_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.state.jsonl_backup_active = True

    @staticmethod
    def _fetch_memory(conn: sqlite3.Connection, memory_id: int) -> Optional[dict[str, Any]]:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return row_to_dict(row) if row else None

    def get_memory(self, memory_id: int) -> Optional[dict[str, Any]]:
        if not self._db_available:
            return None
        with self.connection() as conn:
            return self._fetch_memory(conn, memory_id)

    def update_memory(self, memory_id: int, updates: dict[str, Any]) -> bool:
        if not self._db_available or not self.state.sqlite_writable:
            return False
        allowed = {"source_type", "confidence", "protection_level", "status", "metadata"}
        pairs = [(key, value) for key, value in updates.items() if key in allowed]
        if not pairs:
            return True
        try:
            with self.write_transaction() as conn:
                current = self._fetch_memory(conn, int(memory_id))
                if not current:
                    return False
                changed = any(current.get(key) != value for key, value in pairs)
                if not changed:
                    return True
                status_changed = any(
                    key == "status" and current.get(key) != value
                    for key, value in pairs
                )
                new_status = next((value for key, value in pairs if key == "status"), None)
                # Claim judgments depend not only on extracted text but also on
                # trust/protection/status.  Changing any of those must invalidate
                # the old CAS snapshot.  Metadata only participates when its
                # canonical entity/scope changes; bookkeeping such as
                # confirmed_from does not create a second revision bump.
                judgment_semantics_changed = any(
                    key in {"source_type", "confidence", "protection_level", "status"}
                    and current.get(key) != value
                    for key, value in pairs
                )
                metadata_update = next((value for key, value in pairs if key == "metadata"), None)
                if isinstance(metadata_update, dict):
                    current_md = current.get("metadata") or {}
                    current_md = current_md if isinstance(current_md, dict) else {}
                    judgment_semantics_changed = judgment_semantics_changed or (
                        _canon_entity(current_md.get("entity")) != _canon_entity(metadata_update.get("entity"))
                        or _canon_scope(current_md.get("scope")) != _canon_scope(metadata_update.get("scope"))
                    )
                sql = ", ".join(f"{key} = ?" for key, _ in pairs)
                if judgment_semantics_changed:
                    sql += ", claim_revision = claim_revision + 1"
                values = [
                    json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for _, v in pairs
                ]
                values.append(int(memory_id))
                conn.execute(f"UPDATE memories SET {sql} WHERE id = ?", values)
                if status_changed and self.state.sqlite_vec_available:
                    try:
                        conn.execute(
                            "UPDATE memories_vec SET parent_status = ? WHERE id = ?",
                            (str(new_status or "deleted"), int(memory_id)),
                        )
                        conn.execute(
                            "UPDATE memory_sections_vec SET parent_status = ? WHERE id IN "
                            "(SELECT id FROM memory_sections WHERE memory_id = ?)",
                            (str(new_status or "deleted"), int(memory_id)),
                        )
                    except sqlite3.Error:
                        pass
                return True
        except sqlite3.Error:
            return False

    def list_memories(self, workspace: Optional[str] = None, subject: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        clauses = ["status != 'deleted'"]
        params: list[Any] = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY event_time DESC, ingest_time DESC LIMIT ?",
                params,
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def _filter_clauses(
        self,
        like_status_clause: str,
        tags_filter: Optional[list[str]],
        after_dt: Optional[datetime],
        before_dt: Optional[datetime],
        source_type: Optional[str],
    ) -> Tuple[list[str], list[Any]]:
        """WHERE clause + params shared by count_filtered_memories and recall_by_filters.

        Mirrors search._passes_filters: like_status_clause + per-tag json_each exact
        match (AND semantics) + ingest_time ISO-string bounds + source_type equality.
        workspace is intentionally NOT filtered (v0.7.4 cross-workspace search).
        """
        clauses: list[str] = [like_status_clause]
        params: list[Any] = []
        if tags_filter:
            for tag in tags_filter:
                clauses.append("EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)")
                params.append(tag)
        if after_dt is not None:
            clauses.append("ingest_time >= ?")
            params.append(after_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat())
        if before_dt is not None:
            clauses.append("ingest_time <= ?")
            params.append(before_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat())
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        return clauses, params

    def count_filtered_memories(
        self,
        like_status_clause: str,
        tags_filter: Optional[list[str]],
        after_dt: Optional[datetime],
        before_dt: Optional[datetime],
        source_type: Optional[str],
        ws_canonical: Optional[str] = None,
    ) -> int:
        """v0.7.3: COUNT(*) under the same filters used by search's _passes_filters.

        Only called when has_filters=True. Clauses are built by _filter_clauses so the
        SQL count and the SQL recall (recall_by_filters) share one source of truth.
        Cross-workspace (v0.7.4) — workspace is not filtered, EXCEPT under strict
        isolation (v0.9.7) where ``ws_canonical`` scopes the count to one canonical
        workspace so the total matches the paginated recall.
        """
        if not self._db_available:
            return 0
        clauses, params = self._filter_clauses(like_status_clause, tags_filter, after_dt, before_dt, source_type)
        if ws_canonical:
            clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
            params.append(ws_canonical)
        sql = f"SELECT COUNT(*) AS c FROM memories WHERE {' AND '.join(clauses)}"
        with self.connection() as conn:
            try:
                row = conn.execute(sql, params).fetchone()
                return int(row["c"]) if row else 0
            except sqlite3.Error:
                return 0

    def recall_by_filters(
        self,
        like_status_clause: str,
        tags_filter: Optional[list[str]],
        after_dt: Optional[datetime],
        before_dt: Optional[datetime],
        source_type: Optional[str],
        limit: int,
        offset: int = 0,
        ws_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """G6 (v0.8.5): filter-driven recall for empty-query + filters in memory_search.

        SELECT * mirroring count_filtered_memories' WHERE (via _filter_clauses),
        ordered by ingest_time DESC, capped at ``limit`` (pool_cap). Returns
        row_to_dict rows — same shape as _wide_recall pool rows. Enables
        list-by-tag / by-source_type / by-time when query is empty.

        v0.9.4: ``offset`` adds SQL OFFSET for cursor pagination on the
        exact-count filter path (used by ``memory_search_expired``).
        v0.9.7: ``ws_canonical`` hard-scopes to one canonical workspace under
        strict isolation (filters in SQL so pagination and totals stay correct).
        """
        if not self._db_available:
            return []
        clauses, params = self._filter_clauses(like_status_clause, tags_filter, after_dt, before_dt, source_type)
        if ws_canonical:
            clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
            params.append(ws_canonical)
        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY ingest_time DESC LIMIT ? OFFSET ?"
        with self.connection() as conn:
            try:
                rows = conn.execute(sql, params + [int(limit), int(offset)]).fetchall()
            except sqlite3.Error:
                return []
        return [row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    #  Conflicts
    # ------------------------------------------------------------------

    def record_conflict(self, left_id: int, right_id: int, subject: Optional[str], reason: str, winner_id: Optional[int], status: str = "open") -> Optional[int]:
        return self.conflicts.record_conflict(left_id, right_id, subject, reason, winner_id, status)

    def resolve_conflicts_for(self, memory_id: int) -> int:
        return self.conflicts.resolve_conflicts_for(memory_id)

    def list_conflicts(self, status: str = "open", limit: int = 50, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self.conflicts.list_conflicts(status, limit, source)

    def list_open_conflicts_for_memory_ids(
        self, memory_ids: list[int],
    ) -> list[dict[str, Any]]:
        return self.conflicts.list_open_conflicts_for_memory_ids(memory_ids)

    def get_memory_summaries(
        self, memory_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        return self.audit.get_memory_summaries(memory_ids)

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
        left_claim_revision: Optional[int] = None,
        right_claim_revision: Optional[int] = None,
        judgment_status: Optional[str] = None,
        structured_details: Optional[list[dict[str, Any]]] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
        detection_channel: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.conflicts.record_conflict_enriched(
            left_id,
            right_id,
            conflict_type,
            conflict_point,
            reason,
            suggested_winner=suggested_winner,
            confidence_hint=confidence_hint,
            source=source,
            status=status,
            refresh=refresh,
            left_version=left_version,
            right_version=right_version,
            left_claim_revision=left_claim_revision,
            right_claim_revision=right_claim_revision,
            judgment_status=judgment_status,
            structured_details=structured_details,
            scan_prompt_version=scan_prompt_version,
            scan_model=scan_model,
            detection_channel=detection_channel,
        )

    def resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved",
    ) -> dict[str, Any]:
        return self.conflicts.resolve_conflict(conflict_id, reason=reason, status=status)

    def is_pair_dismissed(self, left_id: int, right_id: int) -> bool:
        return self.conflicts.is_pair_dismissed(left_id, right_id)

    def is_structured_pair_closed_for_snapshot(
        self,
        left_id: int,
        right_id: int,
        left_version: int,
        right_version: int,
        left_claim_revision: int,
        right_claim_revision: int,
    ) -> bool:
        return self.conflicts.is_structured_pair_closed_for_snapshot(
            left_id, right_id, left_version, right_version,
            left_claim_revision, right_claim_revision,
        )

    def get_memory_version(self, memory_id: int) -> Optional[int]:
        return self.conflicts.get_memory_version(memory_id)

    def dismissed_pairs_for(self, memory_ids: list[int]) -> set:
        return self.conflicts.dismissed_pairs_for(memory_ids)

    def get_embedding(self, memory_id: int) -> Optional[list[float]]:
        return self.vectors.get_embedding(memory_id)

    # ------------------------------------------------------------------
    #  Semantic write-time notices
    # ------------------------------------------------------------------

    def record_semantic_notice(
        self,
        *,
        memory_id: int,
        peer_id: Optional[int],
        severity: str,
        notice_type: str,
        title: str,
        message: str,
        payload: dict[str, Any],
        dedupe_key: Optional[str] = None,
        conflict_id: Optional[int] = None,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        left_claim_revision: Optional[int] = None,
        right_claim_revision: Optional[int] = None,
        source: str = "semantic_write_gate",
    ) -> dict[str, Any]:
        return self.semantic_notices.record_semantic_notice(
            memory_id=memory_id,
            peer_id=peer_id,
            severity=severity,
            notice_type=notice_type,
            title=title,
            message=message,
            payload=payload,
            dedupe_key=dedupe_key,
            conflict_id=conflict_id,
            left_version=left_version,
            right_version=right_version,
            left_claim_revision=left_claim_revision,
            right_claim_revision=right_claim_revision,
            source=source,
        )

    def list_semantic_notices(self, status: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        return self.semantic_notices.list_semantic_notices(status=status, limit=limit)

    def semantic_notice_counts(self) -> dict[str, int]:
        return self.semantic_notices.semantic_notice_counts()

    def is_semantic_pair_closed(
        self,
        left_id: int,
        right_id: int,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        notice_type: str = "semantic_pair",
    ) -> bool:
        return self.semantic_notices.is_semantic_pair_closed(
            left_id, right_id, left_version, right_version, notice_type,
        )

    def update_semantic_notice_status(self, notice_id: int, status: str, reason: str = "") -> dict[str, Any]:
        return self.semantic_notices.update_semantic_notice_status(notice_id, status, reason)

    @property
    def scan_log_path(self) -> Path:
        return self.audit.scan_log_path

    @property
    def attention_log_path(self) -> Path:
        return self.audit.attention_log_path

    def log_attention(self, *, trigger: str, source: str, memory_ids: list) -> None:
        return self.audit.log_attention(trigger=trigger, source=source, memory_ids=memory_ids)

    def _scan_log_last_completed(self) -> Optional[dict[str, Any]]:
        return self.audit.scan_log_last_completed()

    # ------------------------------------------------------------------
    #  Legacy vector conflict candidate scan was removed.
    #
    #  Embedding/sqlite-vec remains available for semantic recall, section
    #  recall, and workspace aliasing.  The old KNN conflict-candidate
    #  scanner and its tuning parameters are intentionally not kept here; the
    #  legacy MCP tool is no longer registered.


    # ------------------------------------------------------------------
    #  Edit / History
    # ------------------------------------------------------------------

    def update_tags_low_side_effect(
        self,
        memory_id: int,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """v0.7.6: low-side-effect tag-only update.

        Unlike ``edit_memory``, this does NOT write ``memory_history``,
        does NOT bump ``version``, does NOT touch content/subject/sections/
        split_status/split_revision, and does NOT trigger re-embedding.
        It only updates the ``tags`` column and re-syncs FTS (tags are
        indexed in FTS5).

        Uses ``write_transaction()`` (BEGIN IMMEDIATE) so the re-read +
        protection check + writes share the write lock (TOCTOU-safe).

        Returns an outcome dict:
          ``updated``       — tags changed, FTS re-synced.
          ``no_change``     — add/remove yielded no difference; zero writes.
          ``not_found``     — memory_id absent.
          ``not_active``    — superseded/deleted.
          ``forbidden``     — protected and not authorized.
          ``unavailable``   — DB not writable.
          ``error``         — sqlite3.Error mid-transaction; fully rolled back.
        """
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": memory_id}
        try:
            with self.write_transaction() as conn:
                current = self._fetch_memory(conn, memory_id)
                if not current:
                    return {"outcome": "not_found", "memory_id": memory_id}
                status = current.get("status")
                if status != "active":
                    return {"outcome": "not_active", "memory_id": memory_id, "status": status}
                raw_tags = current.get("tags")
                if isinstance(raw_tags, list):
                    old_tags = raw_tags
                elif isinstance(raw_tags, str):
                    try:
                        parsed = json.loads(raw_tags)
                        old_tags = parsed if isinstance(parsed, list) else []
                    except (json.JSONDecodeError, ValueError):
                        old_tags = []
                else:
                    old_tags = []

                # Protection check inside the transaction (TOCTOU-safe).
                protection = current.get("protection_level")
                source_type = current.get("source_type")
                is_protected = protection == "locked" or source_type == "user_confirmed"
                if is_protected and not authorized:
                    return {
                        "outcome": "forbidden",
                        "memory_id": memory_id,
                        "protection_level": protection,
                        "source_type": source_type,
                    }

                # Compute new tags preserving order + deduping.
                current_set: set[str] = set(old_tags)
                new_tags_list = list(old_tags)
                for t in (remove_tags or []):
                    if t in current_set:
                        current_set.discard(t)
                        new_tags_list = [x for x in new_tags_list if x != t]
                for t in (add_tags or []):
                    if t not in current_set:
                        current_set.add(t)
                        new_tags_list.append(t)

                if new_tags_list == old_tags:
                    # Zero-write no-op (covers idempotent remove of absent tag).
                    return {"outcome": "no_change", "memory_id": memory_id, "tags": old_tags}

                new_tags_json = json.dumps(new_tags_list, ensure_ascii=False)
                from ..claims import resolve_entity
                old_entity, _ = resolve_entity(current)
                new_record = dict(current)
                new_record["tags"] = new_tags_list
                new_entity, _ = resolve_entity(new_record)
                claim_semantics_changed = old_entity != new_entity
                conn.execute(
                    "UPDATE memories SET tags=?, claim_revision=claim_revision+? WHERE id=?",
                    (new_tags_json, 1 if claim_semantics_changed else 0, memory_id),
                )
                if self.state.fts5_available:
                    old_content = current["content"]
                    old_subject = current.get("subject")
                    conn.execute(
                        "INSERT INTO memories_fts(memories_fts, rowid, content, tags, subject) "
                        "VALUES('delete', ?, ?, ?, ?)",
                        (memory_id, old_content, " ".join(old_tags), old_subject or ""),
                    )
                    conn.execute(
                        "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                        (memory_id, old_content, " ".join(new_tags_list), old_subject or ""),
                    )
                # write_transaction() commits on normal exit; rolls back on raise.
                return {
                    "outcome": "updated",
                    "memory_id": memory_id,
                    "tags": new_tags_list,
                    "claim_semantics_changed": claim_semantics_changed,
                }
        except sqlite3.Error:
            return {"outcome": "error", "memory_id": memory_id}

    def update_metadata_fields_low_side_effect(
        self,
        memory_id: int,
        set_fields: Optional[dict] = None,
        clear_fields: Optional[list] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        return self._claim_store.update_metadata_fields_low_side_effect(
            memory_id=memory_id,
            set_fields=set_fields,
            clear_fields=clear_fields,
            authorized=authorized,
        )

    def list_entities(
        self,
        limit: int = 50,
        include_unassigned: bool = True,
    ) -> dict[str, Any]:
        return self._claim_store.list_entities(limit, include_unassigned)

    def find_metadata_overlap_candidates(
        self,
        subject: Optional[str],
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """v0.7.6: recall active memories that might duplicate/evolve the
        given (subject, tags). Used by write_hints.

        Two recall channels (each capped at *limit*):
          - tag overlap: ``json_each`` match on any of *tags*.
          - subject overlap: LIKE on the first few subject tokens.
        Results are merged/deduped, limited, and returned as
        ``{id, subject, tags, content}`` dicts. Never raises.
        """
        if not self._db_available:
            return []
        candidates: dict[int, dict[str, Any]] = {}
        try:
            with self.connection() as conn:
                # Channel 1: tag overlap.
                if tags:
                    clean_tags = [t for t in tags if isinstance(t, str) and t.strip()]
                    if clean_tags:
                        ph = ",".join("?" * len(clean_tags))
                        ph_placeholders = clean_tags
                        rows = conn.execute(
                            f"SELECT id, subject, tags, content FROM memories "
                            f"WHERE status='active' AND id != ? AND "
                            f"EXISTS (SELECT 1 FROM json_each(tags) "
                            f"WHERE json_each.value IN ({ph}) AND json_each.type='text') "
                            f"LIMIT ?",
                            (exclude_id, *ph_placeholders, limit),
                        ).fetchall()
                        for r in rows:
                            candidates[int(r["id"])] = row_to_dict(r)
                # Channel 2: subject overlap.
                if subject:
                    tokens = _subject_tokens(subject)
                    like_clauses: list[str] = []
                    like_params: list[Any] = []
                    for tok in tokens[:4]:  # cap at first 4 tokens
                        if len(tok) >= 2:
                            like_clauses.append("subject LIKE ?")
                            like_params.append(f"%{tok}%")
                    if like_clauses:
                        joined = " OR ".join(like_clauses)
                        rows = conn.execute(
                            f"SELECT id, subject, tags, content FROM memories "
                            f"WHERE status='active' AND id != ? AND ({joined}) "
                            f"LIMIT ?",
                            (exclude_id, *like_params, limit),
                        ).fetchall()
                        for r in rows:
                            candidates[int(r["id"])] = row_to_dict(r)
        except sqlite3.Error:
            return []
        return list(candidates.values())

    def edit_memory(
        self,
        memory_id: int,
        new_content: str,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: Optional[str] = None,
    ) -> Optional[int]:
        """In-place edit a memory's content, archiving the prior version."""
        if not self._db_available or not self.state.sqlite_writable:
            return None
        try:
            # write_transaction (BEGIN IMMEDIATE) serializes concurrent edits: the
            # write lock is taken BEFORE the read, so a second edit blocks until the
            # first commits and then sees the bumped version (#3 — was self.connection(),
            # which let two edits both read v=1 and both write v=2: lost update + dup
            # history version).
            with self.write_transaction() as conn:
                current = self._fetch_memory(conn, memory_id)
                if not current:
                    return None
                old_content = current["content"]
                old_subject = current.get("subject")
                old_tags = current.get("tags") or []
                old_version = int(current.get("version") or 1)
                tags_json = json.dumps(new_tags, ensure_ascii=False) if new_tags is not None else json.dumps(old_tags, ensure_ascii=False)
                # Defense in depth: an empty-string subject would silently wipe the
                # field via the branch below (`new_subject if new_subject is not None
                # else old_subject`). Reject it here too — tools.memory_edit already
                # guards this at the service layer, but direct callers of this DB
                # method must not be able to clear subject either.
                if new_subject is not None and not str(new_subject).strip():
                    raise ValueError("new_subject must be non-empty when provided")
                subject_value = new_subject if new_subject is not None else old_subject
                history_cur = conn.execute(
                    """
                    INSERT INTO memory_history
                    (memory_id, content_snapshot, subject_snapshot, tags_snapshot, version, changed_at, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        old_content,
                        old_subject,
                        json.dumps(old_tags, ensure_ascii=False),
                        old_version,
                        utc_now_iso(),
                        reason,
                    ),
                )
                history_id = int(history_cur.lastrowid)
                conn.execute(
                    "UPDATE memories SET content=?, subject=?, tags=?, version=?, "
                    "claim_revision=claim_revision+1 WHERE id=?",
                    (new_content, subject_value, tags_json, old_version + 1, memory_id),
                )
                # v0.6.0: content changed → clear sections + bump split_revision
                self._delete_sections_for_memory(conn, memory_id)
                conn.execute(
                    "UPDATE memories SET split_status = NULL, "
                    "split_revision = split_revision + 1 WHERE id = ?",
                    (memory_id,),
                )
                if self.state.fts5_available:
                    conn.execute(
                        "INSERT INTO memories_fts(memories_fts, rowid, content, tags, subject) VALUES('delete', ?, ?, ?, ?)",
                        (memory_id, old_content, " ".join(old_tags), old_subject or ""),
                    )
                    conn.execute(
                        "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                        (memory_id, new_content, " ".join(new_tags) if new_tags is not None else " ".join(old_tags), subject_value or ""),
                    )
                return history_id
        except sqlite3.Error:
            return None

    def list_history(self, memory_id: int) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_history WHERE memory_id=? ORDER BY version DESC, id DESC",
                (memory_id,),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def cleanup_history(self, memory_id: Optional[int] = None, older_than_days: Optional[int] = None) -> int:
        """Delete historical snapshots from memory_history.

        SAFETY RED LINE: only ever issues DELETE against memory_history.
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0
        clauses: list[str] = []
        params: list[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if older_than_days is not None:
            from datetime import datetime, timedelta, timezone

            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(older_than_days))).replace(microsecond=0).isoformat()
            clauses.append("changed_at < ?")
            params.append(cutoff)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as conn:
            cur = conn.execute(f"DELETE FROM memory_history {where}", params)
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    #  Audit
    # ------------------------------------------------------------------

    def audit_summary(self) -> dict[str, Any]:
        return self.audit.audit_summary()

    # ==================================================================
    #  v0.6.0: _vec_index_meta + section operations
    # ==================================================================

    # ---- _vec_index_meta CRUD ----

    @staticmethod
    def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
        return MetaStore.get_meta(conn, key)

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        return MetaStore.set_meta(conn, key, value)

    @staticmethod
    def _delete_meta(conn: sqlite3.Connection, key: str) -> None:
        return MetaStore.delete_meta(conn, key)

    def get_vec_index_state(self) -> dict[str, Any]:
        return self.meta.get_vec_index_state()

    def init_vec_index_state(
        self,
        embedding_space_id: Optional[str],
        has_managed_embedder: bool,
    ) -> None:
        return self.meta.init_vec_index_state(embedding_space_id, has_managed_embedder)

    # ---- Section CRUD ----

    @staticmethod
    def _insert_section(
        conn: sqlite3.Connection,
        memory_id: int,
        section_index: int,
        title: Optional[str],
        title_path: Optional[str],
        summary: Optional[str],
        anchor_text: Optional[str],
        occurrence_index: int,
        start_offset: int,
        end_offset: int,
        provenance: str,
        embedding_truncated: int,
        embedding_original_tokens: int,
        embedding_used_tokens: int,
    ) -> int:
        return SectionStore.insert_section(
            conn, memory_id, section_index, title, title_path, summary,
            anchor_text, occurrence_index, start_offset, end_offset,
            provenance, embedding_truncated, embedding_original_tokens,
            embedding_used_tokens,
        )

    @staticmethod
    def _store_section_vec(
        conn: sqlite3.Connection,
        section_id: int,
        embedding: list[float],
    ) -> None:
        return SectionStore.store_section_vec(conn, section_id, embedding)

    @staticmethod
    def _delete_sections_for_memory(conn: sqlite3.Connection, memory_id: int) -> int:
        return SectionStore.delete_sections_for_memory(conn, memory_id)

    @staticmethod
    def _get_sections(conn: sqlite3.Connection, memory_id: int) -> list[dict[str, Any]]:
        return SectionStore.get_sections(conn, memory_id)

    @staticmethod
    def _get_section_vec_ids(conn: sqlite3.Connection, memory_id: int) -> set[int]:
        return SectionStore.get_section_vec_ids(conn, memory_id)

    def get_sections_by_memory(self, memory_id: int) -> list[dict[str, Any]]:
        return self.sections.get_sections_by_memory(memory_id)

    def get_sections_by_ids(
        self, memory_id: int, section_ids: list[int]
    ) -> Tuple[list[dict[str, Any]], list[int]]:
        return self.sections.get_sections_by_ids(memory_id, section_ids)

    def section_vec_distance_match(
        self,
        memory_id: int,
        query_embedding: list[float],
        threshold: float,
    ) -> list[dict[str, Any]]:
        return self.sections.section_vec_distance_match(memory_id, query_embedding, threshold)

def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata", "structured_details"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


def _canon_entity(value: Any) -> str:
    """v0.9 §3.3 entity/scope canonicalisation: strip / lower / collapse whitespace /
    strip trailing punctuation. CJK has no case, so ``lower()`` is a no-op for CJK
    and this is safe on Chinese entity names. Applied at entity-write time (via the
    tool) and re-applied at list/detection read time (idempotent) so storage stays
    deduped regardless of how a value was written. Returns "" for empty/None.
    """
    from ..text import canon_entity
    return canon_entity(value)


def _canon_scope(value: Any) -> str:
    """Same lexical normalisation as entity; kept separate for API clarity."""
    from ..text import canon_scope
    return canon_scope(value)


def _coerce_tags_db(raw: Any) -> list[str]:
    """Normalise a ``tags`` value into a deduped ``list[str]``.

    Implementation lives in text.coerce_tags (Phase 1 single source); thin re-export
    here so db.py scan logic and existing imports keep working.
    """
    from ..text import coerce_tags
    return coerce_tags(raw)


# CJK Unicode range for subject tokenisation (write_hints candidate recall).
# Single source: text.CJK_RE_SUBJECT (contiguous 㐀-鿿 range; a superset of
# text.CJK_RE_SEARCH differing only by U+4DC0-4DFF). Re-exported for back-compat.
from ..text import CJK_RE_SUBJECT as _CJK_CHAR_RE


def _subject_tokens(subject: str) -> list[str]:
    """Split a subject into tokens for LIKE-based candidate recall.

    Implementation lives in text.subject_tokens (Phase 1); thin re-export here.
    """
    from ..text import subject_tokens
    return subject_tokens(subject)

