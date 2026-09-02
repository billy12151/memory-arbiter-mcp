from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from ..config import Settings
from ..acl import WorkspaceScope
from ..degrade import DegradeState
from ..db_generation import (
    LegacyDatabaseError,
    database_startup_lock,
    detect_database_generation,
    legacy_database_message,
    require_current_or_new_database,
)
from ..models import MemoryRecord, utc_now_iso
from .semantic_notices import SemanticNoticeStore
from .audit import AuditStore
from .meta import MetaStore
from .schema import SchemaStore
from .workspaces import WorkspaceStore, _coerce_ws, _normalize_alias_key
from .conflicts import ConflictStore
from .memories import MemoriesStore
from .backup_replay import BackupReplayStore
from .evidence_store import EvidenceStore

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
    "DegradeState",
    "Iterator",
    "MemoryRecord",
    "Optional",
    "Path",
    "Settings",
    "Tuple",
    "contextmanager",
    "datetime",
    "json",
    "re",
    "sqlite3",
    "time",
    "timezone",
    "utc_now_iso",
    "uuid",
]

_BUSY_TIMEOUT_MS = 5000
_INIT_BUSY_RETRIES = 5
_INIT_RETRY_BASE_SECONDS = 0.05


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

    def __init__(self, settings: Settings, *, allow_incomplete: bool = False):
        self.settings = settings
        self.state = DegradeState()
        self._db_available = False
        self._sqlite_vec_loadable = False
        self.semantic_notices = SemanticNoticeStore(self)
        self.audit = AuditStore(self)
        self.meta = MetaStore(self)
        self.schema = SchemaStore(self)
        self.workspaces = WorkspaceStore(self)
        self.conflicts = ConflictStore(self)
        self.memories = MemoriesStore(self)
        self.backup_replay = BackupReplayStore(self)
        self.evidence = EvidenceStore(self)
        # Hold one lock across the generation gate and any first-start schema
        # creation. Current databases skip DDL entirely at normal startup.
        with database_startup_lock(settings.db_path):
            if allow_incomplete:
                generation = detect_database_generation(settings.db_path)
                if generation == "legacy":
                    raise LegacyDatabaseError(legacy_database_message(settings.db_path))
            else:
                generation = require_current_or_new_database(settings.db_path)
            self._init_database(
                initialize_schema=allow_incomplete or generation in {"missing", "empty"},
            )

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
            import sqlite_vec

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
        atomic multi-statement writes (CAS, evidence publish, etc.).
        """
        if not self._db_available:
            raise sqlite3.Error("Database not available")
        conn = self._new_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            # If BEGIN itself failed (e.g. busy timeout) there is no active
            # transaction; a blind ROLLBACK would raise "cannot rollback - no
            # transaction is active" and mask the original error.
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
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
            import sqlite_vec

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    #  One-time init (runs before any tool call)
    # ------------------------------------------------------------------

    def _init_database(self, *, initialize_schema: bool = True) -> None:
        # Only brand-new databases are tightened: never touch permissions of
        # a file the operator may share deliberately.
        db_preexisted = self.settings.db_path.exists()
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: sqlite3.Error | None = None
        for attempt in range(_INIT_BUSY_RETRIES):
            conn: sqlite3.Connection | None = None
            try:
                conn = self._new_connection(init=initialize_schema)
                if initialize_schema:
                    self._init_schema(conn)
                self._probe_features(conn, initialize=initialize_schema)
                if not db_preexisted:
                    try:
                        os.chmod(self.settings.db_path, 0o600)
                    except OSError:
                        pass
                self._db_available = True
                return
            except sqlite3.Error as exc:
                last_error = exc
                message = str(exc).lower()
                transient = "locked" in message or "busy" in message
                if not transient or attempt + 1 >= _INIT_BUSY_RETRIES:
                    break
            finally:
                if conn is not None:
                    conn.close()
            time.sleep(_INIT_RETRY_BASE_SECONDS * (2 ** attempt))

        self._db_available = False
        self.state.sqlite_writable = False
        self.state.mode = "jsonl_backup"
        self.state.jsonl_backup_active = True
        self.state.warn(
            f"SQLite unavailable or not writable: {last_error or 'unknown initialization error'}. "
            "Using JSONL append-only backup when possible."
        )

    # ------------------------------------------------------------------
    #  Schema
    # ------------------------------------------------------------------

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        return self.schema._init_schema(conn)

    def _probe_features(
        self, conn: sqlite3.Connection, *, initialize: bool = True,
    ) -> None:
        return self.schema._probe_features(conn, initialize=initialize)

    def _probe_sqlite_vec_loadable(self) -> bool | None:
        return self.schema._probe_sqlite_vec_loadable()

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        return self.schema._rebuild_fts(conn)

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        return self.schema._ensure_fts(conn)

    def ensure_evidence_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        return self.schema.ensure_evidence_vec_table(conn, dim)

    def ensure_workspace_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        return self.schema.ensure_workspace_vec_table(conn, dim)

    def ensure_vec_tables(self, dim: int) -> list[str]:
        """Lazily create the derived vec0 tables at the model-reported dim."""
        return self.schema.ensure_vec_tables(dim)

    def ensure_vector_tables_for_repair(self) -> tuple[bool, list[str]]:
        return self.schema.ensure_vector_tables_for_repair()

    def missing_vector_tables(self) -> list[str]:
        return self.schema.missing_vector_tables()

    def resolve_workspace_canonical(
        self,
        ws_raw: str | None,
        embedder: Any = None,
        *,
        match_distance: float | None = None,
        register_new: bool = True,
    ) -> dict[str, Any]:
        return self.workspaces.resolve_workspace_canonical(
            ws_raw, embedder, match_distance=match_distance, register_new=register_new,
        )

    def record_workspace_decision(
        self,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.record_workspace_decision(
            workspace_name, canonical, status=status, force=force,
        )

    def record_workspace_decision_on_conn(
        self,
        conn: sqlite3.Connection,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.record_workspace_decision(
            workspace_name, canonical, status=status, force=force, conn=conn,
        )

    def get_workspace_decision(self, workspace_name: str) -> dict[str, Any] | None:
        return self.workspaces.get_workspace_decision(workspace_name)

    def rename_workspace_canonical(
        self, old: str, new: str,
    ) -> tuple[int, list[str]]:
        return self.workspaces.rename_workspace_canonical(old, new)

    def migrate_workspace(
        self, from_ws: str, to_ws: str, *, embedder: Any = None,
    ) -> tuple[int, list[str]]:
        return self.workspaces.migrate_workspace(from_ws, to_ws, embedder=embedder)

    def prepare_workspace_canonical_embedding(
        self, canonical: str, embedder: Any = None,
    ) -> list[float] | None:
        return self.workspaces.prepare_workspace_canonical_embedding(canonical, embedder)

    def rebuild_workspace_canonical_vectors(
        self, embedder: Any, embedding_space_id: str,
    ) -> dict[str, Any]:
        return self.workspaces.rebuild_workspace_canonical_vectors(
            embedder, embedding_space_id,
        )

    def set_memory_workspace_canonical(
        self,
        memory_id: int,
        canonical: str,
        embedder: Any = None,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.set_memory_workspace_canonical(memory_id, canonical, embedder)

    def set_memory_workspace_canonical_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        canonical: str,
        precomputed_embedding: list[float] | None = None,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.set_memory_workspace_canonical(
            memory_id,
            canonical,
            None,
            conn=conn,
            precomputed_embedding=precomputed_embedding,
        )

    def evidence_knn(
        self, query_embedding: list[float], *, k: int = 100, parent_status_filter: str = "active",
        workspace: WorkspaceScope = None, exclude_memory_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.evidence.knn(query_embedding, k=k, parent_status_filter=parent_status_filter, workspace=workspace, exclude_memory_id=exclude_memory_id)

    def scan_rule_candidates(
        self, *, after_memory_id: int = 0, anchor_batch: int = 50, neighbor_k: int = 10,
        include_check: bool = False, max_distance: float | None = None,
        workspace: WorkspaceScope = None, similarity_pool_limit: int = 0,
        include_duplicates: bool = False,
    ) -> dict[str, Any]:
        return self.evidence.scan_rule_candidates(
            after_memory_id=after_memory_id, anchor_batch=anchor_batch,
            neighbor_k=neighbor_k, include_check=include_check,
            max_distance=max_distance, workspace=workspace,
            similarity_pool_limit=similarity_pool_limit,
            include_duplicates=include_duplicates,
        )

    def insert_memory(
        self,
        record: MemoryRecord,
        workspace_canonical: str | None = None,
        workspace_embedding: list[float] | None = None,
        *,
        register_workspace_canonical: bool = True,
    ) -> tuple[int | None, list[str]]:
        return self.memories.insert_memory(
            record, workspace_canonical, workspace_embedding,
            register_workspace_canonical=register_workspace_canonical,
        )

    def insert_memory_on_conn(
        self, conn: sqlite3.Connection, record: MemoryRecord,
        workspace_canonical: str | None = None,
    ) -> int:
        return self.memories.insert_memory_on_conn(conn, record, workspace_canonical)

    def _append_backup(self, record: MemoryRecord) -> None:
        return self.memories._append_backup(record)

    @staticmethod
    def _fetch_memory(conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
        return MemoriesStore._fetch_memory(conn, memory_id)

    def get_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
        return self.memories.get_memory_on_conn(conn, memory_id)

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        return self.memories.get_memory(memory_id)

    def get_memory_for_workspace(
        self, memory_id: int, ws_canonical: str, admitted: "WorkspaceScope" = None,
    ) -> dict[str, Any] | None:
        return self.memories.get_memory_for_workspace(memory_id, ws_canonical, admitted)

    def list_memories_for_workspace(
        self, ws_canonical: str, limit: int = 50, admitted: "WorkspaceScope" = None,
    ) -> list[dict[str, Any]]:
        return self.memories.list_memories_for_workspace(ws_canonical, limit, admitted)

    def update_memory(self, memory_id: int, updates: dict[str, Any]) -> bool:
        return self.memories.update_memory(memory_id, updates)

    def update_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int, updates: dict[str, Any]) -> bool:
        return self.memories.update_memory(memory_id, updates, conn=conn)

    def list_memories(self, workspace: str | None = None, subject: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.memories.list_memories(subject=subject, limit=limit)

    def active_subject_tag_rows(
        self, exclude_memory_id: int, workspace_canonical: str | None,
        *, limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.memories.active_subject_tag_rows(
            exclude_memory_id, workspace_canonical, limit=limit,
        )

    def subject_tags_knn(
        self,
        query_embedding: list[float],
        *,
        k: int,
        exclude_memory_id: int,
        workspace_canonical: str | None,
    ) -> list[dict[str, Any]]:
        return self.memories.subject_tags_knn(
            query_embedding, k=k, exclude_memory_id=exclude_memory_id,
            workspace_canonical=workspace_canonical,
        )

    def upsert_subject_tags_vector(self, memory_id: int, embedding: list[float]) -> bool:
        return self.memories.upsert_subject_tags_vector(memory_id, embedding)

    def delete_subject_tags_vector(self, memory_id: int) -> bool:
        return self.memories.delete_subject_tags_vector(memory_id)

    def missing_subject_tags_rows(self) -> list[dict[str, Any]]:
        return self.memories.missing_subject_tags_rows()

    def _filter_clauses(
        self,
        like_status_clause: str,
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
    ) -> tuple[list[str], list[Any]]:
        return MemoriesStore._filter_clauses(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
        )

    def count_filtered_memories(
        self,
        like_status_clause: str,
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
        ws_canonical: WorkspaceScope = None,
    ) -> int:
        return self.memories.count_filtered_memories(
            like_status_clause, tags_filter, after_dt, before_dt, source_type, ws_canonical,
        )

    def recall_by_filters(
        self,
        like_status_clause: str,
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
        limit: int,
        offset: int = 0,
        ws_canonical: WorkspaceScope = None,
    ) -> list[dict[str, Any]]:
        return self.memories.recall_by_filters(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
            limit, offset, ws_canonical,
        )

    def record_conflict_group(self, **kwargs: Any) -> dict[str, Any]:
        return self.conflicts.record_conflict_group(**kwargs)

    def get_conflict(self, conflict_id: int) -> dict[str, Any] | None:
        return self.conflicts.get_conflict(conflict_id)

    def escalate_structured_notice(self, notice_id: int, **kwargs: Any) -> dict[str, Any]:
        return self.conflicts.escalate_structured_notice(notice_id, **kwargs)

    def judge_conflict(self, conflict_id: int, **kwargs: Any) -> dict[str, Any]:
        return self.conflicts.judge_conflict(conflict_id, **kwargs)

    def resolve_conflicts_for(self, memory_id: int) -> int:
        return self.conflicts.resolve_conflicts_for(memory_id)

    def resolve_conflicts_for_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> int:
        return self.conflicts.resolve_conflicts_for_on_conn(conn, memory_id)

    def list_conflicts(
        self,
        status: str = "open",
        limit: int = 50,
        source: str | None = None,
        workspace: WorkspaceScope = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.conflicts.list_conflicts(status, limit, source, workspace, offset)

    def list_open_conflicts_for_memory_ids(
        self, memory_ids: list[int], *, include_applying: bool = False,
    ) -> list[dict[str, Any]]:
        return self.conflicts.list_open_conflicts_for_memory_ids(memory_ids, include_applying=include_applying)

    def get_memory_summaries(
        self, memory_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        return self.audit.get_memory_summaries(memory_ids)

    def resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved",
        *, expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return self.conflicts.resolve_conflict(
            conflict_id, reason=reason, status=status, expected_revision=expected_revision,
        )

    def is_pair_dismissed(self, left_id: int, right_id: int) -> bool:
        return self.conflicts.is_pair_dismissed(left_id, right_id)

    def get_memory_version(self, memory_id: int) -> int | None:
        return self.conflicts.get_memory_version(memory_id)

    def dismissed_pairs_for(self, memory_ids: list[int]) -> set[tuple[int, int]]:
        return self.conflicts.dismissed_pairs_for(memory_ids)

    # ------------------------------------------------------------------
    #  Semantic write-time notices
    # ------------------------------------------------------------------

    def record_semantic_notice(
        self,
        *,
        memory_id: int,
        peer_id: int | None,
        severity: str,
        notice_type: str,
        title: str,
        message: str,
        payload: dict[str, Any],
        dedupe_key: str | None = None,
        conflict_id: int | None = None,
        left_version: int | None = None,
        right_version: int | None = None,
        source: str = "semantic_evidence",
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
            source=source,
        )

    def claim_next_semantic_notice(self, workspace_canonical: WorkspaceScope = None) -> dict[str, Any] | None:
        return self.semantic_notices.claim_next_semantic_notice(workspace_canonical)

    def read_semantic_notice(
        self, notice_id: int, workspace_canonical: WorkspaceScope = None,
    ) -> dict[str, Any] | None:
        return self.semantic_notices.read_semantic_notice(notice_id, workspace_canonical)

    def list_semantic_notices(
        self, status: str = "open", limit: int = 10, workspace_canonical: WorkspaceScope = None,
    ) -> list[dict[str, Any]]:
        return self.semantic_notices.list_semantic_notices(
            status=status, limit=limit, workspace_canonical=workspace_canonical,
        )

    def semantic_notice_counts(
        self, workspace_canonical: WorkspaceScope = None,
    ) -> dict[str, int]:
        return self.semantic_notices.semantic_notice_counts(workspace_canonical)

    def is_semantic_pair_closed(
        self,
        left_id: int,
        right_id: int,
        left_version: int | None = None,
        right_version: int | None = None,
        notice_type: str = "semantic_evidence",
    ) -> bool:
        return self.semantic_notices.is_semantic_pair_closed(
            left_id, right_id, left_version, right_version, notice_type,
        )

    def update_semantic_notice_status(
        self,
        notice_id: int,
        status: str,
        reason: str = "",
        workspace_canonical: WorkspaceScope = None,
        conflict_id: int | None = None,
    ) -> dict[str, Any]:
        return self.semantic_notices.update_semantic_notice_status(
            notice_id, status, reason, workspace_canonical, conflict_id,
        )

    @property
    def scan_log_path(self) -> Path:
        return self.audit.scan_log_path

    @property
    def attention_log_path(self) -> Path:
        return self.audit.attention_log_path

    def log_attention(self, *, trigger: str, source: str, memory_ids: list[int]) -> None:
        return self.audit.log_attention(trigger=trigger, source=source, memory_ids=memory_ids)

    def log_scan(
        self,
        *,
        duration_sec: float,
        client: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        return self.audit.log_scan(
            duration_sec=duration_sec, client=client, agent_id=agent_id,
        )

    def _scan_log_last_completed(self) -> dict[str, Any] | None:
        return self.audit.scan_log_last_completed()

    # ------------------------------------------------------------------
    #  Legacy vector conflict candidate scan was removed.
    #
    #  Embedding/sqlite-vec remains available for evidence recall and
    #  workspace aliasing.  The old KNN conflict-candidate scanner and its
    #  tuning parameters are intentionally not kept here; the legacy MCP tool
    #  is no longer registered.


    # ------------------------------------------------------------------
    #  Edit / History
    # ------------------------------------------------------------------

    def update_tags_low_side_effect(
        self,
        memory_id: int,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        authorized: bool = False,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        return self.memories.update_tags_low_side_effect(
            memory_id, add_tags, remove_tags, authorized, conn=conn,
        )

    def update_metadata_fields_low_side_effect(
        self,
        memory_id: int,
        set_fields: dict[str, Any] | None = None,
        clear_fields: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        return self.memories.update_metadata_fields_low_side_effect(
            memory_id, set_fields=set_fields, clear_fields=clear_fields, authorized=authorized,
        )

    def update_metadata_fields_low_side_effect_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        set_fields: dict[str, Any] | None = None,
        clear_fields: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        return self.memories.update_metadata_fields_low_side_effect_on_conn(
            conn, memory_id, set_fields=set_fields,
            clear_fields=clear_fields, authorized=authorized,
        )

    def list_entities(
        self,
        limit: int = 50,
        include_unassigned: bool = True,
    ) -> dict[str, Any]:
        return self.memories.list_entities(limit, include_unassigned)

    def find_metadata_overlap_candidates(
        self,
        subject: str | None,
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.memories.find_metadata_overlap_candidates(subject, tags, exclude_id, limit)

    def find_semantic_overlap_candidates(
        self,
        subject: str | None,
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
        canonical_workspace: str | None = None,
        isolation: str = "none",
    ) -> list[dict[str, Any]]:
        return self.memories.find_semantic_overlap_candidates(
            subject, tags, exclude_id, limit,
            canonical_workspace=canonical_workspace,
            isolation=isolation,
        )

    def edit_memory(
        self,
        memory_id: int,
        new_content: str,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str | None = None,
    ) -> int | None:
        return self.memories.edit_memory(memory_id, new_content, new_subject, new_tags, reason)

    def edit_memory_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        new_content: str,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str | None = None,
        *,
        authorized: bool = True,
    ) -> int | None:
        return self.memories.edit_memory(
            memory_id, new_content, new_subject, new_tags, reason,
            conn=conn, authorized=authorized,
        )

    def edit_memory_intent(
        self,
        memory_id: int,
        *,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        expected_version: int | None = None,
        expected_content_hash: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        return self.memories.edit_memory_intent(
            memory_id,
            new_content=new_content,
            old_text=old_text,
            new_text=new_text,
            new_subject=new_subject,
            new_tags=new_tags,
            add_tags=add_tags,
            remove_tags=remove_tags,
            reason=reason,
            authorized=authorized,
            expected_version=expected_version,
            expected_content_hash=expected_content_hash,
            conn=conn,
        )

    def list_history(self, memory_id: int) -> list[dict[str, Any]]:
        return self.memories.list_history(memory_id)

    def cleanup_history(
        self, memory_id: int | None = None, older_than_days: int | None = None,
        *, conn: sqlite3.Connection | None = None,
    ) -> int:
        return self.memories.cleanup_history(memory_id, older_than_days, conn=conn)

    def audit_summary(self) -> dict[str, Any]:
        return self.audit.audit_summary()

    # ==================================================================
    #  v0.6.0: _vec_index_meta + vec-index state machine
    # ==================================================================

    # ---- _vec_index_meta CRUD ----

    @staticmethod
    def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
        return MetaStore.get_meta(conn, key)

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        return MetaStore.set_meta(conn, key, value)

    @staticmethod
    def _delete_meta(conn: sqlite3.Connection, key: str) -> None:
        return MetaStore.delete_meta(conn, key)

    def conflict_scan_state(self) -> dict[str, Any]:
        return self.meta.conflict_scan_state()

    def rearm_conflict_scan_if_drifted(self) -> bool:
        return self.meta.rearm_conflict_scan_if_drifted()

    def record_conflict_scan_page(
        self,
        *,
        epoch: str,
        detector_version: str,
        boundary: dict[str, Any],
        after_memory_id: int,
        next_anchor_memory_id: int | None,
        anchors_scanned: int,
        workspace: WorkspaceScope = None,
    ) -> bool:
        return self.meta.record_conflict_scan_page(
            epoch=epoch,
            detector_version=detector_version,
            boundary=boundary,
            after_memory_id=after_memory_id,
            next_anchor_memory_id=next_anchor_memory_id,
            anchors_scanned=anchors_scanned,
            workspace=workspace,
        )

    def complete_conflict_scan(
        self, *, epoch: str, detector_version: str, boundary: dict[str, Any]
    ) -> bool:
        return self.meta.complete_conflict_scan(
            epoch=epoch, detector_version=detector_version, boundary=boundary,
        )

    def get_vec_index_state(self) -> dict[str, Any]:
        return self.meta.get_vec_index_state()

    def get_active_dim(self) -> int | None:
        return self.meta.get_active_dim()

    def set_active_dim(self, dim: int) -> None:
        return self.meta.set_active_dim(dim)

    def mark_space_rebuild_started(self) -> None:
        return self.meta.mark_space_rebuild_started()

    def space_rebuild_pending_ids(self, limit: int) -> list[int]:
        return self.meta.space_rebuild_pending_ids(limit)

    def stale_index_ids(self, limit: int, workspace: WorkspaceScope = None) -> list[int]:
        return self.meta.stale_index_ids(limit, workspace)

    def maybe_complete_space_rebuild(self, embedding_space_id: str) -> bool:
        return self.meta.maybe_complete_space_rebuild(embedding_space_id)

    def require_space_rebuild(self, embedding_space_id: str, reason: str) -> None:
        return self.meta.require_space_rebuild(embedding_space_id, reason)

    def init_vec_index_state(
        self,
        embedding_space_id: str | None,
        has_managed_embedder: bool,
        active_dim: int | None = None,
    ) -> None:
        return self.meta.init_vec_index_state(
            embedding_space_id, has_managed_embedder, active_dim=active_dim,
        )

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
