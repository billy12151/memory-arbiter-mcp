from __future__ import annotations

import json
import os
import threading
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from ..config import Settings
from ..degrade import DegradeState
from ..db_generation import (
    LegacyDatabaseError,
    database_startup_lock,
    detect_database_generation,
    legacy_database_message,
    require_current_or_new_database,
)
from ..models import MemoryRecord, utc_now_iso
from ._core_delegates import _CoreDelegatesMixin
from .semantic_notices import SemanticNoticeStore
from .audit import AuditStore
from .meta import MetaStore
from .schema import SchemaStore
from .workspaces import WorkspaceStore, _coerce_ws, _normalize_alias_key
from .conflicts import ConflictStore
from .memories import MemoriesStore
from .backup_replay import BackupReplayStore
from .evidence_store import EvidenceStore
from .scan_queue import ScanQueueStore
from .internal_conflicts import InternalConflictStore

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


class MemoryDB(_CoreDelegatesMixin):
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
        # 0.16.12 P1-T6: per-scope linked-open-items df cache. Fingerprint =
        # (COUNT(active), SUM(version)) — most product write paths move one of
        # the two (insert/delete/status flip ⇒ COUNT; edit ⇒ version); the
        # paths that don't (tags-only edits, workspace moves) invalidate
        # explicitly via invalidate_linked_df_cache().
        self._linked_df_cache: dict[tuple[str, ...], tuple[tuple[int, int], dict[str, int]]] = {}
        self._linked_df_cache_lock = threading.Lock()
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
        self.scan_queue = ScanQueueStore(self)
        # 0.17.0 Part 2: write-time conflict backlog.（0.17.1 claims 全退）
        from .conflict_backlog import ConflictBacklogStore
        self.conflict_backlog = ConflictBacklogStore(self)
        self.internal_conflicts = InternalConflictStore(self)
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

    def invalidate_linked_df_cache(self) -> None:
        """Explicit df-cache invalidation for the write paths the
        (COUNT(active), SUM(version)) fingerprint cannot see: tags-only
        edits (no version bump) and workspace moves/renames (scope membership
        changes without count/version movement) — first-round review finding.
        Clearing all scopes is deliberately conservative."""
        with self._linked_df_cache_lock:
            self._linked_df_cache.clear()

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
                # 0.16.0 §6⑲: the additive completion channel runs on BOTH
                # fresh and existing databases — historically an existing DB
                # ran zero DDL at startup, leaving new columns/tables without
                # a creation point. Failures (read-only files) degrade to a
                # warning, never a failed boot.
                try:
                    from .additive import ensure_additive_structures
                    applied = ensure_additive_structures(conn)
                    if applied:
                        self.state.warn(
                            "additive schema completion applied: " + ", ".join(applied)
                        )
                except sqlite3.Error as exc:
                    self.state.warn(
                        f"additive schema completion skipped: {exc}. "
                        "0.16.0 scan structures are unavailable until the "
                        "database is writable."
                    )
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
        # B5 补齐（R3/R4 指出）：另两个置位点同步记时间戳，否则只读库降级
        # 形态下 jsonl_backup_active=True 而 last_used_at=None（观测字段解释不了）。
        self.state.jsonl_backup_last_used_at = utc_now_iso()
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

    # (0.17.0 C6: ensure_evidence_vec_table forward retired with the unit
    # vec table; ensure_memory_row_vec_table owns the evidence channel.)


    # ------------------------------------------------------------------
    #  0.16.0 conflict-scan pipeline (watermarks + judgment queue)
    # ------------------------------------------------------------------


    def scan_queue_counts(self) -> dict[str, int]:
        return self.scan_queue.counts()

    def scan_queue_backlog(self) -> int:
        return self.scan_queue.backlog()

    def scan_queue_refresh_stale_pins(self) -> int:
        return self.scan_queue.refresh_stale_pins()

    def pending_scan_memory_ids(self, *, after_id: int = 0, limit: int = 100) -> list[int]:
        """Active memories whose scan watermark is missing or behind their
        version (never-scanned, edited since the last pass, or moved)."""
        if not self._db_available:
            return []
        with self.connection() as conn:
            return [
                int(row["id"]) for row in conn.execute(
                    """SELECT id FROM memories
                       WHERE status='active' AND id > ?
                         AND (scan_watermark IS NULL OR scan_watermark < version)
                       ORDER BY id LIMIT ?""",
                    (int(after_id), max(1, int(limit))),
                ).fetchall()
            ]

    def least_recently_scanned_ids(self, *, limit: int = 20, exclude_ids: "list[int] | None" = None) -> list[int]:
        """0.17.0 P2-6.2: slow-lane anchors — oldest last_scanned_at first,
        NULL (never scanned) first of all; watermark-current memories only
        (a pending main-batch memory is the fast lane's job)."""
        if not self._db_available:
            return []
        exclude_sql = ""
        params: list[Any] = []
        if exclude_ids:
            marks = ",".join("?" for _ in exclude_ids)
            exclude_sql = f" AND id NOT IN ({marks})"
            params.extend(int(i) for i in exclude_ids)
        # bind order follows the SQL: NOT IN placeholders first, LIMIT last
        # (adversarial review P1-2 — the swapped order silently turned the
        # first exclude id into the LIMIT and re-selected just-scanned rows).
        params.append(max(1, int(limit)))
        try:
            with self.connection() as conn:
                return [
                    int(row[0]) for row in conn.execute(
                        f"""SELECT id FROM memories
                            WHERE status='active'
                              AND (scan_watermark IS NOT NULL AND scan_watermark >= version)
                            {exclude_sql}
                            ORDER BY last_scanned_at IS NOT NULL, last_scanned_at ASC, id ASC
                            LIMIT ?""",
                        params,
                    ).fetchall()
                ]
        except sqlite3.Error:
            return []

    def pending_scan_memory_count(self) -> int:
        if not self._db_available:
            return 0
        with self.connection() as conn:
            return int(conn.execute(
                """SELECT COUNT(*) FROM memories
                   WHERE status='active'
                     AND (scan_watermark IS NULL OR scan_watermark < version)"""
            ).fetchone()[0])

    def mark_scanned(self, memory_id: int, version: int) -> bool:
        """Advance one memory's watermark; a no-op when the version moved on."""
        if not self._db_available or not self.state.sqlite_writable:
            return False
        try:
            with self.write_transaction() as conn:
                # 0.17.0 P2-6.2: the slow lane rotates by wall time — stamp it
                # on the same UPDATE (one transaction, no extra write).
                from ..models import utc_now_iso
                cur = conn.execute(
                    "UPDATE memories SET scan_watermark=?, last_scanned_at=? "
                    "WHERE id=? AND version=?",
                    (int(version), utc_now_iso(), int(memory_id), int(version)),
                )
                return bool(cur.rowcount)
        except sqlite3.Error:
            return False

    def clear_all_scan_watermarks(self) -> int:
        """Arm a full pipeline round: every active memory becomes pending.

        This is the detector/prompt epoch 布防 path (plan §6⑪/commit 8) —
        watermark-NULL first-scan memories were already covered implicitly.
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0
        with self.write_transaction() as conn:
            cur = conn.execute(
                "UPDATE memories SET scan_watermark=NULL WHERE status='active'"
            )
            return int(cur.rowcount or 0)



    # ------------------------------------------------------------------
    #  Legacy vector conflict candidate scan was removed.
    #
    #  Embedding/sqlite-vec remains available for evidence recall and
    #  workspace aliasing.  The old KNN conflict-candidate scanner and its
    #  tuning parameters are intentionally not kept here; the legacy MCP tool
    #  is no longer registered.



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
