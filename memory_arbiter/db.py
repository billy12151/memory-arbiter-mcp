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
from typing import Any, Iterator, Optional, Tuple

from .claims_db import StructuredClaimStore
from .config import Settings
from .conflict_judgments import ConflictJudgmentStore
from .degrade import DegradeState
from .models import MemoryRecord, utc_now_iso

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
        self._claim_store = StructuredClaimStore(self)
        self._judgment_store = ConflictJudgmentStore(self)
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
        # NOTE: executescript() issues an implicit COMMIT before running,
        # so it cannot be wrapped in an explicit BEGIN IMMEDIATE.  The
        # CREATE TABLE IF NOT EXISTS statements are idempotent by design;
        # column migrations below use PRAGMA table_info probes so they are
        # safe even if two processes start simultaneously.
        conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  content TEXT NOT NULL,
                  agent_id TEXT NOT NULL,
                  workspace TEXT NOT NULL,
                  tags TEXT NOT NULL DEFAULT '[]',
                  source_type TEXT NOT NULL,
                  source_ref TEXT,
                  event_time TEXT NOT NULL,
                  ingest_time TEXT NOT NULL,
                  confidence REAL NOT NULL DEFAULT 0.5,
                  protection_level TEXT NOT NULL DEFAULT 'normal',
                  status TEXT NOT NULL DEFAULT 'active',
                  subject TEXT,
                  metadata TEXT NOT NULL DEFAULT '{}',
                  workspace_canonical TEXT,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conflicts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  left_id INTEGER NOT NULL,
                  right_id INTEGER NOT NULL,
                  subject TEXT,
                  status TEXT NOT NULL DEFAULT 'open',
                  reason TEXT NOT NULL,
                  winner_id INTEGER,
                  created_at TEXT NOT NULL,
                  resolved_at TEXT,
                  FOREIGN KEY(left_id) REFERENCES memories(id),
                  FOREIGN KEY(right_id) REFERENCES memories(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(workspace, agent_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(workspace, subject);
                CREATE INDEX IF NOT EXISTS idx_memories_event ON memories(event_time, ingest_time);
                -- v0.7.6: conflict-lookup indexes (ordinary, not unique partial —
                -- uniqueness is enforced by record_conflict_enriched logic).
                CREATE INDEX IF NOT EXISTS idx_conflicts_status_left ON conflicts(status, left_id);
                CREATE INDEX IF NOT EXISTS idx_conflicts_status_right ON conflicts(status, right_id);
                CREATE INDEX IF NOT EXISTS idx_conflicts_status_created ON conflicts(status, created_at);
                CREATE TABLE IF NOT EXISTS memory_history (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  memory_id INTEGER NOT NULL,
                  content_snapshot TEXT NOT NULL,
                  subject_snapshot TEXT,
                  tags_snapshot TEXT,
                  version INTEGER NOT NULL,
                  changed_at TEXT NOT NULL,
                  reason TEXT,
                  FOREIGN KEY(memory_id) REFERENCES memories(id)
                );
                CREATE INDEX IF NOT EXISTS idx_history_memory ON memory_history(memory_id, changed_at);

                -- v0.6.0: section-split derived index (no body column; zero redundancy)
                CREATE TABLE IF NOT EXISTS memory_sections (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  memory_id INTEGER NOT NULL,
                  section_index INTEGER NOT NULL,
                  title TEXT,
                  title_path TEXT,
                  summary TEXT,
                  anchor_text TEXT,
                  occurrence_index INTEGER NOT NULL DEFAULT 0,
                  start_offset INTEGER NOT NULL,
                  end_offset INTEGER NOT NULL,
                  provenance TEXT NOT NULL,
                  embedding_truncated INTEGER NOT NULL DEFAULT 0,
                  embedding_original_tokens INTEGER NOT NULL DEFAULT 0,
                  embedding_used_tokens INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(memory_id) REFERENCES memories(id),
                  UNIQUE(memory_id, section_index)
                );

                -- v0.6.0: global vector-index metadata (KV store)
                CREATE TABLE IF NOT EXISTS _vec_index_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );

                -- workspace alias canonicalization registry. Each distinct
                -- canonical workspace name gets one row; memories.workspace stores
                -- the raw input, memories.workspace_canonical the resolved name.
                CREATE TABLE IF NOT EXISTS workspace_canonicals (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL UNIQUE,
                  created_at TEXT NOT NULL
                );
                """
        )
        # Idempotent column migrations — probe each individually so
        # partial upgrades and concurrent first-starts are safe.
        self._migrate_add_column(conn, "memories", "version",
                                 "INTEGER NOT NULL DEFAULT 1")
        # Workspace alias canonicalization: raw input stays in `workspace`,
        # resolved canonical name lands here (NULL on old rows until backfilled).
        self._migrate_add_column(conn, "memories", "workspace_canonical",
                                 "TEXT")
        # Index created after the column migration so existing DBs (where the
        # CREATE TABLE above is a no-op) don't reference a not-yet-added column.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_canonical "
            "ON memories(workspace_canonical, status)"
        )
        self._migrate_add_column(conn, "memories", "split_status",
                                 "TEXT")
        self._migrate_add_column(conn, "memories", "split_revision",
                                 "INTEGER NOT NULL DEFAULT 0")
        # v0.7.5: conflict-scan enrichment columns on the conflicts table.
        # conflict_type / conflict_point carry the *what* (agent LLM judgement);
        # suggested_winner / confidence_hint / source carry the *suggestion*
        # (who wins + provenance). Distinct from the existing winner_id/reason
        # columns which record the *outcome* of arbitration.
        self._migrate_add_column(conn, "conflicts", "conflict_type",
                                 "TEXT")
        self._migrate_add_column(conn, "conflicts", "conflict_point",
                                 "TEXT")
        self._migrate_add_column(conn, "conflicts", "suggested_winner",
                                 "INTEGER")
        self._migrate_add_column(conn, "conflicts", "confidence_hint",
                                 "TEXT")
        self._migrate_add_column(conn, "conflicts", "source",
                                 "TEXT")
        # v0.7.6: scan-refresh metadata. These columns let the agent-side scan
        # task decide whether to re-run LLM on an existing open conflict (e.g.
        # when memory version or scan model changed). The system itself never
        # auto-triggers refresh — it just stores the provenance and exposes a
        # refresh interface (record_conflict_enriched(refresh=True)).
        self._migrate_add_column(conn, "conflicts", "left_version", "INTEGER")
        self._migrate_add_column(conn, "conflicts", "right_version", "INTEGER")
        self._migrate_add_column(conn, "conflicts", "scan_prompt_version", "TEXT")
        self._migrate_add_column(conn, "conflicts", "scan_model", "TEXT")
        self._migrate_add_column(conn, "conflicts", "refreshed_at", "TEXT")
        conn.commit()
        self._migrate_v090_claims(conn)

    def _migrate_v090_claims(self, conn: sqlite3.Connection) -> None:
        """Install the complete v0.9 claim/judgment schema atomically."""
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._migrate_add_column(conn, "memories", "claim_revision",
                                     "INTEGER NOT NULL DEFAULT 1")
            self._migrate_add_column(conn, "memories", "claims_indexed_revision", "INTEGER")
            self._migrate_add_column(conn, "memories", "claims_reconciled_revision", "INTEGER")
            self._migrate_add_column(conn, "memories", "claim_ambiguous_count",
                                     "INTEGER NOT NULL DEFAULT 0")
            self._migrate_add_column(conn, "memories", "structured_enrich_ms", "REAL")
            self._migrate_add_column(conn, "memories", "structured_candidate_count", "INTEGER")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_claims (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  memory_id INTEGER NOT NULL,
                  entity TEXT NOT NULL,
                  attribute TEXT NOT NULL,
                  scope TEXT NOT NULL DEFAULT '',
                  value TEXT NOT NULL,
                  raw_value TEXT,
                  value_type TEXT,
                  extractor_rule TEXT,
                  evidence TEXT,
                  start_offset INTEGER,
                  end_offset INTEGER,
                  claim_revision INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                  UNIQUE(memory_id, entity, attribute, scope)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_key ON memory_claims(entity, attribute)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_claims_memory_revision "
                "ON memory_claims(memory_id, claim_revision)"
            )
            self._migrate_add_column(conn, "conflicts", "left_claim_revision", "INTEGER")
            self._migrate_add_column(conn, "conflicts", "right_claim_revision", "INTEGER")
            self._migrate_add_column(conn, "conflicts", "judgment_status", "TEXT")
            self._migrate_add_column(conn, "conflicts", "structured_details", "TEXT")
            self._migrate_add_column(conn, "conflicts", "structured_detected_at", "TEXT")
            self._migrate_add_column(conn, "conflicts", "scan_detected_at", "TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conflict_judgments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conflict_id INTEGER NOT NULL,
                  verdict TEXT NOT NULL,
                  recommended_use TEXT NOT NULL,
                  suggested_winner INTEGER,
                  confidence_hint TEXT,
                  reason TEXT NOT NULL,
                  judge_type TEXT NOT NULL,
                  judge_ref TEXT,
                  left_version INTEGER NOT NULL,
                  right_version INTEGER NOT NULL,
                  left_claim_revision INTEGER NOT NULL,
                  right_claim_revision INTEGER NOT NULL,
                  supersedes_judgment_id INTEGER,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(conflict_id) REFERENCES conflicts(id) ON DELETE CASCADE,
                  FOREIGN KEY(suggested_winner) REFERENCES memories(id),
                  FOREIGN KEY(supersedes_judgment_id) REFERENCES conflict_judgments(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_judgments_conflict_created "
                "ON conflict_judgments(conflict_id, created_at)"
            )
            self._migrate_add_column(conn, "conflicts", "active_judgment_id", "INTEGER")
            self._migrate_add_column(conn, "conflicts", "resolution_kind", "TEXT")
            self._migrate_add_column(conn, "conflicts", "conflict_scope", "TEXT")
            self._migrate_add_column(conn, "conflict_judgments", "resolution_kind", "TEXT")
            self._migrate_add_column(conn, "conflict_judgments", "conflict_scope", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conflicts_judgment_status "
                "ON conflicts(status, judgment_status)"
            )
            # Best-effort provenance backfill for databases upgraded from the
            # first v0.9 build. New writes set these channels explicitly.
            conn.execute(
                "UPDATE conflicts SET structured_detected_at=COALESCE(structured_detected_at, created_at) "
                "WHERE left_claim_revision IS NOT NULL"
            )
            conn.execute(
                "UPDATE conflicts SET scan_detected_at=COALESCE(scan_detected_at, created_at) "
                "WHERE left_claim_revision IS NULL "
                "AND source IS NOT NULL "
                "AND source NOT IN ('structured_claim','metadata_write_hint')"
            )
            # SQLite cannot add a REFERENCES clause to an existing column
            # without rebuilding conflicts.  This trigger enforces both target
            # existence and same-conflict ownership, migration-safely.
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_conflicts_active_judgment_fk
                BEFORE UPDATE OF active_judgment_id ON conflicts
                WHEN NEW.active_judgment_id IS NOT NULL
                 AND NOT EXISTS (
                   SELECT 1 FROM conflict_judgments j
                   WHERE j.id=NEW.active_judgment_id AND j.conflict_id=NEW.id
                 )
                BEGIN
                  SELECT RAISE(ABORT, 'invalid active_judgment_id');
                END
                """
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _migrate_add_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        decl: str,
    ) -> None:
        """Add *column* to *table* if it does not yet exist (idempotent)."""
        cols = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ------------------------------------------------------------------
    #  Feature probing
    # ------------------------------------------------------------------

    def _probe_features(self, conn: sqlite3.Connection) -> None:
        # sqlite-vec
        if self.settings.enable_sqlite_vec:
            try:
                import sqlite_vec  # type: ignore

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._sqlite_vec_loadable = True
                self.state.sqlite_vec_available = True
                self.state.mode = "sqlite_vec"
                self._ensure_vec_table(conn)
                self._ensure_section_vec_table(conn)
                self._ensure_workspace_vec_table(conn)
                # v0.9.4: migrate vec0 tables to include parent_status metadata column
                self._migrate_vec_parent_status(conn)
            except Exception as exc:  # pragma: no cover
                self.state.warn(
                    f"sqlite-vec unavailable: {exc}. "
                    "Semantic recall disabled; falling back to FTS5 or keyword search."
                )
        else:
            probe = self._probe_sqlite_vec_loadable()
            if probe is True:
                self.state.warn(
                    "sqlite-vec is installed and loadable but disabled by configuration. "
                    "Set MEMORY_ARBITER_ENABLE_SQLITE_VEC=true to enable semantic recall."
                )
            else:
                self.state.warn(
                    "sqlite-vec disabled by configuration. Semantic recall disabled. "
                    "Install with `pip install '.[vec]'` and set "
                    "MEMORY_ARBITER_ENABLE_SQLITE_VEC=true to enable."
                )

        # FTS5
        try:
            self._ensure_fts(conn)
            self._rebuild_fts(conn)
            self.state.fts5_available = True
            if not self.state.sqlite_vec_available:
                self.state.mode = "fts5"
        except sqlite3.Error as exc:
            self.state.warn(f"SQLite FTS5 unavailable: {exc}. Falling back to LIKE/keyword search.")
            if not self.state.sqlite_vec_available:
                self.state.mode = "like"

        # Write probe
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS write_probe (id INTEGER)")
            conn.execute("INSERT INTO write_probe(id) VALUES (1)")
            conn.execute("DELETE FROM write_probe")
            conn.commit()
        except sqlite3.Error as exc:
            self.state.sqlite_writable = False
            self.state.mode = "jsonl_backup"
            self.state.jsonl_backup_active = True
            self.state.warn(
                f"SQLite opened read-only or write probe failed: {exc}. "
                "Writes will use JSONL backup when possible."
            )

    def _probe_sqlite_vec_loadable(self) -> Optional[bool]:
        """Best-effort: can we import + load sqlite-vec right now?"""
        try:
            import sqlite_vec  # type: ignore

            probe_conn = sqlite3.connect(":memory:")
            probe_conn.enable_load_extension(True)
            sqlite_vec.load(probe_conn)
            probe_conn.close()
            return True
        except ImportError:
            return False
        except Exception:
            return None

    # ------------------------------------------------------------------
    #  FTS / Vec table helpers (called only during init)
    # ------------------------------------------------------------------

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
            conn.commit()
        except sqlite3.Error:
            conn.rollback()

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories_fts'"
        ).fetchone()
        if row:
            sql = str(row["sql"] or "").lower()
            if "tokenize='trigram'" in sql or 'tokenize="trigram"' in sql or "tokenize=trigram" in sql:
                return
            conn.execute("DROP TABLE memories_fts")
            conn.commit()
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE memories_fts USING fts5("
                "content, tags, subject, content='memories', content_rowid='id', "
                "tokenize='trigram')"
            )
        except sqlite3.Error as exc:
            self.state.warn(f"FTS5 trigram tokenizer unavailable: {exc}. Falling back to default FTS5 tokenizer.")
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "content, tags, subject, content='memories', content_rowid='id')"
            )

    def _ensure_vec_table(self, conn: sqlite3.Connection) -> None:
        dim = int(getattr(self.settings, "vec_dim", 768) or 768)
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
                f"USING vec0(id INTEGER PRIMARY KEY, parent_status TEXT, embedding float[{dim}])"
            )
            conn.commit()
        except sqlite3.Error as exc:
            self.state.warn(f"vec0 table creation failed (dim={dim}): {exc}. Semantic recall disabled.")
            self.state.sqlite_vec_available = False
            self._sqlite_vec_loadable = False

    def _ensure_section_vec_table(self, conn: sqlite3.Connection) -> None:
        """Create the section-level vec0 table (v0.6.0)."""
        dim = int(getattr(self.settings, "vec_dim", 768) or 768)
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_sections_vec "
                f"USING vec0(id INTEGER PRIMARY KEY, parent_status TEXT, embedding float[{dim}])"
            )
            conn.commit()
        except sqlite3.Error as exc:
            self.state.warn(
                f"memory_sections_vec creation failed (dim={dim}): {exc}. "
                "Section split will be unavailable."
            )

    def _ensure_workspace_vec_table(self, conn: sqlite3.Connection) -> None:
        """Create the workspace-canonical vec0 table for alias resolution.

        One vector per canonical workspace name (id == workspace_canonicals.id).
        Used by resolve_workspace_canonical to find the nearest existing
        canonical for a raw workspace string. Failure is non-fatal — alias
        resolution degrades to exact string match.
        """
        dim = int(getattr(self.settings, "vec_dim", 768) or 768)
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS workspace_canonicals_vec "
                f"USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
            conn.commit()
        except sqlite3.Error as exc:
            self.state.warn(
                f"workspace_canonicals_vec creation failed (dim={dim}): {exc}. "
                "Workspace alias resolution will fall back to exact match."
            )

    def _migrate_vec_parent_status(self, conn: sqlite3.Connection) -> None:
        """v0.9.4: add parent_status metadata column to vec0 tables (idempotent).

        Detects whether memories_vec / memory_sections_vec carry the
        ``parent_status`` column.  If both are already present the migration
        is a no-op.  If absent the table is rebuilt via DROP+CREATE+re-insert
        (vec0 does not support ALTER ADD COLUMN — the column set is frozen at
        CREATE).  Parent_status values are derived from ``memories.status``
        (or ``memory_sections.memory_id`` → ``memories.status``).  Orphan rows
        (no parent in memories/memory_sections) get ``COALESCE(..., 'deleted')``.

        Requires sqlite-vec >= 0.1.6 for metadata-column filter support.
        """
        dim = int(getattr(self.settings, "vec_dim", 768) or 768)
        mem_has_col = False
        sec_has_col = False
        try:
            # PRAGMA table_info on vec0 virtual tables works for checking columns
            mem_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(memories_vec)")}
            if "parent_status" in mem_cols:
                mem_has_col = True
            sec_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(memory_sections_vec)")}
            if "parent_status" in sec_cols:
                sec_has_col = True
        except sqlite3.Error:
            pass  # table doesn't exist yet — will be created fresh below

        if not mem_has_col:
            try:
                # Backup existing rows (id + raw embedding as returned by vec0).
                conn.execute("DROP TABLE IF EXISTS _vec_mem_bak")
                conn.execute("CREATE TABLE _vec_mem_bak(id INTEGER PRIMARY KEY, embedding BLOB)")
                conn.execute("INSERT INTO _vec_mem_bak SELECT id, embedding FROM memories_vec")
                conn.execute("DROP TABLE memories_vec")
                conn.execute(
                    f"CREATE VIRTUAL TABLE memories_vec "
                    f"USING vec0(id INTEGER PRIMARY KEY, parent_status TEXT, embedding float[{dim}])"
                )
                # Repopulate: get parent status from memories table, with
                # COALESCE for deleted/orphan rows.  Embedding format: vec0 may
                # return struct-packed bytes — convert to JSON list for re-insert.
                raw_rows = conn.execute("SELECT id, embedding FROM _vec_mem_bak").fetchall()
                for row in raw_rows:
                    mem_id = int(row["id"])
                    # Fetch parent status — COALESCE('deleted') for missing memories.
                    parent_row = conn.execute(
                        "SELECT status FROM memories WHERE id = ?", (mem_id,)
                    ).fetchone()
                    parent_status = parent_row["status"] if parent_row else "deleted"
                    embedding_raw = row["embedding"]
                    if isinstance(embedding_raw, bytes):
                        n = len(embedding_raw) // 4
                        embedding = json.dumps(list(struct.unpack(f"<{n}f", embedding_raw)))
                    elif isinstance(embedding_raw, list):
                        embedding = json.dumps(embedding_raw)
                    else:
                        embedding = json.dumps(embedding_raw) if not isinstance(embedding_raw, str) else embedding_raw
                    conn.execute(
                        "INSERT INTO memories_vec(id, parent_status, embedding) VALUES (?, ?, ?)",
                        (mem_id, parent_status, embedding),
                    )
                conn.execute("DROP TABLE _vec_mem_bak")
                # Persist the DROP+CREATE+re-insert.  Python sqlite3's default
                # isolation level opens an implicit transaction on DML; without
                # this commit the entire migration is rolled back when the init
                # connection closes, leaving the vec table on the old (no
                # parent_status) schema on any pre-existing database.
                conn.commit()
            except sqlite3.Error as exc:
                self.state.warn(
                    f"memories_vec parent_status migration failed: {exc}. "
                    "KNN will fall back to post-JOIN filtering."
                )

        if not sec_has_col:
            try:
                conn.execute("DROP TABLE IF EXISTS _vec_sec_bak")
                conn.execute("CREATE TABLE _vec_sec_bak(id INTEGER PRIMARY KEY, embedding BLOB)")
                conn.execute("INSERT INTO _vec_sec_bak SELECT id, embedding FROM memory_sections_vec")
                conn.execute("DROP TABLE memory_sections_vec")
                conn.execute(
                    f"CREATE VIRTUAL TABLE memory_sections_vec "
                    f"USING vec0(id INTEGER PRIMARY KEY, parent_status TEXT, embedding float[{dim}])"
                )
                raw_rows = conn.execute("SELECT id, embedding FROM _vec_sec_bak").fetchall()
                for row in raw_rows:
                    sec_id = int(row["id"])
                    # Fetch parent status via memory_sections JOIN (N17: COALESCE for orphan sections)
                    parent_row = conn.execute(
                        "SELECT COALESCE(m.status, 'deleted') AS status "
                        "FROM memory_sections s "
                        "LEFT JOIN memories m ON m.id = s.memory_id "
                        "WHERE s.id = ?", (sec_id,)
                    ).fetchone()
                    parent_status = parent_row["status"] if parent_row else "deleted"
                    embedding_raw = row["embedding"]
                    if isinstance(embedding_raw, bytes):
                        n = len(embedding_raw) // 4
                        embedding = json.dumps(list(struct.unpack(f"<{n}f", embedding_raw)))
                    elif isinstance(embedding_raw, list):
                        embedding = json.dumps(embedding_raw)
                    else:
                        embedding = json.dumps(embedding_raw) if not isinstance(embedding_raw, str) else embedding_raw
                    conn.execute(
                        "INSERT INTO memory_sections_vec(id, parent_status, embedding) VALUES (?, ?, ?)",
                        (sec_id, parent_status, embedding),
                    )
                conn.execute("DROP TABLE _vec_sec_bak")
                # Persist the section-vec migration for the same reason as the
                # memories_vec branch above (uncommitted DML rolls back on close).
                conn.commit()
            except sqlite3.Error as exc:
                self.state.warn(
                    f"memory_sections_vec parent_status migration failed: {exc}. "
                    "KNN will fall back to post-JOIN filtering."
                )

    # ------------------------------------------------------------------
    #  Embedding operations
    # ------------------------------------------------------------------

    def store_embedding(self, memory_id: int, embedding: list[float]) -> Tuple[bool, list[str]]:
        warnings: list[str] = []
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; embedding not stored."]
        if not self.state.sqlite_vec_available:
            return False, ["sqlite-vec unavailable; embedding not stored."]
        if not embedding:
            return False, ["embedding is empty (encode failed); not stored."]
        try:
            with self.connection() as conn:
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

    def resolve_workspace_canonical(
        self,
        ws_raw: Optional[str],
        embedder: Any = None,
        *,
        match_distance: Optional[float] = None,
        register_new: bool = True,
    ) -> dict[str, Any]:
        """Resolve a raw workspace string to its canonical name (alias merge).

        Strategy (double-store: raw stays in memories.workspace, resolved name
        goes to memories.workspace_canonical):
          1. Exact match against workspace_canonicals.name → reuse it.
          2. If an embedder + sqlite-vec are available, embed the raw string and
             KNN against workspace_canonicals_vec; if the nearest canonical is
             within ``match_distance`` reuse it (handles 金营项目 / 金科营销项目).
          3. Otherwise it is a NEW canonical. When ``register_new`` is True, the
             raw string is registered as a new canonical (+ its vector); when
             False (read/query path) nothing is written.

        Returns a dict:
          {canonical, is_new, matched_by: exact|vector|new|fallback,
           distance, similar: [{name, distance}, ...]}

        Never raises — degrades to exact string identity so callers can rely on
        a canonical always coming back (falls back to the raw string itself).
        """
        raw = (ws_raw or "").strip()
        result: dict[str, Any] = {
            "canonical": raw or "default",
            "is_new": False,
            "matched_by": "fallback",
            "distance": None,
            "similar": [],
        }
        if not raw:
            result["canonical"] = "default"
            return result
        if not self._db_available:
            return result
        if match_distance is None:
            match_distance = float(getattr(self.settings, "workspace_match_distance", 0.25) or 0.25)

        try:
            with self.connection() as conn:
                # 1. Exact canonical hit.
                exact = conn.execute(
                    "SELECT id, name FROM workspace_canonicals WHERE name = ?",
                    (raw,),
                ).fetchone()
                if exact:
                    result.update({"canonical": exact["name"], "is_new": False, "matched_by": "exact", "distance": 0.0})
                    return result

                # 2. Vector nearest-canonical (only when embedding is available).
                vec_ok = self.state.sqlite_vec_available and embedder is not None
                embedding = None
                if vec_ok:
                    try:
                        er = embedder.embed_text(prefix="", body=raw)
                        embedding = list(er.embedding) if er and er.embedding else None
                    except Exception:
                        embedding = None
                if embedding:
                    try:
                        query_json = json.dumps(embedding)
                        # Full-scan cosine (not MATCH/L2): the canonical table is
                        # tiny (one row per project) and embeddinggemma vectors are
                        # unnormalized, so cosine is the scale-invariant choice —
                        # mirrors section_vec_distance_match.
                        rows = conn.execute(
                            """SELECT c.name AS name,
                                      vec_distance_cosine(v.embedding, ?) AS distance
                               FROM workspace_canonicals_vec v
                               JOIN workspace_canonicals c ON c.id = v.id
                               ORDER BY distance
                               LIMIT 5""",
                            (query_json,),
                        ).fetchall()
                        result["similar"] = [{"name": r["name"], "distance": float(r["distance"])} for r in rows]
                        if rows and float(rows[0]["distance"]) <= match_distance:
                            result.update({
                                "canonical": rows[0]["name"],
                                "is_new": False,
                                "matched_by": "vector",
                                "distance": float(rows[0]["distance"]),
                            })
                            return result
                    except sqlite3.Error:
                        pass  # vec query failed — fall through to new-canonical path

                # 3. New canonical.
                result.update({"canonical": raw, "is_new": True, "matched_by": "new"})
                if register_new and self.state.sqlite_writable:
                    try:
                        now = utc_now_iso()
                        cur = conn.execute(
                            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                            (raw, now),
                        )
                        row = conn.execute(
                            "SELECT id FROM workspace_canonicals WHERE name = ?", (raw,)
                        ).fetchone()
                        if row and embedding and self.state.sqlite_vec_available:
                            conn.execute(
                                "INSERT OR REPLACE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                                (int(row["id"]), json.dumps(embedding)),
                            )
                        conn.commit()
                    except sqlite3.Error:
                        pass  # registration best-effort; canonical still returned
                return result
        except sqlite3.Error:
            return result

    def delete_embedding(self, memory_id: int) -> Tuple[bool, list[str]]:
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; embedding not deleted."]
        if not self.state.sqlite_vec_available:
            return False, ["sqlite-vec unavailable; embedding not deleted."]
        try:
            with self.connection() as conn:
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
            with self.write_transaction() as conn:
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
            with self.write_transaction() as conn:
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
            with self.write_transaction() as conn:
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
            with self.connection() as conn:
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

    def _resync_vec_parent_status(self) -> dict[str, int]:
        """Repair mismatched vec.parent_status to match memories.status (v0.9.4)."""
        try:
            with self.write_transaction() as conn:
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
            with self.connection() as conn:
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
            with self.connection() as conn:
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

    def insert_memory(
        self, record: MemoryRecord, workspace_canonical: Optional[str] = None
    ) -> Tuple[Optional[int], list[str]]:
        warnings: list[str] = []
        if not record.content:
            raise ValueError("content is required")
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
        if not self._db_available or not self.state.sqlite_writable:
            return None
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO conflicts(left_id, right_id, subject, status, reason, winner_id, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (left_id, right_id, subject, status, reason, winner_id, utc_now_iso(), utc_now_iso() if status != "open" else None),
            )
            conn.commit()
            return int(cur.lastrowid)

    def resolve_conflicts_for(self, memory_id: int) -> int:
        if not self._db_available or not self.state.sqlite_writable:
            return 0
        with self.connection() as conn:
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
            conn.commit()
            return cur.rowcount

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
            return [row_to_dict(row) for row in rows]

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
        for evolution/compatible verdicts with a live active judgment. This is
        narrower than ``pairs_closed_for_scan``'s close gate by design:
        search must not re-litigate a pair as guidance, whereas scan deliberately
        keeps contradiction-resolved pairs ringable for a second look. Do not
        align the two queries without resolving that intent first.
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
                        f"(c.status='open' AND (c.left_claim_revision IS NULL OR ("
                        f"c.left_claim_revision=(SELECT claim_revision FROM memories WHERE id=c.left_id) AND "
                        f"c.right_claim_revision=(SELECT claim_revision FROM memories WHERE id=c.right_id) AND "
                        f"c.left_version=(SELECT version FROM memories WHERE id=c.left_id) AND "
                        f"c.right_version=(SELECT version FROM memories WHERE id=c.right_id)))) "
                        f"OR (c.status='resolved' AND c.active_judgment_id IS NOT NULL "
                        f"AND j.verdict IN ('evolution','compatible') "
                        f"AND c.left_version IS NOT NULL AND c.right_version IS NOT NULL "
                        f"AND c.left_version=(SELECT version FROM memories WHERE id=c.left_id) "
                        f"AND c.right_version=(SELECT version FROM memories WHERE id=c.right_id) "
                        f"AND (c.left_claim_revision IS NULL OR c.left_claim_revision=(SELECT claim_revision FROM memories WHERE id=c.left_id)) "
                        f"AND (c.right_claim_revision IS NULL OR c.right_claim_revision=(SELECT claim_revision FROM memories WHERE id=c.right_id)))"
                        f") AND (c.left_id IN ({ph}) OR c.right_id IN ({ph}))",
                        (*chunk, *chunk),
                    ).fetchall()
                    results.extend(row_to_dict(r) for r in rows)
        except sqlite3.Error:
            return []
        return results

    def get_memory_summaries(
        self, memory_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Batch-fetch lightweight summaries for a set of memory IDs.

        Returns ``{id: {id, subject, status, source_type, protection_level,
        tags, snippet}}``. Missing / non-active IDs are simply absent from the
        result (callers treat absence as "vanished"). Never raises.
        """
        if not memory_ids or not self._db_available:
            return {}
        unique_ids = sorted(set(int(i) for i in memory_ids if i is not None))
        if not unique_ids:
            return {}
        out: dict[int, dict[str, Any]] = {}
        try:
            with self.connection() as conn:
                for chunk_start in range(0, len(unique_ids), 500):
                    chunk = unique_ids[chunk_start:chunk_start + 500]
                    ph = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT id, subject, status, source_type, "
                        f"protection_level, tags, "
                        f"substr(content, 1, 200) AS snippet "
                        f"FROM memories WHERE id IN ({ph}) AND status = 'active'",
                        chunk,
                    ).fetchall()
                    for r in rows:
                        out[int(r["id"])] = row_to_dict(r)
        except sqlite3.Error:
            return {}
        return out

    # ------------------------------------------------------------------
    # v0.7.5: conflict-scan enrichment (id=243 path-B).
    # Three new helpers — none of them touch memories/FTS/embeddings. They
    # only read/write the conflicts table + memories_vec (read-only KNN).
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
        left_claim_revision: Optional[int] = None,
        right_claim_revision: Optional[int] = None,
        judgment_status: Optional[str] = None,
        structured_details: Optional[list[dict[str, Any]]] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
        detection_channel: Optional[str] = None,
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
        if detection_channel is None:
            detection_channel = (
                "structured" if source == "structured_claim"
                else "metadata" if source == "metadata_write_hint"
                else "scan"
            )
        if detection_channel not in {"structured", "scan", "metadata"}:
            return {"outcome": "invalid_detection_channel"}
        if source == "structured_claim" and any(
            value is None for value in (
                left_version, right_version, left_claim_revision, right_claim_revision,
            )
        ):
            return {"outcome": "invalid_structured_snapshot"}
        raw_left, raw_right = int(left_id), int(right_id)
        if raw_left <= raw_right:
            a, b = raw_left, raw_right
        else:
            a, b = raw_right, raw_left
            left_version, right_version = right_version, left_version
            left_claim_revision, right_claim_revision = right_claim_revision, left_claim_revision
            if structured_details is not None:
                swapped_details: list[dict[str, Any]] = []
                for detail in structured_details:
                    swapped = dict(detail)
                    for suffix in ("value", "raw_value", "evidence", "start_offset", "end_offset", "memory_id"):
                        left_key, right_key = f"left_{suffix}", f"right_{suffix}"
                        if left_key in swapped or right_key in swapped:
                            swapped[left_key], swapped[right_key] = swapped.get(right_key), swapped.get(left_key)
                    swapped_details.append(swapped)
                structured_details = swapped_details
        subject = conflict_point or reason
        now = utc_now_iso()
        structured_detected_at = now if detection_channel == "structured" else None
        scan_detected_at = now if detection_channel == "scan" else None
        with self.write_transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM conflicts WHERE status='open' AND left_id=? AND right_id=?",
                (a, b),
            ).fetchone()
            if existing:
                existing_row = row_to_dict(existing)
                priority = {
                    "metadata_write_hint": 10,
                    "structured_claim": 20,
                    "llm_informed": 30,
                    "policy_informed": 35,
                    "human_confirmed": 40,
                    None: 30,
                }
                incoming_priority = priority.get(source, 25)
                existing_priority = priority.get(existing_row.get("source"), 25)
                existing_structured = (
                    existing_row.get("left_claim_revision") is not None
                    and existing_row.get("right_claim_revision") is not None
                )
                incoming_structured = detection_channel == "structured"
                # Conflict identity/provenance and judgment snapshot are separate
                # ownership domains. A scan may enrich an existing structured
                # conflict, but it must never erase the structured CAS pins or
                # bypass the mandatory host-LLM receipt.
                preserve_structured_snapshot = existing_structured and not incoming_structured
                preserve_judgment_projection = (
                    preserve_structured_snapshot
                    and existing_row.get("active_judgment_id") is not None
                )
                effective_left_version = (
                    existing_row.get("left_version")
                    if preserve_structured_snapshot else left_version
                )
                effective_right_version = (
                    existing_row.get("right_version")
                    if preserve_structured_snapshot else right_version
                )
                effective_left_claim_revision = (
                    existing_row.get("left_claim_revision")
                    if preserve_structured_snapshot else left_claim_revision
                )
                effective_right_claim_revision = (
                    existing_row.get("right_claim_revision")
                    if preserve_structured_snapshot else right_claim_revision
                )
                pins_changed = (
                    effective_left_version != existing_row.get("left_version")
                    or effective_right_version != existing_row.get("right_version")
                    or effective_left_claim_revision != existing_row.get("left_claim_revision")
                    or effective_right_claim_revision != existing_row.get("right_claim_revision")
                )
                # A source-revision change invalidates an old LLM/policy/human
                # projection. Historical judgments remain append-only, but the
                # conflict returns to a structured pending candidate.
                reset_judgment = incoming_structured and pins_changed
                effective_structured_detected_at = (
                    existing_row.get("structured_detected_at") or structured_detected_at
                )
                effective_scan_detected_at = (
                    existing_row.get("scan_detected_at") or scan_detected_at
                )
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
                            left_claim_revision=?, right_claim_revision=?,
                            judgment_status=?, active_judgment_id=?,
                            resolution_kind=?, conflict_scope=?,
                            structured_details=COALESCE(?, structured_details),
                            scan_prompt_version=?, scan_model=?,
                            structured_detected_at=?, scan_detected_at=?, refreshed_at=?
                        WHERE id=?
                        """,
                        (
                            conflict_type, conflict_point, effective_reason,
                            effective_winner, effective_winner, effective_confidence,
                            effective_source, effective_left_version, effective_right_version,
                            effective_left_claim_revision, effective_right_claim_revision,
                            effective_judgment_status, active_judgment_id,
                            effective_resolution_kind, effective_conflict_scope,
                            (json.dumps(structured_details, ensure_ascii=False)
                             if structured_details is not None else None),
                            scan_prompt_version, scan_model,
                            effective_structured_detected_at, effective_scan_detected_at, now,
                            int(existing["id"]),
                        ),
                    )
                    if cur.rowcount == 0:
                        return {"outcome": "not_open", "conflict_id": int(existing["id"])}
                    return {"outcome": "refreshed", "conflict_id": int(existing["id"])}
                provenance_changed = (
                    effective_structured_detected_at != existing_row.get("structured_detected_at")
                    or effective_scan_detected_at != existing_row.get("scan_detected_at")
                )
                if provenance_changed:
                    conn.execute(
                        "UPDATE conflicts SET structured_detected_at=?, scan_detected_at=?, "
                        "refreshed_at=? WHERE id=?",
                        (
                            effective_structured_detected_at,
                            effective_scan_detected_at,
                            now,
                            int(existing["id"]),
                        ),
                    )
                return {"outcome": "deduped", "conflict_id": int(existing["id"])}
            cur = conn.execute(
                """
                INSERT INTO conflicts(
                    left_id, right_id, subject, status, reason, winner_id,
                    created_at, resolved_at,
                    conflict_type, conflict_point, suggested_winner,
                    confidence_hint, source,
                    left_version, right_version, scan_prompt_version, scan_model
                    , left_claim_revision, right_claim_revision, judgment_status,
                    structured_details, structured_detected_at, scan_detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a, b, subject, status, reason, suggested_winner,
                    now, now if status != "open" else None,
                    conflict_type, conflict_point, suggested_winner,
                    confidence_hint, source,
                    left_version, right_version, scan_prompt_version, scan_model,
                    left_claim_revision, right_claim_revision, judgment_status,
                    (json.dumps(structured_details, ensure_ascii=False)
                     if structured_details is not None else None),
                    structured_detected_at, scan_detected_at,
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
                    "AND (c.left_claim_revision IS NULL OR c.left_claim_revision = (SELECT claim_revision FROM memories WHERE id=c.left_id)) "
                    "AND (c.right_claim_revision IS NULL OR c.right_claim_revision = (SELECT claim_revision FROM memories WHERE id=c.right_id)) "
                    "LIMIT 1",
                    (a, b),
                ).fetchone()
                return row is not None
        except sqlite3.Error:
            return False

    def is_structured_pair_closed_for_snapshot(
        self,
        left_id: int,
        right_id: int,
        left_version: int,
        right_version: int,
        left_claim_revision: int,
        right_claim_revision: int,
    ) -> bool:
        """Whether this exact structured snapshot already received a terminal close.

        Compatible/evolution judgments resolve the conflict row while the two
        literal values remain different.  A no-op rebuild must not recreate the
        same candidate and ring again.  Any content/entity/trust change alters a
        version or claim revision and therefore naturally re-enables detection.
        """
        if not self._db_available:
            return False
        a, b = sorted((int(left_id), int(right_id)))
        lv, rv = int(left_version), int(right_version)
        lcr, rcr = int(left_claim_revision), int(right_claim_revision)
        if a != int(left_id):
            lv, rv = rv, lv
            lcr, rcr = rcr, lcr
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM conflicts WHERE left_id=? AND right_id=? "
                    "AND status IN ('resolved','not_a_conflict') "
                    "AND left_version=? AND right_version=? "
                    "AND left_claim_revision=? AND right_claim_revision=? LIMIT 1",
                    (a, b, lv, rv, lcr, rcr),
                ).fetchone()
                return row is not None
        except sqlite3.Error:
            return False

    def pairs_closed_for_scan(self, memory_ids: list[int]) -> set[tuple[int, int]]:
        """Terminal conflict pairs that vector scan should not re-ring on.

        This is the scan-facing close gate.  It combines explicit false-positive
        dismissals with resolved guidance for the same memory snapshot.  Version
        and claim-revision pins decide when a pair naturally reopens.

        Note (v0.10.2): this is intentionally wider than the resolved-guidance
        branch in ``list_open_conflicts_for_memory_ids``.  Scan closes any
        resolved pair whose verdict is evolution/compatible *or* that has no
        active judgment (manual close).  A resolved pair carrying a
        contradiction verdict is deliberately left ringable so a contested
        close can be re-examined; search never surfaces such a pair as
        guidance.  Mirroring the two queries would either re-litigate settled
        evolution/compatible pairs or silence contested ones — keep them
        asymmetric unless that trade-off is revisited.
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
                    """
                    SELECT c.left_id, c.right_id FROM conflicts c
                    LEFT JOIN conflict_judgments j ON j.id = c.active_judgment_id
                    WHERE (c.left_id IN ({ph}) OR c.right_id IN ({ph}))
                    AND (
                      (
                        c.status='not_a_conflict'
                        AND (c.left_version IS NULL OR c.left_version = (SELECT version FROM memories WHERE id=c.left_id))
                        AND (c.right_version IS NULL OR c.right_version = (SELECT version FROM memories WHERE id=c.right_id))
                        AND (c.left_claim_revision IS NULL OR c.left_claim_revision = (SELECT claim_revision FROM memories WHERE id=c.left_id))
                        AND (c.right_claim_revision IS NULL OR c.right_claim_revision = (SELECT claim_revision FROM memories WHERE id=c.right_id))
                      )
                      OR (
                        c.status='resolved'
                        AND c.left_version IS NOT NULL
                        AND c.right_version IS NOT NULL
                        AND c.left_version = (SELECT version FROM memories WHERE id=c.left_id)
                        AND c.right_version = (SELECT version FROM memories WHERE id=c.right_id)
                        AND (c.left_claim_revision IS NULL OR c.left_claim_revision = (SELECT claim_revision FROM memories WHERE id=c.left_id))
                        AND (c.right_claim_revision IS NULL OR c.right_claim_revision = (SELECT claim_revision FROM memories WHERE id=c.right_id))
                        AND (
                          j.verdict IN ('evolution','compatible')
                          OR c.active_judgment_id IS NULL
                        )
                      )
                    )
                    """.format(ph=ph),
                    (*ids, *ids),
                ).fetchall()
                for r in rows:
                    a, b = int(r["left_id"]), int(r["right_id"])
                    out.add((min(a, b), max(a, b)))
        except sqlite3.Error:
            return set()
        return out

    def purge_stale_dismissals(self) -> int:
        """v0.8.8: delete ``not_a_conflict`` rows whose pinned versions no longer
        match the memories' current versions (a side was edited after dismissal).
        Called by scan. Garbage collection only — functional re-enable is already
        handled by ``is_pair_dismissed``'s version CAS at check time. Best-effort.

        Symmetry note (v0.8.8 review): the read path (``is_pair_dismissed`` /
        ``dismissed_pairs_for``) treats a NULL version pin as a *valid* dismissal
        (``IS NULL OR =``). This GC must agree, so a row is reclaimed ONLY when a
        pin is non-NULL AND differs from the memory's current version
        (``IS NOT NULL AND <>`` — a NULL pin is left alone). A NULL pin can't be
        invalidated by version CAS alone; ``_enrich_write_response`` avoids
        creating NULL pins by defaulting missing versions to 1 (``or 1``).
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0
        try:
            with self.connection() as conn:
                cur = conn.execute(
                    "DELETE FROM conflicts WHERE status='not_a_conflict' AND ( "
                    "(left_version IS NOT NULL AND left_version <> "
                    "(SELECT version FROM memories WHERE id=conflicts.left_id)) "
                    "OR (right_version IS NOT NULL AND right_version <> "
                    "(SELECT version FROM memories WHERE id=conflicts.right_id)) "
                    "OR (left_claim_revision IS NOT NULL AND left_claim_revision <> "
                    "(SELECT claim_revision FROM memories WHERE id=conflicts.left_id)) "
                    "OR (right_claim_revision IS NOT NULL AND right_claim_revision <> "
                    "(SELECT claim_revision FROM memories WHERE id=conflicts.right_id)) )"
                )
                conn.commit()
                return cur.rowcount
        except sqlite3.Error:
            return 0

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

    # ------------------------------------------------------------------
    # v0.9: structured claim and judgment store delegation
    # ------------------------------------------------------------------

    def publish_memory_claims(
        self,
        memory_id: int,
        claims: list[dict[str, Any]],
        ambiguous_count: int = 0,
        expected_claim_revision: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._claim_store.publish_memory_claims(
            memory_id, claims, ambiguous_count, expected_claim_revision,
        )

    def mark_claim_index_failed(
        self, memory_id: int, expected_claim_revision: Optional[int] = None,
    ) -> None:
        self._claim_store.mark_claim_index_failed(memory_id, expected_claim_revision)

    def mark_claim_reconciled(
        self,
        memory_id: int,
        expected_claim_revision: int,
        enrich_ms: float,
        candidate_count: int,
    ) -> dict[str, Any]:
        return self._claim_store.mark_claim_reconciled(
            memory_id, expected_claim_revision, enrich_ms, candidate_count,
        )

    def list_memory_claims(
        self, memory_id: int, current_only: bool = True,
    ) -> list[dict[str, Any]]:
        return self._claim_store.list_memory_claims(memory_id, current_only)

    def find_structured_claim_pairs(self, memory_id: int) -> dict[str, Any]:
        return self._claim_store.find_structured_claim_pairs(memory_id)

    def list_structured_open_conflicts_for_memory(
        self, memory_id: int,
    ) -> list[dict[str, Any]]:
        return self._claim_store.list_structured_open_conflicts_for_memory(memory_id)

    def read_structured_open_conflicts_for_memory(
        self, memory_id: int,
    ) -> dict[str, Any]:
        return self._claim_store.read_structured_open_conflicts_for_memory(memory_id)

    def structured_pair_gate_states(
        self, memory_id: int, pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._claim_store.structured_pair_gate_states(memory_id, pairs)

    def build_conflict_judgment_request(
        self, conflict_id: int,
    ) -> Optional[dict[str, Any]]:
        return self._judgment_store.build_conflict_judgment_request(conflict_id)

    def build_conflict_judgment_requests(
        self, conflict_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        return self._judgment_store.build_conflict_judgment_requests(conflict_ids)

    def submit_conflict_judgment(
        self,
        conflict_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        confidence_hint: Optional[str],
        reason: str,
        affects_current_output: bool,
        usage_context: str,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._judgment_store.submit_conflict_judgment(
            conflict_id=conflict_id,
            expected_left_version=expected_left_version,
            expected_right_version=expected_right_version,
            expected_left_claim_revision=expected_left_claim_revision,
            expected_right_claim_revision=expected_right_claim_revision,
            verdict=verdict,
            recommended_use=recommended_use,
            suggested_winner=suggested_winner,
            confidence_hint=confidence_hint,
            reason=reason,
            affects_current_output=affects_current_output,
            usage_context=usage_context,
            judge_ref=judge_ref,
            resolution_kind=resolution_kind,
            conflict_scope=conflict_scope,
        )

    def correct_conflict_judgment(
        self,
        conflict_id: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        reason: str,
        expected_judgment_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._judgment_store.correct_conflict_judgment(
            conflict_id=conflict_id,
            verdict=verdict,
            recommended_use=recommended_use,
            suggested_winner=suggested_winner,
            reason=reason,
            expected_judgment_id=expected_judgment_id,
            expected_left_version=expected_left_version,
            expected_right_version=expected_right_version,
            expected_left_claim_revision=expected_left_claim_revision,
            expected_right_claim_revision=expected_right_claim_revision,
            judge_ref=judge_ref,
            resolution_kind=resolution_kind,
            conflict_scope=conflict_scope,
        )

    def list_conflict_judgments(
        self, conflict_id: int,
    ) -> list[dict[str, Any]]:
        return self._judgment_store.list_conflict_judgments(conflict_id)

    def dismissed_pairs_for(self, memory_ids: list[int]) -> set:
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
        out: set = set()
        try:
            with self.connection() as conn:
                ph = ",".join("?" * len(ids))
                rows = conn.execute(
                    "SELECT left_id, right_id FROM conflicts "
                    "WHERE status='not_a_conflict' "
                    f"AND (left_id IN ({ph}) OR right_id IN ({ph})) "
                    "AND (left_version IS NULL OR left_version = (SELECT version FROM memories WHERE id=conflicts.left_id)) "
                    "AND (right_version IS NULL OR right_version = (SELECT version FROM memories WHERE id=conflicts.right_id)) "
                    "AND (left_claim_revision IS NULL OR left_claim_revision = (SELECT claim_revision FROM memories WHERE id=conflicts.left_id)) "
                    "AND (right_claim_revision IS NULL OR right_claim_revision = (SELECT claim_revision FROM memories WHERE id=conflicts.right_id))",
                    (*ids, *ids),
                ).fetchall()
                for r in rows:
                    a, b = int(r["left_id"]), int(r["right_id"])
                    out.add((min(a, b), max(a, b)))
        except sqlite3.Error:
            return set()
        return out

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
            with self.connection() as conn:
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

    # ------------------------------------------------------------------
    #  v0.7.5: scan_log.jsonl (diagnostic layer for conflict scan + doctor)
    # ------------------------------------------------------------------

    @property
    def scan_log_path(self) -> Path:
        """``scan_log.jsonl`` sits next to ``memory.sqlite3``."""
        return self.settings.db_path.parent / "scan_log.jsonl"

    #  v0.8.8: attention_log.jsonl — diagnostic layer for the attention_required
    #  flag. Doctor reports volume by source so the operator can see whether a
    #  given source (esp. the advisory runtime_metadata_hint) is flooding the
    #  flag — the cry-wolf indicator. Same best-effort discipline as scan_log.
    @property
    def attention_log_path(self) -> Path:
        """``attention_log.jsonl`` sits next to ``memory.sqlite3``."""
        return self.settings.db_path.parent / "attention_log.jsonl"

    def log_attention(self, *, trigger: str, source: str, memory_ids: list) -> None:
        """Append one attention event (v0.8.8). Best-effort diagnostic layer.

        Same append discipline as ``_scan_log_append`` ("a" mode, single
        write, POSIX < PIPE_BUF atomicity). Never raises — a failed
        diagnostic write must not break the write/search that triggered it.
        """
        try:
            entry = {
                "ts": utc_now_iso(),
                "trigger": trigger,
                "source": source,
                "ids": [int(i) for i in memory_ids if i is not None],
            }
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with open(self.attention_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except (OSError, ValueError, TypeError):
            pass

    def _scan_log_append(self, entry: dict[str, Any]) -> None:
        """Append one JSONL line to scan_log. Best-effort (diagnostic layer).

        Uses ``"a"`` mode + a single ``write`` so concurrent scans don't
        corrupt each other's lines under POSIX (< PIPE_BUF atomicity).
        Never raises — a failed diagnostic write must not break the scan.
        """
        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with open(self.scan_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass

    def _scan_log_last_completed(self) -> Optional[dict[str, Any]]:
        """Read the last ``status=completed`` entry from scan_log.

        Tolerant: missing file -> None; corrupted lines skipped; if no
        ``completed`` line exists, returns None. Doctor and the scan tool
        both key off this for the incremental watermark.
        """
        path = self.scan_log_path
        if not path.exists():
            return None
        last_completed: Optional[dict[str, Any]] = None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(rec, dict) and rec.get("status") == "completed":
                        last_completed = rec
        except OSError:
            return None
        return last_completed

    # ------------------------------------------------------------------
    #  v0.7.5: conflict candidate scan (id=243 path-B, tool-side, no LLM)
    # ------------------------------------------------------------------

    def scan_conflict_candidates(
        self,
        workspace: Optional[str] = None,
        top_k: int = 8,
        max_pairs: int = 200,
        max_distance: float = 12.0,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Vector-recall candidate conflict pairs (no LLM, no writes to memories).

        Returns a dict with ``candidates`` (list of pairs with distance/tags/
        excerpts), ``checked_pairs``, ``truncated``, ``max_memory_id``, and
        ``scan_log_written``. When sqlite-vec is unavailable, returns a normal
        (non-error) ``scanned=False`` result with a hint — this is a config
        state, not an exception.
        """
        if not self._db_available:
            return {"scanned": False, "reason": "db_unavailable",
                    "hint": "SQLite unavailable; conflict scan cannot run."}
        # v0.8.8: GC stale advisory dismissals (rows whose pinned versions no
        # longer match — a side was edited). Functional re-enable is already
        # handled by is_pair_dismissed's version CAS; this just reclaims dead rows.
        self.purge_stale_dismissals()
        if not self.state.sqlite_vec_available:
            return {"scanned": False, "reason": "sqlite_vec_unavailable",
                    "hint": "向量未启用，冲突扫描不可用。请先配置 sqlite-vec embedding。"}

        t0 = time.time()
        # Record the scan *start* (not end) so edits during/after this scan
        # are caught next time. Catch uses >= against this baseline; because
        # utc_now_iso has second-level precision, using the end time + strict >
        # would miss same-second edits. Re-processing is idempotent
        # (record_conflict dedupes), so a little overlap is harmless.
        scan_start_iso = utc_now_iso()

        # --- determine incremental watermark + last scan time ---
        watermark = 0
        last_scan_time: Optional[str] = None
        if incremental:
            last = self._scan_log_last_completed()
            if last:
                watermark = int(last.get("max_memory_id") or 0)
                last_scan_time = last.get("scan_time")

        # --- pick memories to scan: new (id > watermark) + edited since last scan ---
        with self.connection() as conn:
            if workspace:
                new_rows = conn.execute(
                    "SELECT id, workspace, subject, tags, content FROM memories "
                    "WHERE status='active' AND id > ? AND workspace=? "
                    "ORDER BY id",
                    (watermark, workspace),
                ).fetchall()
            else:
                new_rows = conn.execute(
                    "SELECT id, workspace, subject, tags, content FROM memories "
                    "WHERE status='active' AND id > ? ORDER BY id",
                    (watermark,),
                ).fetchall()
            scan_ids = {int(r["id"]) for r in new_rows}
            edited_rows: list = []
            if last_scan_time:
                edited_rows = conn.execute(
                    "SELECT DISTINCT h.memory_id FROM memory_history h "
                    "WHERE h.changed_at >= ?",
                    (last_scan_time,),
                ).fetchall()
                for er in edited_rows:
                    mid = int(er["memory_id"])
                    if mid not in scan_ids:
                        scan_ids.add(mid)

        if not scan_ids:
            self._scan_log_append({
                "scan_time": scan_start_iso,
                "duration_sec": round(time.time() - t0, 4),
                "workspace": workspace,
                "status": "completed",
                "max_memory_id": watermark,
                "checked_pairs": 0,
                "truncated": False,
            })
            return {"scanned": True, "candidates": [], "checked_pairs": 0,
                    "truncated": False, "max_memory_id": watermark,
                    "scan_log_written": True, "reason": "no_new_or_edited"}

        # --- cache: id -> (workspace, subject, tags, excerpt) ---
        # Seed from new_rows (already fetched); edited-only memories need a
        # status check — a memory edited then superseded must not be scanned.
        id_meta: dict[int, dict[str, Any]] = {}
        max_seen_id = watermark
        for r in new_rows:
            mid = int(r["id"])
            content = r["content"] or ""
            id_meta[mid] = {
                "workspace": r["workspace"],
                "subject": r["subject"] or f"memory #{mid}",
                "tags": _coerce_tags_db(r["tags"]),
                "excerpt": content[:200],
            }
            if mid > max_seen_id:
                max_seen_id = mid
        # backfill meta for edited-only memories not in new_rows;
        # drop non-active (edited-then-superseded) from the scan set.
        edited_only = [int(er["memory_id"]) for er in edited_rows
                       if int(er["memory_id"]) not in id_meta]
        if edited_only:
            self._bulk_backfill_meta(id_meta, edited_only)
            # prune: a memory edited but now superseded/deleted must not scan.
            scan_ids = {mid for mid in scan_ids
                        if mid not in edited_only or mid in id_meta}

        # --- recall top-K neighbours for each scan memory ---
        seen_pairs: set[tuple[int, int]] = set()
        candidates: list[dict[str, Any]] = []
        # collect neighbour ids whose meta we still need (non-scan side).
        missing_meta_ids: set[int] = set()

        for mid in sorted(scan_ids):
            if mid not in id_meta:
                # pruned (edited-then-superseded) — skip.
                continue
            emb = self.get_embedding(mid)
            if emb is None:
                continue
            neighbours = self.vec_knn(emb, k=top_k)
            for nb in neighbours:
                nb_id = int(nb["id"])
                if nb_id == mid:
                    continue
                # canonicalise pair
                a, b = sorted((mid, nb_id))
                key = (a, b)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                dist = float(nb["distance"])
                if dist > max_distance:
                    continue
                # defer meta fetch for the non-scan neighbour; collect first,
                # bulk-fetch after the loop to avoid N+1 queries.
                if a not in id_meta:
                    missing_meta_ids.add(a)
                if b not in id_meta:
                    missing_meta_ids.add(b)
                candidates.append((a, b, dist))

        # bulk-fetch all missing neighbour meta in one query.
        if missing_meta_ids:
            self._bulk_backfill_meta(id_meta, list(missing_meta_ids))

        # materialise candidate dicts (now that all meta is present) + ws filter.
        resolved: list[dict[str, Any]] = []
        candidate_ids = sorted({mid for a, b, _dist in candidates for mid in (a, b)})
        closed_pairs = self.pairs_closed_for_scan(candidate_ids)
        for a, b, dist in candidates:
            if (a, b) in closed_pairs:
                continue
            ws_a = id_meta.get(a, {}).get("workspace")
            ws_b = id_meta.get(b, {}).get("workspace")
            if ws_a is None or ws_b is None:
                # neighbour memory vanished (deleted between KNN and fetch).
                continue
            if ws_a != ws_b:
                continue
            resolved.append({
                "left_id": a, "right_id": b,
                "left_subject": id_meta.get(a, {}).get("subject", f"memory #{a}"),
                "right_subject": id_meta.get(b, {}).get("subject", f"memory #{b}"),
                "left_tags": id_meta.get(a, {}).get("tags", []),
                "right_tags": id_meta.get(b, {}).get("tags", []),
                "left_excerpt": id_meta.get(a, {}).get("excerpt", ""),
                "right_excerpt": id_meta.get(b, {}).get("excerpt", ""),
                "distance": dist,
                "workspace": ws_a,
            })

        checked_pairs = len(resolved)
        resolved.sort(key=lambda c: c["distance"])
        truncated = False
        if len(resolved) > max_pairs:
            resolved = resolved[:max_pairs]
            truncated = True

        new_max = max(max_seen_id, watermark)
        self._scan_log_append({
            "scan_time": scan_start_iso,
            "duration_sec": round(time.time() - t0, 4),
            "workspace": workspace,
            "status": "completed",
            "max_memory_id": new_max,
            "checked_pairs": checked_pairs,
            "truncated": truncated,
        })
        return {
            "scanned": True,
            "candidates": resolved,
            "checked_pairs": checked_pairs,
            "truncated": truncated,
            "max_memory_id": new_max,
            "scan_log_written": True,
        }

    def _bulk_backfill_meta(
        self, id_meta: dict[int, dict[str, Any]], ids: list[int]
    ) -> None:
        """Bulk-fetch subject/workspace/tags/excerpt for ids missing from ``id_meta``.

        Only active memories are loaded; superseded/deleted ids are simply
        absent from the result (caller treats absence as "pruned"). One query
        instead of N (avoids the N+1 the per-pair backfill caused).
        """
        missing = [i for i in ids if i not in id_meta]
        if not missing:
            return
        # SQLite has a variable limit (999 by default); chunk to be safe.
        try:
            with self.connection() as conn:
                for chunk_start in range(0, len(missing), 500):
                    chunk = missing[chunk_start:chunk_start + 500]
                    placeholders = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT id, workspace, subject, tags, content, status "
                        f"FROM memories WHERE id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for r in rows:
                        if r["status"] != "active":
                            continue
                        content = r["content"] or ""
                        mid = int(r["id"])
                        id_meta[mid] = {
                            "workspace": r["workspace"],
                            "subject": r["subject"] or f"memory #{mid}",
                            "tags": _coerce_tags_db(r["tags"]),
                            "excerpt": content[:200],
                        }
        except sqlite3.Error:
            pass

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
                from .claims import resolve_entity
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
        """Pure-SQL aggregate overview per workspace."""
        empty = {"workspaces": {}, "total_memories": 0, "total_open_conflicts": 0}
        if not self._db_available:
            return empty
        with self.connection() as conn:
            mem_rows = conn.execute(
                """
                SELECT workspace,
                       COUNT(*) AS count,
                       MIN(event_time) AS oldest,
                       MAX(event_time) AS newest,
                       source_type
                FROM memories
                WHERE status != 'deleted'
                GROUP BY workspace, source_type
                """
            ).fetchall()

            open_conflict_rows = conn.execute(
                "SELECT workspace, COUNT(*) AS open_conflicts FROM ("
                " SELECT m.workspace AS workspace FROM conflicts c"
                " JOIN memories m ON m.id IN (c.left_id, c.right_id)"
                " WHERE c.status = 'open' GROUP BY c.id"
                ") GROUP BY workspace"
            ).fetchall()
            open_conflicts_by_ws = {row["workspace"]: int(row["open_conflicts"]) for row in open_conflict_rows}

        workspaces: dict[str, dict[str, Any]] = {}
        total_memories = 0
        for row in mem_rows:
            ws = row["workspace"]
            bucket = workspaces.setdefault(
                ws,
                {"count": 0, "oldest": None, "newest": None, "open_conflicts": 0, "by_source_type": {}},
            )
            count = int(row["count"])
            bucket["count"] += count
            total_memories += count
            oldest, newest = row["oldest"], row["newest"]
            if oldest is not None and (bucket["oldest"] is None or oldest < bucket["oldest"]):
                bucket["oldest"] = oldest
            if newest is not None and (bucket["newest"] is None or newest > bucket["newest"]):
                bucket["newest"] = newest
            if row["source_type"] is not None:
                bucket["by_source_type"][row["source_type"]] = (
                    bucket["by_source_type"].get(row["source_type"], 0) + count
                )

        total_open_conflicts = 0
        for ws, count in open_conflicts_by_ws.items():
            workspaces.setdefault(
                ws,
                {"count": 0, "oldest": None, "newest": None, "open_conflicts": 0, "by_source_type": {}},
            )["open_conflicts"] = count
            total_open_conflicts += count

        return {
            "workspaces": workspaces,
            "total_memories": total_memories,
            "total_open_conflicts": total_open_conflicts,
        }

    # ==================================================================
    #  v0.6.0: _vec_index_meta + section operations
    # ==================================================================

    # ---- _vec_index_meta CRUD ----

    @staticmethod
    def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute("SELECT value FROM _vec_index_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO _vec_index_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def _delete_meta(conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM _vec_index_meta WHERE key = ?", (key,))

    def get_vec_index_state(self) -> dict[str, Any]:
        """Read all _vec_index_meta keys as a dict."""
        if not self._db_available:
            return {"state": "unmanaged"}
        with self.connection() as conn:
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

    def init_vec_index_state(
        self,
        embedding_space_id: Optional[str],
        has_managed_embedder: bool,
    ) -> None:
        """Legacy initialization of _vec_index_meta (design doc §1.1b).

        Called once at startup after schema migration.  Determines the
        initial state based on whether the embedder is available and whether
        the vec tables already have data.
        """
        if not self._db_available:
            return
        with self.write_transaction() as conn:
            if not has_managed_embedder or embedding_space_id is None:
                self._set_meta(conn, "state", "unmanaged")
                return

            rows = conn.execute("SELECT key, value FROM _vec_index_meta").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in rows}
            state = meta.get("state")
            active_space_id = meta.get("active_space_id")
            target_space_id = meta.get("target_space_id")

            # Reconcile the persisted state with the embedder loaded by this
            # process.  Returning merely because ``state`` exists would leave
            # a database marked ready after the model (and vector space) has
            # changed.
            if active_space_id == embedding_space_id:
                self._set_meta(conn, "state", "ready")
                for key in (
                    "target_space_id", "migration_cursor", "migration_epoch",
                    "migration_lease_owner", "migration_lease_expires_at",
                    "last_error",
                ):
                    self._delete_meta(conn, key)
                return

            if state in {"mismatch", "failed"} and target_space_id == embedding_space_id:
                return  # resume the existing migration and preserve its cursor

            # Check if vec tables have data
            mem_vec_count = conn.execute("SELECT COUNT(*) AS c FROM memories_vec").fetchone()["c"]
            sec_vec_count = 0
            try:
                sec_vec_count = conn.execute("SELECT COUNT(*) AS c FROM memory_sections_vec").fetchone()["c"]
            except sqlite3.Error:
                pass

            if not active_space_id and mem_vec_count == 0 and sec_vec_count == 0:
                # Fresh install — trust current embedder
                self._set_meta(conn, "state", "ready")
                self._set_meta(conn, "active_space_id", embedding_space_id)
                self._delete_meta(conn, "target_space_id")
            else:
                # Existing/previous vectors belong to an unknown or different
                # space.  Start a fresh migration towards the current model.
                self._set_meta(conn, "state", "mismatch")
                self._set_meta(conn, "target_space_id", embedding_space_id)
                self._set_meta(conn, "migration_epoch", uuid.uuid4().hex)
                for key in (
                    "migration_cursor", "migration_lease_owner",
                    "migration_lease_expires_at", "last_error",
                ):
                    self._delete_meta(conn, key)

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
        """Insert one section row, return its id."""
        cur = conn.execute(
            """
            INSERT INTO memory_sections
            (memory_id, section_index, title, title_path, summary,
             anchor_text, occurrence_index, start_offset, end_offset,
             provenance, embedding_truncated, embedding_original_tokens,
             embedding_used_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, section_index, title, title_path, summary,
             anchor_text, occurrence_index, start_offset, end_offset,
             provenance, embedding_truncated, embedding_original_tokens,
             embedding_used_tokens, utc_now_iso()),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _store_section_vec(
        conn: sqlite3.Connection,
        section_id: int,
        embedding: list[float],
    ) -> None:
        if not embedding:
            raise ValueError("section embedding is empty (encode failed)")
        conn.execute(
            "DELETE FROM memory_sections_vec WHERE id = ?", (section_id,)
        )
        # v0.9.4: look up parent status via memory_sections JOIN (N17: COALESCE for orphan)
        parent_row = conn.execute(
            "SELECT COALESCE(m.status, 'deleted') AS status "
            "FROM memory_sections s "
            "LEFT JOIN memories m ON m.id = s.memory_id "
            "WHERE s.id = ?", (section_id,)
        ).fetchone()
        parent_status = parent_row["status"] if parent_row else "deleted"
        conn.execute(
            "INSERT INTO memory_sections_vec(id, parent_status, embedding) VALUES (?, ?, ?)",
            (section_id, parent_status, json.dumps(embedding)),
        )

    @staticmethod
    def _delete_sections_for_memory(conn: sqlite3.Connection, memory_id: int) -> int:
        """Delete all sections + section vecs for a memory. Returns section count."""
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM memory_sections WHERE memory_id = ?", (memory_id,)
        ).fetchone()["c"]
        try:
            conn.execute(
                "DELETE FROM memory_sections_vec WHERE id IN "
                "(SELECT id FROM memory_sections WHERE memory_id = ?)",
                (memory_id,),
            )
        except sqlite3.Error:
            pass  # vec table may not exist if sqlite-vec not loaded
        conn.execute("DELETE FROM memory_sections WHERE memory_id = ?", (memory_id,))
        return int(count)

    @staticmethod
    def _get_sections(conn: sqlite3.Connection, memory_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM memory_sections WHERE memory_id = ? ORDER BY section_index",
            (memory_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _get_section_vec_ids(conn: sqlite3.Connection, memory_id: int) -> set[int]:
        rows = conn.execute(
            "SELECT v.id AS id FROM memory_sections_vec v "
            "JOIN memory_sections s ON s.id = v.id WHERE s.memory_id = ?",
            (memory_id,),
        ).fetchall()
        return {int(r["id"]) for r in rows}

    def get_sections_by_memory(self, memory_id: int) -> list[dict[str, Any]]:
        """Public read: all sections for a memory."""
        if not self._db_available:
            return []
        with self.connection() as conn:
            return self._get_sections(conn, memory_id)

    def get_sections_by_ids(
        self, memory_id: int, section_ids: list[int]
    ) -> Tuple[list[dict[str, Any]], list[int]]:
        """Public read: specific sections. Returns (found, missing_ids)."""
        if not self._db_available or not section_ids:
            return [], []
        with self.connection() as conn:
            placeholders = ",".join("?" * len(section_ids))
            rows = conn.execute(
                f"SELECT * FROM memory_sections WHERE memory_id = ? AND id IN ({placeholders})",
                [memory_id] + section_ids,
            ).fetchall()
            found = [dict(row) for row in rows]
            found_ids = {r["id"] for r in found}
            missing = [sid for sid in section_ids if sid not in found_ids]
            return found, missing

    def section_vec_distance_match(
        self,
        memory_id: int,
        query_embedding: list[float],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Section Vec semantic matching via vec_distance_cosine (design doc §2.5).

        Returns sections with distance <= threshold, ordered by distance.
        Only call when Vec gate is open.
        """
        if not self._db_available or not self.state.sqlite_vec_available:
            return []
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT s.id AS section_id, s.title, s.title_path,
                           s.summary, s.start_offset, s.end_offset,
                           s.embedding_truncated, s.embedding_original_tokens,
                           s.embedding_used_tokens,
                           vec_distance_cosine(v.embedding, ?) AS distance
                    FROM memory_sections s
                    JOIN memory_sections_vec v ON v.id = s.id
                    WHERE s.memory_id = ?
                      AND vec_distance_cosine(v.embedding, ?) <= ?
                    ORDER BY distance
                    """,
                    (json.dumps(query_embedding), memory_id,
                     json.dumps(query_embedding), threshold),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []


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
    from .claims import canon_token
    return canon_token(value)


def _canon_scope(value: Any) -> str:
    """Same lexical normalisation as entity; kept separate for API clarity."""
    from .claims import canon_scope
    return canon_scope(value)


def _coerce_tags_db(raw: Any) -> list[str]:
    """Normalise a ``tags`` value into a deduped ``list[str]`` (db-side copy).

    Mirrors ``search._coerce_tags`` but lives here so db.py scan logic doesn't
    reverse-depend on search.py. Accepts list / JSON string / malformed / None;
    never raises.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        tags = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            tags = parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError, TypeError):
            return []
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if isinstance(t, str) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# CJK Unicode range for subject tokenisation (write_hints candidate recall).
import re as _re

_CJK_CHAR_RE = _re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")


def _subject_tokens(subject: str) -> list[str]:
    """Split a subject into tokens for LIKE-based candidate recall.

    CJK: split into 2-char sliding windows (LIKE %xx% is coarse enough).
    ASCII: split on whitespace/punctuation, keep tokens ≥ 2 chars.
    """
    if not subject:
        return []
    tokens: list[str] = []
    for word in subject.split():
        if not word:
            continue
        if _CJK_CHAR_RE.search(word):
            # CJK-heavy token: use 2-char sliding windows.
            chars = "".join(c for c in word if _CJK_CHAR_RE.match(c) or c.isalnum())
            for i in range(len(chars) - 1):
                tokens.append(chars[i:i + 2])
        elif len(word) >= 2:
            tokens.append(word.lower())
    return tokens
