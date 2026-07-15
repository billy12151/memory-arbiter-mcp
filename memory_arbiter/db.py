from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from .config import Settings
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
                """
        )
        # Idempotent column migrations — probe each individually so
        # partial upgrades and concurrent first-starts are safe.
        self._migrate_add_column(conn, "memories", "version",
                                 "INTEGER NOT NULL DEFAULT 1")
        self._migrate_add_column(conn, "memories", "split_status",
                                 "TEXT")
        self._migrate_add_column(conn, "memories", "split_revision",
                                 "INTEGER NOT NULL DEFAULT 0")

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
                f"USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])"
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
                f"USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
            conn.commit()
        except sqlite3.Error as exc:
            self.state.warn(
                f"memory_sections_vec creation failed (dim={dim}): {exc}. "
                "Section split will be unavailable."
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
                conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
                conn.execute(
                    "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                    (memory_id, json.dumps(embedding)),
                )
                conn.commit()
            return True, []
        except sqlite3.Error as exc:
            warnings.append(f"store_embedding failed: {exc}")
            return False, warnings

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

    def vec_knn(self, query_embedding: list[float], k: int = 10) -> list[dict[str, Any]]:
        if not self._db_available or not self.state.sqlite_vec_available:
            return []
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT v.id AS id, v.distance AS distance, m.workspace AS workspace,
                           m.agent_id AS agent_id, m.status AS status, m.subject AS subject,
                           m.tags AS tags, m.content AS content, m.source_type AS source_type,
                           m.confidence AS confidence, m.protection_level AS protection_level,
                           m.event_time AS event_time, m.ingest_time AS ingest_time,
                           m.metadata AS metadata, m.split_status AS split_status
                    FROM memories_vec v
                    JOIN memories m ON m.id = v.id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                    """,
                    (json.dumps(query_embedding), k),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def section_vec_knn(
        self, query_embedding: list[float], k: int = 10
    ) -> list[dict[str, Any]]:
        """v0.6.3 Channel 6: section-level vec KNN recall.

        Returns the k nearest *sections* (not memories), joined to their
        parent memory's metadata. Unlike ``vec_knn`` this does NOT select
        ``m.content`` — Channel 6 candidates score via the vec floor and get
        their content re-fetched by ``_attach_sections`` from
        ``current_mem_map``, so pulling k full texts here would be wasted I/O
        (k ≈ need×3, most rows dedup to the same handful of memories).
        Workspace/status/split_status filtering is done Python-side in
        ``_wide_recall`` (mirroring Channel 5) so like_status_clause semantics
        stay aligned.
        """
        if not self._db_available or not self.state.sqlite_vec_available:
            return []
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT s.memory_id AS memory_id, s.id AS section_id,
                           v.distance AS distance,
                           s.title AS section_title, s.title_path AS section_title_path,
                           m.workspace AS workspace, m.status AS status,
                           m.subject AS subject, m.tags AS tags,
                           m.source_type AS source_type, m.confidence AS confidence,
                           m.protection_level AS protection_level,
                           m.event_time AS event_time, m.ingest_time AS ingest_time,
                           m.metadata AS metadata, m.split_status AS split_status
                    FROM memory_sections_vec v
                    JOIN memory_sections s ON s.id = v.id
                    JOIN memories m ON m.id = s.memory_id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                    """,
                    (json.dumps(query_embedding), k),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    # ------------------------------------------------------------------
    #  Memory CRUD
    # ------------------------------------------------------------------

    def insert_memory(self, record: MemoryRecord) -> Tuple[Optional[int], list[str]]:
        warnings: list[str] = []
        if not record.content:
            raise ValueError("content is required")
        if not self._db_available or not self.state.sqlite_writable:
            self._append_backup(record)
            warnings.append("SQLite write unavailable; wrote append-only JSONL backup.")
            return None, warnings
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO memories
                (content, agent_id, workspace, tags, source_type, source_ref, event_time, ingest_time,
                 confidence, protection_level, status, subject, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.content,
                    record.agent_id,
                    record.workspace,
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
        sql = ", ".join(f"{key} = ?" for key, _ in pairs)
        values = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for _, v in pairs]
        values.append(memory_id)
        with self.connection() as conn:
            conn.execute(f"UPDATE memories SET {sql} WHERE id = ?", values)
            conn.commit()
        return True

    def list_memories(self, workspace: Optional[str] = None, subject: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        clauses = ["status != 'deleted'"]
        params: list[Any] = []
        if workspace:
            clauses.append("workspace = ?")
            params.append(workspace)
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
            conn.commit()
            return cur.rowcount

    def list_conflicts(self, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM conflicts WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    #  Edit / History
    # ------------------------------------------------------------------

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
        with self.connection() as conn:
            current = self._fetch_memory(conn, memory_id)
            if not current:
                return None
            old_content = current["content"]
            old_subject = current.get("subject")
            old_tags = current.get("tags") or []
            old_version = int(current.get("version") or 1)
            tags_json = json.dumps(new_tags, ensure_ascii=False) if new_tags is not None else json.dumps(old_tags, ensure_ascii=False)
            subject_value = new_subject if new_subject is not None else old_subject
            try:
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
                    "UPDATE memories SET content=?, subject=?, tags=?, version=? WHERE id=?",
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
                conn.commit()
                return history_id
            except sqlite3.Error:
                conn.rollback()
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
        conn.execute(
            "INSERT INTO memory_sections_vec(id, embedding) VALUES (?, ?)",
            (section_id, json.dumps(embedding)),
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
    for key in ("tags", "metadata"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data
