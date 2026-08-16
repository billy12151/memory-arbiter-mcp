"""Schema, migration, and feature-probe operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
import struct
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings
    from ..degrade import DegradeState
    from .core import MemoryDB


class SchemaStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    @property
    def settings(self) -> "Settings":
        return self._db.settings

    @property
    def state(self) -> "DegradeState":
        return self._db.state

    @property
    def _sqlite_vec_loadable(self) -> bool:
        return self._db._sqlite_vec_loadable

    @_sqlite_vec_loadable.setter
    def _sqlite_vec_loadable(self, value: bool) -> None:
        self._db._sqlite_vec_loadable = value

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
                CREATE TABLE IF NOT EXISTS semantic_notices (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'open',
                  severity TEXT NOT NULL,
                  source TEXT NOT NULL,
                  memory_id INTEGER NOT NULL,
                  peer_id INTEGER,
                  conflict_id INTEGER,
                  notice_type TEXT NOT NULL,
                  title TEXT NOT NULL,
                  message TEXT NOT NULL,
                  payload TEXT NOT NULL DEFAULT '{}',
                  dedupe_key TEXT,
                  left_version INTEGER,
                  right_version INTEGER,
                  left_claim_revision INTEGER,
                  right_claim_revision INTEGER,
                  delivered_at TEXT,
                  dismissed_at TEXT,
                  resolved_at TEXT,
                  FOREIGN KEY(memory_id) REFERENCES memories(id),
                  FOREIGN KEY(peer_id) REFERENCES memories(id),
                  FOREIGN KEY(conflict_id) REFERENCES conflicts(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(workspace, agent_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(workspace, subject);
                CREATE INDEX IF NOT EXISTS idx_memories_event ON memories(event_time, ingest_time);
                -- v0.7.6: conflict-lookup indexes (ordinary, not unique partial —
                -- uniqueness is enforced by record_conflict_enriched logic).
                CREATE INDEX IF NOT EXISTS idx_conflicts_status_left ON conflicts(status, left_id);
                CREATE INDEX IF NOT EXISTS idx_conflicts_status_right ON conflicts(status, right_id);
                CREATE INDEX IF NOT EXISTS idx_conflicts_status_created ON conflicts(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_semantic_notices_status_created ON semantic_notices(status, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_notices_dedupe ON semantic_notices(dedupe_key) WHERE dedupe_key IS NOT NULL;
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

                -- Workspace alias governance (v0.13, design 637). Deliberately
                -- NOT the conflict_judgments CAS design: alias governance is
                -- human-triggered, low-concurrency, single-key (one
                -- alias_workspace -> one canonical), so a current-state table +
                -- append-only event log + UNIQUE + single transaction is enough.
                -- Concurrency safety comes from the UNIQUE constraint + txn, not
                -- from a version/CAS column.
                CREATE TABLE IF NOT EXISTS workspace_aliases (
                  alias_workspace TEXT NOT NULL,
                  canonical TEXT NOT NULL,
                  relation TEXT NOT NULL,   -- alias|typo|same_project|same_family|related|unrelated|uncertain
                  status TEXT NOT NULL,     -- confirmed|rejected
                  source TEXT NOT NULL,     -- user|agent|rule|qwen
                  updated_at TEXT NOT NULL,
                  UNIQUE(alias_workspace)
                );
                -- Append-only audit trail. Never UPDATE/DELETE these rows.
                CREATE TABLE IF NOT EXISTS workspace_alias_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  alias_workspace TEXT NOT NULL,
                  old_canonical TEXT,
                  new_canonical TEXT,
                  old_status TEXT,
                  new_status TEXT,
                  action TEXT NOT NULL,     -- accept|reject|rename|migrate|correct
                  judge_type TEXT NOT NULL, -- user|agent|rule|qwen
                  reason TEXT,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ws_alias_events
                  ON workspace_alias_events(alias_workspace, created_at);
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
        """Add *column* idempotently, including concurrent first starts.

        SQLite has no ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``. Two
        processes can therefore both observe a missing column before either
        ALTER commits. If the loser gets ``duplicate column name``, re-read the
        schema and accept the race only when the requested column now exists.
        """
        cols = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as exc:
                current_cols = {
                    str(row["name"])
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                if "duplicate column name" not in str(exc).lower() or column not in current_cols:
                    raise

    # ------------------------------------------------------------------
    #  Feature probing
    # ------------------------------------------------------------------

    def _probe_features(self, conn: sqlite3.Connection) -> None:
        # sqlite-vec
        if self.settings.enable_sqlite_vec:
            try:
                import sqlite_vec  # type: ignore[import-untyped]

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
            import sqlite_vec

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
        dim = int(self.settings.vec_dim or 768)
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
        dim = int(self.settings.vec_dim or 768)
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
        dim = int(self.settings.vec_dim or 768)
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
        dim = int(self.settings.vec_dim or 768)
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
