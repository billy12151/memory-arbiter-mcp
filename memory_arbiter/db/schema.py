"""Database schema and optional SQLite extension probes."""
from __future__ import annotations

import sqlite3
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings
    from ..degrade import DegradeState
    from .core import MemoryDB


class SchemaStore:
    """Own the single local-text storage schema.

    Historical databases are converted side by side by ``migrate-vnext``.
    Runtime startup only performs additive migrations for current fields.
    """

    def __init__(self, db: "MemoryDB") -> None:
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
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              content TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              workspace TEXT NOT NULL,
              workspace_canonical TEXT,
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
              version INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL
            );

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

            CREATE TABLE IF NOT EXISTS conflicts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              left_id INTEGER NOT NULL,
              right_id INTEGER NOT NULL,
              subject TEXT,
              status TEXT NOT NULL DEFAULT 'open',
              reason TEXT NOT NULL,
              winner_id INTEGER,
              conflict_type TEXT,
              conflict_point TEXT,
              suggested_winner INTEGER,
              confidence_hint TEXT,
              source TEXT,
              left_version INTEGER,
              right_version INTEGER,
              judgment_status TEXT,
              active_judgment_id INTEGER,
              resolution_kind TEXT,
              conflict_scope TEXT,
              scan_prompt_version TEXT,
              scan_model TEXT,
              refreshed_at TEXT,
              created_at TEXT NOT NULL,
              resolved_at TEXT,
              FOREIGN KEY(left_id) REFERENCES memories(id),
              FOREIGN KEY(right_id) REFERENCES memories(id)
            );

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
              supersedes_judgment_id INTEGER,
              resolution_kind TEXT,
              conflict_scope TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(conflict_id) REFERENCES conflicts(id) ON DELETE CASCADE,
              FOREIGN KEY(suggested_winner) REFERENCES memories(id),
              FOREIGN KEY(supersedes_judgment_id) REFERENCES conflict_judgments(id)
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
              left_version INTEGER NOT NULL,
              right_version INTEGER NOT NULL,
              delivered_at TEXT,
              dismissed_at TEXT,
              resolved_at TEXT,
              resolution_reason TEXT,
              FOREIGN KEY(memory_id) REFERENCES memories(id),
              FOREIGN KEY(peer_id) REFERENCES memories(id),
              FOREIGN KEY(conflict_id) REFERENCES conflicts(id)
            );

            CREATE TABLE IF NOT EXISTS memory_evidence (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              memory_id INTEGER NOT NULL,
              memory_version INTEGER NOT NULL,
              content_hash TEXT NOT NULL,
              unit_index INTEGER NOT NULL,
              kind TEXT NOT NULL,
              text TEXT NOT NULL,
              start_offset INTEGER NOT NULL,
              end_offset INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
              UNIQUE(memory_id, unit_index)
            );

            CREATE TABLE IF NOT EXISTS backup_replay_log (
              replay_key TEXT PRIMARY KEY,
              memory_id INTEGER NOT NULL,
              payload_hash TEXT NOT NULL,
              replayed_at TEXT NOT NULL,
              postprocess_status TEXT NOT NULL DEFAULT 'complete',
              postprocess_stages TEXT NOT NULL DEFAULT '{}',
              postprocess_error_code TEXT,
              FOREIGN KEY(memory_id) REFERENCES memories(id)
            );
            CREATE TABLE IF NOT EXISTS _vec_index_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS migration_state (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS workspace_canonicals (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workspace_aliases (
              alias_workspace TEXT NOT NULL UNIQUE,
              canonical TEXT NOT NULL,
              relation TEXT NOT NULL,
              status TEXT NOT NULL,
              source TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workspace_alias_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              alias_workspace TEXT NOT NULL,
              old_canonical TEXT,
              new_canonical TEXT,
              old_status TEXT,
              new_status TEXT,
              action TEXT NOT NULL,
              judge_type TEXT NOT NULL,
              reason TEXT,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(workspace, agent_id, status);
            CREATE INDEX IF NOT EXISTS idx_memories_canonical ON memories(workspace_canonical, status);
            CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(workspace, subject);
            CREATE INDEX IF NOT EXISTS idx_memories_event ON memories(event_time, ingest_time);
            CREATE INDEX IF NOT EXISTS idx_history_memory ON memory_history(memory_id, changed_at);
            CREATE INDEX IF NOT EXISTS idx_conflicts_status_left ON conflicts(status, left_id);
            CREATE INDEX IF NOT EXISTS idx_conflicts_status_right ON conflicts(status, right_id);
            CREATE INDEX IF NOT EXISTS idx_conflicts_status_created ON conflicts(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_conflicts_judgment_status ON conflicts(status, judgment_status);
            CREATE INDEX IF NOT EXISTS idx_judgments_conflict_created ON conflict_judgments(conflict_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_semantic_notices_status_created ON semantic_notices(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_semantic_notices_open_undelivered_priority
              ON semantic_notices(
                CASE lower(severity)
                  WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'warning' THEN 2
                  WHEN 'normal' THEN 3 WHEN 'info' THEN 4 ELSE 5 END,
                created_at, id, severity, notice_type, memory_id, peer_id,
                left_version, right_version
              ) WHERE status='open' AND delivered_at IS NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_notices_dedupe
              ON semantic_notices(dedupe_key) WHERE dedupe_key IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_ws_alias_events
              ON workspace_alias_events(alias_workspace, created_at);

            CREATE TRIGGER IF NOT EXISTS trg_conflicts_active_judgment_fk
            BEFORE UPDATE OF active_judgment_id ON conflicts
            WHEN NEW.active_judgment_id IS NOT NULL
             AND NOT EXISTS (
               SELECT 1 FROM conflict_judgments j
               WHERE j.id=NEW.active_judgment_id AND j.conflict_id=NEW.id
             )
            BEGIN
              SELECT RAISE(ABORT, 'invalid active_judgment_id');
            END;
            """
        )
        conn.commit()

    def _probe_features(self, conn: sqlite3.Connection) -> None:
        if self.settings.enable_sqlite_vec:
            try:
                import sqlite_vec  # type: ignore[import-untyped]

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._sqlite_vec_loadable = True
                self.state.sqlite_vec_available = True
                self.state.mode = "sqlite_vec"
                self._ensure_evidence_vec_table(conn)
                self._ensure_workspace_vec_table(conn)
            except Exception as exc:  # pragma: no cover - environment dependent
                self.state.warn(
                    f"sqlite-vec unavailable: {exc}. Semantic recall disabled; "
                    "falling back to FTS5 or keyword search."
                )
        else:
            probe = self._probe_sqlite_vec_loadable()
            suffix = (
                "Set MEMORY_ARBITER_ENABLE_SQLITE_VEC=true to enable semantic recall."
                if probe is True
                else "Install with `pip install '.[vec]'` and enable vec.enabled."
            )
            self.state.warn(f"sqlite-vec disabled by configuration. {suffix}")

        try:
            self._ensure_fts(conn)
            self._rebuild_fts(conn)
            self.state.fts5_available = True
            if not self.state.sqlite_vec_available:
                self.state.mode = "fts5"
        except sqlite3.Error as exc:
            self.state.warn(f"SQLite FTS5 unavailable: {exc}. Falling back to LIKE search.")
            if not self.state.sqlite_vec_available:
                self.state.mode = "like"

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
        try:
            import sqlite_vec  # type: ignore[import-untyped]

            probe = sqlite3.connect(":memory:")
            probe.enable_load_extension(True)
            sqlite_vec.load(probe)
            probe.close()
            return True
        except ImportError:
            return False
        except Exception:
            return None

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
            conn.commit()
        except sqlite3.Error:
            conn.rollback()

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_fts'"
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
            self.state.warn(
                f"FTS5 trigram tokenizer unavailable: {exc}. Using default tokenizer."
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "content, tags, subject, content='memories', content_rowid='id')"
            )

    def _ensure_evidence_vec_table(self, conn: sqlite3.Connection) -> None:
        dim = int(self.settings.vec_dim or 768)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_evidence_vec "
            f"USING vec0(id INTEGER PRIMARY KEY, parent_status TEXT, embedding float[{dim}])"
        )
        conn.commit()

    def _ensure_workspace_vec_table(self, conn: sqlite3.Connection) -> None:
        dim = int(self.settings.vec_dim or 768)
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS workspace_canonicals_vec "
                f"USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
            conn.commit()
        except sqlite3.Error as exc:
            self.state.warn(
                f"workspace vector index unavailable: {exc}. "
                "Workspace alias resolution will use exact matches."
            )
