"""Database schema and optional SQLite extension probes."""
from __future__ import annotations

import sqlite3
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings
    from ..degrade import DegradeState
    from .core import MemoryDB

from ..db_generation import CURRENT_SCHEMA_GENERATION


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
              revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
              workspace_canonical TEXT NOT NULL,
              slot_key TEXT CHECK(slot_key IS NULL OR (json_valid(slot_key) AND json_type(slot_key)='object' AND length(slot_key) <= 4096)),
              slot_key_hash TEXT CHECK(slot_key_hash IS NULL OR length(slot_key_hash)=64),
              candidate_key TEXT NOT NULL CHECK(json_valid(candidate_key) AND json_type(candidate_key)='object' AND length(candidate_key) <= 65536),
              candidate_key_hash TEXT NOT NULL CHECK(length(candidate_key_hash)=64),
              conflict_point TEXT,
              status TEXT NOT NULL CHECK(status IN ('candidate','open','applying','resolved','not_a_conflict'))
                CHECK(status NOT IN ('open','applying') OR (slot_key IS NOT NULL AND slot_key_hash IS NOT NULL)),
              member_versions TEXT NOT NULL CHECK(json_valid(member_versions) AND json_type(member_versions)='array' AND length(member_versions) <= 262144),
              member_fingerprint TEXT NOT NULL CHECK(length(member_fingerprint)=64),
              value_groups TEXT NOT NULL CHECK(json_valid(value_groups) AND json_type(value_groups)='array' AND length(value_groups) <= 131072),
              detection_reason TEXT NOT NULL,
              source TEXT NOT NULL,
              detector_version TEXT NOT NULL,
              prompt_version TEXT,
              overflow INTEGER NOT NULL DEFAULT 0 CHECK(overflow IN (0,1)),
              chosen_value TEXT,
              resolution_memory_id INTEGER,
              resolution_memory_version INTEGER,
              decided_by TEXT CHECK(decided_by IS NULL OR decided_by IN ('user','agent')),
              decided_ref TEXT,
              decision_reason TEXT,
              decided_at TEXT,
              apply_summary TEXT NOT NULL DEFAULT '{"plan":[]}' CHECK(json_valid(apply_summary) AND json_type(apply_summary)='object' AND length(apply_summary) <= 131072),
              notice_severity TEXT,
              notice_type TEXT,
              notice_title TEXT,
              notice_message TEXT,
              notice_payload TEXT CHECK(notice_payload IS NULL OR (json_valid(notice_payload) AND json_type(notice_payload)='object' AND length(notice_payload) <= 131072)),
              notice_task_id TEXT,
              notice_dedupe_key TEXT,
              notice_delivery_status TEXT NOT NULL DEFAULT 'pending' CHECK(notice_delivery_status IN ('pending','delivered','dismissed','resolved','stale','not_applicable')),
              notice_delivered_at TEXT,
              notice_resolution_reason TEXT,
              notice_slot_provenance TEXT CHECK(notice_slot_provenance IS NULL OR (json_valid(notice_slot_provenance) AND json_type(notice_slot_provenance)='object' AND length(notice_slot_provenance) <= 32768)),
              created_at TEXT NOT NULL,
              refreshed_at TEXT NOT NULL,
              resolved_at TEXT,
              FOREIGN KEY(resolution_memory_id) REFERENCES memories(id)
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
              alias_workspace TEXT NOT NULL,
              canonical TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('confirmed','rejected')),
              updated_at TEXT NOT NULL,
              PRIMARY KEY(alias_workspace, canonical)
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(workspace, agent_id, status);
            CREATE INDEX IF NOT EXISTS idx_memories_canonical ON memories(workspace_canonical, status);
            CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(workspace, subject);
            CREATE INDEX IF NOT EXISTS idx_memories_event ON memories(event_time, ingest_time);
            CREATE INDEX IF NOT EXISTS idx_history_memory ON memory_history(memory_id, changed_at);
            CREATE INDEX IF NOT EXISTS idx_conflicts_status_created ON conflicts(status, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_conflicts_candidate_identity
              ON conflicts(candidate_key_hash);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_conflicts_active_slot
              ON conflicts(workspace_canonical, slot_key_hash)
              WHERE slot_key_hash IS NOT NULL AND status IN ('open','applying');
            CREATE UNIQUE INDEX IF NOT EXISTS idx_conflicts_event_snapshot
              ON conflicts(workspace_canonical, slot_key_hash, member_fingerprint)
              WHERE slot_key_hash IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_conflicts_notice_dedupe
              ON conflicts(notice_dedupe_key) WHERE notice_dedupe_key IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_conflicts_notice_delivery
              ON conflicts(notice_delivery_status, created_at, id)
              WHERE notice_delivery_status='pending';
            """
        )
        self._normalize_workspace_alias_schema(conn)
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) "
            "VALUES('schema_generation',?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value=CASE WHEN migration_state.value LIKE '%:building' "
            "THEN migration_state.value ELSE excluded.value END,"
            "updated_at=CURRENT_TIMESTAMP",
            (CURRENT_SCHEMA_GENERATION,),
        )
        conn.execute(
            """INSERT INTO migration_state(key,value,updated_at)
               VALUES('migration_completed_at',strftime('%Y-%m-%dT%H:%M:%fZ','now'),CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO NOTHING"""
        )
        conn.commit()

    @staticmethod
    def _normalize_workspace_alias_schema(conn: sqlite3.Connection) -> None:
        """Compact legacy alias rows into the internal decision-state table."""
        table_info = conn.execute("PRAGMA table_info(workspace_aliases)").fetchall()
        columns = [str(row[1]) for row in table_info]
        required = {"alias_workspace", "canonical", "status"}
        if not required <= set(columns):
            raise sqlite3.DatabaseError(
                "workspace_aliases is missing required decision-state columns"
            )
        pk_columns = [
            str(row[1])
            for row in sorted(table_info, key=lambda item: int(item[5] or 0))
            if int(row[5] or 0) > 0
        ]
        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='workspace_aliases'"
        ).fetchone()
        table_sql = str(sql_row[0] or "").replace(" ", "").casefold() if sql_row else ""
        compact_shape = columns == ["alias_workspace", "canonical", "status", "updated_at"]
        compact_constraints = (
            pk_columns == ["alias_workspace", "canonical"]
            and all(int(row[3] or 0) == 1 for row in table_info)
            and all(str(row[2] or "").casefold() == "text" for row in table_info)
            and "check(statusin('confirmed','rejected'))" in table_sql
        )
        legacy_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_aliases_legacy'"
        ).fetchone() is not None
        if not compact_shape or not compact_constraints or legacy_exists:
            sources = ["workspace_aliases_legacy", "workspace_aliases"] if legacy_exists else ["workspace_aliases"]
            for source in sources:
                source_columns = {
                    str(row[1]) for row in conn.execute(f"PRAGMA table_info({source})")
                }
                if not required <= source_columns:
                    raise sqlite3.DatabaseError(
                        f"{source} is missing required decision-state columns"
                    )
            conn.execute("SAVEPOINT compact_workspace_aliases")
            try:
                conn.execute("DROP TABLE IF EXISTS workspace_aliases_compact")
                conn.execute(
                    "CREATE TABLE workspace_aliases_compact ("
                    "alias_workspace TEXT NOT NULL,canonical TEXT NOT NULL,"
                    "status TEXT NOT NULL CHECK(status IN ('confirmed','rejected')),"
                    "updated_at TEXT NOT NULL,PRIMARY KEY(alias_workspace,canonical))"
                )
                for source in sources:
                    source_columns = {
                        str(row[1]) for row in conn.execute(f"PRAGMA table_info({source})")
                    }
                    invalid = int(conn.execute(
                        f"SELECT COUNT(*) FROM {source} "
                        "WHERE status NOT IN ('confirmed','rejected') OR status IS NULL"
                    ).fetchone()[0])
                    if invalid:
                        raise sqlite3.DatabaseError(
                            f"{source} contains invalid workspace decision states"
                        )
                    updated_expr = (
                        "COALESCE(updated_at,CURRENT_TIMESTAMP)"
                        if "updated_at" in source_columns else "CURRENT_TIMESTAMP"
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO workspace_aliases_compact("
                        "alias_workspace,canonical,status,updated_at) "
                        "SELECT alias_workspace,canonical,status," + updated_expr + " "
                        f"FROM {source}"
                    )
                conn.execute("DROP TABLE workspace_aliases")
                conn.execute("DROP TABLE IF EXISTS workspace_aliases_legacy")
                conn.execute("ALTER TABLE workspace_aliases_compact RENAME TO workspace_aliases")
                conn.execute("RELEASE compact_workspace_aliases")
            except BaseException:
                conn.execute("ROLLBACK TO compact_workspace_aliases")
                conn.execute("RELEASE compact_workspace_aliases")
                raise
        conn.execute("DROP INDEX IF EXISTS idx_ws_alias_events")
        conn.execute("DROP TABLE IF EXISTS workspace_alias_events")

    def _probe_features(self, conn: sqlite3.Connection, *, initialize: bool = True) -> None:
        if self.settings.enable_sqlite_vec:
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._sqlite_vec_loadable = True
                self.state.sqlite_vec_available = True
                self.state.mode = "sqlite_vec"
                if initialize:
                    self._ensure_evidence_vec_table(conn)
                    self._ensure_workspace_vec_table(conn)
                else:
                    # Constant-size capability probe only. Missing/corrupt
                    # vector tables are a doctor/repair concern, not startup DDL.
                    conn.execute("SELECT id FROM memory_evidence_vec LIMIT 0")
                    conn.execute("SELECT id FROM workspace_canonicals_vec LIMIT 0")
            except Exception as exc:  # pragma: no cover - environment dependent
                self._sqlite_vec_loadable = False
                self.state.sqlite_vec_available = False
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
            if initialize:
                self._ensure_fts(conn)
                self._rebuild_fts(conn)
            else:
                conn.execute("SELECT rowid FROM memories_fts LIMIT 0")
            self.state.fts5_available = True
            if not self.state.sqlite_vec_available:
                self.state.mode = "fts5"
        except sqlite3.Error as exc:
            self.state.warn(f"SQLite FTS5 unavailable: {exc}. Falling back to LIKE search.")
            if not self.state.sqlite_vec_available:
                self.state.mode = "like"

        if initialize:
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

    def _probe_sqlite_vec_loadable(self) -> bool | None:
        try:
            import sqlite_vec

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

    def ensure_vector_tables_for_repair(self) -> tuple[bool, list[str]]:
        """Create missing derived vec0 tables only from an explicit repair."""
        if not self.settings.enable_sqlite_vec:
            return False, ["vec.enabled must be true to repair vector tables"]
        conn: sqlite3.Connection | None = None
        try:
            conn = self._db._new_connection()
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            before = {
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('memory_evidence_vec','workspace_canonicals_vec')"
                )
            }
            self._ensure_evidence_vec_table(conn)
            self._ensure_workspace_vec_table(conn)
            self._db._sqlite_vec_loadable = True
            self.state.sqlite_vec_available = True
            self.state.mode = "sqlite_vec"
            return before != {"memory_evidence_vec", "workspace_canonicals_vec"}, []
        except (ImportError, sqlite3.Error) as exc:
            return False, [f"vector table repair unavailable: {exc}"]
        finally:
            if conn is not None:
                conn.close()

    def missing_vector_tables(self) -> list[str]:
        """Read-only preview of derived vec0 tables requiring recreation."""
        expected = {"memory_evidence_vec", "workspace_canonicals_vec"}
        try:
            with self._db.connection() as conn:
                present = {
                    str(row[0]) for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name IN ('memory_evidence_vec','workspace_canonicals_vec')"
                    )
                }
        except sqlite3.Error:
            return sorted(expected)
        return sorted(expected - present)
