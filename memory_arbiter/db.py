from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, Tuple

from .config import Settings
from .degrade import DegradeState
from .models import MemoryRecord, utc_now_iso


class MemoryDB:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = DegradeState()
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(str(self.settings.db_path))
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
            self._probe_features()
        except sqlite3.Error as exc:
            self.conn = None
            self.state.sqlite_writable = False
            self.state.mode = "jsonl_backup"
            self.state.jsonl_backup_active = True
            self.state.warn(f"SQLite unavailable or not writable: {exc}. Using JSONL append-only backup when possible.")

    def _init_schema(self) -> None:
        assert self.conn is not None
        self.conn.executescript(
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
            """
        )
        self._migrate_add_version_column()
        self.conn.commit()

    def _migrate_add_version_column(self) -> None:
        """Idempotently add the `version` column to the memories table.

        CREATE TABLE IF NOT EXISTS only affects fresh DBs; existing on-disk
        databases need an ALTER TABLE to pick up the column. Probed via
        PRAGMA table_info so it's safe to run repeatedly.
        """
        assert self.conn is not None
        cols = {str(row["name"]) for row in self.conn.execute("PRAGMA table_info(memories)")}
        if "version" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN version INTEGER NOT NULL DEFAULT 1")

    def _probe_features(self) -> None:
        assert self.conn is not None
        if self.settings.enable_sqlite_vec:
            try:
                import sqlite_vec  # type: ignore

                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.state.sqlite_vec_available = True
                self.state.mode = "sqlite_vec"
                self._ensure_vec_table()
            except Exception as exc:  # pragma: no cover - depends on local optional package
                self.state.warn(f"sqlite-vec unavailable: {exc}. Semantic recall disabled; falling back to FTS5 or keyword search.")
        else:
            self.state.warn("sqlite-vec disabled by configuration. Semantic recall disabled.")

        try:
            self._ensure_fts()
            self._rebuild_fts()
            self.state.fts5_available = True
            if not self.state.sqlite_vec_available:
                self.state.mode = "fts5"
        except sqlite3.Error as exc:
            self.state.warn(f"SQLite FTS5 unavailable: {exc}. Falling back to LIKE/keyword search.")
            if not self.state.sqlite_vec_available:
                self.state.mode = "like"

        try:
            self.conn.execute("CREATE TABLE IF NOT EXISTS write_probe (id INTEGER)")
            self.conn.execute("INSERT INTO write_probe(id) VALUES (1)")
            self.conn.execute("DELETE FROM write_probe")
            self.conn.commit()
        except sqlite3.Error as exc:
            self.state.sqlite_writable = False
            self.state.mode = "jsonl_backup"
            self.state.jsonl_backup_active = True
            self.state.warn(f"SQLite opened read-only or write probe failed: {exc}. Writes will use JSONL backup when possible.")

    def _rebuild_fts(self) -> None:
        assert self.conn is not None
        try:
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
            self.conn.commit()
        except sqlite3.Error:
            self.conn.rollback()

    def _ensure_fts(self) -> None:
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories_fts'"
        ).fetchone()
        if row:
            sql = str(row["sql"] or "").lower()
            if "tokenize='trigram'" in sql or 'tokenize="trigram"' in sql or "tokenize=trigram" in sql:
                return
            self.conn.execute("DROP TABLE memories_fts")
            self.conn.commit()
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE memories_fts USING fts5(content, tags, subject, content='memories', content_rowid='id', tokenize='trigram')"
            )
        except sqlite3.Error as exc:
            self.state.warn(f"FTS5 trigram tokenizer unavailable: {exc}. Falling back to default FTS5 tokenizer.")
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tags, subject, content='memories', content_rowid='id')"
            )

    def _ensure_vec_table(self) -> None:
        """Create the vec0 table for optional semantic recall.

        Dimension is read from settings (env MEMORY_ARBITER_VEC_DIM, default 768).
        Users are responsible for backfilling embeddings; the table starts empty.
        If the dimension changes, the user must drop + recreate the table.
        """
        assert self.conn is not None
        dim = int(getattr(self.settings, "vec_dim", 768) or 768)
        try:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
            self.conn.commit()
        except sqlite3.Error as exc:
            self.state.warn(f"vec0 table creation failed (dim={dim}): {exc}. Semantic recall disabled.")
            self.state.sqlite_vec_available = False

    def store_embedding(self, memory_id: int, embedding: list[float]) -> Tuple[bool, list[str]]:
        """Store or replace an embedding for a memory. Returns (ok, warnings)."""
        warnings: list[str] = []
        if self.conn is None or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; embedding not stored."]
        if not self.state.sqlite_vec_available:
            return False, ["sqlite-vec unavailable; embedding not stored."]
        try:
            # vec0 has no UPSERT; delete then insert to replace.
            self.conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            self.conn.execute(
                "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                (memory_id, json.dumps(embedding)),
            )
            self.conn.commit()
            return True, []
        except sqlite3.Error as exc:
            self.conn.rollback()
            warnings.append(f"store_embedding failed: {exc}")
            return False, warnings

    def vec_knn(self, query_embedding: list[float], k: int = 10) -> list[dict[str, Any]]:
        """Return up to k nearest neighbors by cosine distance. Empty if vec unavailable."""
        if self.conn is None or not self.state.sqlite_vec_available:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT v.id AS id, v.distance AS distance, m.workspace AS workspace,
                       m.agent_id AS agent_id, m.status AS status, m.subject AS subject,
                       m.tags AS tags, m.content AS content, m.source_type AS source_type,
                       m.confidence AS confidence, m.protection_level AS protection_level,
                       m.event_time AS event_time, m.ingest_time AS ingest_time,
                       m.metadata AS metadata
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

    def insert_memory(self, record: MemoryRecord) -> Tuple[Optional[int], list[str]]:
        warnings: list[str] = []
        if not record.content:
            raise ValueError("content is required")
        if self.conn is None or not self.state.sqlite_writable:
            self._append_backup(record)
            warnings.append("SQLite write unavailable; wrote append-only JSONL backup.")
            return None, warnings
        cur = self.conn.execute(
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
            self.conn.execute(
                "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                (memory_id, record.content, " ".join(record.tags), record.subject or ""),
            )
        self.conn.commit()
        return memory_id, warnings

    def _append_backup(self, record: MemoryRecord) -> None:
        self.settings.backup_jsonl.parent.mkdir(parents=True, exist_ok=True)
        payload = record.__dict__.copy()
        payload["backup_written_at"] = utc_now_iso()
        with self.settings.backup_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.state.jsonl_backup_active = True

    def get_memory(self, memory_id: int) -> Optional[dict[str, Any]]:
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return row_to_dict(row) if row else None

    def update_memory(self, memory_id: int, updates: dict[str, Any]) -> bool:
        if self.conn is None or not self.state.sqlite_writable:
            return False
        allowed = {"source_type", "confidence", "protection_level", "status", "metadata"}
        pairs = [(key, value) for key, value in updates.items() if key in allowed]
        if not pairs:
            return True
        sql = ", ".join(f"{key} = ?" for key, _ in pairs)
        values = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for _, v in pairs]
        values.append(memory_id)
        self.conn.execute(f"UPDATE memories SET {sql} WHERE id = ?", values)
        self.conn.commit()
        return True

    def list_memories(self, workspace: Optional[str] = None, subject: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if self.conn is None:
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
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY event_time DESC, ingest_time DESC LIMIT ?",
            params,
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def record_conflict(self, left_id: int, right_id: int, subject: Optional[str], reason: str, winner_id: Optional[int], status: str = "open") -> Optional[int]:
        if self.conn is None or not self.state.sqlite_writable:
            return None
        cur = self.conn.execute(
            """
            INSERT INTO conflicts(left_id, right_id, subject, status, reason, winner_id, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (left_id, right_id, subject, status, reason, winner_id, utc_now_iso(), utc_now_iso() if status != "open" else None),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def resolve_conflicts_for(self, memory_id: int) -> int:
        """Mark all open conflicts involving memory_id as resolved.

        Used when a memory is explicitly superseded: dangling open conflicts
        that reference it no longer need human review. Returns affected row count.
        """
        if self.conn is None or not self.state.sqlite_writable:
            return 0
        cur = self.conn.execute(
            "UPDATE conflicts SET status='resolved', resolved_at=? "
            "WHERE status='open' AND (left_id=? OR right_id=?)",
            (utc_now_iso(), memory_id, memory_id),
        )
        self.conn.commit()
        return cur.rowcount

    def list_conflicts(self, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        if self.conn is None:
            return []
        rows = self.conn.execute(
            "SELECT * FROM conflicts WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def edit_memory(
        self,
        memory_id: int,
        new_content: str,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: Optional[str] = None,
    ) -> Optional[int]:
        """In-place edit a memory's content, archiving the prior version to memory_history.

        Stores a snapshot of the old content/subject/tags at the old version,
        bumps memories.version, and rewrites the FTS row so search sees the new
        text immediately (update_memory does not sync FTS for content edits).
        Returns the history row id, or None if the DB is unwritable.
        """
        if self.conn is None or not self.state.sqlite_writable:
            return None
        current = self.get_memory(memory_id)
        if not current:
            return None
        old_content = current["content"]
        old_subject = current.get("subject")
        old_tags = current.get("tags") or []
        old_version = int(current.get("version") or 1)
        tags_json = json.dumps(new_tags, ensure_ascii=False) if new_tags is not None else json.dumps(old_tags, ensure_ascii=False)
        subject_value = new_subject if new_subject is not None else old_subject
        try:
            history_cur = self.conn.execute(
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
            self.conn.execute(
                "UPDATE memories SET content=?, subject=?, tags=?, version=? WHERE id=?",
                (new_content, subject_value, tags_json, old_version + 1, memory_id),
            )
            if self.state.fts5_available:
                # External-content FTS5: remove the old row via the special 'delete'
                # command (requires the row's pre-edit values), then re-insert fresh.
                self.conn.execute(
                    "INSERT INTO memories_fts(memories_fts, rowid, content, tags, subject) VALUES('delete', ?, ?, ?, ?)",
                    (memory_id, old_content, " ".join(old_tags), old_subject or ""),
                )
                self.conn.execute(
                    "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                    (memory_id, new_content, " ".join(new_tags) if new_tags is not None else " ".join(old_tags), subject_value or ""),
                )
            self.conn.commit()
            return history_id
        except sqlite3.Error:
            self.conn.rollback()
            return None

    def list_history(self, memory_id: int) -> list[dict[str, Any]]:
        """Return version snapshots for a memory, newest version first."""
        if self.conn is None:
            return []
        rows = self.conn.execute(
            "SELECT * FROM memory_history WHERE memory_id=? ORDER BY version DESC, id DESC",
            (memory_id,),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def cleanup_history(self, memory_id: Optional[int] = None, older_than_days: Optional[int] = None) -> int:
        """Delete historical snapshots from memory_history.

        SAFETY RED LINE: this method only ever issues DELETE against
        memory_history. It must never touch the memories table, regardless of
        arguments. Full cleanup (no args) wipes the entire history table.
        """
        if self.conn is None or not self.state.sqlite_writable:
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
        cur = self.conn.execute(f"DELETE FROM memory_history {where}", params)
        self.conn.commit()
        return cur.rowcount

    def audit_summary(self) -> dict[str, Any]:
        """Pure-Sql aggregate overview per workspace. No semantic judgement.

        Returns memory counts, oldest/newest event_time, open conflicts and
        source_type distribution for every workspace, plus global totals.
        """
        empty = {"workspaces": {}, "total_memories": 0, "total_open_conflicts": 0}
        if self.conn is None:
            return empty

        mem_rows = self.conn.execute(
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

        open_conflict_rows = self.conn.execute(
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


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data
