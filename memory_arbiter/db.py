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
            """
        )
        self.conn.commit()

    def _probe_features(self) -> None:
        assert self.conn is not None
        if self.settings.enable_sqlite_vec:
            try:
                import sqlite_vec  # type: ignore

                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.state.sqlite_vec_available = True
                self.state.mode = "sqlite_vec"
            except Exception as exc:  # pragma: no cover - depends on local optional package
                self.state.warn(f"sqlite-vec unavailable: {exc}. Semantic recall disabled; falling back to FTS5 or keyword search.")
        else:
            self.state.warn("sqlite-vec disabled by configuration. Semantic recall disabled.")

        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tags, subject, content='memories', content_rowid='id')"
            )
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

    def record_conflict(self, left_id: int, right_id: int, subject: Optional[str], reason: str, winner_id: Optional[int]) -> Optional[int]:
        if self.conn is None or not self.state.sqlite_writable:
            return None
        cur = self.conn.execute(
            """
            INSERT INTO conflicts(left_id, right_id, subject, reason, winner_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (left_id, right_id, subject, reason, winner_id, utc_now_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_conflicts(self, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        if self.conn is None:
            return []
        rows = self.conn.execute(
            "SELECT * FROM conflicts WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data
