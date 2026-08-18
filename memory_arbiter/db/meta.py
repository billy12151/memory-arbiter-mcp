"""Vector-index metadata persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MemoryDB


class MetaStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    @staticmethod
    def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute("SELECT value FROM _vec_index_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    @staticmethod
    def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO _vec_index_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def delete_meta(conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM _vec_index_meta WHERE key = ?", (key,))

    def get_vec_index_state(self) -> dict[str, Any]:
        db = self._db
        if not db._db_available:
            return {"state": "unmanaged"}
        with db.connection() as conn:
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
        db = self._db
        if not db._db_available:
            return
        with db.write_transaction() as conn:
            if not has_managed_embedder or embedding_space_id is None:
                self.set_meta(conn, "state", "unmanaged")
                return

            rows = conn.execute("SELECT key, value FROM _vec_index_meta").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in rows}
            state = meta.get("state")
            active_space_id = meta.get("active_space_id")
            target_space_id = meta.get("target_space_id")

            if active_space_id == embedding_space_id:
                self.set_meta(conn, "state", "ready")
                for key in (
                    "target_space_id", "migration_cursor", "migration_epoch",
                    "migration_lease_owner", "migration_lease_expires_at",
                    "last_error",
                ):
                    self.delete_meta(conn, key)
                return

            if state in {"mismatch", "failed"} and target_space_id == embedding_space_id:
                return

            try:
                mem_vec_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_evidence_vec"
                ).fetchone()["c"]
            except sqlite3.Error:
                mem_vec_count = 0
            sec_vec_count = 0

            if not active_space_id and mem_vec_count == 0 and sec_vec_count == 0:
                self.set_meta(conn, "state", "ready")
                self.set_meta(conn, "active_space_id", embedding_space_id)
                self.delete_meta(conn, "target_space_id")
            else:
                self.set_meta(conn, "state", "mismatch")
                self.set_meta(conn, "target_space_id", embedding_space_id)
                self.set_meta(conn, "migration_epoch", uuid.uuid4().hex)
                for key in (
                    "migration_cursor", "migration_lease_owner",
                    "migration_lease_expires_at", "last_error",
                ):
                    self.delete_meta(conn, key)
