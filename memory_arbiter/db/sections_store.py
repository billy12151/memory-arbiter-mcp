"""Section storage operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional, Tuple, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


class SectionStore:
    """CRUD + vec matching for ``memory_sections`` and ``memory_sections_vec``."""

    def __init__(self, db: "MemoryDB"):
        self._db = db

    @staticmethod
    def insert_section(
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
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    @staticmethod
    def store_section_vec(
        conn: sqlite3.Connection,
        section_id: int,
        embedding: list[float],
    ) -> None:
        if not embedding:
            raise ValueError("section embedding is empty (encode failed)")
        conn.execute("DELETE FROM memory_sections_vec WHERE id = ?", (section_id,))
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
    def delete_sections_for_memory(conn: sqlite3.Connection, memory_id: int) -> int:
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
            pass
        conn.execute("DELETE FROM memory_sections WHERE memory_id = ?", (memory_id,))
        return int(count)

    @staticmethod
    def get_sections(conn: sqlite3.Connection, memory_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM memory_sections WHERE memory_id = ? ORDER BY section_index",
            (memory_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def get_section_vec_ids(conn: sqlite3.Connection, memory_id: int) -> set[int]:
        rows = conn.execute(
            "SELECT v.id AS id FROM memory_sections_vec v "
            "JOIN memory_sections s ON s.id = v.id WHERE s.memory_id = ?",
            (memory_id,),
        ).fetchall()
        return {int(r["id"]) for r in rows}

    def get_sections_by_memory(self, memory_id: int) -> list[dict[str, Any]]:
        db = self._db
        if not db._db_available:
            return []
        with db.connection() as conn:
            return self.get_sections(conn, memory_id)

    def get_sections_by_ids(
        self, memory_id: int, section_ids: list[int]
    ) -> Tuple[list[dict[str, Any]], list[int]]:
        db = self._db
        if not db._db_available or not section_ids:
            return [], []
        with db.connection() as conn:
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
        db = self._db
        if not db._db_available or not db.state.sqlite_vec_available:
            return []
        try:
            with db.connection() as conn:
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
