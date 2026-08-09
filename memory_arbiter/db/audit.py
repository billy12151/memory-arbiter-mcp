"""Audit / diagnostic-log persistence for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata", "structured_details"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


class AuditStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    def get_memory_summaries(self, memory_ids: list[int]) -> dict[int, dict[str, Any]]:
        db = self._db
        if not memory_ids or not db._db_available:
            return {}
        unique_ids = sorted(set(int(i) for i in memory_ids if i is not None))
        if not unique_ids:
            return {}
        out: dict[int, dict[str, Any]] = {}
        try:
            with db.connection() as conn:
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
                        out[int(r["id"])] = _row_to_dict(r)
        except sqlite3.Error:
            return {}
        return out

    @property
    def scan_log_path(self) -> Path:
        return self._db.settings.db_path.parent / "scan_log.jsonl"

    @property
    def attention_log_path(self) -> Path:
        return self._db.settings.db_path.parent / "attention_log.jsonl"

    def log_attention(self, *, trigger: str, source: str, memory_ids: list[Any]) -> None:
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

    def scan_log_last_completed(self) -> Optional[dict[str, Any]]:
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

    def audit_summary(self) -> dict[str, Any]:
        db = self._db
        empty = {"workspaces": {}, "total_memories": 0, "total_open_conflicts": 0}
        if not db._db_available:
            return empty
        with db.connection() as conn:
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
