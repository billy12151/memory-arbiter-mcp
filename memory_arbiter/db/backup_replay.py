"""Deterministic JSONL backup inspection and replay."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..models import MemoryRecord, utc_now_iso
from ..timeutil import parse_iso8601_utc
from ..validation import MAX_TEXT_FIELD_CHARS, validate_product_payload

if TYPE_CHECKING:
    from .core import MemoryDB


class BackupReplayStore:
    MAX_BACKUP_LINE_BYTES = 3 * 1024 * 1024

    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    @property
    def path(self) -> Path:
        return self._db.settings.backup_jsonl

    @staticmethod
    def _payload_hash(envelope: dict[str, Any]) -> str:
        raw = json.dumps(
            {
                "backup_schema": envelope.get("backup_schema"),
                "workspace_canonical": envelope.get("workspace_canonical"),
                "record": envelope.get("record"),
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def state_signature(self) -> tuple[int, int, int]:
        """Cheap change detector for response-time backup notices."""
        try:
            stat = self.path.stat()
            file_state = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            file_state = (0, 0)
        receipt_count = 0
        if self._db.db_available:
            try:
                with self._db.connection() as conn:
                    row = conn.execute("SELECT COUNT(*) FROM backup_replay_log").fetchone()
                    receipt_count = int(row[0] or 0) if row else 0
            except sqlite3.Error:
                receipt_count = -1
        return (*file_state, receipt_count)

    def inspect(self, limit: int = 1_000, offset: int = 0) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        seen: set[str] = set()
        max_entries = max(1, min(int(limit), 10_000))
        offset = max(0, int(offset))
        if not self.path.exists():
            return {"entries": [], "invalid_entries": [], "total": 0, "importable": 0, "already_replayed": 0, "conflicts": 0, "invalid": 0, "offset": offset, "next_offset": None, "has_more": False}
        fh = self.path.open("rb")
        has_more = False
        try:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            except ImportError:  # pragma: no cover
                fcntl = None  # type: ignore[assignment]
            for line_number, raw_line in enumerate(fh, 1):
                if line_number <= offset:
                    continue
                if len(entries) + len(invalid) >= max_entries:
                    has_more = True
                    break
                if len(raw_line) > self.MAX_BACKUP_LINE_BYTES:
                    invalid.append({"line": line_number, "reason": "backup line exceeds size limit"})
                    continue
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    invalid.append({"line": line_number, "reason": f"invalid UTF-8: {exc}"})
                    continue
                try:
                    envelope = json.loads(line)
                    if not isinstance(envelope, dict) or type(envelope.get("backup_schema")) is not int or envelope.get("backup_schema") != 1:
                        raise ValueError("unsupported backup_schema")
                    replay_key = envelope.get("replay_key")
                    record = envelope.get("record")
                    if not isinstance(replay_key, str) or not replay_key.strip() or len(replay_key) > 200:
                        raise ValueError("replay_key must be a non-empty string of at most 200 characters")
                    if parse_iso8601_utc(envelope.get("backup_written_at")) is None:
                        raise ValueError("backup_written_at must be ISO 8601")
                    if replay_key in seen:
                        raise ValueError("duplicate replay_key in backup file")
                    seen.add(replay_key)
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    validation_payload = dict(record)
                    validation = validate_product_payload(
                        "memory", "remember", validation_payload,
                        vec_dim=int(self._db.settings.vec_dim),
                    )
                    if validation.error is not None:
                        raise ValueError(f"invalid record: {validation.error.get('field')}: {validation.error.get('reason')}")
                    canonical = envelope.get("workspace_canonical")
                    if not isinstance(canonical, str) or not canonical.strip() or len(canonical) > MAX_TEXT_FIELD_CHARS:
                        raise ValueError("workspace_canonical must be a non-empty bounded string")
                    entries.append({
                        "line": line_number,
                        "replay_key": replay_key,
                        "payload_hash": self._payload_hash(envelope),
                        "workspace_canonical": envelope.get("workspace_canonical"),
                        "record": record,
                    })
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    invalid.append({"line": line_number, "reason": str(exc)})
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
        replayed: dict[str, dict[str, Any]] = {}
        if entries and self._db.db_available:
            keys = [entry["replay_key"] for entry in entries]
            with self._db.connection() as conn:
                for start in range(0, len(keys), 500):
                    chunk = keys[start:start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT replay_key, memory_id, payload_hash FROM backup_replay_log WHERE replay_key IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    replayed.update({str(row["replay_key"]): dict(row) for row in rows})
        for entry in entries:
            prior = replayed.get(entry["replay_key"])
            if prior is None:
                entry["status"] = "importable"
            elif prior["payload_hash"] == entry["payload_hash"]:
                entry["status"] = "already_replayed"
                entry["memory_id"] = int(prior["memory_id"])
            else:
                entry["status"] = "receipt_hash_conflict"
        consumed = len(entries) + len(invalid)
        return {
            "entries": entries,
            "invalid_entries": invalid,
            "total": len(entries) + len(invalid),
            "importable": sum(entry["status"] == "importable" for entry in entries),
            "already_replayed": sum(entry["status"] == "already_replayed" for entry in entries),
            "conflicts": sum(entry["status"] == "receipt_hash_conflict" for entry in entries),
            "invalid": len(invalid),
            "offset": offset,
            "next_offset": offset + consumed if has_more else None,
            "has_more": has_more,
        }

    def replay_one(self, entry: dict[str, Any]) -> dict[str, Any]:
        if not self._db.db_available or not self._db.state.sqlite_writable:
            return {"outcome": "sqlite_unavailable", "replay_key": entry["replay_key"]}
        record = MemoryRecord.from_input(entry["record"], self._db.settings.defaults())
        replay_key = str(entry["replay_key"])
        payload_hash = str(entry["payload_hash"])
        try:
            with self._db.write_transaction() as conn:
                prior = conn.execute(
                    "SELECT memory_id, payload_hash FROM backup_replay_log WHERE replay_key=?",
                    (replay_key,),
                ).fetchone()
                if prior is not None:
                    outcome = "already_replayed" if prior["payload_hash"] == payload_hash else "receipt_hash_conflict"
                    return {"outcome": outcome, "replay_key": replay_key, "memory_id": int(prior["memory_id"])}
                canonical = str(entry.get("workspace_canonical") or record.workspace)
                if self._db.settings.isolation == "strict":
                    known = conn.execute(
                        "SELECT 1 FROM workspace_canonicals WHERE name=?", (canonical,),
                    ).fetchone()
                    if known is None:
                        record.status = "pending"
                memory_id = self._db.insert_memory_on_conn(conn, record, canonical)
                conn.execute(
                    "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                    (canonical, utc_now_iso()),
                )
                conn.execute(
                    "INSERT INTO backup_replay_log(replay_key, memory_id, payload_hash, replayed_at) VALUES (?, ?, ?, ?)",
                    (replay_key, memory_id, payload_hash, utc_now_iso()),
                )
            return {"outcome": "imported", "replay_key": replay_key, "memory_id": memory_id, "record": record}
        except sqlite3.IntegrityError:
            # A concurrent replayer may have committed the receipt first.
            with self._db.connection() as conn:
                prior = conn.execute(
                    "SELECT memory_id, payload_hash FROM backup_replay_log WHERE replay_key=?",
                    (replay_key,),
                ).fetchone()
            if prior is not None:
                outcome = "already_replayed" if prior["payload_hash"] == payload_hash else "receipt_hash_conflict"
                return {"outcome": outcome, "replay_key": replay_key, "memory_id": int(prior["memory_id"])}
            raise
