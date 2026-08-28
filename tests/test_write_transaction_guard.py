"""write_transaction cleanup must never mask the original error.

When BEGIN IMMEDIATE itself fails (busy timeout), there is no active
transaction; a blind ROLLBACK raises "cannot rollback - no transaction is
active" and replaces the real cause. The guarded rollback fixes that while
keeping normal body-failure rollback intact (adversarial-review leftover,
2026-08-28).
"""
import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB


def make_db(tmp_path: Path) -> MemoryDB:
    settings = Settings(
        db_path=tmp_path / "txn.sqlite3",
        backup_jsonl=tmp_path / "txn.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        enable_sqlite_vec=False, vec_dim=2, isolation="none",
    )
    return MemoryDB(settings)


class _LockedConnection:
    """Connection stub whose BEGIN IMMEDIATE always fails."""

    in_transaction = False

    def __init__(self) -> None:
        self.rollback_attempts = 0

    def execute(self, sql: str):
        if sql.startswith("BEGIN"):
            raise sqlite3.OperationalError("database is locked")
        if sql.startswith("ROLLBACK"):
            self.rollback_attempts += 1
            raise sqlite3.OperationalError("cannot rollback - no transaction is active")
        return None

    def close(self) -> None:
        pass


def test_begin_failure_preserves_original_error(tmp_path: Path, monkeypatch) -> None:
    db = make_db(tmp_path)
    stub = _LockedConnection()
    monkeypatch.setattr(db, "_new_connection", lambda: stub)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with db.write_transaction():
            pass
    assert stub.rollback_attempts == 0


def test_body_failure_still_rolls_back(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    with pytest.raises(ValueError, match="boom"):
        with db.write_transaction() as conn:
            conn.execute(
                "INSERT INTO workspace_canonicals(name, created_at) "
                "VALUES ('ghost-bucket', '2026-01-01T00:00:00Z')"
            )
            raise ValueError("boom")
    with db.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='ghost-bucket'"
        ).fetchone()
    assert row is None
