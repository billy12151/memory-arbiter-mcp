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
from memory_arbiter.tools import MemoryTools


def make_db(tmp_path: Path) -> MemoryDB:
    settings = Settings(
        db_path=tmp_path / "txn.sqlite3",
        backup_jsonl=tmp_path / "txn.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        isolation="none",
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


def test_post_commit_failure_after_insert_reports_ok_with_warning(tmp_path: Path) -> None:
    """Adversarial round 2: an exception in post-commit processing (e.g. the
    semantic worker reserve) AFTER the insert committed must not be reported
    as {written: False} — the row is durable, so a failure response makes the
    caller retry and duplicate the memory. The write degrades to an ok
    response carrying the new memory id plus a warning."""
    settings = Settings(
        db_path=tmp_path / "post.sqlite3",
        backup_jsonl=tmp_path / "post.jsonl",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)

    baseline = tools.memory_write(content="baseline", subject="s0", tags=[])
    assert baseline["ok"] is True

    def boom(task_id):
        raise RuntimeError("injected worker fault")

    tools._semantic_worker.reserve = boom
    failed = tools.memory_write(content="important fact", subject="dup-test", tags=[])

    assert failed["ok"] is True, failed
    assert failed["data"].get("written") is not False
    memory_id = failed["data"]["id"]
    assert memory_id is not None
    assert any(
        "post-commit processing failed" in warning
        for warning in failed.get("warnings") or []
    ), failed

    # The row committed exactly once; a caller that only retries on
    # ok=False never duplicates it.
    with db.connection() as conn:
        rows = conn.execute("SELECT id FROM memories WHERE subject='dup-test'").fetchall()
    assert [int(r["id"]) for r in rows] == [memory_id]
