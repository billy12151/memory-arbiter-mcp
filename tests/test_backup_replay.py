from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import MemoryRecord
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "memory.backup.jsonl",
        enable_sqlite_vec=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write_envelope(path: Path, replay_key: str, content: str = "backup fact") -> None:
    record = MemoryRecord(
        content=content, subject="backup", agent_id="agent", workspace="project",
    )
    envelope = {
        "backup_schema": 1,
        "replay_key": replay_key,
        "backup_written_at": "2026-08-16T10:00:00.123456+00:00",
        "workspace_canonical": "project",
        "record": record.__dict__,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(envelope, ensure_ascii=False) + "\n")


def test_backup_envelope_has_replay_key_and_canonical(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.db._db_available = False
    tools.db.state.sqlite_writable = False
    result = tools.memory_write(content="backup", subject="backup", workspace="project")
    assert result["ok"] is True
    assert result["data"]["backup_only"] is True
    envelope = json.loads(tools.settings.backup_jsonl.read_text(encoding="utf-8"))
    assert envelope["backup_schema"] == 1
    assert envelope["replay_key"]
    assert envelope["workspace_canonical"] == "project"
    assert envelope["record"]["content"] == "backup"


def test_replay_dry_run_authorization_and_idempotency(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "entry-1")
    preview = tools.memory_repair("replay_backup", {"dry_run": True})
    assert preview["data"]["importable"] == 1
    denied = tools.memory_repair("replay_backup", {"dry_run": False})
    assert denied["ok"] is False
    assert denied["data"]["action_required"] == "ask_user_for_authorization"
    replayed = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert replayed["ok"] is True
    memory_id = replayed["data"]["imported"][0]["memory_id"]
    assert tools.db.get_memory(memory_id)["content"] == "backup fact"
    repeated = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert repeated["data"]["imported_count"] == 0
    assert repeated["data"]["already_replayed_count"] == 1


def test_replay_bad_line_does_not_block_valid_entry(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.backup_jsonl.write_text("not-json\n", encoding="utf-8")
    _write_envelope(tools.settings.backup_jsonl, "entry-valid")
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert result["data"]["imported_count"] == 1
    assert len(result["data"]["invalid_entries"]) == 1


def test_replay_receipt_failure_rolls_back_memory(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "entry-rollback")
    original = tools.db.insert_memory_on_conn

    def insert_then_break(conn, record, workspace_canonical=None):
        memory_id = original(conn, record, workspace_canonical)
        conn.execute("DROP TABLE backup_replay_log")
        return memory_id

    monkeypatch.setattr(tools.db, "insert_memory_on_conn", insert_then_break)
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert result["ok"] is False
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='backup_replay_log'").fetchone() is not None


def test_concurrent_replay_imports_once(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "entry-race")
    barrier = threading.Barrier(2)
    results = []

    def run() -> None:
        barrier.wait()
        results.append(tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True}))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM backup_replay_log").fetchone()[0] == 1


def test_strict_replay_unknown_workspace_stays_pending(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "strict.sqlite3",
        backup_jsonl=tmp_path / "strict.jsonl",
        isolation="strict",
    )
    tools = MemoryTools(settings, MemoryDB(settings))
    _write_envelope(settings.backup_jsonl, "strict-entry")
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    memory_id = result["data"]["imported"][0]["memory_id"]
    assert tools.db.get_memory(memory_id)["status"] == "pending"


def test_oversized_backup_line_rejected_before_json_parse(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.backup_jsonl.write_bytes(b"{" + b"x" * (3 * 1024 * 1024) + b"}\n")
    preview = tools.memory_repair("replay_backup", {"dry_run": True})
    assert preview["data"]["invalid"] == 1
    assert "size limit" in preview["data"]["invalid_entries"][0]["reason"]


def test_replay_pagination_reaches_entries_after_replayed_prefix(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for index in range(3):
        _write_envelope(tools.settings.backup_jsonl, f"entry-{index}", content=f"fact-{index}")
    first = tools.memory_repair(
        "replay_backup", {"dry_run": False, "authorized": True, "limit": 2, "offset": 0},
    )
    assert first["data"]["imported_count"] == 2
    assert first["data"]["next_offset"] == 2
    second = tools.memory_repair(
        "replay_backup", {"dry_run": False, "authorized": True, "limit": 2, "offset": 2},
    )
    assert second["data"]["imported_count"] == 1
    assert second["data"]["has_more"] is False
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3


def test_backup_envelope_rejects_bad_time_and_unbounded_key(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = MemoryRecord(content="x", subject="s", agent_id="a", workspace="w")
    bad = {
        "backup_schema": 1,
        "replay_key": "x" * 201,
        "backup_written_at": "not-a-time",
        "workspace_canonical": "w",
        "record": record.__dict__,
    }
    tools.settings.backup_jsonl.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    preview = tools.memory_repair("replay_backup", {"dry_run": True})
    assert preview["data"]["invalid"] == 1


def test_backup_notice_scan_is_cached_until_file_or_receipts_change(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "notice-entry")
    calls = 0
    original = tools.db.backup_replay.inspect

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tools.db.backup_replay, "inspect", counted)
    first = tools._consume_notices()
    second = tools._consume_notices()
    assert first and first[0]["type"] == "backup_replay_pending"
    assert second == []
    assert calls == 1
    tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    tools._consume_notices()
    assert calls >= 2
