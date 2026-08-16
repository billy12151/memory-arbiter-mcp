from __future__ import annotations

import json
import os
import sqlite3
import stat
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


def test_strict_replay_same_unknown_canonical_all_stay_pending(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "strict-many.sqlite3",
        backup_jsonl=tmp_path / "strict-many.jsonl",
        isolation="strict",
    )
    tools = MemoryTools(settings, MemoryDB(settings))
    _write_envelope(settings.backup_jsonl, "strict-1", content="one")
    _write_envelope(settings.backup_jsonl, "strict-2", content="two")
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert result["data"]["imported_count"] == 2
    with tools.db.connection() as conn:
        statuses = [row[0] for row in conn.execute("SELECT status FROM memories ORDER BY id")]
    assert statuses == ["pending", "pending"]


def test_oversized_unterminated_line_is_bounded_and_next_line_survives(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.backup_jsonl.write_bytes(b"x" * (3 * 1024 * 1024 + 50) + b"\n")
    _write_envelope(tools.settings.backup_jsonl, "after-large")
    calls: list[int] = []
    original = tools.db.backup_replay._read_bounded_line

    def tracked(fh):
        result = original(fh)
        calls.append(len(result[0]))
        return result

    monkeypatch.setattr(tools.db.backup_replay, "_read_bounded_line", tracked)
    preview = tools.memory_repair("replay_backup", {"dry_run": True})
    assert preview["data"]["invalid"] == 1
    assert preview["data"]["importable"] == 1
    assert max(calls) <= tools.db.backup_replay.MAX_BACKUP_LINE_BYTES + 1


def test_replay_offset_above_20000_and_cross_page(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.backup_jsonl.write_text("not-json\n" * 20_001, encoding="utf-8")
    _write_envelope(tools.settings.backup_jsonl, "deep-1")
    _write_envelope(tools.settings.backup_jsonl, "deep-2")
    first = tools.memory_repair("replay_backup", {"dry_run": True, "limit": 1, "offset": 20_001})
    assert first["ok"] is True
    assert first["data"]["importable"] == 1
    assert first["data"]["next_offset"] == 20_002
    second = tools.memory_repair("replay_backup", {"dry_run": True, "limit": 1, "offset": first["data"]["next_offset"]})
    assert second["data"]["importable"] == 1
    assert second["data"]["has_more"] is False


def test_backup_writer_uses_single_append_write_and_0600(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    calls: list[tuple[int, int]] = []
    original = os.write

    def tracked(fd: int, data: bytes) -> int:
        calls.append((fd, len(data)))
        return original(fd, data)

    monkeypatch.setattr(os, "write", tracked)
    tools.db.memories._append_backup(MemoryRecord(content="x", subject="s", agent_id="a", workspace="w"))
    assert len(calls) == 1
    assert stat.S_IMODE(tools.settings.backup_jsonl.stat().st_mode) == 0o600


def test_backup_writer_short_write_fails(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    monkeypatch.setattr(os, "write", lambda fd, data: len(data) - 1)
    try:
        tools.db.memories._append_backup(MemoryRecord(content="x", subject="s", agent_id="a", workspace="w"))
    except OSError as exc:
        assert "short JSONL backup write" in str(exc)
    else:
        raise AssertionError("short write must fail")


def test_flat_legacy_row_is_explicitly_unsupported(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.backup_jsonl.write_text(json.dumps({"content": "old flat row", "workspace": "w"}) + "\n")
    preview = tools.memory_repair("replay_backup", {"dry_run": True})
    assert preview["data"]["invalid"] == 1
    assert "legacy_entry" in preview["data"]["invalid_entries"][0]["reason"]
    assert "not replayable" in preview["data"]["invalid_entries"][0]["reason"]


def test_replay_postprocess_warning_retries_idempotently(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "retry-postprocess")
    original = tools._operations._index_and_reconcile_claims
    attempts = 0

    def flaky(memory_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return original(memory_id)

    monkeypatch.setattr(tools._operations, "_index_and_reconcile_claims", flaky)
    first = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert first["data"]["imported_count"] == 1
    with tools.db.connection() as conn:
        assert conn.execute("SELECT postprocess_status FROM backup_replay_log").fetchone()[0] == "pending"
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    second = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert second["data"]["imported_count"] == 0
    assert second["data"]["already_replayed_count"] == 1
    with tools.db.connection() as conn:
        assert conn.execute("SELECT postprocess_status FROM backup_replay_log").fetchone()[0] == "complete"
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


def test_replay_terminal_warning_does_not_repeat_completed_stages(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "terminal-warning")
    calls = {"claims": 0, "split": 0, "semantic": 0}

    def claims(memory_id):
        calls["claims"] += 1
        return {"diagnostic": {"claim_indexed": True, "claim_reconciled": True}, "warnings": []}

    def split(memory_id):
        calls["split"] += 1
        return ({"reindex_pending": False}, {"memory_id": memory_id}, [])

    def semantic(memory_id, record):
        calls["semantic"] += 1
        return {"status": "disabled"}

    monkeypatch.setattr(tools._operations, "_index_and_reconcile_claims", claims)
    monkeypatch.setattr(tools._operations, "_after_write_split", split)
    monkeypatch.setattr(tools._operations, "_enqueue_semantic_conflict_check", semantic)
    first = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert first["data"]["imported"][0]["postprocess_status"] == "warning"
    second = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert second["data"]["already_replayed"][0]["postprocess_status"] == "warning"
    assert calls == {"claims": 1, "split": 1, "semantic": 1}


def test_replay_partial_success_retries_only_failed_stage(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "partial-stage")
    calls = {"claims": 0, "split": 0, "semantic": 0}

    def claims(memory_id):
        calls["claims"] += 1
        return {"diagnostic": {"claim_indexed": True, "claim_reconciled": True}, "warnings": []}

    def split(memory_id):
        calls["split"] += 1
        return ({"required": False}, None, [])

    semantic_results = iter(({"status": "queue_full"}, {"status": "queued"}))

    def semantic(memory_id, record):
        calls["semantic"] += 1
        return next(semantic_results)

    monkeypatch.setattr(tools._operations, "_index_and_reconcile_claims", claims)
    monkeypatch.setattr(tools._operations, "_after_write_split", split)
    monkeypatch.setattr(tools._operations, "_enqueue_semantic_conflict_check", semantic)
    first = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    receipt = first["data"]["imported"][0]
    assert receipt["postprocess_status"] == "pending"
    assert receipt["postprocess_error_code"] == "semantic_queue_full"
    second = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert second["data"]["already_replayed"][0]["postprocess_status"] == "complete"
    assert calls == {"claims": 1, "split": 1, "semantic": 2}


def test_replay_exception_after_checkpoint_retries_only_remaining_stages(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "exception-after-checkpoint")
    calls = {"claims": 0, "split": 0}

    def claims(memory_id):
        calls["claims"] += 1
        return {"diagnostic": {"claim_indexed": True, "claim_reconciled": True}, "warnings": []}

    def split(memory_id):
        calls["split"] += 1
        if calls["split"] == 1:
            raise RuntimeError("temporary split failure")
        return ({"required": False}, None, [])

    monkeypatch.setattr(tools._operations, "_index_and_reconcile_claims", claims)
    monkeypatch.setattr(tools._operations, "_after_write_split", split)
    monkeypatch.setattr(
        tools._operations, "_enqueue_semantic_conflict_check",
        lambda memory_id, record: {"status": "disabled"},
    )
    first = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert first["data"]["imported"][0]["postprocess_status"] == "pending"
    second = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert second["data"]["already_replayed"][0]["postprocess_status"] == "complete"
    assert calls == {"claims": 1, "split": 2}


def test_replay_semantic_nonterminal_enqueue_states_remain_pending(tmp_path: Path, monkeypatch) -> None:
    for status in ("queue_full", "runtime_disabled", "shutdown"):
        case = tmp_path / status
        case.mkdir()
        tools = make_tools(case)
        _write_envelope(tools.settings.backup_jsonl, f"semantic-{status}")
        monkeypatch.setattr(
            tools._operations, "_index_and_reconcile_claims",
            lambda memory_id: {"diagnostic": {}, "warnings": []},
        )
        monkeypatch.setattr(
            tools._operations, "_after_write_split",
            lambda memory_id: ({"required": False}, None, []),
        )
        monkeypatch.setattr(
            tools._operations, "_enqueue_semantic_conflict_check",
            lambda memory_id, record, result=status: {"status": result},
        )
        result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
        receipt = result["data"]["imported"][0]
        assert receipt["postprocess_status"] == "pending"
        assert receipt["postprocess_error_code"] == f"semantic_{status}"


def test_replay_async_split_records_enqueue_not_completion(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    _write_envelope(tools.settings.backup_jsonl, "async-split")
    monkeypatch.setattr(
        tools._operations, "_index_and_reconcile_claims",
        lambda memory_id: {"diagnostic": {}, "warnings": []},
    )
    monkeypatch.setattr(
        tools._operations, "_after_write_split",
        lambda memory_id: ({"reindex_pending": True, "mode": "rules_async"}, None, []),
    )
    monkeypatch.setattr(
        tools._operations, "_enqueue_semantic_conflict_check",
        lambda memory_id, record: {"status": "disabled"},
    )
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    receipt = result["data"]["imported"][0]
    assert receipt["postprocess_stages"]["split"] == "queued"
    assert receipt["postprocess_status"] == "complete"


def test_replay_schema_migrates_existing_receipts_as_complete(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "legacy.sqlite3",
        backup_jsonl=tmp_path / "legacy.jsonl",
        enable_sqlite_vec=False,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    _write_envelope(settings.backup_jsonl, "legacy-receipt")
    first = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    memory_id = first["data"]["imported"][0]["memory_id"]
    tools.shutdown(timeout=1)
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("ALTER TABLE backup_replay_log RENAME TO backup_replay_log_new")
        conn.execute(
            "CREATE TABLE backup_replay_log (replay_key TEXT PRIMARY KEY, memory_id INTEGER NOT NULL, "
            "payload_hash TEXT NOT NULL, replayed_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO backup_replay_log SELECT replay_key, memory_id, payload_hash, replayed_at "
            "FROM backup_replay_log_new"
        )
        conn.execute("DROP TABLE backup_replay_log_new")
        conn.commit()
    reopened = MemoryTools(settings=settings, db=MemoryDB(settings))
    repeated = reopened.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    receipt = repeated["data"]["already_replayed"][0]
    assert receipt["memory_id"] == memory_id
    assert receipt["postprocess_status"] == "complete"


def test_formal_replay_caps_page_at_200(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for index in range(201):
        _write_envelope(tools.settings.backup_jsonl, f"cap-{index}")
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True, "limit": 1000})
    assert result["data"]["processed"] == 200
    assert result["data"]["imported_count"] == 200
    assert result["data"]["remaining"] is True
    assert result["data"]["next_offset"] == 200


def test_backup_notice_scan_failure_emits_cached_degradation_notice(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    monkeypatch.setattr(tools.db.backup_replay, "state_signature", lambda: (_ for _ in ()).throw(OSError("unreadable")))
    first = tools._consume_notices()
    second = tools._consume_notices()
    assert first[0]["type"] == "backup_replay_notice_degraded"
    assert first[0]["severity"] == "warning"
    assert second == []


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
