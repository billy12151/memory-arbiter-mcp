"""0.16.6 write dedup gate e2e (owner spec: 只管活的 — ACTIVE-only partial
unique index, no pre-check on the happy path; the DB error IS the detector).

Pinned scenarios from the design review:
- transport replay (the m986/m987 incident species) → idempotent success
- cross-workspace same content → legal
- retired counterpart → rewrite lands (乙 semantics)
- edit INTO another active row's bytes → structured refusal
- pending activation colliding with an active twin → named error
- move into a workspace holding the same active content → refused with ids
- error-but-no-twin race → one retry commits cleanly
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.db import workspaces as workspaces_module

from test_scan_pipeline import make_tools, _write


@pytest.fixture(autouse=True)
def _no_fake_vector_workspace_match(monkeypatch):
    """make_tools embeds with a fake GGUF, so workspace-name vector matching
    is meaningless noise here — pin the distance beyond any match and keep
    workspace names exact."""
    monkeypatch.setattr(workspaces_module, "WORKSPACE_MATCH_DISTANCE", -1.0)


def _notice_types(res: dict) -> set[str]:
    return {str(n.get("type")) for n in res.get("notices", [])}


def _set_status(tools, memory_id: int, status: str) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status=? WHERE id=?", (status, memory_id))


def test_transport_replay_returns_same_id(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory_write(content="逐字节相同的闭环记录", subject="s", workspace="ws", tags=[])
    assert first["ok"]
    second = tools.memory_write(content="逐字节相同的闭环记录", subject="s", workspace="ws", tags=[])
    assert second["ok"], second
    assert second["data"]["duplicate_replay"] is True
    assert second["data"]["id"] == first["data"]["id"]
    assert "duplicate_replay" in _notice_types(second)
    with tools.db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content=?", ("逐字节相同的闭环记录",)
        ).fetchone()[0]
    assert count == 1, "replay must not create a second row"


def test_same_content_across_workspaces_is_legal(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "s", "shared bytes", workspace="memory-arbiter")
    b = _write(tools, "s", "shared bytes", workspace="金营项目")
    assert a != b
    res = tools.memory_write(content="shared bytes", subject="s", workspace="金营项目", tags=[])
    assert res["ok"] and res["data"]["id"] == b


def test_retired_counterpart_allows_rewrite(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "s", "version one", workspace="ws")
    _set_status(tools, a, "superseded")
    res = tools.memory_write(content="version one", subject="s", workspace="ws", tags=[])
    assert res["ok"], res
    assert "duplicate_replay" not in res["data"]
    assert res["data"]["id"] != a


def test_edit_into_active_twin_bytes_refused(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "A", "正文甲", workspace="ws")
    b = _write(tools, "B", "正文乙", workspace="ws")
    res = tools.memory(
        "update",
        {"memory_id": b, "new_content": "正文甲", "reason": "edit into twin"},
    )
    assert not res["ok"], res
    data = res["data"]
    assert data.get("outcome") == "duplicate_content"
    assert "byte-identical" in str(data.get("error", ""))
    # B unchanged
    with tools.db.connection() as conn:
        row = conn.execute("SELECT content FROM memories WHERE id=?", (b,)).fetchone()
    assert row["content"] == "正文乙"


def test_noop_edit_of_own_content_still_allowed(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "A", "稳定正文", workspace="ws")
    res = tools.memory(
        "update",
        {"memory_id": a, "new_content": "稳定正文", "reason": "subject-touching no-op"},
    )
    assert res["ok"], res


def test_pending_twin_activates_into_named_error(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "A", "激活前就存在的内容", workspace="ws")
    pending = tools.memory_write(
        content="激活前就存在的内容", subject="P", workspace="ws", tags=[], status="pending",
    )
    assert pending["ok"], "pending insert sits outside the ACTIVE-only index"
    res = tools.memory_repair(task="activate_pending", data={"id": pending["data"]["id"], "authorized": True})
    assert not res["ok"], res
    assert "duplicate_active_content" in str(res["data"].get("error", "")), res["data"]
    # the pending row is still pending — nothing half-applied
    with tools.db.connection() as conn:
        status = conn.execute(
            "SELECT status FROM memories WHERE id=?", (pending["data"]["id"],)
        ).fetchone()["status"]
    assert status == "pending"


def test_move_into_same_content_workspace_refused(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "s", "跨区同文", workspace="memory-arbiter")
    _write(tools, "s", "跨区同文", workspace="金营项目")
    res = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "金营项目",
        "reason": "relocate", "authorized": True,
    })
    assert not res["ok"], res
    reasons = [str(e.get("reason", "")) for e in res["data"].get("errors", [])]
    assert any("content duplicate collision" in reason for reason in reasons), res["data"]
    # moving the NON-duplicate sibling still works
    other = _write(tools, "s2", "唯一内容", workspace="memory-arbiter")
    ok = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [other], "new_workspace": "金营项目",
        "reason": "relocate", "authorized": True,
    })
    assert ok["ok"], ok


def test_race_window_retries_once_and_commits(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    real = tools.db.insert_memory
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.IntegrityError(
                "UNIQUE constraint failed: index 'idx_memories_content_sha'"
            )
        return real(*args, **kwargs)

    tools.db.insert_memory = flaky  # type: ignore[method-assign]
    try:
        res = tools.memory_write(content="竞态重试正文", subject="s", workspace="ws", tags=[])
    finally:
        tools.db.insert_memory = real  # type: ignore[method-assign]
    assert res["ok"], res
    assert calls["n"] == 2, "exactly one retry, never a loop"
    assert "duplicate_replay" not in res["data"]
    with tools.db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content=?", ("竞态重试正文",)
        ).fetchone()[0]
    assert count == 1


def test_backup_replay_duplicate_returns_duplicate_content(tmp_path: Path) -> None:
    """Round-1 review P0 pin: the backup-replay hook must actually reach its
    duplicate_content outcome (a missing `as exc` used to NameError here), and
    the outcome lands in the idempotent bucket, not a replay conflict."""
    tools = make_tools(tmp_path)
    _write(tools, "s", "备份里已有的正文", workspace="ws")
    entry = {
        "replay_key": "rk-dup-1",
        "payload_hash": "h1",
        "workspace_canonical": "ws",
        "record": {
            "content": "备份里已有的正文", "subject": "s", "agent_id": "a",
            "workspace": "ws", "tags": [], "source_type": "agent_generated",
            "source_ref": None, "event_time": "2026-01-01T00:00:00+00:00",
            "ingest_time": "2026-01-01T00:00:00+00:00", "confidence": 0.5,
            "protection_level": "normal", "status": "active",
            "metadata": {}, "version": 1,
        },
    }
    with open(tools.db.settings.backup_jsonl, "w", encoding="utf-8") as fh:
        fh.write('{"backup_schema": 1, "replay_key": "rk-dup-1", "payload_hash": "h1", '
                 '"backup_written_at": "2026-01-01T00:00:00+00:00", "workspace_canonical": "ws", '
                 '"record": ' + __import__("json").dumps(entry["record"], ensure_ascii=False) + '}\n')
    res = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert res["ok"], res
    data = res["data"]
    conflicts = [c for c in data.get("conflicts", []) if c.get("replay_key") == "rk-dup-1"]
    assert not conflicts, f"duplicate_content must not be a replay conflict: {conflicts}"
    idem = [r for r in data.get("already_replayed", []) if r.get("replay_key") == "rk-dup-1"]
    assert idem and idem[0].get("outcome") == "duplicate_content", data
    with tools.db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content=?", ("备份里已有的正文",)
        ).fetchone()[0]
    assert count == 1, "replay of a duplicate line must not insert"
